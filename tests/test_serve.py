"""Serving surfaces end to end: HTTP API (search, ask, ACL headers, reload, widget),
MCP over stdio in a subprocess, incremental sync, export bundle verified over HTTP, and
monitoring on a real query log."""

import json
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from ragtuner.config import RagConfig
from ragtuner.evalgen import HeuristicGenerator
from ragtuner.export import build_bundle, verify_bundle
from ragtuner.monitor import monitor
from ragtuner.parse import parse_file
from ragtuner.run import load_corpus, tune
from ragtuner.search import Space
from ragtuner.serve.http import make_server
from ragtuner.serve.mcp import MCPServer
from ragtuner.serve.runtime import Runtime

CFG = RagConfig(parser="plaintext", chunker="structure", chunk_params={"target_words": 60, "max_words": 120}, embedder=None, retrieval="sparse", top_k=3)


def make_corpus(root: Path, acl: bool = False) -> Path:
    (root / "docs").mkdir(parents=True)
    for i in range(3):
        (root / "docs" / f"guide{i}.md").write_text("# Guide\n\n" + "\n\n".join(
            f"## Topic {i}{j}\n\nForm F{i}{j} must be filed within {j + 10} days by every applicant{i}{j}. "
            f"Late filing of form F{i}{j} costs a penalty of {j * 25} dollars, waived for first offenses." for j in range(4)), encoding="utf-8")
    if acl:
        (root / "acl.json").write_text(json.dumps({"guide0.md": {"tenant": "a", "principals": ["alice"]},
                                                    "guide1.md": {"tenant": "a", "principals": []},
                                                    "guide2.md": {"tenant": "b", "principals": []}}))
    return root


def post(port, path, body, headers=None):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


@pytest.fixture
def server(tmp_path):
    root = make_corpus(tmp_path / "c", acl=True)
    rt = Runtime(root, CFG, cache_root=tmp_path / "cache", query_log=tmp_path / "q.jsonl")
    srv = make_server(rt, port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield rt, srv.server_address[1], tmp_path
    srv.shutdown()


def test_http_search_ask_acl_reload_widget(server):
    rt, port, tmp = server
    assert json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/health"))["status"] == "ok"
    alice = post(port, "/search", {"query": "form F01 penalty", "k": 3}, {"X-Tenant": "a", "X-Principal": "alice"})["hits"]
    assert alice and alice[0]["doc_id"].startswith("guide0")
    anon = post(port, "/search", {"query": "form F01 penalty", "k": 5}, {"X-Tenant": "a"})["hits"]
    assert anon and all(h["doc_id"].startswith("guide1") for h in anon)
    other = post(port, "/search", {"query": "form F01 penalty", "k": 5}, {"X-Tenant": "b", "X-Principal": "alice"})["hits"]
    assert all(h["doc_id"].startswith("guide2") for h in other)
    ans = post(port, "/ask", {"query": "What is the penalty for late filing of form F12?"}, {"X-Tenant": "a"})
    assert ans["mode"] == "extractive" and "[1]" in ans["answer"] and ans["citations"]
    with pytest.raises(urllib.error.HTTPError) as e:
        post(port, "/search", {"query": "x", "k": 0})
    assert e.value.code == 400
    js = urllib.request.urlopen(f"http://127.0.0.1:{port}/widget.js").read().decode()
    assert "/ask" in js and "aria-live" in js
    log = [json.loads(line) for line in (tmp / "q.jsonl").read_text().splitlines()]
    assert len(log) == 4 and log[0]["tenant"] == "a" and log[0]["top_score"] is not None
    # incremental reload: edit one file, add one, only those are parsed
    (tmp / "c" / "docs" / "guide1.md").write_text("# Guide\n\nForm Z99 is new and replaces everything.", encoding="utf-8")
    (tmp / "c" / "docs" / "guide3.md").write_text("# Extra\n\nA brand new document about form Q7 deadlines.", encoding="utf-8")
    r = post(port, "/admin/reload", {})
    assert r["changed"] and r["added"] and r["parsed"] == 2 and r["parse_cache_hits"] == 2
    # fail closed: the new file has no acl.json entry, so tenant-scoped callers must not see it
    assert all(not h["doc_id"].startswith("guide3") for h in post(port, "/search", {"query": "form Q7 deadlines", "k": 5}, {"X-Tenant": "a"})["hits"])


def test_mcp_stdio_subprocess(tmp_path):
    root = make_corpus(tmp_path / "c")
    cfg = tmp_path / "cfg.yaml"
    CFG.to_yaml(cfg)
    proc = subprocess.Popen([sys.executable, "-m", "ragtuner", "mcp", str(root), "--config", str(cfg), "--cache", str(tmp_path / "cache")],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    msgs = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "search", "arguments": {"query": "form F21 penalty", "k": 2}}},
        {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "ask", "arguments": {"query": "When must form F03 be filed?"}}},
        {"jsonrpc": "2.0", "id": 5, "method": "nope"},
    ]
    out, err = proc.communicate("".join(json.dumps(m) + "\n" for m in msgs), timeout=120)
    replies = {r["id"]: r for r in (json.loads(line) for line in out.splitlines() if line.strip())}
    assert set(replies) == {1, 2, 3, 4, 5}, err[-2000:]
    assert replies[1]["result"]["serverInfo"]["name"] == "ragtuner"
    assert {t["name"] for t in replies[2]["result"]["tools"]} == {"search", "ask"}
    hits = replies[3]["result"]["structuredContent"]["hits"]
    assert len(hits) == 2 and hits[0]["doc_id"].startswith("guide2")
    assert "Sources:" in replies[4]["result"]["content"][0]["text"]
    assert replies[5]["error"]["code"] == -32601


def test_mcp_handle_empty_query_is_tool_error():
    srv = MCPServer(lambda: None)
    r = srv.handle({"jsonrpc": "2.0", "id": 9, "method": "tools/call", "params": {"name": "search", "arguments": {"query": " "}}})
    assert r["result"]["isError"] is True


def test_sync_cli_reports_incremental_work(tmp_path):
    from ragtuner.cli import main

    root = make_corpus(tmp_path / "c")
    cfg = tmp_path / "cfg.yaml"
    CFG.to_yaml(cfg)
    st = tmp_path / "state.json"
    args = ["sync", str(root), "--config", str(cfg), "--cache", str(tmp_path / "cache"), "--state", str(st)]
    assert main(args) == 0
    first = json.loads(st.read_text())
    assert len(first) == 3
    (root / "docs" / "guide2.md").write_text("# Changed\n\nNew content for guide two.", encoding="utf-8")
    rt = Runtime(root, CFG, cache_root=tmp_path / "cache", previous_state=first)
    assert rt.last_reload["changed"] == [str(root / "docs" / "guide2.md")] and rt.last_reload["parsed"] == 1


def test_export_bundle_verifies_over_http_and_monitor(tmp_path):
    root = make_corpus(tmp_path / "c")
    corpus = load_corpus("c", root)
    docs = [parse_file(f) for f in corpus.files]
    es = HeuristicGenerator(seed=0, unanswerable_fraction=0.0).generate(docs, 12)
    es.metadata = {"parser": "plaintext"}
    space = Space(parsers=["plaintext"], chunkers=[("fixed", {"size": 40, "overlap": 5}), ("structure", {"target_words": 60, "max_words": 120})],
                  embedders=[None], retrievals=["sparse"], rerankers=[None])
    run = tmp_path / "run"
    tune(corpus, es, out=run, cache_root=tmp_path / "cache", space=space, default=CFG)
    assert json.loads((run / "manifest.json").read_text())["corpus_state"]
    info = build_bundle(run, tmp_path / "bundle")
    assert (tmp_path / "bundle" / "Dockerfile").exists() and any(r.startswith("ragtuner @") for r in info["requirements"])
    v = verify_bundle(tmp_path / "bundle", root, cache_root=tmp_path / "cache")
    assert v["identical"] and v["checked"] == len(es.questions)
    # monitor: a log of off-topic queries drifts and flags low confidence; a changed corpus is stale
    rt = Runtime(root, RagConfig.from_yaml(run / "best_config.yaml"), cache_root=tmp_path / "cache", query_log=tmp_path / "q.jsonl")
    for q in ["weather forecast tomorrow", "football scores", "form", "applicant penalty dollars waived"] * 5:
        try:
            rt.search(q)
        except ValueError:
            pass
    (root / "docs" / "guide0.md").write_text("# Gone\n\nReplaced.", encoding="utf-8")
    m = monitor(tmp_path / "q.jsonl", run, corpus_dir=root, candidates_out=tmp_path / "cand.jsonl")
    signals = {p["signal"] for p in m["proposals"]}
    assert "stale_index" in signals and m["freshness"]["changed"] == 1
    assert m["queries"] == 20 and ("low_confidence" in signals or "empty" in signals)


def test_extractive_answer_strips_table_markup():
    from ragtuner.pipeline import Hit

    h = Hit("c", "d", 0, 10, 1.0, 1, "| Item | Fee |\n|---|---|\n| Restocking fee for returns | $5 |\n\nThe restocking fee applies to opened items.")
    ans = Runtime._extractive("What is the restocking fee?", [h])
    assert "|" not in ans and "---" not in ans and "restocking fee" in ans.lower()


def test_dense_config_serves_from_worker_threads(tmp_path):
    """Regression: the SQLite embedding cache was bound to the thread that built the index,
    so every HTTP request for a dense or hybrid config failed in a worker thread."""
    root = make_corpus(tmp_path / "c")
    cfg = RagConfig(parser="plaintext", chunker="structure", chunk_params={"target_words": 60, "max_words": 120},
                    embedder="hash-64", retrieval="hybrid", top_k=3)
    rt = Runtime(root, cfg, cache_root=tmp_path / "cache")
    srv = make_server(rt, port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        results = []
        workers = [threading.Thread(target=lambda: results.append(post(srv.server_address[1], "/search", {"query": f"form F{i}1 penalty"})))
                   for i in range(3)]
        for w in workers:
            w.start()
        for w in workers:
            w.join()
        assert len(results) == 3 and all(r["hits"] for r in results)
    finally:
        srv.shutdown()


def test_verify_rejects_a_deployment_that_returns_nothing(tmp_path):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    root = make_corpus(tmp_path / "c")
    corpus = load_corpus("c", root)
    es = HeuristicGenerator(seed=0, unanswerable_fraction=0.0).generate([parse_file(f) for f in corpus.files], 6)
    es.metadata = {"parser": "plaintext"}
    space = Space(parsers=["plaintext"], chunkers=[("structure", {"target_words": 60, "max_words": 120})], embedders=[None],
                  retrievals=["sparse"], rerankers=[None])
    tune(corpus, es, out=tmp_path / "run", cache_root=tmp_path / "cache", space=space, default=CFG)
    build_bundle(tmp_path / "run", tmp_path / "bundle")

    class Empty(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _reply(self, obj):
            body = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._reply({"status": "ok", "config": "x"})

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            self._reply({"hits": []})

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Empty)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        v = verify_bundle(tmp_path / "bundle", url=f"http://127.0.0.1:{srv.server_address[1]}")
    finally:
        srv.shutdown()
    assert v["identical"] is False and v["mismatched"] == v["checked"] > 0
