"""Standalone deployment bundle for a tuned config, with an over-the-wire verification.

The bundle holds the exported config and manifest, pinned requirements for exactly the
backends the config uses, a Dockerfile serving the HTTP API, an MCP client entry, and the
eval set plus the tuner's recorded hits. `verify_bundle` starts the API from the bundle
and checks that every eval question returns the same top chunk ids the tuner recorded,
so a deployment can be proven to be the configuration that was evaluated.
"""

from __future__ import annotations

import json
import shutil
import threading
import urllib.request
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from ragtuner.config import RagConfig

_DOCKERFILE = """# Override the base when Docker Hub rate-limits you, e.g.
#   --build-arg BASE_IMAGE=public.ecr.aws/docker/library/python:3.12-slim
ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app/bundle
# Mount the corpus at /corpus (and a cache volume at /cache to keep re-indexing incremental)
EXPOSE 8080
CMD ["ragtuner", "serve", "/corpus", "--config", "/app/bundle/best_config.yaml", "--host", "0.0.0.0", "--port", "8080", "--cache", "/cache", "--query-log", "/cache/queries.jsonl"]
"""


def _pin(pkg: str) -> str | None:
    try:
        return f"{pkg}=={version(pkg)}"
    except PackageNotFoundError:
        return None


def requirements_for(config: RagConfig, commit: str | None) -> list[str]:
    # an archive URL pinned to the tuning commit: reproducible, and needs no git in the image
    src = f"https://github.com/fernandogarzaaa/ragtuner/archive/{commit or 'main'}.tar.gz"
    reqs = [f"ragtuner @ {src}"]
    pkgs = ["pymupdf", "numpy", "pyyaml"]
    if config.parser == "docling":
        pkgs += ["docling", "torch", "torchvision"]
    if config.embedder == "bge-small" or config.reranker:
        pkgs += ["sentence-transformers", "torch"]
    for p in dict.fromkeys(pkgs):
        pin = _pin(p)
        if pin:
            reqs.append(pin)
    if any("+cpu" in r for r in reqs):
        # CPU-only PyTorch builds live on PyTorch's own index, not on PyPI
        reqs.insert(0, "--extra-index-url https://download.pytorch.org/whl/cpu")
    return reqs


def build_bundle(run_dir: str | Path, out_dir: str | Path) -> dict[str, Any]:
    run, out = Path(run_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = RagConfig.from_yaml(run / "best_config.yaml")
    manifest = json.loads((run / "manifest.json").read_text()) if (run / "manifest.json").exists() else {}
    for name in ("best_config.yaml", "manifest.json", "eval_set.jsonl", "winner_results.json", "report.md"):
        if (run / name).exists():
            shutil.copy(run / name, out / name)
    reqs = requirements_for(cfg, manifest.get("ragtuner_commit"))
    (out / "requirements.txt").write_text("\n".join(reqs) + "\n")
    (out / "Dockerfile").write_text(_DOCKERFILE)
    (out / "mcp.json").write_text(json.dumps({"mcpServers": {"ragtuner": {
        "command": "ragtuner", "args": ["mcp", "/path/to/corpus", "--config", str((out / "best_config.yaml").resolve())]}}}, indent=2) + "\n")
    (out / "README.md").write_text(f"""# ragtuner bundle: {cfg.key}

The configuration below was selected by `ragtuner tune` (see report.md) and is pinned here
with the exact package versions that evaluated it (requirements.txt, manifest.json).

## Run the API

```bash
pip install -r requirements.txt
ragtuner serve /path/to/corpus --config best_config.yaml --port 8080 --query-log queries.jsonl
```

Or with Docker: `docker build -t rag . && docker run -p 8080:8080 -v /path/to/corpus:/corpus -v rag-cache:/cache rag`.

Endpoints: `POST /search`, `POST /ask`, `GET /manifest`, `POST /admin/reload`, chat widget at `/widget.js`.
Permission filtering reads `X-Tenant` and `X-Principal`; put an authenticating gateway in front.

## MCP

Add the entry in mcp.json to your client's MCP config, with the corpus path filled in.

## Verify this deployment is the evaluated config

```bash
ragtuner export --run . --corpus /path/to/corpus --out /tmp/check --verify
```

## Monitor

```bash
ragtuner monitor --log queries.jsonl --run . --corpus /path/to/corpus
```
""")
    return {"bundle": str(out), "config": cfg.key, "requirements": reqs}


def verify_bundle(bundle: str | Path, corpus_dir: str | Path | None = None, *, cache_root: str | Path = ".ragtuner-cache", k: int = 10,
                  url: str | None = None) -> dict[str, Any]:
    """Replay the bundle's eval set over HTTP and compare top-k chunk ids with the tuner's
    record. With `url`, check a running deployment (a container, a server elsewhere);
    otherwise serve the bundle locally on an ephemeral port."""
    from ragtuner.evalset import EvalSet
    from ragtuner.serve.http import make_server
    from ragtuner.serve.runtime import Runtime

    b = Path(bundle)
    recorded = {r["question_id"]: [h["chunk_id"] for h in r["hits"]][:k] for r in json.loads((b / "winner_results.json").read_text())}
    es = EvalSet.load(b / "eval_set.jsonl")
    server = None
    if url:
        base = url.rstrip("/")
        with urllib.request.urlopen(f"{base}/health", timeout=60) as resp:
            served = json.load(resp)["config"]
    else:
        if corpus_dir is None:
            raise ValueError("need corpus_dir or url")
        rt = Runtime(corpus_dir, b / "best_config.yaml", cache_root=cache_root)
        server = make_server(rt, port=0)
        base = f"http://127.0.0.1:{server.server_address[1]}"
        threading.Thread(target=server.serve_forever, daemon=True).start()
        served = rt.config.key
    checked, mismatched = 0, []
    try:
        for q in es.questions:
            if q.id not in recorded:
                continue
            req = urllib.request.Request(f"{base}/search", data=json.dumps({"query": q.question, "k": k}).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                got = [h["chunk_id"] for h in json.load(resp)["hits"]]
            checked += 1
            # the full recorded top-k, length included: an empty or partial index must not pass
            if got != recorded[q.id][:k]:
                mismatched.append(q.id)
    finally:
        if server is not None:
            server.shutdown()
    return {"config": served, "checked": checked, "mismatched": len(mismatched), "identical": checked > 0 and not mismatched,
            "mismatch_ids": mismatched[:10]}
