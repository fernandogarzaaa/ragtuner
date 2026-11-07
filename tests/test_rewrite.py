from pathlib import Path

from ragtuner.config import RagConfig
from ragtuner.index.bm25 import BM25Index
from ragtuner.pipeline import Corpus, build_index
from ragtuner.rewrite import LLMRewriter, prf_expand

TEXTS = [
    "Capital expenditures: purchases of property, plant and equipment were 1,577 million in 2018.",
    "Purchases of property plant equipment rose as the company expanded its plants.",
    "The dividend policy was unchanged; dividends per share rose to 5.44 dollars.",
]


def test_prf_adds_feedback_terms_not_in_query():
    bm25 = BM25Index(TEXTS)
    q = prf_expand("capital expenditures", bm25, TEXTS)
    assert q.startswith("capital expenditures ")
    extra = q.split()[2:]
    assert extra and "purchas" in " ".join(extra)


def test_llm_rewriter_caches_and_falls_back():
    calls = []

    def fake(system, user):
        calls.append(user)
        if "boom" in user:
            raise RuntimeError("down")
        return "purchases of property, plant and equipment"

    rw = LLMRewriter(caller=fake)
    a = rw("What was 3M capex in 2018?")
    b = rw("What was 3M capex in 2018?")
    assert a == b and len(calls) == 1 and "property, plant" in a and a.startswith("What was 3M capex")
    assert rw("boom") == "boom"


def test_rewrite_config_key_and_pipeline(tmp_path: Path):
    base = RagConfig(parser="plaintext", embedder=None, retrieval="sparse")
    assert "qr-" not in base.key
    prf = RagConfig(parser="plaintext", embedder=None, retrieval="sparse", query_rewrite="prf")
    assert "|qr-prf|" in prf.key and prf.key != base.key
    root = tmp_path / "c"
    root.mkdir()
    for i, t in enumerate(TEXTS):
        (root / f"d{i}.md").write_text(t, encoding="utf-8")
    idx, _ = build_index(Corpus.from_dir("c", root), prf, cache_root=tmp_path / "cache")
    assert idx.rewrite("capital expenditures") != "capital expenditures"
    hits = idx.retrieve("capital expenditures", k=2)
    assert hits and hits[0].doc_id.startswith("d0")
