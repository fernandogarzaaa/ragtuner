from pathlib import Path

import pytest

from ragtuner.config import RagConfig
from ragtuner.pipeline import Corpus, build_index

DOCS = {
    "refunds.md": "# Refunds\n\nCustomers may return items within 30 days for a full refund.\n\n# Shipping\n\nOrders ship in two business days.",
    "pets.md": "# Pets\n\nDogs and cats are welcome in the office on Fridays.",
}


@pytest.fixture
def corpus(tmp_path: Path) -> Corpus:
    root = tmp_path / "corpus"
    root.mkdir()
    for name, text in DOCS.items():
        (root / name).write_text(text, encoding="utf-8")
    return Corpus.from_dir("t", root)


@pytest.mark.parametrize("retrieval", ["sparse", "dense", "hybrid", "better-rag"])
def test_each_retrieval_mode_finds_the_refund_chunk(corpus, tmp_path, retrieval):
    cfg = RagConfig(parser="plaintext", chunker="structure", chunk_params={"target_words": 50, "max_words": 100},
                    embedder="hash-64" if retrieval in ("dense", "hybrid") else None, retrieval=retrieval, top_k=2)
    index, stats = build_index(corpus, cfg, cache_root=tmp_path / "cache")
    hits = index.retrieve("refund within 30 days")
    assert hits and hits[0].doc_id.startswith("refunds-")
    assert "refund" in hits[0].text
    assert stats.docs == 2 and stats.chunks >= 3


def test_config_key_is_stable_and_yaml_round_trips(tmp_path):
    a = RagConfig(chunker="fixed", chunk_params={"size": 120, "overlap": 24}, embedder=None, retrieval="sparse")
    b = RagConfig.from_dict(a.to_dict())
    assert a.key == b.key
    a.to_yaml(tmp_path / "c.yaml")
    assert RagConfig.from_yaml(tmp_path / "c.yaml") == a
    with pytest.raises(ValueError):
        RagConfig(embedder=None, retrieval="dense").validate()


def test_caches_make_second_build_free(corpus, tmp_path):
    cfg = RagConfig(parser="plaintext", chunker="recursive", chunk_params={"size": 40, "overlap": 5}, embedder="hash-64", retrieval="dense")
    _, s1 = build_index(corpus, cfg, cache_root=tmp_path / "cache")
    _, s2 = build_index(corpus, cfg, cache_root=tmp_path / "cache")
    assert s1.embedded_texts > 0 and s2.embedded_texts == 0
    assert s2.parse_cache_hits == 2
