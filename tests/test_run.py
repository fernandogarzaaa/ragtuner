import json
from pathlib import Path

from ragtuner.config import RagConfig
from ragtuner.run import load_corpus, resolve_default, space_preset


def test_load_corpus_reads_doc_id_mapping(tmp_path: Path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.md").write_text("# A\n\ntext")
    (tmp_path / "doc_ids.json").write_text(json.dumps({"a.md": "ACME_2024_10K"}))
    c = load_corpus("t", tmp_path)
    assert len(c.files) == 1 and c.doc_id_for(c.files[0]) == "ACME_2024_10K"


def test_space_presets_only_use_available_backends():
    s = space_preset("smoke", parsers=["plaintext"])
    assert s.parsers == ["plaintext"] and None in s.embedders
    p = space_preset("phase1", parsers=["plaintext"])
    assert all(cfg.parser == "plaintext" for cfg in p.stage1())


def test_resolve_default_falls_back_when_embedder_missing(monkeypatch):
    cfg = RagConfig(parser="plaintext", embedder="openai-small", retrieval="dense")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    resolved, notes = resolve_default(cfg)
    assert resolved.embedder == "bge-small" and notes and "unavailable" in notes[0]
