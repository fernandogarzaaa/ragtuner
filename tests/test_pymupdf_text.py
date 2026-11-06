from pathlib import Path

from ragtuner.cache import ParseCache
from ragtuner.parse import parse_file
from ragtuner.schema import BlockType


def test_headings_pages_and_text_only(sample_pdf: Path):
    doc = parse_file(sample_pdf, "pymupdf-text")
    doc.validate()
    assert doc.pages == 2
    headings = [b.text for b in doc.blocks_of(BlockType.HEADING)]
    assert "Annual Report 2025" in headings
    assert "Outlook" in headings
    assert doc.blocks_of(BlockType.LIST), "bullet lines should be classified as list blocks"
    # the text-only arm must not emit structured tables, but the table text must survive
    assert doc.tables() == []
    assert "North" in doc.text and "140" in doc.text
    pages = {b.page for b in doc.blocks}
    assert pages == {1, 2}


def test_parse_cache_hits_on_same_bytes(sample_pdf: Path, tmp_path: Path):
    cache = ParseCache(tmp_path / "cache")
    doc1, hit1 = cache.parse(sample_pdf, "pymupdf-text")
    doc2, hit2 = cache.parse(sample_pdf, "pymupdf-text")
    assert (hit1, hit2) == (False, True)
    assert doc1 == doc2


def test_parse_cache_returns_requested_id_on_hit(sample_pdf, tmp_path):
    cache = ParseCache(tmp_path / "cache")
    first, _ = cache.parse(sample_pdf, "pymupdf-text")  # default id
    named, hit = cache.parse(sample_pdf, "pymupdf-text", doc_id="ACME_2024_10K")
    assert hit and named.id == "ACME_2024_10K" and named.blocks[0].id == "ACME_2024_10K:b0"
    named.validate()
    again, hit2 = cache.parse(sample_pdf, "pymupdf-text")
    assert hit2 and again.id == first.id
