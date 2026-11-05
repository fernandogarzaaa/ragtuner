import pytest

pytest.importorskip("docling", reason="docling optional dependency not installed")


def test_docling_parses_sample_pdf(sample_pdf):
    from ragtuner.parse import parse_file

    doc = parse_file(sample_pdf, "docling")
    doc.validate()
    assert doc.pages == 2
    assert doc.tables(), "docling should extract the ruled table"
