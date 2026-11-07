from pathlib import Path

import pytest

from ragtuner.schema import BlockType, ParsedDocument, RawBlock, Table, assemble


def test_assemble_computes_exact_offsets_and_drops_empty_blocks():
    doc = assemble(
        doc_id="d",
        source="x",
        content_hash="h",
        parser="p",
        parser_version="1",
        raw_blocks=[
            RawBlock(BlockType.HEADING, "  Title  ", level=1),
            RawBlock(BlockType.PARAGRAPH, ""),
            RawBlock(BlockType.PARAGRAPH, "Body text."),
        ],
    )
    assert [b.type for b in doc.blocks] == [BlockType.HEADING, BlockType.PARAGRAPH]
    assert doc.text == "Title\n\nBody text."
    for b in doc.blocks:
        assert doc.text[b.start : b.end] == b.text
    assert doc.blocks[0].id == "d:b0" and doc.blocks[1].id == "d:b1"


def test_validate_rejects_mismatched_span():
    doc = assemble(doc_id="d", source="x", content_hash="h", parser="p", parser_version="1",
                   raw_blocks=[RawBlock(BlockType.PARAGRAPH, "abc")])
    bad = ParsedDocument(**{**doc.__dict__, "text": "xyz"})
    with pytest.raises(ValueError):
        bad.validate()


def test_round_trip_through_json(tmp_path: Path):
    table = Table(rows=[["a", "b"], ["1", "2"]], header_rows=1, caption="T1")
    doc = assemble(doc_id="d", source="x", content_hash="h", parser="p", parser_version="1",
                   raw_blocks=[RawBlock(BlockType.TABLE, table.to_markdown(), page=3, bbox=(0, 0, 1, 1), table=table)])
    path = tmp_path / "doc.json"
    doc.save(path)
    back = ParsedDocument.load(path)
    assert back == doc
    assert back.tables()[0].cells() == ["a", "b", "1", "2"]
    assert back.blocks[0].bbox == (0, 0, 1, 1)


def test_table_markdown_pads_ragged_rows():
    md = Table(rows=[["h1", "h2", "h3"], ["x"]], header_rows=1).to_markdown()
    assert md.splitlines()[0] == "| h1 | h2 | h3 |"
    assert md.splitlines()[2] == "| x |  |  |"
