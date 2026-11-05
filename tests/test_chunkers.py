from pathlib import Path

import pytest

from ragtuner.chunk import get_chunker
from ragtuner.parse import parse_file
from ragtuner.schema import BlockType, RawBlock, Table, assemble

MD = "# Title\n\n" + "\n\n".join(f"Paragraph {i}. " + " ".join(f"w{i}_{j}" for j in range(40)) + "." for i in range(10)) + \
     "\n\n## Data\n\n| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n\nTail paragraph.\n"


@pytest.fixture
def doc(tmp_path: Path):
    p = tmp_path / "d.md"
    p.write_text(MD, encoding="utf-8")
    return parse_file(p)


@pytest.mark.parametrize("name,params", [
    ("fixed", {"size": 60, "overlap": 10}),
    ("recursive", {"size": 80, "overlap": 15}),
    ("structure", {"target_words": 100, "max_words": 200}),
])
def test_chunks_map_back_to_document_text(doc, name, params):
    chunks = get_chunker(name, **params).chunk(doc)
    assert chunks
    for c in chunks:
        assert doc.text[c.start:c.end] == c.text
        assert c.words <= max(params.get("size", 0), params.get("max_words", 0)) + 5
    # full coverage: every non-whitespace char of the doc is inside some chunk
    covered = set()
    for c in chunks:
        covered.update(range(c.start, c.end))
    # structure turns headings into index prefixes, not chunks, so their chars are exempt
    exempt = set()
    if name == "structure":
        for b in doc.blocks_of(BlockType.HEADING):
            exempt.update(range(b.start, b.end))
    missing = [i for i, ch in enumerate(doc.text) if not ch.isspace() and i not in covered and i not in exempt]
    assert not missing, f"{name} left {len(missing)} chars uncovered"


def test_fixed_matches_better_rag_window_semantics(doc):
    chunks = get_chunker("fixed", size=50, overlap=10).chunk(doc)
    assert chunks[0].words == 50
    # consecutive windows advance by size - overlap words
    first_words = chunks[0].text.split()
    second_words = chunks[1].text.split()
    assert second_words[:10] == first_words[40:50]


def test_recursive_never_splits_inside_a_sentence_when_it_fits(doc):
    chunks = get_chunker("recursive", size=80, overlap=0).chunk(doc)
    for c in chunks:
        assert c.text.rstrip().endswith((".", "|")) or c.text.endswith("Title") or c.text.endswith("Data")


def test_structure_keeps_table_whole_and_prefixes_heading_path(doc):
    chunks = get_chunker("structure", target_words=100, max_words=200).chunk(doc)
    table_chunks = [c for c in chunks if "table" in c.block_types]
    assert len(table_chunks) == 1
    tc = table_chunks[0]
    assert tc.block_types == ("table",)
    assert tc.index_text.startswith("Title > Data\n")
    assert tc.text.startswith("| a | b |")
    # headings never become chunks of their own
    assert all("heading" not in c.block_types for c in chunks)


def test_structure_splits_big_table_and_repeats_header():
    rows = [["item", "value"]] + [[f"row{i}", str(i)] for i in range(400)]
    t = Table(rows=rows, header_rows=1)
    doc = assemble(doc_id="t", source="x", content_hash="h", parser="p", parser_version="1",
                   raw_blocks=[RawBlock(BlockType.HEADING, "Big", level=1), RawBlock(BlockType.TABLE, t.to_markdown(), table=t)])
    chunks = get_chunker("structure", target_words=100, max_words=150).chunk(doc)
    assert len(chunks) > 3
    for c in chunks:
        assert c.metadata.get("table_piece")
        assert "| item | value |" in c.index_text
        assert doc.text[c.start:c.end] == c.text
    assert "| row399 | 399 |" in chunks[-1].text
