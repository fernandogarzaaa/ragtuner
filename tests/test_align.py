"""Cross-parser span projection. A second test parser ('flat') renders the same markdown
with different offsets (tables as space-separated reading order, everything one line per
block, a page header injected), the way pymupdf-text differs from docling."""

from pathlib import Path

from ragtuner.align import Projector, project_questions
from ragtuner.config import RagConfig
from ragtuner.evalgen import HeuristicGenerator
from ragtuner.parse import parse_file
from ragtuner.parse.base import register
from ragtuner.pipeline import Corpus
from ragtuner.schema import BlockType, ParsedDocument, RawBlock, assemble, hash_file
from ragtuner.search import Evaluator


class FlatParser:
    name = "flat"
    version = "1"
    suffixes = frozenset({".md"})

    def parse(self, path: Path, *, doc_id: str | None = None) -> ParsedDocument:
        src = parse_file(path, "plaintext", doc_id=doc_id)
        raw = [RawBlock(BlockType.PARAGRAPH, "ACME CORP ANNUAL REPORT page header")]
        for b in src.blocks:
            if b.table is not None:
                text = " ".join(" ".join(r) for r in b.table.rows)
            else:
                text = " ".join(b.text.split())
            raw.append(RawBlock(BlockType.PARAGRAPH, text))
        return assemble(doc_id=src.id, source=str(path), content_hash=hash_file(path), parser=self.name,
                        parser_version=self.version, raw_blocks=raw)


register("flat", __name__, "FlatParser")

MD = "# Report\n\n" + "\n\n".join(
    f"## Region {i}\n\nRegion {i} revenue grew because distributor{i} expanded into market{i}. "
    f"Management expects contract{i} renewals to continue through the next fiscal year, with risk{i} noted.\n\n"
    f"| metric | q1 | q2 |\n|---|---|---|\n| sales{i} | {i}10 | {i}20 |\n| units{i} | {i}30 | {i}40 |"
    for i in range(8))


def test_projector_exact_markdown_table_and_lost():
    doc = assemble(doc_id="d", source="x", content_hash="h", parser="p", parser_version="1",
                   raw_blocks=[RawBlock(BlockType.PARAGRAPH, "header noise. Revenue grew 10% in Q2 2023, driven by sales."),
                               RawBlock(BlockType.PARAGRAPH, "metric q1 q2 sales 110 120 units 130 140")])
    pr = Projector(doc)
    s, e, q = pr.project("Revenue grew 10% in Q2 2023")
    assert doc.text[s:e] == "Revenue grew 10% in Q2 2023" and q == 1.0
    s, e, q = pr.project("| metric | q1 | q2 |\n|---|---|---|\n| sales | 110 | 120 |\n| units | 130 | 140 |")
    assert doc.text[s:e] == "metric q1 q2 sales 110 120 units 130 140" and q == 1.0
    assert pr.project("completely different words about penguins migrating") is None


def test_evaluator_projects_spans_for_a_different_parser(tmp_path: Path):
    root = tmp_path / "c"
    root.mkdir()
    (root / "r.md").write_text(MD, encoding="utf-8")
    corpus = Corpus.from_dir("c", root)
    anchor_docs = [parse_file(f, "plaintext") for f in corpus.files]
    es = HeuristicGenerator(seed=0, table_fraction=0.4, unanswerable_fraction=0.0).generate(anchor_docs, 12)
    assert es.questions

    ev = Evaluator(corpus, cache_root=tmp_path / "cache", anchor_parser="plaintext")
    projected = ev.questions_for("flat", es.questions)
    flat = {d.id: d for d in ev.docs("flat")}
    anchor = {d.id: d for d in anchor_docs}
    for q, p in zip(es.questions, projected):
        for a, b in zip(q.spans, p.spans):
            assert b.length > 0, "every span survives a parser that only reformats"
            src_words = anchor[a.doc_id].text[a.start:a.end].replace("|", " ").replace("-", " ").split()
            assert flat[b.doc_id].text[b.start:b.end].split() == [w for w in src_words if w]
    st = ev.projection_stats["flat"]
    assert st["lost"] == 0 and st["spans"] == sum(len(q.spans) for q in es.questions)

    # with projection, a whole-document chunk covers every span; without it, offsets are meaningless
    cfg = RagConfig(parser="flat", chunker="fixed", chunk_params={"size": 1000, "overlap": 0}, embedder=None, retrieval="sparse")
    with_proj = ev.evaluate(cfg, es.questions)
    assert with_proj.metrics["hit@50"] == 1.0
    ev_raw = Evaluator(corpus, cache_root=tmp_path / "cache", anchor_parser=None)
    shifted = [q for q in es.questions if q.spans[0].start > 0]
    raw = ev_raw.evaluate(RagConfig(parser="flat", chunker="fixed", chunk_params={"size": 40, "overlap": 0}, embedder=None, retrieval="sparse"), shifted)
    proj = ev.evaluate(RagConfig(parser="flat", chunker="fixed", chunk_params={"size": 40, "overlap": 0}, embedder=None, retrieval="sparse"), shifted)
    assert proj.metrics["recall@5"] > raw.metrics["recall@5"]


def test_lost_spans_are_parse_failures():
    a = assemble(doc_id="d", source="x", content_hash="h", parser="p", parser_version="1",
                 raw_blocks=[RawBlock(BlockType.TABLE, "| alpha | 1 |\n|---|---|\n| beta | 2 |")])
    b = assemble(doc_id="d", source="x", content_hash="h", parser="q", parser_version="1",
                 raw_blocks=[RawBlock(BlockType.PARAGRAPH, "Only prose survived this parser, the table is gone entirely.")])
    from ragtuner.chunk import get_chunker
    from ragtuner.diagnose import stage_of
    from ragtuner.evalset import EvalQuestion, GoldSpan
    from ragtuner.metrics import score_question

    q = EvalQuestion("q", "beta?", (GoldSpan("d", 0, len(a.text), block_type="table"),))
    (pq,), st = project_questions([q], {"d": a}, {"d": b})
    assert st["lost"] == 1 and pq.spans[0].length == 0 and pq.spans[0].block_type == "table"
    chunks = get_chunker("fixed", size=50, overlap=0).chunk(b)
    assert stage_of(pq, score_question(pq, []), 5, {"d": chunks}) == "parse"
