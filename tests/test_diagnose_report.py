from pathlib import Path

from ragtuner.chunk import get_chunker
from ragtuner.config import RagConfig
from ragtuner.diagnose import analyze, stage_of
from ragtuner.evalgen import HeuristicGenerator
from ragtuner.evalset import EvalQuestion, EvalSet, GoldSpan
from ragtuner.metrics import score_question
from ragtuner.parse import parse_file
from ragtuner.pipeline import Corpus, Hit, build_index
from ragtuner.report import write_report
from ragtuner.schema import BlockType, RawBlock, Table, assemble
from ragtuner.search import Evaluator, Space, staged_search


def _doc_with_table():
    t = Table(rows=[["item", "value"]] + [[f"row{i}", str(i)] for i in range(10)], header_rows=1)
    raw = [RawBlock(BlockType.HEADING, "H", level=1),
           RawBlock(BlockType.PARAGRAPH, "A long paragraph about alpha and beta values " * 6),
           RawBlock(BlockType.TABLE, t.to_markdown(), table=t)]
    return assemble(doc_id="d", source="x", content_hash="h", parser="p", parser_version="1", raw_blocks=raw)


def test_stage_labels_parse_chunk_deep_rank():
    doc = _doc_with_table()
    chunks = get_chunker("fixed", size=10, overlap=0).chunk(doc)
    by_doc = {"d": chunks}
    table_block = doc.blocks[2]
    q = EvalQuestion("q", "?", (GoldSpan("d", table_block.start, table_block.end, block_type="table"),))
    # chunk: the table is ~ 50 words; no fixed-10 window covers half of it
    r = score_question(q, [])
    assert stage_of(q, r, 5, by_doc) == "chunk"
    big = get_chunker("structure", target_words=100, max_words=200).chunk(doc)
    assert stage_of(q, r, 5, {"d": big}) == "deep"
    # rank: covered at rank 7
    cov = next(c for c in big if "table" in c.block_types)
    hits = [Hit(f"x{i}", "d", 0, 5, 1.0, i, "") for i in range(1, 7)] + [Hit(cov.id, "d", cov.start, cov.end, 0.5, 7, "")]
    assert stage_of(q, score_question(q, hits), 5, {"d": big}) == "rank"
    # parse: the gold span's doc has no chunks at all (parser lost it)
    assert stage_of(q, r, 5, {}) == "parse"
    # hit -> None
    assert stage_of(q, score_question(q, [Hit(cov.id, "d", cov.start, cov.end, 1.0, 1, "")]), 5, {"d": big}) is None


def test_fault_injection_attributes_table_misses(tmp_path: Path):
    """Destroy tables in the index (drop every table chunk) and check the analyzer blames
    table-sourced questions at the parse stage."""
    md = "# T\n\n" + "\n\n".join(
        f"## Sec {i}\n\nParagraph {i} describes gadget{i} procedures and form G{i} with several details about deadlines and fees for gadget{i}. Applicants file the form at the regional office, keep a stamped copy, and wait for written confirmation before shipping any gadget{i} units.\n\n"
        f"| name | fee{i} |\n|---|---|\n| row{i}a | {i}1 |\n| row{i}b | {i}2 |"
        for i in range(12))
    p = tmp_path / "t.md"
    p.write_text(md, encoding="utf-8")
    doc = parse_file(p)
    es = HeuristicGenerator(seed=3, table_fraction=0.5, unanswerable_fraction=0.0).generate([doc], 24)
    chunks = [c for c in get_chunker("structure", target_words=60, max_words=120).chunk(doc) if "table" not in c.block_types]
    results = []
    for q in es.questions:
        # a good retriever: chunks overlapping the gold span come first, so prose questions hit
        span = q.spans[0]
        ordered = sorted(chunks, key=lambda c: -(min(c.end, span.end) - max(c.start, span.start)))
        hits = [Hit(c.id, c.doc_id, c.start, c.end, 1.0 / rank, rank, c.text) for rank, c in enumerate(ordered, start=1)]
        results.append(score_question(q, hits[:50]))
    rep = analyze(es.questions, results, chunks, k=5)
    assert rep.n_misses > 0
    top = rep.diagnoses[0]
    assert top.attribute == "block_type" and top.value == "table" and top.stage == "parse"
    assert "layout parser" in top.fix or "Tables" in top.fix
    assert all(qid.startswith("h-") for qid in top.question_ids)


def test_write_report_end_to_end(tmp_path: Path):
    root = tmp_path / "corpus"
    root.mkdir()
    for i in range(3):
        (root / f"d{i}.md").write_text("# G\n\n" + "\n\n".join(f"## S{j}\n\nSection {i}{j} about widget{i}{j} rules and form W{i}{j} plus deadlines of {j + 3} days. Applicants send the form to the regional office and keep a copy for their records. Late submissions are charged a fee and may be rejected." for j in range(5)), encoding="utf-8")
    corpus = Corpus.from_dir("t", root)
    docs = [parse_file(f) for f in corpus.files]
    synth = HeuristicGenerator(seed=0).generate(docs, 12)
    real = EvalSet("real", [EvalQuestion(q.id + "-r", q.question, q.spans, source="real") for q in synth.questions[:4] if q.spans])
    es = EvalSet.merge("t", [synth, real])
    space = Space(parsers=["plaintext"], chunkers=[("fixed", {"size": 40, "overlap": 5})], embedders=[None, "hash-64"],
                  retrievals=["sparse", "dense"], rerankers=[None])
    ev = Evaluator(corpus, cache_root=tmp_path / "cache")
    default = RagConfig(parser="plaintext", chunker="fixed", chunk_params={"size": 40, "overlap": 5}, embedder="hash-64", retrieval="dense")
    rep = staged_search(ev, space, es, default)
    fails = analyze(es.questions, rep.winner.results, ev.chunks(rep.winner.config), k=5)
    paths = write_report(tmp_path / "out", rep, es, fails, None, notes=["test run"])
    text = paths["report_md"].read_text()
    assert "## Verdict" in text and "Hypothesis" in text and "best_config" in str(paths["best_config"])
    assert (tmp_path / "out" / "manifest.json").exists() and (tmp_path / "out" / "eval_set.jsonl").exists()
    idx, _ = build_index(corpus, RagConfig.from_yaml(paths["best_config"]), cache_root=tmp_path / "cache")
    assert idx.config.key == rep.winner.config.key
