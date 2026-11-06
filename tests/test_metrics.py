import math

from ragtuner.evalset import EvalQuestion, EvalSet, GoldSpan
from ragtuner.metrics import aggregate, covers, score_question
from ragtuner.pipeline import Hit


def hit(rank, start, end, doc="d", score=1.0):
    return Hit(chunk_id=f"c{rank}", doc_id=doc, start=start, end=end, score=score, rank=rank, text="")


def test_covers_requires_half_the_span_in_the_same_doc():
    span = GoldSpan("d", 100, 200)
    assert covers(hit(1, 0, 150), span)          # 50 chars of 100
    assert not covers(hit(1, 0, 149), span)      # 49 chars
    assert not covers(hit(1, 100, 200, doc="e"), span)
    # a small chunk lying inside a page-sized span is relevant; one barely touching it is not
    page = GoldSpan("d", 0, 5000)  # longer than LONG_SPAN_CHARS
    assert covers(hit(1, 1000, 1400), page)
    assert not covers(hit(1, 4900, 5300), page)
    # a table-sized span keeps the strict rule: a chunk inside only part of it does not count
    table = GoldSpan("d", 0, 1000)
    assert not covers(hit(1, 100, 400), table) and covers(hit(1, 0, 600), table)


def test_question_metrics_single_and_multi_span():
    q1 = EvalQuestion("q1", "?", (GoldSpan("d", 100, 200),))
    r1 = score_question(q1, [hit(1, 0, 50), hit(2, 90, 210), hit(3, 150, 300)])
    assert r1.span_ranks == [2] and r1.recall_at(5) == 1.0 and r1.rr_at(5) == 0.5 and r1.recall_at(1) == 0.0
    q2 = EvalQuestion("q2", "?", (GoldSpan("d", 0, 100), GoldSpan("d", 500, 600)))
    r2 = score_question(q2, [hit(1, 0, 100), hit(2, 300, 400)])
    assert r2.span_ranks == [1, None] and r2.recall_at(5) == 0.5 and r2.hit_at(5) == 1.0
    assert 0 < r2.ndcg_at(5) < 1


def test_aggregate_handles_unanswerable_and_refusal_gate():
    qs = [EvalQuestion(f"a{i}", "?", (GoldSpan("d", 0, 100),)) for i in range(10)]
    rs = [score_question(q, [hit(1, 0, 100, score=0.9)]) for q in qs]
    un = [EvalQuestion(f"u{i}", "?", ()) for i in range(4)]
    rs += [score_question(q, [hit(1, 0, 100, score=s)]) for q, s in zip(un, [0.1, 0.2, 0.95, 0.3])]
    m = aggregate(rs)
    assert m["recall@5"] == 1.0 and m["n_answerable"] == 10 and m["n_unanswerable"] == 4
    assert m["refusal_rate@p10"] == 0.75
    assert math.isnan(score_question(un[0], []).recall_at(5))


def test_evalset_round_trip_and_filters(tmp_path):
    qs = [
        EvalQuestion("r1", "real?", (GoldSpan("d", 0, 10, page=1, block_type="table"),), source="real"),
        EvalQuestion("s1", "syn?", (GoldSpan("d", 20, 30),), source="synthetic-llm", lexical_overlap=0.2),
        EvalQuestion("s2", "none?", (), source="synthetic-llm", question_type="unanswerable"),
    ]
    es = EvalSet("t", qs, {"corpus": "x"})
    es.save(tmp_path / "e.jsonl")
    back = EvalSet.load(tmp_path / "e.jsonl")
    assert back.questions == qs and back.metadata == {"corpus": "x"} and back.name == "t"
    assert len(back.real) == 1 and len(back.synthetic) == 2
    assert back.summary()["by_block_type"] == {"table": 1, "none": 2}
