import json
from pathlib import Path

from ragtuner.evalgen import (
    HeuristicGenerator,
    LLMGenerator,
    import_financebench,
    import_qa_pairs,
    lexical_overlap,
    locate,
)
from ragtuner.parse import parse_file

MD = """# Refund policy

Customers may return purchased items within 30 days for a full refund when the receipt is included. Items damaged by misuse are excluded from refunds entirely.

# Fees

| Item | Fee |
|---|---|
| Restocking | $5 |
| Shipping | $12 |

# Shipping

Orders ship within two business days from the Ohio warehouse. Expedited shipping costs an additional twenty dollars per order.
"""


def make_doc(tmp_path: Path):
    p = tmp_path / "policy.md"
    p.write_text(MD, encoding="utf-8")
    return parse_file(p)


def test_locate_exact_normalized_and_fuzzy(tmp_path):
    doc = make_doc(tmp_path)
    s, e, q = locate(doc, "within 30 days for a full refund")
    assert doc.text[s:e] == "within 30 days for a full refund" and q == 1.0
    s, e, q = locate(doc, "within   30 DAYS\nfor a full refund")
    assert doc.text[s:e] == "within 30 days for a full refund" and q == 1.0
    fuzzy = locate(doc, "Orders ship within two bussiness days from the Ohio warehouse.")
    assert fuzzy is not None and fuzzy[2] > 0.9 and "Ohio warehouse" in doc.text[fuzzy[0]:fuzzy[1]]
    assert locate(doc, "completely unrelated sentence about penguins and glaciers") is None


def test_heuristic_generator_anchors_spans_and_marks_source(tmp_path):
    doc = make_doc(tmp_path)
    es = HeuristicGenerator(seed=1, table_fraction=0.5, unanswerable_fraction=0.25).generate([doc], 8)
    assert es.questions
    for q in es.questions:
        assert q.source == "synthetic-heuristic"
        for s in q.spans:
            assert doc.text[s.start:s.end].strip()
    summary = es.summary()
    assert summary["unanswerable"] == 2
    assert "table" in summary["by_block_type"]


def test_llm_generator_with_fake_caller_locates_and_paraphrases(tmp_path):
    doc = make_doc(tmp_path)
    calls = []

    def fake(system, user):
        calls.append(system[:20])
        if "Rewrite" in system:
            return json.dumps({"question": "How long after buying something can I get my money back?"})
        if "two passages" in system:
            return json.dumps({"question": "x", "answer": "y", "evidence_1": "Restocking", "evidence_2": "Ohio warehouse"})
        if "does NOT answer" in system:
            return json.dumps({"question": "What is the refund policy for gift cards?"})
        return json.dumps({"question": "Can customers return purchased items within 30 days for a full refund?",
                           "answer": "yes", "evidence": "Customers may return purchased items within 30 days for a full refund when the receipt is included.",
                           "question_type": "factual"})

    gen = LLMGenerator(caller=fake, seed=0, table_fraction=0.0, multihop_fraction=0.0, unanswerable_fraction=0.0, max_overlap=0.5)
    es = gen.generate([doc], 1)
    assert len(es.questions) == 1
    q = es.questions[0]
    assert q.source == "synthetic-llm" and q.lexical_overlap is not None and q.lexical_overlap <= 0.5
    assert q.question.startswith("How long after")
    assert q.metadata["paraphrase_rounds"] == 1
    assert doc.text[q.spans[0].start:q.spans[0].end].startswith("Customers may return")


def test_llm_generator_drops_unlocatable_evidence(tmp_path):
    doc = make_doc(tmp_path)
    gen = LLMGenerator(caller=lambda s, u: json.dumps({"question": "q?", "answer": "a", "evidence": "this text is not in the doc at all"}),
                       table_fraction=0.0, multihop_fraction=0.0, unanswerable_fraction=0.0)
    es = gen.generate([doc], 1)
    assert not es.questions and gen.dropped


def test_financebench_importer(tmp_path):
    doc = make_doc(tmp_path)
    rows = [{"financebench_id": "fb1", "company": "ACME", "doc_name": "ACME_2024_10K", "question_type": "metrics-generated",
             "question": "What is the restocking fee?", "answer": "$5",
             "evidence": [{"evidence_text": "| Restocking | $5 |", "doc_name": "ACME_2024_10K", "evidence_page_num": 0, "evidence_text_full_page": ""}]},
            {"financebench_id": "fb2", "company": "ACME", "doc_name": "OTHER", "question": "?", "answer": "", "evidence": []}]
    p = tmp_path / "fb.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    es = import_financebench(p, {"ACME_2024_10K": doc})
    assert len(es.questions) == 1
    q = es.questions[0]
    assert q.source == "real" and q.spans[0].page == 1 and q.spans[0].block_type == "table" and q.question_type == "numeric"


def test_qa_pairs_importer_uses_answer_not_question(tmp_path):
    doc = make_doc(tmp_path)
    rows = [{"question": "How fast do you deliver?", "answer": "Orders ship within two business days from the Ohio warehouse and expedited shipping costs twenty dollars more per order."},
            {"question": "Do you sell cars?", "answer": "No."}]
    es = import_qa_pairs(rows, [doc], min_score=0.3)
    assert len(es.questions) == 1
    q = es.questions[0]
    assert q.gold_quality == "weak" and "Ohio" in doc.text[q.spans[0].start:q.spans[0].end]
    assert lexical_overlap("How fast do you deliver?", doc.text[q.spans[0].start:q.spans[0].end]) < 0.5
