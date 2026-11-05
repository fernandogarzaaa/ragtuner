"""Real-query importers. Real questions are the final arbiter; their gold spans come
from human-provided evidence where it exists (FinanceBench) or from locating a known
answer text in the corpus (marked gold_quality='weak')."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from ragtuner.evalgen.common import lexical_overlap, locate, make_id
from ragtuner.evalset import EvalQuestion, EvalSet, GoldSpan
from ragtuner.index.bm25 import tokenize
from ragtuner.schema import ParsedDocument


def _block_at(doc: ParsedDocument, pos: int):
    for b in doc.blocks:
        if b.start <= pos < b.end:
            return b
    return None


def import_financebench(jsonl: str | Path, docs: dict[str, ParsedDocument], *, name: str = "financebench-real") -> EvalSet:
    """docs maps FinanceBench doc_name (e.g. '3M_2018_10K') to its ParsedDocument."""
    questions: list[EvalQuestion] = []
    skipped: list[dict[str, Any]] = []
    for line in Path(jsonl).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["doc_name"] not in docs:
            continue
        doc = docs[row["doc_name"]]
        spans: list[GoldSpan] = []
        quality = "strong"
        for ev in row.get("evidence", []):
            if ev.get("doc_name") and ev["doc_name"] != row["doc_name"]:
                continue
            loc = locate(doc, ev.get("evidence_text", ""), min_ratio=0.6)
            page = ev.get("evidence_page_num")
            page = int(page) + 1 if page is not None else None  # FinanceBench pages are 0-indexed
            if loc is None and ev.get("evidence_text_full_page"):
                loc = locate(doc, ev["evidence_text_full_page"], min_ratio=0.5)
            if loc is None:
                continue
            start, end, q = loc
            b = _block_at(doc, start)
            if q < 0.95:
                quality = "weak"
            spans.append(GoldSpan(doc.id, start, end, page=page, block_type=b.type.value if b else None, block_id=b.id if b else None))
        if not spans:
            skipped.append({"id": row["financebench_id"], "reason": "no evidence located"})
            continue
        qtext = row["question"]
        questions.append(EvalQuestion(
            id=row["financebench_id"], question=qtext, spans=tuple(spans), answer=str(row.get("answer", "")), source="real",
            question_type="multi-hop" if len(spans) > 1 else ("numeric" if any(ch.isdigit() for ch in str(row.get("answer", ""))) else "factual"),
            lexical_overlap=round(max(lexical_overlap(qtext, doc.text[s.start : s.end]) for s in spans), 3),
            gold_quality=quality, metadata={"fb_question_type": row.get("question_type"), "company": row.get("company")},
        ))
    return EvalSet(name, questions, {"source": "financebench", "skipped": skipped})


def import_qa_pairs(rows: Iterable[dict[str, Any]], docs: list[ParsedDocument], *, name: str = "real", min_score: float = 0.35) -> EvalSet:
    """rows: {'question', 'answer', optional 'source'}. The gold span is the block whose
    content tokens best overlap the ANSWER text (not the question, which would favor sparse
    retrieval). Jaccard below min_score drops the row. Always gold_quality='weak'."""
    blocks = [(d, b, set(tokenize(b.text))) for d in docs for b in d.blocks if len(b.text.split()) >= 15]
    questions: list[EvalQuestion] = []
    skipped = 0
    for row in rows:
        ans = set(tokenize(str(row.get("answer", ""))))
        if len(ans) < 8:
            skipped += 1
            continue
        best, best_s = None, 0.0
        for d, b, toks in blocks:
            if not toks:
                continue
            s = len(ans & toks) / len(ans | toks)
            if s > best_s:
                best, best_s = (d, b), s
        if best is None or best_s < min_score:
            skipped += 1
            continue
        d, b = best
        span = GoldSpan(d.id, b.start, b.end, page=b.page, block_type=b.type.value, block_id=b.id)
        questions.append(EvalQuestion(
            id=make_id("real", row["question"]), question=row["question"], spans=(span,), answer=str(row.get("answer", "")),
            source="real", question_type="factual", lexical_overlap=round(lexical_overlap(row["question"], b.text), 3),
            gold_quality="weak", metadata={"match_jaccard": round(best_s, 3), "origin": row.get("source")},
        ))
    return EvalSet(name, questions, {"source": "qa-pairs", "skipped": skipped})
