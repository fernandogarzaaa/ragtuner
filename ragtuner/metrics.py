"""Retrieval metrics by span containment.

A hit covers a gold span when, in the same document, their character overlap is at least
COVER_FRACTION of the span. For long spans (over LONG_SPAN_CHARS, about 250 words: the
page-level evidence FinanceBench ships) the denominator is the shorter of span and chunk,
so a chunk lying mostly inside the evidence also counts; otherwise page-sized evidence
would be unreachable for small chunkers and the metric would reward big chunks by
construction. Sentence- and table-sized spans keep the strict rule, so a table split
across chunks still shows up as a chunking failure.
Relevance depends on offsets only, never on chunk ids or index text.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from ragtuner.evalset import EvalQuestion, GoldSpan
from ragtuner.pipeline import Hit

COVER_FRACTION = 0.5
LONG_SPAN_CHARS = 1500


def coverage(start: int, end: int, span: GoldSpan) -> float:
    """Fraction used by the relevance rule for a chunk [start, end) against a span."""
    overlap = min(end, span.end) - max(start, span.start)
    if overlap <= 0 or span.length <= 0:
        return 0.0
    denom = span.length if span.length <= LONG_SPAN_CHARS else min(span.length, max(end - start, 1))
    return overlap / denom


def covers(hit: Hit, span: GoldSpan, fraction: float = COVER_FRACTION) -> bool:
    if hit.doc_id != span.doc_id or span.length <= 0:
        return False
    return coverage(hit.start, hit.end, span) >= fraction


@dataclass
class QuestionResult:
    """What one config did for one question, over the full candidate list."""

    question_id: str
    hits: list[Hit]
    # for each gold span, the first rank (1-based) at which it is covered, else None
    span_ranks: list[int | None]
    top_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def first_rank(self) -> int | None:
        ranks = [r for r in self.span_ranks if r is not None]
        return min(ranks) if ranks else None

    def recall_at(self, k: int) -> float:
        if not self.span_ranks:
            return float("nan")
        return sum(1 for r in self.span_ranks if r is not None and r <= k) / len(self.span_ranks)

    def hit_at(self, k: int) -> float:
        if not self.span_ranks:
            return float("nan")
        return 1.0 if any(r is not None and r <= k for r in self.span_ranks) else 0.0

    def rr_at(self, k: int) -> float:
        if not self.span_ranks:
            return float("nan")
        r = self.first_rank
        return 1.0 / r if r is not None and r <= k else 0.0

    def ndcg_at(self, k: int) -> float:
        if not self.span_ranks:
            return float("nan")
        gains = sorted(r for r in self.span_ranks if r is not None and r <= k)
        dcg = sum(1.0 / math.log2(r + 1) for r in gains)
        ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(self.span_ranks), k)))
        return dcg / ideal if ideal else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "span_ranks": self.span_ranks,
            "first_rank": self.first_rank,
            "top_score": self.top_score,
            "hits": [h.to_dict() for h in self.hits[:10]],
            **self.metadata,
        }


def score_question(q: EvalQuestion, hits: Sequence[Hit]) -> QuestionResult:
    ranks: list[int | None] = []
    for span in q.spans:
        found = None
        for h in hits:
            if covers(h, span):
                found = h.rank
                break
        ranks.append(found)
    return QuestionResult(question_id=q.id, hits=list(hits), span_ranks=ranks, top_score=hits[0].score if hits else None)


def _mean(xs: list[float]) -> float:
    xs = [x for x in xs if not math.isnan(x)]
    return sum(xs) / len(xs) if xs else float("nan")


def aggregate(results: list[QuestionResult], *, ks: Sequence[int] = (5, 10), questions: dict[str, EvalQuestion] | None = None) -> dict[str, float]:
    """Mean metrics over answerable questions, plus the refusal gap on unanswerable ones."""
    answerable = [r for r in results if r.span_ranks]
    out: dict[str, float] = {"n_answerable": float(len(answerable))}
    for k in ks:
        out[f"recall@{k}"] = _mean([r.recall_at(k) for r in answerable])
        out[f"hit@{k}"] = _mean([r.hit_at(k) for r in answerable])
        out[f"mrr@{k}"] = _mean([r.rr_at(k) for r in answerable])
        out[f"ndcg@{k}"] = _mean([r.ndcg_at(k) for r in answerable])
    out["hit@50"] = _mean([r.hit_at(50) for r in answerable])
    unanswerable = [r for r in results if not r.span_ranks]
    out["n_unanswerable"] = float(len(unanswerable))
    if unanswerable and answerable:
        # refusal gate: threshold at the 10th percentile of answerable top scores; how many
        # unanswerable questions fall below it (would be refused by a score threshold)
        a_scores = sorted(r.top_score for r in answerable if r.top_score is not None)
        u_scores = [r.top_score for r in unanswerable if r.top_score is not None]
        if a_scores and u_scores:
            thr = a_scores[max(0, int(0.1 * len(a_scores)) - 1)]
            out["refusal_rate@p10"] = sum(1 for s in u_scores if s < thr) / len(u_scores)
    return out


def delta(a: dict[str, float], b: dict[str, float], metric: str) -> float:
    return a.get(metric, float("nan")) - b.get(metric, float("nan"))
