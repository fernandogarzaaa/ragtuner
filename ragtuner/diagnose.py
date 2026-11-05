"""Failure analysis: turn per-question misses into diagnoses with a named fix.

Every miss gets a deterministic stage label from offsets and ranks alone:
  parse      gold span text is not present in any chunk's text (parser lost it)
  chunk      some chunk contains part of it, but no single chunk covers >= 50% of the span
  deep       an intact chunk exists but is not in the top CANDIDATES (embedding/lexical miss)
  rank       in the candidate list but not in the top k (ranking problem)
Then, for each attribute (block type, question type, lexical overlap bucket, page bucket,
document), compare the miss rate to the base rate and report attributes with lift and
enough support. Each (stage, attribute) maps to a fix recipe. Rules compute; prose only
phrases. Every diagnosis lists the question ids behind it.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from ragtuner.chunk import Chunk
from ragtuner.evalset import EvalQuestion, GoldSpan
from ragtuner.metrics import COVER_FRACTION, QuestionResult

STAGES = ("parse", "chunk", "deep", "rank")
MIN_SUPPORT = 5

FIXES: dict[tuple[str, str], str] = {
    ("parse", "block_type=table"): "Tables are lost or flattened before chunking: switch to a layout parser (docling or a hosted parser) and keep tables as structured blocks.",
    ("parse", "*"): "Gold text is missing from every chunk: the parser drops or garbles it. Compare parsers on these pages with `ragtuner gate`.",
    ("chunk", "block_type=table"): "Tables are split across chunks: use the structure chunker (table kept whole, header repeated on splits).",
    ("chunk", "*"): "Gold spans straddle chunk boundaries: increase chunk size or overlap, or use structure-aware chunking so blocks stay whole.",
    ("deep", "overlap=low"): "Questions with little word overlap miss entirely: lexical retrieval cannot bridge paraphrase; add a dense embedder or go hybrid.",
    ("deep", "overlap=high"): "Even near-verbatim questions miss: embeddings blur exact terms; add a sparse arm (hybrid) or raise candidate depth.",
    ("deep", "block_type=table"): "Table chunks are never retrieved: index tables with heading context (structure chunker) and prefer hybrid retrieval for numeric cells.",
    ("deep", "question_type=numeric"): "Numeric questions miss at depth: dense embeddings underweight numbers; hybrid retrieval restores exact-token matching.",
    ("deep", "*"): "Relevant chunks are outside the candidate pool: change the embedding model, switch to hybrid retrieval, or raise the candidate depth.",
    ("rank", "*"): "Relevant chunks are retrieved but ranked below top-k: add a cross-encoder reranker, or lower the candidate pool noise with hybrid fusion.",
    ("rank", "block_type=table"): "Tables are found but outranked by prose: a reranker usually fixes this; also try heading-prefixed table chunks.",
}


@dataclass
class Diagnosis:
    stage: str
    attribute: str
    value: str
    misses: int  # misses with this attribute value
    questions_with_value: int
    total_misses: int
    miss_share: float  # fraction of all misses that carry this value
    base_share: float  # fraction of all questions that carry this value
    lift: float
    fix: str
    question_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    def sentence(self) -> str:
        return (f"{self.attribute}={self.value}: {self.miss_share:.0%} of misses ({self.misses} of {self.total_misses}) "
                f"but {self.base_share:.0%} of questions; {self.stage} failures. Fix: {self.fix}")


@dataclass
class FailureReport:
    k: int
    n_questions: int
    n_misses: int
    stage_counts: dict[str, int]
    diagnoses: list[Diagnosis]
    per_question: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"k": self.k, "n_questions": self.n_questions, "n_misses": self.n_misses, "stage_counts": self.stage_counts,
                "diagnoses": [d.to_dict() for d in self.diagnoses], "per_question": self.per_question}

    def to_markdown(self) -> str:
        lines = [f"Misses at k={self.k}: {self.n_misses} of {self.n_questions} answerable questions.", ""]
        if self.stage_counts:
            lines.append("| stage | misses |")
            lines.append("|---|---|")
            for s in STAGES:
                if self.stage_counts.get(s):
                    lines.append(f"| {s} | {self.stage_counts[s]} |")
            lines.append("")
        if not self.diagnoses:
            lines.append("No attribute reaches the support threshold; misses look evenly spread.")
        for d in self.diagnoses:
            lines.append(f"- {d.sentence()} Questions: {', '.join(d.question_ids[:8])}{' ...' if len(d.question_ids) > 8 else ''}")
        return "\n".join(lines)


def _span_text_present(span: GoldSpan, chunks_by_doc: dict[str, list[Chunk]], doc_text: str | None) -> tuple[bool, bool]:
    """(any chunk overlaps the span at all, some chunk covers >= COVER_FRACTION)."""
    any_overlap = False
    covered = False
    for c in chunks_by_doc.get(span.doc_id, []):
        overlap = min(c.end, span.end) - max(c.start, span.start)
        if overlap > 0:
            any_overlap = True
            if overlap / max(span.length, 1) >= COVER_FRACTION:
                covered = True
                break
    return any_overlap, covered


def stage_of(q: EvalQuestion, r: QuestionResult, k: int, chunks_by_doc: dict[str, list[Chunk]]) -> str | None:
    """None when the question is a hit at k or unanswerable."""
    if not q.spans or r.hit_at(k) >= 1.0:
        return None
    # judge by the easiest span (the one that got closest)
    best = None
    for span, rank in zip(q.spans, r.span_ranks):
        if rank is not None:
            return "rank" if best is None or best == "rank" else best
        any_overlap, covered = _span_text_present(span, chunks_by_doc, None)
        s = "parse" if not any_overlap else ("chunk" if not covered else "deep")
        order = {"deep": 0, "chunk": 1, "parse": 2}
        if best is None or order[s] < order[best]:
            best = s
    return best or "deep"


def _attributes(q: EvalQuestion) -> dict[str, str]:
    attrs = {
        "block_type": q.block_type or "none",
        "question_type": q.question_type,
        "source": q.source,
        "doc": q.doc_ids[0] if q.doc_ids else "none",
    }
    if q.lexical_overlap is not None:
        attrs["overlap"] = "low" if q.lexical_overlap < 0.34 else ("mid" if q.lexical_overlap < 0.67 else "high")
    pages = [s.page for s in q.spans if s.page]
    if pages:
        p = min(pages)
        attrs["page"] = "1-10" if p <= 10 else ("11-50" if p <= 50 else "51+")
    return attrs


def analyze(questions: list[EvalQuestion], results: list[QuestionResult], chunks: list[Chunk], *, k: int = 5,
            min_support: int = MIN_SUPPORT, min_lift: float = 1.3) -> FailureReport:
    by_id = {r.question_id: r for r in results}
    chunks_by_doc: dict[str, list[Chunk]] = defaultdict(list)
    for c in chunks:
        chunks_by_doc[c.doc_id].append(c)
    answerable = [q for q in questions if q.spans and q.id in by_id]
    stages: dict[str, str] = {}
    per_q = []
    for q in answerable:
        s = stage_of(q, by_id[q.id], k, chunks_by_doc)
        if s:
            stages[q.id] = s
        per_q.append({"id": q.id, "stage": s, "first_rank": by_id[q.id].first_rank, **_attributes(q)})
    n_miss = len(stages)
    stage_counts = dict(Counter(stages.values()))
    diagnoses: list[Diagnosis] = []
    if n_miss:
        attr_values: dict[tuple[str, str], list[str]] = defaultdict(list)  # (attr, value) -> all question ids
        for q in answerable:
            for a, v in _attributes(q).items():
                attr_values[(a, v)].append(q.id)
        for (a, v), ids in attr_values.items():
            missed = [i for i in ids if i in stages]
            if len(missed) < min_support:
                continue
            miss_share = len(missed) / n_miss
            base_share = len(ids) / len(answerable)
            lift = miss_share / base_share if base_share else 0.0
            if lift < min_lift:
                continue
            stage = Counter(stages[i] for i in missed).most_common(1)[0][0]
            fix = FIXES.get((stage, f"{a}={v}")) or FIXES.get((stage, "*")) or "No recipe for this combination yet."
            diagnoses.append(Diagnosis(stage=stage, attribute=a, value=v, misses=len(missed), questions_with_value=len(ids),
                                       total_misses=n_miss, miss_share=miss_share, base_share=base_share, lift=lift, fix=fix,
                                       question_ids=sorted(missed)))
        diagnoses.sort(key=lambda d: (-d.misses * d.lift, d.attribute))
        # one diagnosis per attribute family keeps the list readable; doc-level ones last
        seen: set[str] = set()
        kept = []
        for d in diagnoses:
            if d.attribute in seen and d.attribute != "doc":
                continue
            seen.add(d.attribute)
            kept.append(d)
        diagnoses = kept[:8]
    # stage-level fallback when no attribute stands out
    if n_miss and not diagnoses:
        stage = Counter(stages.values()).most_common(1)[0][0]
        ids = sorted(i for i, s in stages.items() if s == stage)
        diagnoses.append(Diagnosis(stage=stage, attribute="stage", value=stage, misses=len(ids), questions_with_value=len(answerable),
                                   total_misses=n_miss, miss_share=len(ids) / n_miss, base_share=1.0, lift=1.0,
                                   fix=FIXES[(stage, "*")], question_ids=ids))
    return FailureReport(k=k, n_questions=len(answerable), n_misses=n_miss, stage_counts=stage_counts, diagnoses=diagnoses, per_question=per_q)
