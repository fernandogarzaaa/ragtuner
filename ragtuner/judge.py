"""Stage 3: generation quality on the finalists. Builds an answer from the retrieved
context with Claude and scores it against the gold answer with a Claude judge.

Both calls are injectable so the module is testable offline. Without credentials the
tuner skips this stage and the report says so; retrieval metrics are never affected.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from ragtuner.evalset import EvalQuestion
from ragtuner.pipeline import Hit

Caller = Callable[[str, str], str]

_ANSWER_SYSTEM = (
    "Answer the question using only the provided context passages. If the context does not "
    "contain the answer, reply exactly: NOT IN CONTEXT. Be concise: one to three sentences, "
    "and quote numbers exactly as they appear."
)

_JUDGE_SYSTEM = (
    "You grade an answer against a reference answer. Score 1 if the answer is correct and "
    "consistent with the reference (numbers must match, units may differ if equivalent), 0.5 if "
    "partially correct or incomplete, 0 if wrong or unsupported. For a question marked "
    "unanswerable, score 1 only if the answer declines (for example says NOT IN CONTEXT). "
    'Reply with JSON only: {"score": 0|0.5|1, "reason": str}'
)


def default_caller(model: str = "claude-opus-5-5", effort: str = "high") -> Caller:
    from ragtuner.llm import default_caller as _dc

    return _dc(model, effort)


@dataclass
class JudgedAnswer:
    question_id: str
    answer: str
    score: float
    reason: str
    declined: bool

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class GenerationResult:
    config_key: str
    judged: list[JudgedAnswer] = field(default_factory=list)
    calls: int = 0
    errors: int = 0

    @property
    def mean_score(self) -> float:
        xs = [j.score for j in self.judged]
        return sum(xs) / len(xs) if xs else float("nan")

    @property
    def refusal_accuracy(self) -> float:
        xs = [j for j in self.judged if j.question_id.endswith("unanswerable") or j.declined]
        return float("nan") if not xs else sum(1 for j in xs if j.declined) / len(xs)

    def to_dict(self) -> dict[str, Any]:
        return {"config_key": self.config_key, "mean_score": self.mean_score, "n": len(self.judged), "calls": self.calls,
                "errors": self.errors, "judged": [j.to_dict() for j in self.judged]}


def _json(text: str) -> dict[str, Any]:
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError("no JSON in reply")
    return json.loads(m.group(0))


class Judge:
    def __init__(self, answer_caller: Caller | None = None, judge_caller: Caller | None = None) -> None:
        self.answer_caller = answer_caller or default_caller(effort="medium")
        self.judge_caller = judge_caller or default_caller(effort="high")

    def answer(self, q: EvalQuestion, hits: list[Hit], *, k: int = 5) -> str:
        ctx = "\n\n".join(f"[{h.rank}] (doc {h.doc_id}, page {h.page})\n{h.text}" for h in hits[:k])
        return self.answer_caller(_ANSWER_SYSTEM, f"Question: {q.question}\n\nContext:\n{ctx}").strip()

    def grade(self, q: EvalQuestion, answer: str) -> tuple[float, str]:
        ref = q.answer if q.answer else ("(unanswerable: the correct behavior is to decline)" if not q.answerable else "(no reference)")
        out = _json(self.judge_caller(_JUDGE_SYSTEM, f"Question: {q.question}\nReference answer: {ref}\nCandidate answer: {answer}"))
        return float(out.get("score", 0.0)), str(out.get("reason", ""))

    def run(self, config_key: str, questions: list[EvalQuestion], hits_by_id: dict[str, list[Hit]], *, k: int = 5) -> GenerationResult:
        res = GenerationResult(config_key=config_key)
        for q in questions:
            try:
                ans = self.answer(q, hits_by_id.get(q.id, []), k=k)
                score, reason = self.grade(q, ans)
                res.calls += 2
            except Exception as e:  # noqa: BLE001
                res.errors += 1
                res.judged.append(JudgedAnswer(q.id, "", 0.0, f"error: {e}", False))
                continue
            res.judged.append(JudgedAnswer(q.id, ans, score, reason, "NOT IN CONTEXT" in ans.upper()))
        return res
