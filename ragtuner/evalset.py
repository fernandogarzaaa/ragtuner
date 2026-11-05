"""Eval set schema. Gold answers are character spans in a ParsedDocument's canonical text,
never chunk ids, so the same set scores every chunker and parser fairly.

A question may have several spans (multi-hop) or none (unanswerable).
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

SOURCES = ("real", "synthetic-llm", "synthetic-heuristic")


@dataclass(frozen=True)
class GoldSpan:
    doc_id: str
    start: int
    end: int
    page: int | None = None
    block_type: str | None = None
    block_id: str | None = None

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class EvalQuestion:
    id: str
    question: str
    spans: tuple[GoldSpan, ...]
    answer: str | None = None
    source: str = "synthetic-llm"
    question_type: str = "factual"  # factual | numeric | multi-hop | unanswerable | ...
    lexical_overlap: float | None = None  # token overlap between question and gold span, 0..1
    gold_quality: str = "strong"  # strong | weak (automatically located, unverified)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def answerable(self) -> bool:
        return bool(self.spans)

    @property
    def doc_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(s.doc_id for s in self.spans))

    @property
    def block_type(self) -> str | None:
        types = [s.block_type for s in self.spans if s.block_type]
        return types[0] if types else None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["spans"] = [asdict(s) for s in self.spans]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EvalQuestion":
        d = dict(d)
        d["spans"] = tuple(GoldSpan(**s) for s in d.get("spans", []))
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class EvalSet:
    name: str
    questions: list[EvalQuestion]
    metadata: dict[str, Any] = field(default_factory=dict)

    def by_source(self, *sources: str) -> "EvalSet":
        wanted = set(sources)
        return EvalSet(self.name, [q for q in self.questions if q.source in wanted], dict(self.metadata))

    @property
    def real(self) -> "EvalSet":
        return self.by_source("real")

    @property
    def synthetic(self) -> "EvalSet":
        return self.by_source("synthetic-llm", "synthetic-heuristic")

    def sample(self, fraction: float, *, seed: int = 0) -> "EvalSet":
        rng = random.Random(seed)
        n = max(1, int(round(len(self.questions) * fraction)))
        return EvalSet(self.name, rng.sample(self.questions, min(n, len(self.questions))), dict(self.metadata))

    def __len__(self) -> int:
        return len(self.questions)

    def __iter__(self):
        return iter(self.questions)

    def summary(self) -> dict[str, Any]:
        from collections import Counter

        return {
            "n": len(self.questions),
            "by_source": dict(Counter(q.source for q in self.questions)),
            "by_type": dict(Counter(q.question_type for q in self.questions)),
            "by_block_type": dict(Counter(q.block_type or "none" for q in self.questions)),
            "unanswerable": sum(1 for q in self.questions if not q.answerable),
            "weak_gold": sum(1 for q in self.questions if q.gold_quality == "weak"),
        }

    def save(self, path: str | Path) -> None:
        p = Path(path)
        with p.open("w", encoding="utf-8") as f:
            f.write(json.dumps({"_meta": {"name": self.name, **self.metadata}}) + "\n")
            for q in self.questions:
                f.write(json.dumps(q.to_dict(), ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "EvalSet":
        rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
        meta = {}
        if rows and "_meta" in rows[0]:
            meta = rows.pop(0)["_meta"]
        name = meta.pop("name", Path(path).stem)
        return cls(name, [EvalQuestion.from_dict(r) for r in rows], meta)

    @classmethod
    def merge(cls, name: str, sets: Iterable["EvalSet"]) -> "EvalSet":
        qs: list[EvalQuestion] = []
        seen: set[str] = set()
        for s in sets:
            for q in s.questions:
                if q.id not in seen:
                    seen.add(q.id)
                    qs.append(q)
        return cls(name, qs)
