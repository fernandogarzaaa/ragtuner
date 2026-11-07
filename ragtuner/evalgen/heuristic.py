"""Offline question generator. No LLM, so it is lexically leaky by construction: the
question names words that appear in the gold span. It exists so the loop runs end to
end without credentials. Its questions carry source="synthetic-heuristic" and are
reported separately; they must never be mixed into a headline number."""

from __future__ import annotations

import random
from collections import Counter

from ragtuner.evalgen.common import heading_path, make_id, sample_blocks, sentences
from ragtuner.evalset import EvalQuestion, EvalSet, GoldSpan
from ragtuner.index.bm25 import tokenize
from ragtuner.schema import BlockType, ParsedDocument

_OFF_DOMAIN = [
    "What is the boiling point of ethanol at sea level?",
    "Who won the 1998 FIFA World Cup?",
    "How many moons does Jupiter have?",
    "What year was the Eiffel Tower completed?",
    "What is the chemical symbol for tungsten?",
    "How long is a marathon in kilometers?",
    "Which composer wrote the Brandenburg Concertos?",
    "What is the tallest mountain in Africa?",
    "How many strings does a standard violin have?",
    "What is the largest desert in Asia?",
]


class HeuristicGenerator:
    name = "heuristic"

    def __init__(self, *, seed: int = 0, table_fraction: float = 0.25, unanswerable_fraction: float = 0.1) -> None:
        self.seed = seed
        self.table_fraction = table_fraction
        self.unanswerable_fraction = unanswerable_fraction

    def generate(self, docs: list[ParsedDocument], n: int, *, name: str = "synthetic") -> EvalSet:
        rng = random.Random(self.seed)
        n_un = int(round(n * self.unanswerable_fraction))
        df = _doc_freq(docs)
        questions: list[EvalQuestion] = []
        for doc, block in sample_blocks(docs, n - n_un, seed=self.seed, table_fraction=self.table_fraction):
            q = self._table_question(doc, block, rng) if block.type is BlockType.TABLE else self._prose_question(doc, block, df, rng)
            if q is not None:
                questions.append(q)
        for i in range(n_un):
            text = _OFF_DOMAIN[i % len(_OFF_DOMAIN)]
            questions.append(EvalQuestion(id=make_id("h-un", text, str(i)), question=text, spans=(), source="synthetic-heuristic",
                                          question_type="unanswerable", metadata={"generator": self.name}))
        return EvalSet(name, questions, {"generator": self.name, "seed": self.seed})

    def _prose_question(self, doc: ParsedDocument, block, df: Counter, rng: random.Random) -> EvalQuestion | None:
        sents = [(s, e) for s, e in sentences(block.text) if 6 <= len(block.text[s:e].split()) <= 60]
        if not sents:
            return None
        s, e = rng.choice(sents)
        sent = block.text[s:e]
        toks = [t for t in tokenize(sent) if len(t) > 3]
        if len(toks) < 2:
            return None
        # rarest two content words across the corpus
        rare = sorted(set(toks), key=lambda t: (df.get(t, 0), t))[:2]
        path = heading_path(doc, block)
        topic = f" in the section on {path[-1]}" if path else ""
        question = f"What does the document say about {' and '.join(rare)}{topic}?"
        span = GoldSpan(doc.id, block.start + s, block.start + e, page=block.page, block_type=block.type.value, block_id=block.id)
        return EvalQuestion(id=make_id("h", doc.id, str(span.start)), question=question, spans=(span,), answer=sent,
                            source="synthetic-heuristic", question_type="factual", lexical_overlap=None,
                            metadata={"generator": self.name, "heading_path": path})

    def _table_question(self, doc: ParsedDocument, block, rng: random.Random) -> EvalQuestion | None:
        t = block.table
        assert t is not None
        rows = [r for r in t.rows[t.header_rows :] if len(r) >= 2 and r[0].strip()]
        header = t.rows[0] if t.header_rows else None
        if not rows:
            return None
        row = rng.choice(rows)
        cols = [i for i in range(1, len(row)) if row[i].strip()]
        if not cols:
            return None
        c = rng.choice(cols)
        col_name = header[c] if header and c < len(header) and header[c].strip() else f"column {c + 1}"
        path = heading_path(doc, block)
        caption = f" in the table {t.caption}" if t.caption else (f" under {path[-1]}" if path else "")
        question = f"What is the {col_name} for {row[0]}{caption}?"
        span = GoldSpan(doc.id, block.start, block.end, page=block.page, block_type="table", block_id=block.id)
        return EvalQuestion(id=make_id("h", doc.id, str(block.start), row[0], str(c)), question=question, spans=(span,), answer=row[c],
                            source="synthetic-heuristic", question_type="numeric" if any(ch.isdigit() for ch in row[c]) else "factual",
                            metadata={"generator": self.name, "heading_path": path})


def _doc_freq(docs: list[ParsedDocument]) -> Counter:
    df: Counter = Counter()
    for d in docs:
        df.update(set(tokenize(d.text)))
    return df
