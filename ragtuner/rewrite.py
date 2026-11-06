"""Query rewriting arms.

prf: pseudo-relevance feedback (RM3-style, offline). Retrieve the top FEEDBACK_DOCS chunks
     with BM25, score their terms by tf-idf, append the top EXPANSION_TERMS terms not in
     the query. Cheap, deterministic, no model.
llm: an LLM reformulates the question into the vocabulary a document would use
     ("capital expenditure" -> "purchases of property, plant and equipment"). Cached per
     query string, so every config in a search pays once per question.

Rewrites change only first-stage retrieval. The reranker always scores the original
question, and relevance is always judged on the original question's gold spans.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Callable

from ragtuner.index.bm25 import BM25Index, tokenize

FEEDBACK_DOCS = 3
EXPANSION_TERMS = 8

_LLM_SYSTEM = (
    "Rewrite the user's question as a search query for a document index. Keep every entity, "
    "period and number. Add the formal terms the source document is likely to use (line-item "
    "names, legal or technical terms, synonyms). Reply with the query only, one line, no quotes."
)


def prf_expand(query: str, bm25: BM25Index, texts: list[str]) -> str:
    hits = bm25.search(query, FEEDBACK_DOCS)
    if not hits:
        return query
    q = set(tokenize(query))
    tf: Counter[str] = Counter()
    for i, _ in hits:
        tf.update(tokenize(texts[i]))
    scored = []
    for term, f in tf.items():
        if term in q or len(term) < 3 or term.isdigit():
            continue
        idf = bm25.idf.get(term, 0.0)
        scored.append((f * idf, term))
    scored.sort(reverse=True)
    extra = [t for _, t in scored[:EXPANSION_TERMS]]
    return query + (" " + " ".join(extra) if extra else "")


class LLMRewriter:
    def __init__(self, caller: Callable[[str, str], str] | None = None) -> None:
        if caller is None:
            from ragtuner.llm import default_caller

            caller = default_caller(effort="low")
        self.caller = caller
        self.cache: dict[str, str] = {}

    def __call__(self, query: str) -> str:
        if query not in self.cache:
            try:
                out = self.caller(_LLM_SYSTEM, query).strip().splitlines()
                rewritten = out[0].strip().strip('"') if out else ""
            except Exception:  # noqa: BLE001 - a failed rewrite falls back to the original
                rewritten = ""
            self.cache[query] = f"{query} {rewritten}" if rewritten else query
        return self.cache[query]


def idf_weight(df: int, n: int) -> float:
    return math.log(1 + (n - df + 0.5) / (df + 0.5))
