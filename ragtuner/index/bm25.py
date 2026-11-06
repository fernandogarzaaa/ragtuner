"""Okapi BM25 over chunk index_text. Standalone sparse arm. Tokenization is lowercase
alphanumerics with a small stopword list and a light suffix stemmer, so "refunds" and
"refund" meet. BETTER-RAG's exact-term scorer stays available as the 'better-rag'
retrieval mode for the baseline comparison."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence

_TOK = re.compile(r"[a-z0-9]+")
_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "for", "from", "how", "i", "in", "is", "it",
    "of", "on", "or", "that", "the", "to", "what", "when", "where", "with", "was", "were", "this", "these", "which",
}


def stem(tok: str) -> str:
    for suf in ("ations", "ation", "ings", "ing", "ies", "ers", "er", "ed", "es", "s"):
        if len(tok) > len(suf) + 3 and tok.endswith(suf):
            return tok[: -len(suf)] + ("y" if suf == "ies" else "")
    return tok


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in _TOK.findall(text.lower()) if t not in _STOP]


class BM25Index:
    def __init__(self, texts: list[str], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(t)) for t in texts]
        self.lengths = [sum(c.values()) for c in self.docs]
        self.avgdl = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        df: Counter[str] = Counter()
        for c in self.docs:
            df.update(c.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for i, c in enumerate(self.docs):
            for t, f in c.items():
                self.postings.setdefault(t, []).append((i, f))

    def search(self, query: str, k: int, allowed: "Sequence[bool] | None" = None) -> list[tuple[int, float]]:
        scores: dict[int, float] = {}
        for t in set(tokenize(query)):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, f in self.postings[t]:
                if allowed is not None and not allowed[i]:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.lengths[i] / max(self.avgdl, 1e-9))
                scores[i] = scores.get(i, 0.0) + idf * f * (self.k1 + 1) / denom
        return sorted(scores.items(), key=lambda x: -x[1])[:k]
