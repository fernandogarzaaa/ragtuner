"""BETTER-RAG's scorer as a retrieval arm, without its MMR selection.

The tuner runs BETTER-RAG with mmr_lambda=1.0, where MMR reduces to sorting by score,
but BETTER-RAG's _mmr_select still compares every candidate against every selected
chunk (O(k * N * k) sparse cosines per query). This module computes the exact same
per-chunk score as BetterRAG.search and sorts it. tests/test_better_scorer.py asserts the
ranking is identical to BetterRAG.search at mmr_lambda=1.0.
"""

from __future__ import annotations

from typing import Any, Sequence

from better_rag import core as br


def better_rag_scores(engine: Any, query: str) -> list[tuple[str, float]]:
    """(BETTER-RAG chunk id, score) for every chunk, best first, ties in index order."""
    q_tokens = br._tokens(query)
    if not q_tokens or not engine._chunks:
        return []
    q_vec = br._term_vector(q_tokens)
    chunks = engine._chunks
    avgdl = sum(len(c.tokens) for c in chunks) / len(chunks)
    scored = []
    for c in chunks:
        score = (0.48 * br._cosine_sparse(q_vec, c.vector)) + (0.37 * engine._bm25(q_tokens, c.tokens, avgdl)) + (0.15 * br._overlap(q_tokens, c.tokens))
        scored.append((c.document_id, max(0.0, min(1.0, score)), score))
    order = sorted(range(len(scored)), key=lambda i: -scored[i][2])
    return [(scored[i][0], scored[i][1]) for i in order]


def better_rag_search(engine: Any, query: str, n: int, allowed: Sequence[bool] | None, by_id: dict[str, int]) -> list[tuple[int, float]]:
    out = []
    for doc_id, score in better_rag_scores(engine, query):
        i = by_id[doc_id]
        if allowed is not None and not allowed[i]:
            continue
        out.append((i, score))
        if len(out) >= n:
            break
    return out
