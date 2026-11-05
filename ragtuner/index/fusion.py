from __future__ import annotations


def rrf(rankings: list[list[int]], *, k: int = 60, limit: int | None = None) -> list[tuple[int, float]]:
    """Reciprocal rank fusion of several ranked id lists."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, i in enumerate(ranking, start=1):
            scores[i] = scores.get(i, 0.0) + 1.0 / (k + rank)
    fused = sorted(scores.items(), key=lambda x: -x[1])
    return fused[:limit] if limit else fused
