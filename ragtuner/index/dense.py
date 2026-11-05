from __future__ import annotations

import numpy as np


class DenseIndex:
    """Brute-force cosine over L2-normalized float32 vectors. Fine to ~200k chunks."""

    def __init__(self, vectors: np.ndarray) -> None:
        v = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        self.vectors = v / np.maximum(norms, 1e-9)

    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[int, float]]:
        q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        q = q / max(float(np.linalg.norm(q)), 1e-9)
        sims = self.vectors @ q
        k = min(k, sims.shape[0])
        if k == 0:
            return []
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [(int(i), float(sims[i])) for i in idx]
