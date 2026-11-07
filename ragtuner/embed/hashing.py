"""Dependency-free hashed bag-of-words embedder. Only for tests and smoke runs; it is
not a config-space arm. Deterministic, 64 dims, L2-normalized."""

from __future__ import annotations

import hashlib
import re
from typing import Sequence

import numpy as np

_TOK = re.compile(r"[a-z0-9]+")


class HashingEmbedder:
    name = "hash-64"
    dim = 64

    def embed(self, texts: Sequence[str], *, kind: str = "document") -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in _TOK.findall(t.lower()):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0 if (h >> 8) % 2 else -1.0
            n = np.linalg.norm(out[i])
            if n:
                out[i] /= n
        return out
