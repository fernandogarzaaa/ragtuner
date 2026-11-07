"""Local sentence-transformers embedders (CPU is fine for Phase 1 corpus sizes)."""

from __future__ import annotations

from typing import Sequence

import numpy as np


class BgeSmallEmbedder:
    name = "BAAI/bge-small-en-v1.5"
    dim = 384
    # bge recommends a query instruction for short query -> passage retrieval
    query_prefix = "Represent this sentence for searching relevant passages: "

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.name)

    def embed(self, texts: Sequence[str], *, kind: str = "document") -> np.ndarray:
        inputs = [self.query_prefix + t for t in texts] if kind == "query" else list(texts)
        return np.asarray(self._model.encode(inputs, batch_size=32, normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)
