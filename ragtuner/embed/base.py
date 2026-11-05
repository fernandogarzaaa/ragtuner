"""Embedding backends and a content-addressed embedding cache.

Cache key = (model id, sha256 of the exact text embedded). Configs that share chunks
share embeddings, which is what makes the staged search affordable.
"""

from __future__ import annotations

import hashlib
import importlib
import sqlite3
from pathlib import Path
from typing import Callable, Protocol, Sequence

import numpy as np


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: Sequence[str], *, kind: str = "document") -> np.ndarray: ...


_REGISTRY: dict[str, tuple[str, str]] = {
    "bge-small": ("ragtuner.embed.local", "BgeSmallEmbedder"),
    "openai-small": ("ragtuner.embed.openai_backend", "OpenAISmallEmbedder"),
    "hash-64": ("ragtuner.embed.hashing", "HashingEmbedder"),
}


def get_embedder(name: str) -> Embedder:
    if name not in _REGISTRY:
        raise KeyError(f"unknown embedder {name!r}; known: {sorted(_REGISTRY)}")
    module, cls = _REGISTRY[name]
    factory: Callable[[], Embedder] = getattr(importlib.import_module(module), cls)
    return factory()


def available_embedders() -> dict[str, str]:
    out = {}
    for name in sorted(_REGISTRY):
        try:
            get_embedder(name)
            out[name] = "ok"
        except Exception as e:  # noqa: BLE001
            out[name] = f"unavailable: {type(e).__name__}: {e}"
    return out


def text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EmbeddingCache:
    """SQLite-backed float32 vector cache. One table, keyed by model and text hash."""

    def __init__(self, root: str | Path = ".ragtuner-cache") -> None:
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / "embeddings.sqlite"
        self._db = sqlite3.connect(self.path)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS vectors (model TEXT, key TEXT, dim INTEGER, vec BLOB, PRIMARY KEY (model, key))"
        )
        self._db.commit()
        self.hits = 0
        self.misses = 0

    def embed(self, embedder: Embedder, texts: Sequence[str], *, kind: str = "document", batch: int = 64) -> np.ndarray:
        keys = [text_key(f"{kind}\x00{t}") for t in texts]
        out = np.zeros((len(texts), embedder.dim), dtype=np.float32)
        missing: list[int] = []
        cur = self._db.cursor()
        for i, k in enumerate(keys):
            row = cur.execute("SELECT vec FROM vectors WHERE model=? AND key=?", (embedder.name, k)).fetchone()
            if row is None:
                missing.append(i)
            else:
                out[i] = np.frombuffer(row[0], dtype=np.float32)
        self.hits += len(texts) - len(missing)
        self.misses += len(missing)
        for s in range(0, len(missing), batch):
            idx = missing[s : s + batch]
            vecs = embedder.embed([texts[i] for i in idx], kind=kind).astype(np.float32)
            for j, i in enumerate(idx):
                out[i] = vecs[j]
            cur.executemany(
                "INSERT OR REPLACE INTO vectors (model, key, dim, vec) VALUES (?, ?, ?, ?)",
                [(embedder.name, keys[i], embedder.dim, vecs[j].tobytes()) for j, i in enumerate(idx)],
            )
            self._db.commit()
        return out

    def close(self) -> None:
        self._db.close()
