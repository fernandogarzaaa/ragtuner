"""Declarative RAG configuration. One of these fully determines a pipeline; the tuner
searches over them and the export ships the winner."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

RETRIEVAL_MODES = ("sparse", "dense", "hybrid", "better-rag")


@dataclass(frozen=True)
class RagConfig:
    parser: str = "pymupdf-text"
    chunker: str = "recursive"
    chunk_params: dict[str, Any] = field(default_factory=lambda: {"size": 512, "overlap": 50})
    embedder: str | None = "bge-small"
    retrieval: str = "dense"
    reranker: str | None = None
    top_k: int = 5
    candidates: int = 50
    rerank_depth: int = 20  # the reranker reorders only the top rerank_depth candidates
    label: str | None = None

    def validate(self) -> None:
        if self.retrieval not in RETRIEVAL_MODES:
            raise ValueError(f"retrieval must be one of {RETRIEVAL_MODES}")
        if self.retrieval in ("dense", "hybrid") and not self.embedder:
            raise ValueError("dense and hybrid retrieval need an embedder")
        if self.top_k <= 0 or self.candidates < self.top_k:
            raise ValueError("need 0 < top_k <= candidates")
        if self.rerank_depth < self.top_k:
            raise ValueError("rerank_depth must be >= top_k")

    @property
    def chunker_key(self) -> str:
        params = "-".join(str(v) for _, v in sorted(self.chunk_params.items()))
        return f"{self.chunker}-{params}" if params else self.chunker

    @property
    def key(self) -> str:
        """Short stable id used in reports and cache paths."""
        d = self.to_dict()
        d.pop("label", None)
        h = hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:8]
        emb = self.embedder or "none"
        rr = f"{self.reranker}@{self.rerank_depth}" if self.reranker else "norerank"
        return f"{self.parser}|{self.chunker_key}|{emb}|{self.retrieval}|{rr}|k{self.top_k}|{h}"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["chunk_params"] = dict(self.chunk_params)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RagConfig":
        cfg = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        cfg.validate()
        return cfg

    def to_yaml(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RagConfig":
        return cls.from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


# The frozen default this project must beat. Written down before any run (see docs/phase1-spec.md).
# The API embedding arm (text-embedding-3-small) needs OPENAI_API_KEY; without it the default
# falls back to the local bge-small model and the report says so.
FROZEN_DEFAULT = RagConfig(
    parser="pymupdf-text",
    chunker="recursive",
    chunk_params={"size": 512, "overlap": 50},
    embedder="openai-small",
    retrieval="dense",
    reranker=None,
    top_k=5,
    label="frozen-default",
)
