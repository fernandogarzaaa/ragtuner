"""text-embedding-3-small via OpenAI directly (OPENAI_API_KEY) or via OpenRouter
(OPENROUTER_API_KEY, model openai/text-embedding-3-small). Same model either way, so
both share one cache name. Stdlib HTTP with retries."""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np

from ragtuner.llm import _post_json


class OpenAISmallEmbedder:
    name = "text-embedding-3-small"
    dim = 1536

    def __init__(self) -> None:
        if os.environ.get("OPENAI_API_KEY"):
            self.url, self.model, key = "https://api.openai.com/v1/embeddings", "text-embedding-3-small", os.environ["OPENAI_API_KEY"]
        elif os.environ.get("OPENROUTER_API_KEY"):
            self.url, self.model, key = "https://openrouter.ai/api/v1/embeddings", "openai/text-embedding-3-small", os.environ["OPENROUTER_API_KEY"]
        else:
            raise RuntimeError("neither OPENAI_API_KEY nor OPENROUTER_API_KEY is set")
        self.headers = {"Authorization": f"Bearer {key}"}
        self.tokens = 0

    def embed(self, texts: Sequence[str], *, kind: str = "document") -> np.ndarray:
        # the API rejects empty strings; embed a single space instead
        inputs = [t if t.strip() else " " for t in texts]
        out = _post_json(self.url, {"model": self.model, "input": inputs}, self.headers)
        self.tokens += int((out.get("usage") or {}).get("total_tokens", 0))
        rows = sorted(out["data"], key=lambda d: d["index"])
        return np.asarray([r["embedding"] for r in rows], dtype=np.float32)
