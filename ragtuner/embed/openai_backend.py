"""OpenAI text-embedding-3-small via raw HTTPS (stdlib only). Needs OPENAI_API_KEY."""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Sequence

import numpy as np


class OpenAISmallEmbedder:
    name = "text-embedding-3-small"
    dim = 1536

    def __init__(self) -> None:
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

    def embed(self, texts: Sequence[str], *, kind: str = "document") -> np.ndarray:
        req = urllib.request.Request(
            "https://api.openai.com/v1/embeddings",
            data=json.dumps({"model": self.name, "input": list(texts)}).encode(),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = json.load(resp)
        rows = sorted(payload["data"], key=lambda d: d["index"])
        return np.asarray([r["embedding"] for r in rows], dtype=np.float32)
