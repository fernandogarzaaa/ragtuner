from __future__ import annotations

import hashlib
from typing import Sequence

_MODELS = {
    "minilm": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "bge-base": "BAAI/bge-reranker-base",
}


class CrossEncoderReranker:
    def __init__(self, model: str = "minilm") -> None:
        from sentence_transformers import CrossEncoder

        self.key = model
        self.name = _MODELS.get(model, model)
        self._model = CrossEncoder(self.name)
        # (query, text) -> score. Configs that share a chunker see the same pairs again,
        # so the search would otherwise rescore identical pairs many times.
        self._cache: dict[tuple[str, str], float] = {}
        self.scored = 0
        self.cache_hits = 0

    def rerank(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        keys = [(query, hashlib.sha1(t.encode("utf-8")).hexdigest()) for t in texts]
        todo = [i for i, k in enumerate(keys) if k not in self._cache]
        if todo:
            scores = self._model.predict([(query, texts[i]) for i in todo], batch_size=32, show_progress_bar=False)
            for i, sc in zip(todo, scores):
                self._cache[keys[i]] = float(sc)
        self.scored += len(todo)
        self.cache_hits += len(texts) - len(todo)
        return [self._cache[k] for k in keys]


def get_reranker(name: str) -> CrossEncoderReranker:
    return CrossEncoderReranker(name)


def available_rerankers() -> list[str]:
    return sorted(_MODELS)
