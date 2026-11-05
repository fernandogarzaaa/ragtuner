from __future__ import annotations

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

    def rerank(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        return [float(s) for s in self._model.predict([(query, t) for t in texts], batch_size=32, show_progress_bar=False)]


def get_reranker(name: str) -> CrossEncoderReranker:
    return CrossEncoderReranker(name)


def available_rerankers() -> list[str]:
    return sorted(_MODELS)
