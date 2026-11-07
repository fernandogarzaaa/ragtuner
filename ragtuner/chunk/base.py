from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from ragtuner.schema import ParsedDocument

_WORD = re.compile(r"\S+")


def word_count(text: str) -> int:
    return len(_WORD.findall(text))


def word_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) char offsets of each whitespace-delimited token."""
    return [(m.start(), m.end()) for m in _WORD.finditer(text)]


@dataclass(frozen=True)
class Chunk:
    """A retrievable unit.

    text is exactly doc.text[start:end]. index_text is what gets embedded or indexed and
    may carry extra context (heading path, repeated table header). Scoring uses the span,
    never index_text, so added context can never fake relevance.
    """

    id: str
    doc_id: str
    text: str
    start: int
    end: int
    index_text: str
    block_ids: tuple[str, ...] = ()
    block_types: tuple[str, ...] = ()
    page: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def words(self) -> int:
        return word_count(self.text)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "doc_id": self.doc_id, "text": self.text, "start": self.start, "end": self.end,
            "index_text": self.index_text, "block_ids": list(self.block_ids),
            "block_types": list(self.block_types), "page": self.page, "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Chunk":
        return cls(
            id=d["id"], doc_id=d["doc_id"], text=d["text"], start=d["start"], end=d["end"],
            index_text=d["index_text"], block_ids=tuple(d.get("block_ids", ())),
            block_types=tuple(d.get("block_types", ())), page=d.get("page"), metadata=dict(d.get("metadata", {})),
        )


class Chunker(Protocol):
    name: str

    @property
    def key(self) -> str: ...

    def chunk(self, doc: ParsedDocument) -> list[Chunk]: ...


def validate_chunks(doc: ParsedDocument, chunks: list[Chunk]) -> None:
    for c in chunks:
        if doc.text[c.start : c.end] != c.text:
            raise ValueError(f"chunk {c.id} text does not match its span")
        if c.doc_id != doc.id:
            raise ValueError(f"chunk {c.id} belongs to {c.doc_id}, not {doc.id}")


_REGISTRY: dict[str, Callable[..., Chunker]] = {}


def register(name: str, factory: Callable[..., Chunker]) -> None:
    _REGISTRY[name] = factory


def get_chunker(name: str, **params: Any) -> Chunker:
    if name not in _REGISTRY:
        raise KeyError(f"unknown chunker {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**params)


def available_chunkers() -> list[str]:
    return sorted(_REGISTRY)
