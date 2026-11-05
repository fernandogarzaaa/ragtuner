"""Fixed word window with overlap over the canonical text. This is BETTER-RAG's chunker
(chunk_size words, chunk_overlap words) reproduced with offsets. The baseline arm."""

from __future__ import annotations

from ragtuner.chunk.base import Chunk, register, validate_chunks, word_spans
from ragtuner.schema import ParsedDocument


class FixedWordChunker:
    name = "fixed"

    def __init__(self, size: int = 120, overlap: int = 24) -> None:
        if size <= 0 or overlap < 0 or overlap >= size:
            raise ValueError("need size > 0 and 0 <= overlap < size")
        self.size = size
        self.overlap = overlap

    @property
    def key(self) -> str:
        return f"fixed-{self.size}-{self.overlap}"

    def chunk(self, doc: ParsedDocument) -> list[Chunk]:
        spans = word_spans(doc.text)
        if not spans:
            return []
        step = self.size - self.overlap
        out: list[Chunk] = []
        i = 0
        n = len(spans)
        while True:
            window = spans[i : i + self.size]
            start, end = window[0][0], window[-1][1]
            text = doc.text[start:end]
            out.append(
                Chunk(
                    id=f"{doc.id}:{self.key}:{len(out)}", doc_id=doc.id, text=text, start=start, end=end,
                    index_text=text, block_ids=_blocks_in(doc, start, end), block_types=_types_in(doc, start, end),
                    page=_page_at(doc, start),
                )
            )
            if i + self.size >= n:
                break
            i += step
        validate_chunks(doc, out)
        return out


def _blocks_in(doc: ParsedDocument, start: int, end: int) -> tuple[str, ...]:
    return tuple(b.id for b in doc.blocks if b.start < end and b.end > start)


def _types_in(doc: ParsedDocument, start: int, end: int) -> tuple[str, ...]:
    return tuple(sorted({b.type.value for b in doc.blocks if b.start < end and b.end > start}))


def _page_at(doc: ParsedDocument, pos: int) -> int | None:
    for b in doc.blocks:
        if b.start <= pos < b.end or b.start >= pos:
            return b.page
    return None


register("fixed", FixedWordChunker)
