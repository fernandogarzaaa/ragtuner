"""Recursive splitter: block boundaries, then sentences, then words, packed to a target
size in words with overlap carried from the tail of the previous chunk. The common
"recursive character splitter" default, expressed on the block model."""

from __future__ import annotations

import re

from ragtuner.chunk.base import Chunk, register, validate_chunks, word_count, word_spans
from ragtuner.schema import ParsedDocument

_SENT_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])")


class RecursiveChunker:
    name = "recursive"

    def __init__(self, size: int = 512, overlap: int = 50) -> None:
        if size <= 0 or overlap < 0 or overlap >= size:
            raise ValueError("need size > 0 and 0 <= overlap < size")
        self.size = size
        self.overlap = overlap

    @property
    def key(self) -> str:
        return f"recursive-{self.size}-{self.overlap}"

    def chunk(self, doc: ParsedDocument) -> list[Chunk]:
        # 1. atomic pieces: sentences within blocks, words within oversized sentences
        pieces: list[tuple[int, int]] = []
        for b in doc.blocks:
            for s, e in _split_block(doc.text, b.start, b.end, self.size):
                pieces.append((s, e))
        if not pieces:
            return []
        # 2. pack pieces greedily to size, with overlap from the previous chunk's tail
        out: list[Chunk] = []
        cur: list[tuple[int, int]] = []
        cur_words = 0
        i = 0
        while i < len(pieces):
            s, e = pieces[i]
            w = word_count(doc.text[s:e])
            if cur and cur_words + w > self.size:
                out.append(self._emit(doc, cur, len(out)))
                # overlap: keep trailing pieces whose words sum to <= overlap
                tail: list[tuple[int, int]] = []
                tw = 0
                for ps, pe in reversed(cur):
                    pw = word_count(doc.text[ps:pe])
                    if tw + pw > self.overlap:
                        break
                    tail.insert(0, (ps, pe))
                    tw += pw
                cur, cur_words = tail, tw
                continue
            cur.append((s, e))
            cur_words += w
            i += 1
        if cur:
            out.append(self._emit(doc, cur, len(out)))
        validate_chunks(doc, out)
        return out

    def _emit(self, doc: ParsedDocument, pieces: list[tuple[int, int]], idx: int) -> Chunk:
        start, end = pieces[0][0], pieces[-1][1]
        text = doc.text[start:end]
        blocks = [b for b in doc.blocks if b.start < end and b.end > start]
        return Chunk(
            id=f"{doc.id}:{self.key}:{idx}", doc_id=doc.id, text=text, start=start, end=end, index_text=text,
            block_ids=tuple(b.id for b in blocks), block_types=tuple(sorted({b.type.value for b in blocks})),
            page=blocks[0].page if blocks else None,
        )


def _split_block(text: str, start: int, end: int, size: int) -> list[tuple[int, int]]:
    """Sentences inside [start, end); any sentence longer than size is split by words."""
    out: list[tuple[int, int]] = []
    seg = text[start:end]
    pos = 0
    bounds = [m.end() for m in _SENT_END.finditer(seg)] + [len(seg)]
    for b in bounds:
        s_rel, e_rel = pos, b
        # trim whitespace
        while s_rel < e_rel and seg[s_rel].isspace():
            s_rel += 1
        while e_rel > s_rel and seg[e_rel - 1].isspace():
            e_rel -= 1
        pos = b
        if s_rel >= e_rel:
            continue
        if word_count(seg[s_rel:e_rel]) <= size:
            out.append((start + s_rel, start + e_rel))
        else:
            ws = word_spans(seg[s_rel:e_rel])
            for i in range(0, len(ws), size):
                win = ws[i : i + size]
                out.append((start + s_rel + win[0][0], start + s_rel + win[-1][1]))
    return out


register("recursive", RecursiveChunker)
