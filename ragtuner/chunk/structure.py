"""Structure-aware chunker.

Rules, in order:
- A table is one chunk. If it is larger than max_words it is split by rows and the
  header rows are repeated in index_text (never in the span text) so every piece stays
  interpretable.
- Headings are not chunks. They prefix index_text of the chunks under them as a path
  ("Chapter 2 > Accounting Periods"), and attach to the following content.
- Paragraphs and list blocks under the same heading are packed up to target_words.
  A single oversized block is split by sentences like the recursive chunker.
"""

from __future__ import annotations

from ragtuner.chunk.base import Chunk, register, validate_chunks, word_count
from ragtuner.chunk.recursive import _split_block
from ragtuner.schema import Block, BlockType, ParsedDocument


class StructureChunker:
    name = "structure"

    def __init__(self, target_words: int = 300, max_words: int = 600, heading_prefix: bool = True) -> None:
        if target_words <= 0 or max_words < target_words:
            raise ValueError("need 0 < target_words <= max_words")
        self.target_words = target_words
        self.max_words = max_words
        self.heading_prefix = heading_prefix

    @property
    def key(self) -> str:
        return f"structure-{self.target_words}-{self.max_words}{'-h' if self.heading_prefix else ''}"

    def chunk(self, doc: ParsedDocument) -> list[Chunk]:
        out: list[Chunk] = []
        path: list[tuple[int, str]] = []  # (level, heading text)
        group: list[Block] = []
        group_words = 0

        def flush() -> None:
            nonlocal group, group_words
            if group:
                out.append(self._emit(doc, group, path, len(out)))
            group, group_words = [], 0

        for b in doc.blocks:
            if b.type is BlockType.HEADING:
                flush()
                level = b.level or 1
                path = [(lv, t) for lv, t in path if lv < level] + [(level, b.text)]
                continue
            if b.type is BlockType.TABLE:
                flush()
                out.extend(self._table_chunks(doc, b, path, len(out)))
                continue
            w = word_count(b.text)
            if w > self.max_words:
                flush()
                for s, e in _packed_sentences(doc, b, self.target_words):
                    out.append(self._emit_span(doc, s, e, [b], path, len(out)))
                continue
            if group and group_words + w > self.target_words:
                flush()
            group.append(b)
            group_words += w
        flush()
        validate_chunks(doc, out)
        return out

    def _prefix(self, path: list[tuple[int, str]]) -> str:
        if not self.heading_prefix or not path:
            return ""
        return " > ".join(t for _, t in path) + "\n"

    def _emit(self, doc: ParsedDocument, blocks: list[Block], path: list[tuple[int, str]], idx: int) -> Chunk:
        return self._emit_span(doc, blocks[0].start, blocks[-1].end, blocks, path, idx)

    def _emit_span(self, doc: ParsedDocument, start: int, end: int, blocks: list[Block], path: list[tuple[int, str]], idx: int) -> Chunk:
        text = doc.text[start:end]
        return Chunk(
            id=f"{doc.id}:{self.key}:{idx}", doc_id=doc.id, text=text, start=start, end=end,
            index_text=self._prefix(path) + text,
            block_ids=tuple(b.id for b in blocks), block_types=tuple(sorted({b.type.value for b in blocks})),
            page=blocks[0].page, metadata={"heading_path": [t for _, t in path]},
        )

    def _table_chunks(self, doc: ParsedDocument, b: Block, path: list[tuple[int, str]], idx: int) -> list[Chunk]:
        if word_count(b.text) <= self.max_words or b.table is None or len(b.table.rows) <= 2:
            return [self._emit_span(doc, b.start, b.end, [b], path, idx)]
        # split a big table by markdown lines, repeating the header in index_text
        lines = b.text.split("\n")
        header_lines = []
        body_start = 0
        if b.table.caption and lines and lines[0] == b.table.caption:
            header_lines.append(lines[0])
            body_start = 1
        header_lines.extend(lines[body_start : body_start + 2])  # header row + separator
        body = lines[body_start + 2 :]
        header_text = "\n".join(header_lines)
        out: list[Chunk] = []
        pos = b.start
        # compute offsets of each body line inside doc.text
        line_spans: list[tuple[int, int]] = []
        cursor = b.start
        for ln in lines:
            line_spans.append((cursor, cursor + len(ln)))
            cursor += len(ln) + 1
        body_spans = line_spans[body_start + 2 :]
        group: list[tuple[int, int]] = []
        words = 0
        for (s, e), ln in zip(body_spans, body):
            w = word_count(ln)
            if group and words + w > self.target_words:
                out.append(self._table_piece(doc, b, group, header_text, path, idx + len(out)))
                group, words = [], 0
            group.append((s, e))
            words += w
        if group:
            out.append(self._table_piece(doc, b, group, header_text, path, idx + len(out)))
        del pos
        return out

    def _table_piece(self, doc: ParsedDocument, b: Block, spans: list[tuple[int, int]], header: str, path: list[tuple[int, str]], idx: int) -> Chunk:
        start, end = spans[0][0], spans[-1][1]
        text = doc.text[start:end]
        return Chunk(
            id=f"{doc.id}:{self.key}:{idx}", doc_id=doc.id, text=text, start=start, end=end,
            index_text=self._prefix(path) + header + "\n" + text,
            block_ids=(b.id,), block_types=(BlockType.TABLE.value,), page=b.page,
            metadata={"heading_path": [t for _, t in path], "table_piece": True},
        )


def _packed_sentences(doc: ParsedDocument, b: Block, target: int) -> list[tuple[int, int]]:
    pieces = _split_block(doc.text, b.start, b.end, target)
    out: list[tuple[int, int]] = []
    cur: list[tuple[int, int]] = []
    words = 0
    for s, e in pieces:
        w = word_count(doc.text[s:e])
        if cur and words + w > target:
            out.append((cur[0][0], cur[-1][1]))
            cur, words = [], 0
        cur.append((s, e))
        words += w
    if cur:
        out.append((cur[0][0], cur[-1][1]))
    return out


register("structure", StructureChunker)
