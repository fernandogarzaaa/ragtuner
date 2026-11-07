"""Deterministic parser for .txt and .md files. No dependencies.

Recognizes ATX headings (#), pipe tables with a separator row, bullet and numbered
lists, and blank-line separated paragraphs. Used for tests and for corpora that are
already text.
"""

from __future__ import annotations

import re
from pathlib import Path

from ragtuner.schema import BlockType, ParsedDocument, RawBlock, Table, assemble, hash_file

_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_LIST_ITEM = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+\S")
_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")


class PlainTextParser:
    name = "plaintext"
    version = "1"
    suffixes = frozenset({".txt", ".md", ".markdown"})

    def parse(self, path: Path, *, doc_id: str | None = None) -> ParsedDocument:
        text = path.read_text(encoding="utf-8-sig")
        raw = list(_blocks_from_text(text))
        return assemble(
            doc_id=doc_id or f"{path.stem}-{hash_file(path)[:8]}",
            source=str(path),
            content_hash=hash_file(path),
            parser=self.name,
            parser_version=self.version,
            raw_blocks=raw,
            pages=None,
        )


def _blocks_from_text(text: str):
    lines = text.splitlines()
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        m = _HEADING.match(line)
        if m:
            yield RawBlock(BlockType.HEADING, m.group(2), level=len(m.group(1)))
            i += 1
            continue
        if "|" in line and i + 1 < n and _TABLE_SEP.match(lines[i + 1]):
            rows = []
            j = i
            while j < n and lines[j].strip() and "|" in lines[j]:
                if not _TABLE_SEP.match(lines[j]):
                    rows.append(_split_pipe_row(lines[j]))
                j += 1
            table = Table(rows=rows, header_rows=1 if rows else 0)
            yield RawBlock(BlockType.TABLE, table.to_markdown(), table=table)
            i = j
            continue
        if _LIST_ITEM.match(line):
            items = []
            j = i
            while j < n and lines[j].strip() and (_LIST_ITEM.match(lines[j]) or lines[j].startswith((" ", "\t"))):
                items.append(lines[j].rstrip())
                j += 1
            yield RawBlock(BlockType.LIST, "\n".join(items))
            i = j
            continue
        para = []
        j = i
        while j < n and lines[j].strip() and not _HEADING.match(lines[j]) and not _LIST_ITEM.match(lines[j]):
            if "|" in lines[j] and j + 1 < n and _TABLE_SEP.match(lines[j + 1]):
                break
            para.append(lines[j].rstrip())
            j += 1
        if not para:  # defensive: guarantees progress
            para = [line]
            j = i + 1
        yield RawBlock(BlockType.PARAGRAPH, " ".join(p.strip() for p in para))
        i = j


def _split_pipe_row(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in re.split(r"(?<!\\)\|", s)]
