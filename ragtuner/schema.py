"""Intermediate document model shared by every parser backend.

Every parser normalizes its output into a ParsedDocument: an ordered list of typed
Blocks plus one canonical text string. Each block records its character span inside
that canonical text, so eval gold answers can be anchored to spans that are
independent of any chunking strategy. That independence is what lets one eval set
score every config fairly.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

BLOCK_SEPARATOR = "\n\n"


class BlockType(str, Enum):
    PARAGRAPH = "paragraph"
    HEADING = "heading"
    TABLE = "table"
    LIST = "list"
    CAPTION = "caption"
    OTHER = "other"


@dataclass(frozen=True)
class Table:
    """A table as rows of cell strings. Row 0..header_rows-1 are header rows."""

    rows: list[list[str]]
    header_rows: int = 0
    caption: str | None = None

    def to_markdown(self) -> str:
        if not self.rows:
            return ""
        width = max(len(r) for r in self.rows)
        rows = [list(r) + [""] * (width - len(r)) for r in self.rows]
        header = rows[0] if self.header_rows else [f"col{i + 1}" for i in range(width)]
        body = rows[1:] if self.header_rows else rows
        lines = ["| " + " | ".join(_md_cell(c) for c in header) + " |"]
        lines.append("|" + "---|" * width)
        for r in body:
            lines.append("| " + " | ".join(_md_cell(c) for c in r) + " |")
        if self.caption:
            lines.insert(0, self.caption)
        return "\n".join(lines)

    def cells(self) -> list[str]:
        return [c for r in self.rows for c in r]


def _md_cell(cell: str) -> str:
    return cell.replace("\n", " ").replace("|", "\\|").strip()


@dataclass(frozen=True)
class RawBlock:
    """What a parser backend emits. assemble() turns these into Blocks with offsets."""

    type: BlockType
    text: str
    page: int | None = None
    level: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    table: Table | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Block:
    id: str
    type: BlockType
    text: str
    start: int
    end: int
    page: int | None = None
    level: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    table: Table | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["type"] = self.type.value
        d["bbox"] = list(self.bbox) if self.bbox else None
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Block":
        table = Table(**d["table"]) if d.get("table") else None
        bbox = tuple(d["bbox"]) if d.get("bbox") else None
        return cls(
            id=d["id"],
            type=BlockType(d["type"]),
            text=d["text"],
            start=d["start"],
            end=d["end"],
            page=d.get("page"),
            level=d.get("level"),
            bbox=bbox,  # type: ignore[arg-type]
            table=table,
            metadata=dict(d.get("metadata", {})),
        )


@dataclass(frozen=True)
class ParsedDocument:
    id: str
    source: str
    content_hash: str
    parser: str
    parser_version: str
    text: str
    blocks: list[Block]
    pages: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Offsets must reproduce block text exactly, in order, without overlap."""
        prev_end = -1
        for b in self.blocks:
            if b.start < 0 or b.end > len(self.text) or b.start > b.end:
                raise ValueError(f"block {b.id} has offsets outside the document text")
            if b.start < prev_end:
                raise ValueError(f"block {b.id} overlaps the previous block")
            if self.text[b.start : b.end] != b.text:
                raise ValueError(f"block {b.id} text does not match its span")
            prev_end = b.end

    def with_id(self, new_id: str) -> "ParsedDocument":
        """Same content under another document id (block ids follow the document id)."""
        if new_id == self.id:
            return self
        from dataclasses import replace

        blocks = [replace(b, id=f"{new_id}:b{i}") for i, b in enumerate(self.blocks)]
        return replace(self, id=new_id, blocks=blocks)

    def blocks_of(self, *types: BlockType) -> list[Block]:
        wanted = set(types)
        return [b for b in self.blocks if b.type in wanted]

    def tables(self) -> list[Table]:
        return [b.table for b in self.blocks if b.table is not None]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "content_hash": self.content_hash,
            "parser": self.parser,
            "parser_version": self.parser_version,
            "pages": self.pages,
            "metadata": self.metadata,
            "text": self.text,
            "blocks": [b.to_dict() for b in self.blocks],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ParsedDocument":
        doc = cls(
            id=d["id"],
            source=d["source"],
            content_hash=d["content_hash"],
            parser=d["parser"],
            parser_version=d["parser_version"],
            text=d["text"],
            blocks=[Block.from_dict(b) for b in d["blocks"]],
            pages=d.get("pages"),
            metadata=dict(d.get("metadata", {})),
        )
        doc.validate()
        return doc

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ParsedDocument":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def assemble(
    *,
    doc_id: str,
    source: str,
    content_hash: str,
    parser: str,
    parser_version: str,
    raw_blocks: Iterable[RawBlock],
    pages: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> ParsedDocument:
    """Build a ParsedDocument from raw blocks, computing the canonical text and offsets.

    Empty blocks are dropped. Block text is stripped of surrounding whitespace so that
    offsets stay stable across backends that differ only in trailing newlines.
    """
    blocks: list[Block] = []
    parts: list[str] = []
    cursor = 0
    index = 0
    for rb in raw_blocks:
        text = rb.text.strip()
        if not text:
            continue
        if index > 0:
            parts.append(BLOCK_SEPARATOR)
            cursor += len(BLOCK_SEPARATOR)
        start = cursor
        end = start + len(text)
        blocks.append(
            Block(
                id=f"{doc_id}:b{index}",
                type=rb.type,
                text=text,
                start=start,
                end=end,
                page=rb.page,
                level=rb.level,
                bbox=rb.bbox,
                table=rb.table,
                metadata=dict(rb.metadata),
            )
        )
        parts.append(text)
        cursor = end
        index += 1
    doc = ParsedDocument(
        id=doc_id,
        source=source,
        content_hash=content_hash,
        parser=parser,
        parser_version=parser_version,
        text="".join(parts),
        blocks=blocks,
        pages=pages,
        metadata=dict(metadata or {}),
    )
    doc.validate()
    return doc


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
