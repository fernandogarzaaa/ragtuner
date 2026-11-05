"""Docling layout parser: tables, headings, lists, captions with page provenance.

Optional dependency: pip install "ragtuner[docling]". Docling pulls in PyTorch and
downloads layout models on first use, so this backend is imported lazily.

STATUS: written against the Docling 2.x item API (iterate_items, TableItem,
SectionHeaderItem, ListItem, TextItem, prov). It has NOT been executed in the
environment this file was authored in; the first run against a real PDF is the
verification step. Keep the mapping below small and defensive until that run passes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ragtuner.schema import BlockType, ParsedDocument, RawBlock, Table, assemble, hash_file


class DoclingParser:
    name = "docling"
    version = "1"
    suffixes = frozenset({".pdf", ".docx", ".pptx", ".html", ".md"})

    def __init__(self) -> None:
        try:
            from docling.document_converter import DocumentConverter
        except ImportError as e:  # pragma: no cover - depends on optional install
            raise ImportError("docling is not installed; pip install 'ragtuner[docling]'") from e
        self._converter = DocumentConverter()
        try:
            import docling

            self.version = str(getattr(docling, "__version__", "unknown"))
        except Exception:  # noqa: BLE001
            pass

    def parse(self, path: Path, *, doc_id: str | None = None) -> ParsedDocument:
        from docling_core.types.doc import (
            DocItemLabel,
            ListItem,
            SectionHeaderItem,
            TableItem,
            TextItem,
        )

        result = self._converter.convert(str(path))
        dl = result.document
        raw: list[RawBlock] = []
        for item, _level in dl.iterate_items():
            page, bbox = _provenance(item)
            if isinstance(item, TableItem):
                table = _table_from_item(item, dl)
                raw.append(RawBlock(BlockType.TABLE, table.to_markdown(), page=page, bbox=bbox, table=table))
            elif isinstance(item, SectionHeaderItem):
                raw.append(RawBlock(BlockType.HEADING, item.text, page=page, level=int(getattr(item, "level", 1)), bbox=bbox))
            elif isinstance(item, ListItem):
                raw.append(RawBlock(BlockType.LIST, item.text, page=page, bbox=bbox))
            elif isinstance(item, TextItem):
                label = getattr(item, "label", None)
                if label == DocItemLabel.CAPTION:
                    btype = BlockType.CAPTION
                elif label == DocItemLabel.TITLE:
                    btype = BlockType.HEADING
                else:
                    btype = BlockType.PARAGRAPH
                level = 1 if btype is BlockType.HEADING else None
                raw.append(RawBlock(btype, item.text, page=page, level=level, bbox=bbox))
        pages = None
        try:
            pages = len(dl.pages) or None
        except Exception:  # noqa: BLE001
            pass
        return assemble(
            doc_id=doc_id or f"{path.stem}-{hash_file(path)[:8]}",
            source=str(path),
            content_hash=hash_file(path),
            parser=self.name,
            parser_version=self.version,
            raw_blocks=raw,
            pages=pages,
        )


def _provenance(item: Any) -> tuple[int | None, tuple[float, float, float, float] | None]:
    prov = getattr(item, "prov", None) or []
    if not prov:
        return None, None
    p = prov[0]
    page = getattr(p, "page_no", None)
    bb = getattr(p, "bbox", None)
    bbox = None
    if bb is not None:
        try:
            bbox = (float(bb.l), float(bb.t), float(bb.r), float(bb.b))
        except AttributeError:
            bbox = None
    return page, bbox


def _table_from_item(item: Any, dl: Any) -> Table:
    """Prefer the dataframe export; fall back to the raw cell grid."""
    caption = None
    try:
        caption = item.caption_text(dl) or None
    except Exception:  # noqa: BLE001
        caption = None
    try:
        try:
            df = item.export_to_dataframe(doc=dl)
        except TypeError:
            df = item.export_to_dataframe()
        header = [str(c) for c in df.columns]
        body = [[_cell(v) for v in row] for row in df.values.tolist()]
        return Table(rows=[header] + body, header_rows=1, caption=caption)
    except Exception:  # noqa: BLE001
        pass
    data = getattr(item, "data", None)
    grid = getattr(data, "grid", None) or []
    rows = [[_cell(getattr(c, "text", "")) for c in row] for row in grid]
    header_rows = 0
    if rows and all(getattr(c, "column_header", False) for c in grid[0]):
        header_rows = 1
    return Table(rows=rows, header_rows=header_rows, caption=caption)


def _cell(v: Any) -> str:
    if v is None:
        return ""
    s = str(v)
    return "" if s == "nan" else s.strip()
