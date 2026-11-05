"""PyMuPDF text-layout parser. Fast, local, born-digital PDFs only.

Blocks come from the PDF text layout. Headings are inferred from font size relative to
the document's body size. Tables are NOT extracted: this backend is the deliberate
"text only" arm of the parser comparison. Table text still appears, as paragraphs in
reading order, which is exactly the failure mode the 20-page gate is meant to measure.
"""

from __future__ import annotations

import re
import statistics
from pathlib import Path
from typing import Any

from ragtuner.schema import BlockType, ParsedDocument, RawBlock, assemble, hash_file

_LIST_START = re.compile(r"^\s*(?:[-*•–▪◦]|\d{1,3}[.)]|\([a-z0-9]{1,3}\))\s+")
_HEADING_SIZE_RATIO = 1.15
_HEADING_MAX_CHARS = 200


class PyMuPDFTextParser:
    name = "pymupdf-text"
    version = "1"
    suffixes = frozenset({".pdf"})

    def parse(self, path: Path, *, doc_id: str | None = None) -> ParsedDocument:
        import pymupdf

        doc = pymupdf.open(path)
        try:
            spans: list[tuple[str, float, float, tuple[float, float, float, float], int, bool]] = []
            # (text, max_font_size, char_weight, bbox, page_no, is_bold)
            for pno, page in enumerate(doc, start=1):
                for b in page.get_text("dict")["blocks"]:
                    if b.get("type") != 0:
                        continue
                    text, size, bold = _block_text(b)
                    if not text.strip():
                        continue
                    spans.append((text, size, float(len(text)), tuple(b["bbox"]), pno, bold))
            body_size = _body_font_size(spans)
            raw = [_classify(text, size, bbox, pno, bold, body_size) for text, size, _, bbox, pno, bold in spans]
            pages = doc.page_count
            meta = {k: v for k, v in (doc.metadata or {}).items() if v}
        finally:
            doc.close()
        return assemble(
            doc_id=doc_id or f"{path.stem}-{hash_file(path)[:8]}",
            source=str(path),
            content_hash=hash_file(path),
            parser=self.name,
            parser_version=self.version,
            raw_blocks=raw,
            pages=pages,
            metadata=meta,
        )


def _block_text(block: dict[str, Any]) -> tuple[str, float, bool]:
    lines_out: list[str] = []
    max_size = 0.0
    bold_chars = 0
    total_chars = 0
    for line in block.get("lines", []):
        parts = []
        for span in line.get("spans", []):
            t = span.get("text", "")
            if not t:
                continue
            parts.append(t)
            max_size = max(max_size, float(span.get("size", 0.0)))
            total_chars += len(t)
            if span.get("flags", 0) & 16:  # bold flag bit
                bold_chars += len(t)
        if parts:
            lines_out.append("".join(parts))
    text = _join_lines(lines_out)
    return text, max_size, total_chars > 0 and bold_chars / total_chars > 0.6


def _join_lines(lines: list[str]) -> str:
    """Join wrapped lines with spaces, de-hyphenating obvious soft breaks."""
    out = ""
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        if not out:
            out = ln
        elif out.endswith("-") and ln[:1].islower():
            out = out[:-1] + ln
        else:
            out = out + " " + ln
    return out


def _body_font_size(spans: list[tuple[str, float, float, Any, int, bool]]) -> float:
    if not spans:
        return 10.0
    # char-weighted median: the most common size by volume of text is the body size
    weighted: list[float] = []
    for _, size, weight, _, _, _ in spans:
        weighted.extend([size] * max(1, int(weight // 20)))
    return statistics.median(weighted)


def _classify(text: str, size: float, bbox: Any, pno: int, bold: bool, body_size: float) -> RawBlock:
    short = len(text) <= _HEADING_MAX_CHARS and "\n" not in text
    if short and size >= body_size * _HEADING_SIZE_RATIO:
        level = 1 if size >= body_size * 1.6 else 2
        return RawBlock(BlockType.HEADING, text, page=pno, level=level, bbox=bbox, metadata={"font_size": size})
    if short and bold and len(text) < 80 and not text.endswith("."):
        return RawBlock(BlockType.HEADING, text, page=pno, level=3, bbox=bbox, metadata={"font_size": size})
    if _LIST_START.match(text):
        return RawBlock(BlockType.LIST, text, page=pno, bbox=bbox, metadata={"font_size": size})
    return RawBlock(BlockType.PARAGRAPH, text, page=pno, bbox=bbox, metadata={"font_size": size})
