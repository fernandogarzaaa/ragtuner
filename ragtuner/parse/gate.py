"""The 20-page parser gate.

Before any tuning run, sample pages from the corpus, run every candidate parser on each
sampled page, and write a labeling sheet plus page images. A human fills in the gold
table cells per page. `score()` then reports cell-level precision, recall and F1 per
parser, and whether the gate threshold is met.

Sampling: uniform random pages plus an oversample of pages PyMuPDF flags as containing
table-like layout. The oversample is biased toward tables PyMuPDF can see, so the sheet
records which pool each page came from.
"""

from __future__ import annotations

import json
import random
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ragtuner.parse.base import get_parser
from ragtuner.schema import ParsedDocument, Table

_NUM_JUNK = re.compile(r"[\$,%\s()]")


@dataclass(frozen=True)
class SampledPage:
    source: str
    page: int  # 1-indexed
    pool: str  # "random" or "table_candidate"


@dataclass
class GateReport:
    threshold: float
    per_parser: dict[str, dict[str, float]] = field(default_factory=dict)
    per_page: list[dict[str, Any]] = field(default_factory=list)

    def passed(self, parser: str) -> bool:
        return self.per_parser.get(parser, {}).get("cell_f1", 0.0) >= self.threshold

    def to_markdown(self) -> str:
        lines = ["| parser | pages | gold cells | cell precision | cell recall | cell F1 | tables found | gate |", "|---|---|---|---|---|---|---|---|"]
        for name, m in sorted(self.per_parser.items()):
            gate = "pass" if self.passed(name) else "fail"
            lines.append(
                f"| {name} | {int(m['pages'])} | {int(m['gold_cells'])} | {m['cell_precision']:.3f} | "
                f"{m['cell_recall']:.3f} | {m['cell_f1']:.3f} | {int(m['pred_tables'])} | {gate} |"
            )
        lines.append("")
        lines.append(f"Threshold: cell F1 >= {self.threshold:.2f}")
        return "\n".join(lines)


def sample_pages(pdfs: Iterable[Path], n: int, *, seed: int = 0, table_fraction: float = 0.6) -> list[SampledPage]:
    import pymupdf

    rng = random.Random(seed)
    all_pages: list[tuple[str, int]] = []
    table_pages: list[tuple[str, int]] = []
    for pdf in pdfs:
        doc = pymupdf.open(pdf)
        try:
            for pno, page in enumerate(doc, start=1):
                all_pages.append((str(pdf), pno))
                try:
                    if page.find_tables().tables:
                        table_pages.append((str(pdf), pno))
                except Exception:  # noqa: BLE001 - table finder can choke on odd pages
                    pass
        finally:
            doc.close()
    if not all_pages:
        return []
    n = min(n, len(all_pages))
    n_table = min(int(round(n * table_fraction)), len(table_pages))
    chosen: dict[tuple[str, int], str] = {}
    for key in rng.sample(table_pages, n_table):
        chosen[key] = "table_candidate"
    remaining = [p for p in all_pages if p not in chosen]
    for key in rng.sample(remaining, min(n - len(chosen), len(remaining))):
        chosen[key] = "random"
    out = [SampledPage(src, pno, pool) for (src, pno), pool in chosen.items()]
    out.sort(key=lambda s: (s.source, s.page))
    return out


def extract_page(pdf: Path, page: int, out: Path) -> Path:
    """Write a single-page PDF so parsers only pay for the sampled page."""
    import pymupdf

    src = pymupdf.open(pdf)
    dst = pymupdf.open()
    try:
        dst.insert_pdf(src, from_page=page - 1, to_page=page - 1)
        dst.save(out)
    finally:
        dst.close()
        src.close()
    return out


def render_page(pdf: Path, page: int, out: Path, *, zoom: float = 2.0) -> Path:
    import pymupdf

    doc = pymupdf.open(pdf)
    try:
        pix = doc[page - 1].get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
        pix.save(out)
    finally:
        doc.close()
    return out


def build_sheet(samples: list[SampledPage], parsers: list[str], out_dir: Path) -> dict[str, Path]:
    """Run parsers on each sampled page. Writes:
    sheet.jsonl        one row per (page, parser) with the tables each parser found
    gold_template.jsonl one row per page for the human to fill in: {"tables": [[["cell", ...], ...]]}
    pages/*.png        rendered pages for labeling
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pages_dir = out_dir / "pages"
    pages_dir.mkdir(exist_ok=True)
    backends = {name: get_parser(name) for name in parsers}
    sheet_rows: list[dict[str, Any]] = []
    gold_rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmp:
        for s in samples:
            pdf = Path(s.source)
            tag = f"{pdf.stem}_p{s.page}"
            single = extract_page(pdf, s.page, Path(tmp) / f"{tag}.pdf")
            png = render_page(pdf, s.page, pages_dir / f"{tag}.png")
            gold_rows.append({"source": s.source, "page": s.page, "pool": s.pool, "image": str(png), "tables": []})
            for name, backend in backends.items():
                row: dict[str, Any] = {"source": s.source, "page": s.page, "pool": s.pool, "parser": name}
                try:
                    doc = backend.parse(single, doc_id=tag)
                    row["tables"] = [t.rows for t in doc.tables()]
                    row["blocks"] = len(doc.blocks)
                    row["text_chars"] = len(doc.text)
                except Exception as e:  # noqa: BLE001 - record parser failures, do not abort the sheet
                    row["tables"] = []
                    row["error"] = f"{type(e).__name__}: {e}"
                sheet_rows.append(row)
    sheet = out_dir / "sheet.jsonl"
    gold = out_dir / "gold_template.jsonl"
    _write_jsonl(sheet, sheet_rows)
    _write_jsonl(gold, gold_rows)
    return {"sheet": sheet, "gold_template": gold, "pages": pages_dir}


def normalize_cell(cell: str) -> str:
    s = cell.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("−", "-").replace("—", "-").replace("–", "-")
    # numeric cells: strip currency, thousands separators, percent and spaces so
    # "$ 1,234" == "1234" and "(12)" == "-12"
    if re.fullmatch(r"[\$\(\)\-\d,\.\s%]+", s) and any(ch.isdigit() for ch in s):
        neg = s.startswith("(") and s.endswith(")") or s.startswith("-")
        s = _NUM_JUNK.sub("", s).lstrip("-")
        s = s.rstrip(".")
        if neg and s:
            s = "-" + s
    return s


def _cell_multiset(rows: list[list[str]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        for c in r:
            k = normalize_cell(str(c))
            if k:
                out[k] = out.get(k, 0) + 1
    return out


def _prf(pred: dict[str, int], gold: dict[str, int]) -> tuple[int, int, int]:
    """Return (true positives, predicted count, gold count) over cell multisets."""
    tp = sum(min(n, gold.get(k, 0)) for k, n in pred.items())
    return tp, sum(pred.values()), sum(gold.values())


def score_page(pred_tables: list[list[list[str]]], gold_tables: list[list[list[str]]]) -> tuple[int, int, int]:
    """Greedy one-to-one alignment of predicted to gold tables by cell overlap, then
    cell-level counts. Unmatched predicted tables count as false positives."""
    gold_sets = [_cell_multiset(t) for t in gold_tables]
    pred_sets = [_cell_multiset(t) for t in pred_tables]
    used: set[int] = set()
    tp_total = 0
    pred_total = sum(sum(p.values()) for p in pred_sets)
    gold_total = sum(sum(g.values()) for g in gold_sets)
    for g in gold_sets:
        best, best_tp = -1, 0
        for i, p in enumerate(pred_sets):
            if i in used:
                continue
            tp, _, _ = _prf(p, g)
            if tp > best_tp:
                best, best_tp = i, tp
        if best >= 0:
            used.add(best)
            tp_total += best_tp
    return tp_total, pred_total, gold_total


def score(sheet_path: Path, gold_path: Path, *, threshold: float = 0.8) -> GateReport:
    sheet = _read_jsonl(sheet_path)
    gold = {(r["source"], r["page"]): r for r in _read_jsonl(gold_path)}
    report = GateReport(threshold=threshold)
    agg: dict[str, dict[str, float]] = {}
    for row in sheet:
        key = (row["source"], row["page"])
        g = gold.get(key)
        if g is None or g.get("tables") is None:
            continue
        tp, pred_n, gold_n = score_page(row.get("tables", []), g["tables"])
        a = agg.setdefault(row["parser"], {"tp": 0, "pred": 0, "gold": 0, "pages": 0, "pred_tables": 0})
        a["tp"] += tp
        a["pred"] += pred_n
        a["gold"] += gold_n
        a["pages"] += 1
        a["pred_tables"] += len(row.get("tables", []))
        report.per_page.append(
            {"source": row["source"], "page": row["page"], "parser": row["parser"], "tp": tp, "pred": pred_n, "gold": gold_n}
        )
    for name, a in agg.items():
        p = a["tp"] / a["pred"] if a["pred"] else 0.0
        r = a["tp"] / a["gold"] if a["gold"] else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        report.per_parser[name] = {
            "pages": a["pages"],
            "gold_cells": a["gold"],
            "pred_cells": a["pred"],
            "pred_tables": a["pred_tables"],
            "cell_precision": p,
            "cell_recall": r,
            "cell_f1": f1,
        }
    return report


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


__all__ = ["GateReport", "SampledPage", "ParsedDocument", "Table", "build_sheet", "sample_pages", "score", "score_page"]
