"""Build the Phase 1 corpora from public sources already downloaded locally.

Usage: python scripts/prepare_corpora.py --downloads DIR --out data/
Layout written:
  data/financebench/docs/*.pdf   data/financebench/doc_ids.json   data/financebench/real.jsonl (FinanceBench rows)
  data/irs/docs/p334.pdf         data/irs/qa.jsonl (question/answer pairs from IRS FAQ pages)
  data/slides/docs/*.pdf         (no real queries available; synthetic only)
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path


def financebench(downloads: Path, out: Path, n_docs: int, max_pages: int) -> None:
    import pymupdf

    src = downloads / "financebench"
    rows = [json.loads(line) for line in (src / "data" / "financebench_open_source.jsonl").read_text().splitlines() if line.strip()]
    counts = Counter(r["doc_name"] for r in rows)
    chosen: list[str] = []
    for name, _ in counts.most_common():
        pdf = src / "pdfs" / f"{name}.pdf"
        if not pdf.exists():
            continue
        try:
            pages = pymupdf.open(pdf).page_count
        except Exception:  # noqa: BLE001
            continue
        if pages > max_pages:
            continue
        chosen.append(name)
        if len(chosen) >= n_docs:
            break
    docs = out / "financebench" / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    for name in chosen:
        shutil.copy(src / "pdfs" / f"{name}.pdf", docs / f"{name}.pdf")
    (out / "financebench" / "doc_ids.json").write_text(json.dumps({f"{name}.pdf": name for name in chosen}, indent=2))
    kept = [r for r in rows if r["doc_name"] in chosen]
    with (out / "financebench" / "real.jsonl").open("w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    print(f"financebench: {len(chosen)} docs, {len(kept)} real questions, pages<= {max_pages}")


def irs(downloads: Path, out: Path) -> None:
    docs = out / "irs" / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    shutil.copy(downloads / "p334.pdf", docs / "p334.pdf")
    shutil.copy(downloads / "irs_qa.jsonl", out / "irs" / "qa.jsonl")
    n = sum(1 for _ in (out / "irs" / "qa.jsonl").open())
    print(f"irs: 1 doc, {n} qa pairs")


def slides(downloads: Path, out: Path) -> None:
    docs = out / "slides" / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    n = 0
    for pdf in sorted((downloads / "slides").glob("*.pdf")):
        shutil.copy(pdf, docs / pdf.name)
        n += 1
    print(f"slides: {n} docs (synthetic questions only)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--downloads", required=True)
    ap.add_argument("--out", default="data")
    ap.add_argument("--n-docs", type=int, default=8)
    ap.add_argument("--max-pages", type=int, default=160)
    a = ap.parse_args()
    d, o = Path(a.downloads), Path(a.out)
    financebench(d, o, a.n_docs, a.max_pages)
    irs(d, o)
    slides(d, o)


if __name__ == "__main__":
    main()
