# ragtuner

Eval-driven RAG configuration tuner. Point it at a corpus; it profiles the data,
generates a span-anchored eval set, searches the config space, diagnoses failures,
and exports the winning config as code.

Status: Phase 1, parser layer. See `docs/phase1-spec.md` for every decision.

## What exists today

- `ragtuner.schema`: the intermediate document model. Typed blocks (paragraph, heading,
  table, list, caption) with page, bbox, structured table cells, and exact character
  offsets into one canonical text. Gold eval answers anchor to these offsets, so one
  eval set scores every chunking strategy fairly.
- `ragtuner.parse`: pluggable parser backends behind one interface.
  - `plaintext` (.txt, .md): deterministic, no dependencies.
  - `pymupdf-text` (.pdf): fast local text layout, headings by font size. Deliberately
    extracts no tables; it is the text-only arm of the comparison.
  - `docling` (.pdf and more): layout model with tables. Optional install, not yet
    executed in CI. First real-PDF run is the verification step.
- `ragtuner.cache.ParseCache`: content-addressed cache keyed by parser, parser
  version, and file hash.
- `ragtuner.parse.gate`: the 20-page parser gate. Samples pages, builds a labeling
  sheet with page images, and scores cell-level precision/recall/F1 per parser once a
  human fills in the gold cells.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pip install -e ".[docling]"   # optional, pulls in PyTorch
```

## Use

```bash
ragtuner parsers
ragtuner parse report.pdf --parser pymupdf-text --out report.json
ragtuner gate sample corpus/ --n 20 --parsers pymupdf-text,docling --out gate_out
# fill in gate_out/gold_template.jsonl using the PNGs in gate_out/pages/
ragtuner gate score --sheet gate_out/sheet.jsonl --gold gate_out/gold.jsonl --threshold 0.8
```

## Gate rule

Docling must reach cell F1 >= 0.80 on the FinanceBench sample or a hosted parser arm is
added to the config space. Nothing downstream runs before the gate has a result.

## Verified so far (2026-11-05)

- 14 tests pass on Python 3.13; the Docling test skips when Docling is not installed.
- IRS Publication 334 (57 pages, born-digital) parses with `pymupdf-text` in about
  0.6 seconds, all pages covered, 1,563 blocks.
- `ragtuner gate sample` on that PDF produced 20 pages (12 table candidates, 8 random)
  with page images in about 26 seconds. A visual check of one candidate page confirmed
  a real table.

## Known gaps

- `pymupdf-text` heading heuristic tags table-of-contents lines with dot leaders and
  some bullet lines as headings. Left as is: it is the text-only baseline arm, and the
  eval, not the parser, decides whether that matters.
- The Docling backend has not been executed yet. Installing it and running
  `pytest tests/test_docling_backend.py` against a real PDF is the next verification.
