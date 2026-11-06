# Phase 1 results

Status: runs in progress. Sections fill in as each corpus completes. Everything here
comes from runs recorded in `data/` and `runs/` of this repo or from the test suite.

## Environment

4 CPU cores, no GPU, no Anthropic or OpenAI credentials. Consequences:

- Synthetic questions came from the heuristic generator (lexically leaky; favors sparse
  retrieval). The Claude generator exists and is unit-tested with fake callers but has
  not been run live.
- The frozen default's embedding arm (text-embedding-3-small) was unavailable; the
  default ran with the local bge-small model instead, and the report notes this.
- Generation judging (stage 3) was skipped.

## Parser gate (FinanceBench)

12 pages sampled (7 table candidates, 5 random) from 8 FinanceBench filings. Four pages
with complete, small tables were hand-labeled (106 gold cells). Scoring is cell-level
after numeric normalization.

| parser | pages | gold cells | cell precision | cell recall | cell F1 | tables found | gate |
|---|---|---|---|---|---|---|---|
| docling | 4 | 106 | 0.922 | 0.887 | 0.904 | 5 | pass |
| pymupdf-text | 4 | 106 | 0.000 | 0.000 | 0.000 | 0 | fail |

Docling passes the 0.80 threshold, so no hosted parser arm was added. Its misses are
mostly merged hierarchical headers ("June 30, 2023.Carrying Value" where gold lists the
two levels as separate cells) and currency symbols glued to numbers, both of which are
arguably correct extractions. Support is four pages; treat the number as indicative.
Artifacts: `data/gate/financebench/`.

Docling throughput on this CPU: about 2 seconds per page (57-page IRS publication in
116 s; 151-page 10-K in 424 s).

## Corpus runs

(filled in below as runs finish)
