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

All runs: 4-core CPU, heuristic synthetic questions anchored on Docling text, selection
on synthetic recall@5, real questions held out. Gold spans are projected onto
PyMuPDF text for PyMuPDF configs (token alignment; counts in each report's notes).

### IRS Publication 334 (57 pages, 1 document)

Artifacts: `runs/irs/`. 100 synthetic questions (90 answerable, 10 unanswerable).
No real set: the 62 IRS FAQ question/answer pairs could not be anchored to reliable gold
spans by lexical matching (hand inspection showed wrong matches), so they were dropped
rather than reported as weak gold.

| config | recall@5 (synthetic) |
|---|---|
| frozen default: pymupdf-text, recursive 512/50, bge-small, dense | 0.778 |
| staged winner: docling, structure 300/600, BM25 | 0.933 |
| full-grid best: docling, structure 300/600, bge-small, hybrid | 0.944 |

- Regret of the staged search against the 64-config grid: 0.011 recall@5. The spec's
  criterion was within 1 to 2 points; this passes. Staged search took 47 evaluations and
  about 2 minutes; the grid took about 15 minutes more.
- The winner's 6 misses are all rank failures (retrieved, not in top 5). The default's
  20 misses are 17 rank and 3 deep (never in the top 50).
- `ragtuner replay` rebuilt the exported config and reproduced the tuner's top-50 chunk
  ids for all 100 questions.
- Caveat: the heuristic generator copies rare words from the gold span into the
  question, which favors BM25. The size of the gap over the dense default is inflated
  by that; the direction is not evidence about real users.
