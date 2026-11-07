# Phase 1 results

Status: complete. All three corpus runs and the FinanceBench cross-validation are recorded below. Everything here
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

### FinanceBench (8 filings, 30 real analyst questions)

Artifacts: `runs/financebench/`. 95 synthetic questions (85 answerable) for selection;
30 real FinanceBench questions held out, gold spans from FinanceBench's own evidence,
located by token alignment within the cited page (29 strong, 1 weak).

| recall@5 | default | tuned winner |
|---|---|---|
| real questions, held out (n=30) | 0.367 | 0.267 |
| synthetic questions, selection (n=95) | 0.506 | 0.953 |

Winner: docling, structure 300/600, BM25, MiniLM reranker on the top 20. Default:
pymupdf-text, recursive 512/50, bge-small, dense.

**Pre-registered hypothesis: FAIL on this corpus.** The tuned config is 10 points worse
than the default on real questions. On other real metrics the winner has higher MRR@10
(0.208 vs 0.163) but lower recall@10 (0.267 vs 0.467) and hit@50 (0.500 vs 0.733).

Why: the heuristic generator writes questions from rare words in the gold span, so the
selection set rewards exact-term matching, and BM25 wins it by a wide margin. Real
analyst questions share few words with the filing text ("What is the FY2018 capital
expenditure amount" against a line item labeled "Purchases of property, plant and
equipment"). The tool's own failure analysis on the winner says this: 22 of its 26
misses are real questions, concentrated in low-overlap, numeric and table questions, all
failing at depth (never in the top 50), and its recipe is to add a dense embedder or go
hybrid. This is the area-3 risk from the spec, observed: a biased synthetic set selects
the wrong config, and a small real set is what catches it.

`ragtuner replay` reproduced the winner's top-50 chunk ids for all 125 questions.

### Slide deck (1 deck, 138 pages, image-only)

Artifacts: `runs/slides/`. Only 31 synthetic questions could be generated (21
answerable): slide text is short, so few blocks pass the generator's eligibility rule.
No real questions exist for this deck.

The deck is image-only: PyMuPDF extracts 18 characters from 138 pages; Docling's OCR
recovers 23,257. Every gold span is therefore absent from the PyMuPDF parse (21 of 21
projections lost), the default scores 0.000 recall@5, and its failure analysis labels all
21 misses as parse failures with the "compare parsers" recipe. Every Docling config
reaches 1.000 on this small set. This is the scanned-document case from area 2: the
parser dimension decides the outcome, and the search surfaces it without a human
looking at the PDF.

`ragtuner replay` reproduced the winner's results for all 31 questions.

### FinanceBench cross-validation: is selection the problem, or the eval set?

Artifacts: `runs/financebench_crossval/`. Every one of the 64 configs in the space was
scored on both the synthetic and the real questions (`ragtuner crossval`).

| measure | value |
|---|---|
| Spearman, synthetic vs real recall@5 ranking of the 64 configs | -0.121 |
| Default, real recall@5 | 0.367 |
| Synthetic-selected winner, real recall@5 | 0.267 |
| Best config on the full real set (optimistic: selected on the test questions) | 0.467 |
| Select on a random half of the real questions, score on the other half: mean delta vs default | -0.049 |
| Same, share of 500 splits where the pick beats the default | 21% |

Mean real recall@5 by family: dense with reranker 0.362, hybrid with reranker 0.329,
hybrid 0.317, dense 0.312, BM25 with reranker 0.296, BETTER-RAG with reranker 0.267,
BM25 0.217, BETTER-RAG 0.179. The real questions reward the opposite of what the
heuristic set rewards.

## FinanceBench with LLM-generated questions

Artifacts: `runs/financebench_crossval_llm/`. The LLM generator (evidence must be quoted
verbatim and is verified, paraphrase pass, table oversampling) ran on free OpenRouter
models: 31 of its requests were served by Nemotron 3 Ultra 550B and 2 by Gemma 4 31B. The
free tier allows 50 requests a day, so it stopped at 23 questions (15 attempts dropped
for unverifiable evidence or the quota). Same 30 real questions held out, same 64 configs.

| measure | heuristic questions (n=85) | LLM questions (n=23) |
|---|---|---|
| Spearman, synthetic vs real config ranking | -0.121 | +0.340 |
| Real recall@5 of the config the synthetic set selects | 0.267 | 0.367 |
| Real recall@5 of the frozen default | 0.367 | 0.367 |
| Median question-to-evidence word overlap (real questions: 0.25) | 0.38 | 0.32 |

The LLM set removes the selection bias that made tuning harmful: its pick (docling,
recursive 512, BM25 with the reranker) ties the default on real questions instead of
losing 10 points. It does not yet find a better config than the default. Two reasons
are visible in the data: it still overrates plain BM25 (its second-ranked family, 0.701
synthetic, is 0.217 on real questions), and 23 questions cannot separate the top configs
(three tie at 0.870; one question is worth 4.3 points).

### Extended to 51 LLM questions

Artifacts: `runs/financebench_crossval_llm2/`. After the free tier's daily reset, the
generator resumed from its checkpoint and added 28 questions (48 calls, all served by
Nemotron 3 Ultra; 2 dropped). Same 30 real questions, same 64 configs.

| selection set | questions | Spearman vs real ranking | real recall@5 of its pick |
|---|---|---|---|
| heuristic | 85 | -0.121 | 0.267 |
| free LLM | 23 | +0.340 | 0.367 |
| free LLM, extended | 51 | +0.351 | 0.300 |
| frozen default (no tuning) | | | 0.367 |

The agreement held as the set doubled, so +0.35 looks like a property of this generator
rather than noise. At the family level the 51 questions now rank hybrid plus reranker
(0.637) and dense plus reranker (0.618) first, which are also the two best families on
real questions (0.329 and 0.362). The single pick did not improve: the reranked
BETTER-RAG arm topped the synthetic set by one question (0.765 against 0.745) and scores
0.300 on real questions. With a correlation around 0.35, selecting the one top config is
fragile; this is the strongest argument for a selection rule that is robust to ranking
noise (for example, choosing the best family first, or requiring a margin over the
default before switching away from it) before spending on a larger Claude-written set.

## Query rewriting on real questions

`runs/prf_check.json`. Each of the 8 configs with the best real recall@5 on FinanceBench was
scored with and without pseudo-relevance feedback on the same 30 held-out questions.

| result | value |
|---|---|
| mean change in real recall@5 | -0.062 |
| configs improved / unchanged / worse | 0 / 2 / 6 |
| largest drop | 0.367 to 0.200 (structure, bge-small, dense) |

When the first BM25 pass misses, as it does for paraphrased analyst questions, the
expansion terms come from the wrong passages and pull retrieval further off. PRF stays
in the `phase2` space so the search can reject it per corpus; the LLM rewrite arm is
built and tested offline but was not run live (no paid LLM credit).

## Deployment, verified end to end

| check | result |
|---|---|
| Bundle served in-process, IRS winner (Docling), eval replayed over HTTP | 100 of 100 questions, identical top-10 chunk ids |
| Bundle built with its own Dockerfile, run as a container, replayed with `export --verify --url` (PyMuPDF config, 613 MB image) | 39 of 39 identical |
| Same for the real IRS winner (Docling, 2.9 GB image, parsed from scratch inside the container) | 100 of 100 identical |
| Official MCP Python SDK client against `ragtuner mcp` on the IRS index | initialize, tools/list, search and ask all succeed |
| Monitor on the container's real query log | drift and low-confidence signals computed; freshness clean |

Problems the Docker build exposed, all fixed: Docker Hub returned 429 to anonymous pulls
(the base image is now a build argument), the slim image has no git (requirements now
install pinned archives, which also pinned BETTER-RAG to a commit instead of main), and
CPU-only PyTorch pins need PyTorch's own index (now written into requirements). This
sandbox intercepts TLS, so the test build used a local base image that trusts the
sandbox CA; the bundle's Dockerfile itself is unchanged by that.

## Conclusions

1. **The hypothesis failed where it could be tested, and the cause is identified.** The
   heuristic synthetic set's config ranking is uncorrelated with the real ranking
   (rho = -0.12), so selecting on it is worse than not tuning. This is a property of the
   offline generator, which copies rare words from the evidence, not of the search,
   which recovers the grid optimum within about one point (IRS regret 0.011).
   Replacing it with an LLM generator moved the correlation to about +0.35, stable from
   23 to 51 free-model questions, but the single top pick still does not beat the
   default (a tie at 23 questions, 6.7 points worse at 51).
2. **There is headroom over the default, but 30 real questions cannot find it reliably.**
   The best config beats the default by 10 points on the full real set, but that is
   selected on the test questions. Selecting on 15 real questions and testing on the
   other 15 loses to the default on average. Tuning needs either an unbiased synthetic
   set or a larger real set.
3. **The parser dimension is decisive on scanned content.** On the image-only deck the
   text-only default retrieves nothing and the search finds that without inspection.
4. **The diagnosis layer earned its place.** On FinanceBench it attributed the winner's
   real misses to low-overlap numeric and table questions failing at depth and named
   dense or hybrid retrieval as the fix, which the cross-validation then confirmed.

## What would change the verdict (next steps, in order)

1. Run the LLM generator at full size (100 or more questions) with a strong model and
   repeat `crossval`. With 23 free-model questions the correlation is +0.34 and the pick
   ties the default; the question is whether a larger, stronger set pushes the pick past
   the default on held-out real questions. About $5 of OpenRouter credit covers it with
   Claude Opus 5.5. If it still only ties, the thesis reduces to "eval plus diagnosis",
   as the spec anticipated.
2. Grow the real set: all 150 open FinanceBench questions span 84 filings. The 8-filing
   subset here was chosen for CPU time.
3. Add `OPENAI_API_KEY` so the frozen default runs with its intended embedder.

## Not done, and why

- Claude question generation, the judge, and LLM query rewriting were not run live with
  Claude. No Anthropic key is available here, and the OpenRouter key provided is on the
  free tier with a balance too small for one Claude request. The generator did run live
  on free open models (above); the judge and LLM rewriting have offline tests only.
- The OpenAI embedding arm works through OpenRouter (verified with a live call), but
  embedding FinanceBench costs about $0.26, above the free balance, so the frozen default
  still ran with bge-small.
- IRS has no real set: lexical matching of FAQ answers to the publication produced wrong
  gold spans on inspection, so those questions were dropped instead of reported.
