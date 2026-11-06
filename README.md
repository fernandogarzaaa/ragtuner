# ragtuner

Eval-driven RAG configuration tuner. Point it at a corpus; it parses the documents into
a block model, generates a span-anchored eval set, searches the config space (parser,
chunker, embedder, retrieval mode, reranker) with a staged successive-halving search,
scores every config by span containment, diagnoses the winner's misses with named
fixes, and exports the winning config with a manifest that ties it to the exact runtime
that evaluated it. See `docs/phase1-spec.md` for the decisions and the pre-registered
hypothesis, and `docs/phase1-results.md` for what the runs showed.

## The loop

```
corpus/ --parse--> blocks (offsets) --chunk--> chunks (offsets) --index--> retrieve
                      |                                                        |
                      +--evalgen--> questions with gold char spans --score (span containment)
```

The one invariant that makes this fair: gold answers are character spans in the parsed
document, never chunk ids. Any chunker's output maps back to those offsets, so one eval
set scores every configuration.

## Commands

Tune:

```bash
ragtuner parsers                                   # which backends import
ragtuner parse doc.pdf --parser docling --summary  # block model for one file
ragtuner gate sample corpus/ --n 20 --parsers pymupdf-text,docling --out gate_out
ragtuner gate score --sheet gate_out/sheet.jsonl --gold gate_out/gold.jsonl
ragtuner evalgen corpus/ --parser docling --n 100 --real-financebench rows.jsonl --out eval.jsonl
ragtuner tune corpus/ --evalset eval.jsonl --space phase1 --with-grid --out runs/corpus
ragtuner crossval corpus/ --evalset eval.jsonl --out runs/corpus-crossval   # does synthetic agree with real?
ragtuner replay corpus/ --run runs/corpus          # exported config reproduces the tuner's hits
```

Deploy and operate:

```bash
ragtuner export --run runs/corpus --out bundle/ --corpus corpus/ --verify   # bundle + replay over HTTP
ragtuner serve corpus/ --config bundle/best_config.yaml --port 8080 --query-log queries.jsonl
ragtuner mcp corpus/ --config bundle/best_config.yaml                       # stdio MCP server
ragtuner sync corpus/ --config bundle/best_config.yaml --state state.json   # incremental re-index
ragtuner monitor --log queries.jsonl --run runs/corpus --corpus corpus/ --candidates label_me.jsonl
ragtuner export --run runs/corpus --out /tmp/chk --verify --url http://host:8080   # check a live deployment
```

`tune` writes `report.md`, `report.json`, `best_config.yaml`, `manifest.json` (with the
corpus file hashes and the exact ragtuner commit), `diagnoses.json`, `eval_set.jsonl`,
`search.json`, and per-question results for the winner and the default.

### Serving surfaces

- **HTTP API** (`serve`): `POST /search`, `POST /ask` (extractive answer with numbered
  citations; set `RAGTUNER_LLM_ANSWERS=1` for an LLM answer over the same hits),
  `GET /manifest`, `POST /admin/reload` (guarded by `RAGTUNER_ADMIN_TOKEN`), `GET /widget.js`.
  Permission filtering reads `X-Tenant` and `X-Principal` from a trusted gateway and fails
  closed. It runs the same build path the tuner evaluated.
- **Chat widget**: `<script src="https://host/widget.js" data-api="https://host"></script>`.
- **MCP server** (`mcp`): `search` and `ask` tools over stdio, tested with the official MCP
  Python SDK client.
- **Bundle** (`export`): config, manifest, pinned requirements (ragtuner and BETTER-RAG
  pinned to exact commits), a Dockerfile, and an MCP client entry. `--verify` replays the
  eval set over HTTP and requires identical top-10 hits. Docker Hub rate-limits anonymous
  pulls; pass `--build-arg BASE_IMAGE=public.ecr.aws/docker/library/python:3.12-slim`.
- **Monitoring** (`monitor`): low-confidence share against the eval baseline, score drift,
  empty results, latency, corpus freshness, and proposed fixes, including a file of
  low-confidence queries to label as new real eval questions.

### LLM and embedding providers

The Claude question generator, the judge, LLM query rewriting and LLM answers use
`ANTHROPIC_API_KEY` when set, otherwise `OPENROUTER_API_KEY` (any OpenRouter model, a
comma-separated list for fallback routing). `text-embedding-3-small` runs with
`OPENAI_API_KEY` or `OPENROUTER_API_KEY`. Free OpenRouter models work with
`--llm-model` and `--max-calls`.

## Modules

| module | what it does |
|---|---|
| `ragtuner.schema` | Block model: typed blocks with page, bbox, table cells, exact offsets into one canonical text |
| `ragtuner.parse` | Backends `plaintext`, `pymupdf-text` (text only, the baseline arm), `docling` (layout + tables); the 20-page parser gate |
| `ragtuner.chunk` | `fixed` word window (BETTER-RAG semantics), `recursive` (blocks, sentences, words), `structure` (tables whole, heading-path prefixes) |
| `ragtuner.embed` | `bge-small` local, `openai-small` (needs `OPENAI_API_KEY`), `hash-64` for tests; SQLite embedding cache by (model, text hash) |
| `ragtuner.index` | BM25 with light stemming, brute-force dense index, reciprocal rank fusion, cross-encoder reranker |
| `ragtuner.pipeline` | `RagConfig` to a retrieval index with caches and a model pool; `better-rag` mode runs BETTER-RAG's scorer over any chunker |
| `ragtuner.evalset`, `ragtuner.metrics` | Span-anchored questions; recall, MRR, nDCG, hit@50 by span containment; refusal gate on unanswerable questions |
| `ragtuner.evalgen` | `LLMGenerator` (Claude, evidence must be located verbatim, paraphrase pass lowers lexical overlap), `HeuristicGenerator` (offline, lexically leaky, reported separately), FinanceBench and QA-pair importers |
| `ragtuner.search` | Config space with constraints, evaluator, staged search with successive halving, full grid and regret |
| `ragtuner.diagnose` | Deterministic failure stage per miss (parse, chunk, deep, rank), attribute lift with minimum support, fix recipes, question ids |
| `ragtuner.judge` | Stage 3 generation quality with Claude (answer from context, judged against gold); skipped without credentials |
| `ragtuner.report` | Markdown and JSON report, verdict on the pre-registered hypothesis, config export with manifest |
| `ragtuner.rewrite` | Query rewriting arms: `prf` (pseudo-relevance feedback) and `llm` (cached reformulation) |
| `ragtuner.access` | Tenant and principal pre-filtering applied inside every index before ranking |
| `ragtuner.align` | Projects gold spans from the anchor parse onto another parser's text by token alignment |
| `ragtuner.crossval` | Synthetic-vs-real ranking agreement and split-half selection on real queries |
| `ragtuner.serve` | Serving runtime, HTTP API, chat widget, MCP server |
| `ragtuner.export`, `ragtuner.monitor` | Deployable bundles with HTTP verification; query-log monitoring with fix proposals |
| `ragtuner.llm` | Anthropic or OpenRouter caller with retries, fallback models and a usage tally |

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,local]"     # tests, local embedder and reranker (PyTorch)
pip install -e ".[docling]"       # layout parser with tables (PyTorch, model download on first use)
pip install -e ".[llm]"           # Claude question generator and judge
```

On a CPU-only machine install PyTorch and torchvision from the CPU index first so they
match (`pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`);
a mismatched torchvision breaks Docling's image processor import.

## Honest status

Measured outcomes, including the failed hypothesis on FinanceBench and why, are in
`docs/phase1-results.md`. In short: the search machinery works (staged search within one
point of the full grid; exported configs reproduce exactly, including from a Docker
container), but a selection set from the offline heuristic generator does not agree with
real questions, and 30 real questions are too few to tune on.
