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

```bash
ragtuner parsers                                   # which backends import
ragtuner parse doc.pdf --parser docling --summary  # block model for one file
ragtuner gate sample corpus/ --n 20 --parsers pymupdf-text,docling --out gate_out
ragtuner gate score --sheet gate_out/sheet.jsonl --gold gate_out/gold.jsonl
ragtuner evalgen corpus/ --parser docling --n 100 --real-financebench rows.jsonl --out eval.jsonl
ragtuner tune corpus/ --evalset eval.jsonl --space phase1 --with-grid --out runs/corpus
ragtuner replay corpus/ --run runs/corpus          # exported config reproduces the tuner's hits
ragtuner query corpus/ --config runs/corpus/best_config.yaml "your question"
```

`tune` writes `report.md`, `report.json`, `best_config.yaml`, `manifest.json`,
`diagnoses.json`, `eval_set.jsonl`, `search.json`, and per-question results for the
winner and the default.

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

What is verified by tests or by runs in this repo's history is listed in
`docs/phase1-results.md`. Two gaps are structural rather than bugs:

- The Claude-based generator and judge have unit tests with fake callers but have not
  been run against the live API from this environment (no credentials here). The runs
  used the heuristic generator, whose questions name words from the gold span and
  therefore favor sparse retrieval. The report labels this on every run.
- Real queries only count when their gold spans are trustworthy. FinanceBench ships
  human evidence, so its real set is anchored. IRS FAQ answers could not be anchored by
  lexical matching (the matches were wrong on inspection), so that corpus has no real
  set rather than a misleading one.
