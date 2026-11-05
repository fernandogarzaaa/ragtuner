# RAG Tuner: Phase 1 Spec and Decisions

Date: 2026-11-05. All decisions below are made; none are open.

## Repo grounding (BETTER-RAG, commit 19e3c31)

Code-verified: 420-line stdlib-only engine. Sparse-only scoring (cosine over term
counts 0.48, normalized BM25 0.37, term overlap 0.15). Whitespace word-window chunker
(120/24). MMR over sparse vectors. Extractive answers (top 3 sentences by overlap).
Guardrails: min_score threshold, confidence = 0.6*top + 0.4*coverage, 8-word blocklist.
JSON persistence of raw docs; re-chunks on load. Brute-force query, ~0.19s at 3,000 chunks.
24 tests pass on Python 3.13. No eval, no dense retrieval, no reranker, no ACL, no sync.

Role in new project: zero-dependency baseline arm of the search space, plus the data
contracts (RagConfig, Document, SearchHit, Citation, RagResult extended with span
offsets and block metadata).

## Positioning (decided)

Internal accelerator for consulting work, built with product-grade interfaces:
declarative config, machine-readable report, code export. Not a no-code product.
Go/no-go on open-sourcing and selling is gated on the Phase 1 hypothesis.

## Pre-registered hypothesis

Tuned config beats the frozen default by >= 5 points recall@5 on the held-out
real-query set, on >= 2 of 3 corpora. If false: reposition as eval + diagnosis tool.

## Frozen default config (the thing to beat)

parser: pymupdf-text | chunker: recursive 512 tokens, 50 overlap |
embedding: OpenAI text-embedding-3-small | retrieval: dense only | reranker: none |
top_k: 5 | query rewriting: none.
This is the common indie-RAG default. If Fernando's actual default differs, overwrite
this block BEFORE the first run and never after.

## Corpora (decided, all public)

1. FinanceBench subset: 10-K PDFs, table-heavy, ships real analyst questions.
2. IRS publications (Pub 17, 334, 505 and similar PDFs with tables and forms).
   Real queries harvested from IRS public FAQ pages.
3. Public investor-day / earnings slide decks (PDF exports). Real queries taken from
   the matching earnings-call Q&A transcripts (analyst questions are genuine human
   queries about the same material).
Client corpora are out of Phase 1. They can be added later without changing anything.

## Eval set per corpus

100 synthetic: generated from parsed blocks (not chunks), gold answer anchored to a
character span in the source. Stratified by block type; >= 25% table-sourced where
tables exist; 10% unanswerable. Paraphrase pass until token overlap with the gold span
falls below 0.5. 30 real questions, held out, scored separately, never averaged with
synthetic. Unanswerable questions are a secondary gate, not part of the primary metric.

Generation and judge model: claude-opus-5-5, adaptive thinking (default), effort medium
for generation, high for the judge. Server-side fallbacks enabled.

## Parsers

Default Docling. Optional hosted backend behind the same interface. Gate: before the
full run, 20 pages per corpus hand-labeled for table cell accuracy, Docling vs
PyMuPDF text. If Docling table accuracy < 80% on corpus 1, add one hosted parser arm.

## Config space

parser {docling, pymupdf-text}
chunker {fixed 120/24 words (BETTER-RAG), recursive 256, recursive 512, structure-aware}
embedding {sparse-only (BETTER-RAG), bge-small-en-v1.5 local, text-embedding-3-small}
retrieval {sparse, dense, hybrid RRF}   (sparse-only embedding pairs only with sparse)
reranker {none, bge-reranker-base}
top_k {5, 10}
Query rewriting deferred to Phase 2.

## Search procedure

Stage 1: parser x chunker x embedding on retrieval metrics, docs containing gold spans
plus a 2x distractor sample. Carry top 2.
Stage 2: retrieval mode x reranker x top_k on full eval set. Carry top 3.
Stage 3: LLM-judge answer quality on top 3 only.
Successive halving inside each stage: 20% of questions, drop bottom half, double.
Caches: parsed docs by (parser, file hash); chunks by (parser, chunker, file hash);
embeddings by (chunk content hash, model). Cost estimate printed before run.
Regret check: on corpus 2, run the full valid grid once and report the gap between
staged winner and grid optimum.

## Metrics

Primary: recall@5 and MRR@10 by span containment (chunk holds >= 50% of gold span chars).
Secondary: nDCG@10, refusal accuracy on unanswerable, judge score on top 3.

## Failure analysis

Per miss, deterministic stage label: parse (span text in no chunk), chunk (no chunk
holds >= 50% of span), retrieval-deep (intact chunk not in top 50), retrieval-rank
(in top 50, not top k), generation (retrieved, answer wrong per judge).
Per attribute (block type, page, doc, question type, overlap bucket): miss rate vs base
rate, report lift with min support 5. Each (stage, attribute) maps to a fix recipe.
LLM phrases the diagnosis; rules compute it; every diagnosis lists question ids.
Fault-injection test: destroy tables in parse, assert attribution to table/parse.

## Deployment (Phase 1: export only, no server)

best_config.yaml plus manifest (config hash, lockfile, model versions, report hash).
API, MCP server, widget deferred to Phase 2. Runtime and tuner share one package so
evaluated code is shipped code.

## Production concerns (design only, build none)

Chunk schema carries doc_id, source_uri, content_hash, principals[], tenant_id.
Store choice for Phase 2: LanceDB local, Qdrant hosted (both pre-filter on metadata).
ACL leak test on 2-principal synthetic corpus is written in Phase 1 against the schema.

## Budget caps per corpus per full run (caps I set, not measurements)

Question generation + paraphrase: $5. Judge on top 3 configs: $10. API embeddings: $5.
Total cap $20 per corpus, $60 for Phase 1. Local models are free beyond compute.
Rough token math at $4/$20 per MTok: 200 generation calls at ~2.5k tokens each is about
$2 to $3; 390 judge calls at ~3k tokens each is about $6 to $8.

## Deliverables

report.md, report.json, best_config.yaml, eval_set.jsonl, diagnoses.json, cache/.

## Effort estimate

60 to 80 focused hours. Estimate, not measurement.

## Order of work

1. Data contracts + parser interface + Docling/PyMuPDF backends + 20-page gate.
2. Chunkers + embeddings + sparse/dense/hybrid retrieval + caches.
3. Eval generation with span anchors, paraphrase, stratification, real-query import.
4. Metrics + staged search + successive halving.
5. Failure analysis + fault-injection test.
6. Report writer. Run all three corpora. Record hypothesis result.
