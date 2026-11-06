# ragtuner report: financebench

Mode: staged. Evaluations: 47. Wall time: 20.1 min.

## Verdict

Hypothesis (winner beats default by >= 0.05 recall@5 on the held-out real set): **FAIL**.
Real set (n=30): default 0.367, winner 0.267, delta -0.100.
Synthetic set (n=95, used for selection only): default 0.506, winner 0.953, delta 0.447.

## Configs

Default: `pymupdf-text|recursive-50-512|bge-small|dense|norerank|k5|4813b9e7`

Winner: `docling|structure-600-300|none|sparse|minilm@20|k5|ca889504`

```yaml
parser: docling
chunker: structure
chunk_params:
  target_words: 300
  max_words: 600
embedder: null
retrieval: sparse
reranker: minilm
top_k: 5
candidates: 50
rerank_depth: 20
label: null
```

## Real queries (held out)

| config | recall@5 | mrr@10 | recall@10 | ndcg@10 | hit@50 | refusal_rate@p10 |
|---|---|---|---|---|---|---|
| default | 0.367 | 0.163 | 0.467 | 0.236 | 0.733 | n/a |
| winner | 0.267 | 0.208 | 0.267 | 0.223 | 0.500 | n/a |

## Synthetic queries (selection set)

| config | recall@5 | mrr@10 | recall@10 | ndcg@10 | hit@50 | refusal_rate@p10 |
|---|---|---|---|---|---|---|
| default | 0.506 | 0.339 | 0.600 | 0.401 | 0.812 | 1.000 |
| winner | 0.953 | 0.783 | 1.000 | 0.837 | 1.000 | 1.000 |

## Eval set

```json
{
  "n": 125,
  "by_source": {
    "synthetic-heuristic": 95,
    "real": 30
  },
  "by_type": {
    "factual": 70,
    "numeric": 44,
    "unanswerable": 10,
    "multi-hop": 1
  },
  "by_block_type": {
    "list": 12,
    "paragraph": 70,
    "table": 33,
    "none": 10
  },
  "unanswerable": 10,
  "weak_gold": 1
}
```

## Failure analysis (winner)

Misses at k=5: 26 of 115 answerable questions.

| stage | misses |
|---|---|
| chunk | 1 |
| deep | 14 |
| rank | 11 |

- source=real: 85% of misses (22 of 26) but 26% of questions; deep failures. Fix: Relevant chunks are outside the candidate pool: change the embedding model, switch to hybrid retrieval, or raise the candidate depth. Questions: financebench_id_00222, financebench_id_00460, financebench_id_00563, financebench_id_00601, financebench_id_00605, financebench_id_00669, financebench_id_00684, financebench_id_00685 ...
- overlap=low: 65% of misses (17 of 26) but 17% of questions; deep failures. Fix: Questions with little word overlap miss entirely: lexical retrieval cannot bridge paraphrase; add a dense embedder or go hybrid. Questions: financebench_id_00222, financebench_id_00460, financebench_id_00563, financebench_id_00605, financebench_id_00669, financebench_id_00684, financebench_id_00685, financebench_id_00711 ...
- question_type=numeric: 69% of misses (18 of 26) but 38% of questions; deep failures. Fix: Numeric questions miss at depth: dense embeddings underweight numbers; hybrid retrieval restores exact-token matching. Questions: financebench_id_00222, financebench_id_00460, financebench_id_00601, financebench_id_00605, financebench_id_00669, financebench_id_00684, financebench_id_00685, financebench_id_00757 ...
- page=11-50: 58% of misses (15 of 26) but 43% of questions; deep failures. Fix: Relevant chunks are outside the candidate pool: change the embedding model, switch to hybrid retrieval, or raise the candidate depth. Questions: financebench_id_00460, financebench_id_00563, financebench_id_00669, financebench_id_00684, financebench_id_00685, financebench_id_00711, financebench_id_00757, financebench_id_00917 ...
- block_type=table: 46% of misses (12 of 26) but 29% of questions; deep failures. Fix: Table chunks are never retrieved: index tables with heading context (structure chunker) and prefer hybrid retrieval for numeric cells. Questions: financebench_id_00222, financebench_id_00460, financebench_id_00563, financebench_id_00684, financebench_id_00685, financebench_id_00711, financebench_id_00799, financebench_id_00807 ...
- doc=AMD_2022_10K: 27% of misses (7 of 26) but 20% of questions; deep failures. Fix: Relevant chunks are outside the candidate pool: change the embedding model, switch to hybrid retrieval, or raise the candidate depth. Questions: financebench_id_00222, financebench_id_00563, financebench_id_00757, financebench_id_00917, financebench_id_00995, financebench_id_01198, financebench_id_01279

## Failure analysis (default)

Misses at k=5: 61 of 115 answerable questions.

| stage | misses |
|---|---|
| deep | 24 |
| rank | 37 |

- overlap=low: 25% of misses (15 of 61) but 17% of questions; rank failures. Fix: Relevant chunks are retrieved but ranked below top-k: add a cross-encoder reranker, or lower the candidate pool noise with hybrid fusion. Questions: financebench_id_00222, financebench_id_00563, financebench_id_00605, financebench_id_00669, financebench_id_00685, financebench_id_00711, financebench_id_00757, financebench_id_00799 ...
- doc=3M_2023Q2_10Q: 16% of misses (10 of 61) but 12% of questions; rank failures. Fix: Relevant chunks are retrieved but ranked below top-k: add a cross-encoder reranker, or lower the candidate pool noise with hybrid fusion. Questions: financebench_id_00807, financebench_id_00941, h-1dfbfaca21, h-2c186830d1, h-75fae8ba56, h-7b735f264a, h-834343d9b4, h-9bcbb26852 ...
- doc=BESTBUY_2024Q2_10Q: 11% of misses (7 of 61) but 9% of questions; rank failures. Fix: Relevant chunks are retrieved but ranked below top-k: add a cross-encoder reranker, or lower the candidate pool noise with hybrid fusion. Questions: financebench_id_00288, financebench_id_01902, h-23f3860a7f, h-51c3e51bb6, h-d6fd5a97eb, h-dce38dd396, h-e8d3d92b9c

## Search trace

| stage | config | n | recall@5 | seconds | embedded texts |
|---|---|---|---|---|---|
| s1/r3 | `docling|structure-600-300|none|sparse|norerank|k5|58027d4d` | 95 | 0.941 | 0.9 | 0 |
| s1/r3 | `pymupdf-text|recursive-50-512|none|sparse|norerank|k5|026d4f5a` | 95 | 0.929 | 0.6 | 0 |
| s1/r3 | `docling|recursive-50-512|none|sparse|norerank|k5|a69d9a8a` | 95 | 0.929 | 0.4 | 0 |
| s1/r3 | `pymupdf-text|structure-600-300|none|sparse|norerank|k5|ca6a65cf` | 95 | 0.800 | 0.5 | 0 |
| s2/final | `docling|structure-600-300|none|sparse|minilm@20|k5|ca889504` | 95 | 0.953 | 31.8 | 0 |
| s2/final | `docling|structure-600-300|none|sparse|norerank|k5|58027d4d` | 95 | 0.941 | 0.3 | 0 |
| s2/final | `docling|structure-600-300|none|better-rag|minilm@20|k5|e13c6395` | 95 | 0.929 | 12.5 | 0 |

## Notes

- Default embedder 'openai-small' unavailable (unavailable: RuntimeError: OPENAI_API_KEY is not set); default run with bge-small instead.
- Gold spans anchored in docling text were projected onto pymupdf-text text: 93 exact, 23 fuzzy, 0 lost of 116. Lost spans count as pymupdf-text parse failures.
- Synthetic questions generated by heuristic; selection used synthetic only, real set held out.
- Heuristic generator is lexically leaky by construction; synthetic numbers favor sparse retrieval. Judge on the real set.
- 1 real questions have weak (automatically located) gold spans.
- Generation judge skipped: no Anthropic credentials in this environment.
