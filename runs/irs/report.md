# ragtuner report: irs

Mode: staged. Evaluations: 47. Wall time: 2.0 min.

## Verdict

No real-query set, so the pre-registered hypothesis cannot be judged. Synthetic recall@5: default 0.778, winner 0.933 (delta 0.156).

## Configs

Default: `pymupdf-text|recursive-50-512|bge-small|dense|norerank|k5|4813b9e7`

Winner: `docling|structure-600-300|none|sparse|norerank|k5|58027d4d`

```yaml
parser: docling
chunker: structure
chunk_params:
  target_words: 300
  max_words: 600
embedder: null
retrieval: sparse
reranker: null
top_k: 5
candidates: 50
rerank_depth: 20
label: null
```

## Real queries (held out)

| config | recall@5 | mrr@10 | recall@10 | ndcg@10 | hit@50 | refusal_rate@p10 |
|---|---|---|---|---|---|---|
| default | n/a | n/a | n/a | n/a | n/a | n/a |
| winner | n/a | n/a | n/a | n/a | n/a | n/a |

## Synthetic queries (selection set)

| config | recall@5 | mrr@10 | recall@10 | ndcg@10 | hit@50 | refusal_rate@p10 |
|---|---|---|---|---|---|---|
| default | 0.778 | 0.589 | 0.800 | 0.642 | 0.967 | 1.000 |
| winner | 0.933 | 0.758 | 0.967 | 0.809 | 1.000 | 1.000 |

## Eval set

```json
{
  "n": 100,
  "by_source": {
    "synthetic-heuristic": 100
  },
  "by_type": {
    "factual": 74,
    "numeric": 16,
    "unanswerable": 10
  },
  "by_block_type": {
    "paragraph": 63,
    "table": 22,
    "list": 5,
    "none": 10
  },
  "unanswerable": 10,
  "weak_gold": 0
}
```

## Failure analysis (winner)

Misses at k=5: 6 of 90 answerable questions.

| stage | misses |
|---|---|
| rank | 6 |

- block_type=paragraph: 100% of misses (6 of 6) but 70% of questions; rank failures. Fix: Relevant chunks are retrieved but ranked below top-k: add a cross-encoder reranker, or lower the candidate pool noise with hybrid fusion. Questions: h-0d82d5874d, h-1ba5208273, h-352b15c929, h-831539ad18, h-931c6eeb9e, h-9e0bcd1539

## Failure analysis (default)

Misses at k=5: 20 of 90 answerable questions.

| stage | misses |
|---|---|
| deep | 3 |
| rank | 17 |

- page=1-10: 25% of misses (5 of 20) but 17% of questions; rank failures. Fix: Relevant chunks are retrieved but ranked below top-k: add a cross-encoder reranker, or lower the candidate pool noise with hybrid fusion. Questions: h-4801a4cd90, h-495a9d9c5a, h-732daac8c0, h-879df544d9, h-d2f0f12f86

## Search trace

| stage | config | n | recall@5 | seconds | embedded texts |
|---|---|---|---|---|---|
| s1/r3 | `docling|structure-600-300|none|sparse|norerank|k5|58027d4d` | 100 | 0.933 | 0.0 | 0 |
| s1/r3 | `pymupdf-text|structure-600-300|none|sparse|norerank|k5|ca6a65cf` | 100 | 0.900 | 0.1 | 0 |
| s1/r3 | `docling|structure-600-300|bge-small|dense|norerank|k5|97480c13` | 100 | 0.822 | 0.0 | 0 |
| s1/r3 | `pymupdf-text|recursive-50-512|none|sparse|norerank|k5|026d4f5a` | 100 | 0.822 | 0.0 | 0 |
| s2/final | `docling|structure-600-300|none|sparse|norerank|k5|58027d4d` | 100 | 0.933 | 0.0 | 0 |
| s2/final | `docling|structure-600-300|none|sparse|minilm@20|k5|ca889504` | 100 | 0.933 | 33.7 | 0 |
| s2/final | `docling|structure-600-300|none|better-rag|norerank|k5|3f24b641` | 100 | 0.867 | 0.5 | 0 |
| grid | `docling|structure-600-300|bge-small|hybrid|norerank|k5|611a0b7f` | 100 | 0.944 | 0.1 | 0 |
| grid | `docling|structure-600-300|none|sparse|norerank|k5|58027d4d` | 100 | 0.933 | 0.1 | 0 |
| grid | `docling|structure-600-300|none|sparse|minilm@20|k5|ca889504` | 100 | 0.933 | 0.1 | 0 |
| grid | `docling|structure-600-300|bge-small|hybrid|minilm@20|k5|7c3b0784` | 100 | 0.933 | 8.3 | 0 |
| grid | `pymupdf-text|structure-600-300|none|sparse|minilm@20|k5|a9f961d2` | 100 | 0.922 | 62.6 | 0 |
| grid | `pymupdf-text|structure-600-300|bge-small|hybrid|minilm@20|k5|6c4a2ccb` | 100 | 0.922 | 8.0 | 0 |
| grid | `docling|structure-600-300|none|better-rag|minilm@20|k5|e13c6395` | 100 | 0.922 | 8.8 | 0 |
| grid | `pymupdf-text|structure-600-300|none|better-rag|minilm@20|k5|e53051fe` | 100 | 0.911 | 12.9 | 0 |
| grid | `docling|recursive-50-512|none|better-rag|minilm@20|k5|4209d02d` | 100 | 0.911 | 21.7 | 0 |
| grid | `pymupdf-text|structure-600-300|none|sparse|norerank|k5|ca6a65cf` | 100 | 0.900 | 0.1 | 0 |
| grid | `pymupdf-text|structure-600-300|bge-small|hybrid|norerank|k5|f0fc28f4` | 100 | 0.900 | 0.1 | 0 |
| grid | `docling|recursive-50-512|none|sparse|minilm@20|k5|498b07a0` | 100 | 0.889 | 88.6 | 0 |
| grid | `docling|structure-600-300|bge-small|dense|minilm@20|k5|9b9298a4` | 100 | 0.889 | 38.0 | 0 |
| grid | `pymupdf-text|recursive-50-512|none|better-rag|norerank|k5|e761dd85` | 100 | 0.878 | 0.6 | 0 |
| grid | `pymupdf-text|recursive-50-512|none|better-rag|minilm@20|k5|d0178e15` | 100 | 0.878 | 24.0 | 0 |
| grid | `docling|recursive-50-512|bge-small|hybrid|minilm@20|k5|bec37034` | 100 | 0.878 | 5.8 | 0 |
| grid | `pymupdf-text|recursive-50-512|none|sparse|minilm@20|k5|a9214306` | 100 | 0.867 | 91.0 | 0 |
| grid | `pymupdf-text|structure-600-300|bge-small|dense|minilm@20|k5|5e8d5894` | 100 | 0.867 | 42.2 | 0 |
| grid | `docling|structure-600-300|none|better-rag|norerank|k5|3f24b641` | 100 | 0.867 | 0.5 | 0 |
| grid | `pymupdf-text|recursive-50-512|bge-small|dense|minilm@20|k5|d7cafd8a` | 100 | 0.856 | 46.3 | 0 |
| grid | `pymupdf-text|recursive-50-512|bge-small|hybrid|norerank|k5|5ee7c8a0` | 100 | 0.856 | 0.1 | 0 |
| grid | `pymupdf-text|recursive-50-512|bge-small|hybrid|minilm@20|k5|37d94570` | 100 | 0.856 | 4.8 | 0 |
| grid | `docling|recursive-50-512|bge-small|hybrid|norerank|k5|e4da06f6` | 100 | 0.856 | 0.1 | 0 |
| grid | `docling|recursive-30-256|none|sparse|minilm@20|k5|96a0d7fb` | 100 | 0.844 | 67.5 | 0 |
| grid | `docling|recursive-30-256|none|better-rag|minilm@20|k5|d411dce9` | 100 | 0.844 | 17.5 | 0 |
| grid | `pymupdf-text|recursive-30-256|none|sparse|minilm@20|k5|5c0be380` | 100 | 0.833 | 65.5 | 0 |
| grid | `pymupdf-text|recursive-30-256|none|better-rag|minilm@20|k5|6fa8129e` | 100 | 0.833 | 17.4 | 0 |
| grid | `pymupdf-text|recursive-30-256|bge-small|dense|minilm@20|k5|71478cef` | 100 | 0.833 | 34.6 | 0 |
| grid | `pymupdf-text|recursive-30-256|bge-small|hybrid|norerank|k5|7a9c12c6` | 100 | 0.833 | 0.1 | 0 |
| grid | `docling|recursive-50-512|none|sparse|norerank|k5|a69d9a8a` | 100 | 0.833 | 0.1 | 0 |
| grid | `docling|recursive-50-512|none|better-rag|norerank|k5|90632552` | 100 | 0.833 | 0.4 | 0 |
| grid | `docling|recursive-50-512|bge-small|dense|minilm@20|k5|d4210ca9` | 100 | 0.833 | 48.5 | 0 |
| grid | `pymupdf-text|recursive-50-512|none|sparse|norerank|k5|026d4f5a` | 100 | 0.822 | 0.0 | 0 |
| grid | `docling|structure-600-300|bge-small|dense|norerank|k5|97480c13` | 100 | 0.822 | 0.0 | 0 |
| grid | `pymupdf-text|recursive-30-256|bge-small|hybrid|minilm@20|k5|ff91a1e9` | 100 | 0.811 | 8.1 | 0 |
| grid | `pymupdf-text|structure-600-300|none|better-rag|norerank|k5|e18f9265` | 100 | 0.811 | 0.5 | 0 |
| grid | `docling|recursive-30-256|bge-small|hybrid|minilm@20|k5|aadd7e1d` | 100 | 0.800 | 9.1 | 0 |
| grid | `pymupdf-text|recursive-30-256|none|sparse|norerank|k5|425c1be0` | 100 | 0.789 | 0.1 | 0 |
| grid | `docling|recursive-30-256|bge-small|hybrid|norerank|k5|5a1d9e3f` | 100 | 0.789 | 0.1 | 0 |
| grid | `pymupdf-text|recursive-50-512|bge-small|dense|norerank|k5|4813b9e7` | 100 | 0.778 | 0.0 | 0 |
| grid | `docling|recursive-30-256|none|sparse|norerank|k5|231b72ca` | 100 | 0.767 | 0.1 | 0 |
| grid | `docling|recursive-50-512|bge-small|dense|norerank|k5|c756194a` | 100 | 0.767 | 0.0 | 0 |
| grid | `docling|recursive-30-256|bge-small|dense|minilm@20|k5|dcd39e0c` | 100 | 0.756 | 39.9 | 0 |
| grid | `pymupdf-text|recursive-30-256|none|better-rag|norerank|k5|2a68acd4` | 100 | 0.733 | 0.5 | 0 |
| grid | `docling|recursive-30-256|none|better-rag|norerank|k5|bbd145d4` | 100 | 0.733 | 0.6 | 0 |
| grid | `pymupdf-text|structure-600-300|bge-small|dense|norerank|k5|f8eb591e` | 100 | 0.722 | 0.0 | 0 |
| grid | `docling|fixed-24-120|none|sparse|minilm@20|k5|a4aefd8c` | 100 | 0.722 | 36.6 | 0 |
| grid | `pymupdf-text|fixed-24-120|none|sparse|minilm@20|k5|76e773df` | 100 | 0.711 | 27.7 | 0 |
| grid | `pymupdf-text|fixed-24-120|bge-small|hybrid|norerank|k5|f7e29f62` | 100 | 0.711 | 0.1 | 0 |
| grid | `pymupdf-text|fixed-24-120|bge-small|hybrid|minilm@20|k5|22932efa` | 100 | 0.711 | 4.0 | 0 |
| grid | `pymupdf-text|fixed-24-120|none|better-rag|minilm@20|k5|b79500a4` | 100 | 0.700 | 9.8 | 0 |
| grid | `docling|recursive-30-256|bge-small|dense|norerank|k5|aeac01a1` | 100 | 0.700 | 0.0 | 0 |
| grid | `docling|fixed-24-120|bge-small|hybrid|minilm@20|k5|d46c3dfe` | 100 | 0.689 | 3.5 | 0 |
| grid | `pymupdf-text|fixed-24-120|none|sparse|norerank|k5|65bfd0f2` | 100 | 0.678 | 0.1 | 0 |
| grid | `docling|fixed-24-120|none|better-rag|minilm@20|k5|e611da7d` | 100 | 0.678 | 9.9 | 0 |
| grid | `pymupdf-text|fixed-24-120|bge-small|dense|minilm@20|k5|f63def47` | 100 | 0.667 | 19.2 | 0 |
| grid | `docling|fixed-24-120|none|sparse|norerank|k5|b544200e` | 100 | 0.656 | 0.1 | 0 |
| grid | `docling|fixed-24-120|bge-small|hybrid|norerank|k5|f61b3e25` | 100 | 0.656 | 0.1 | 0 |
| grid | `pymupdf-text|recursive-30-256|bge-small|dense|norerank|k5|3d030ac7` | 100 | 0.644 | 0.0 | 0 |
| grid | `docling|fixed-24-120|bge-small|dense|minilm@20|k5|9b498b9a` | 100 | 0.633 | 18.3 | 0 |
| grid | `pymupdf-text|fixed-24-120|bge-small|dense|norerank|k5|8f142184` | 100 | 0.589 | 0.0 | 0 |
| grid | `pymupdf-text|fixed-24-120|none|better-rag|norerank|k5|81958d9a` | 100 | 0.567 | 0.7 | 0 |
| grid | `docling|fixed-24-120|none|better-rag|norerank|k5|18660e63` | 100 | 0.567 | 0.6 | 0 |
| grid | `docling|fixed-24-120|bge-small|dense|norerank|k5|251671f5` | 100 | 0.500 | 0.0 | 0 |

## Regret vs full grid

Grid best 0.944, staged winner 0.933, regret 0.011 (recall@5, synthetic).

## Notes

- IRS real FAQ questions could not be anchored to reliable gold spans by lexical matching; the real set is empty for this corpus.
- Default embedder 'openai-small' unavailable (unavailable: RuntimeError: OPENAI_API_KEY is not set); default run with bge-small instead.
- Gold spans anchored in docling text were projected onto pymupdf-text text: 75 exact, 15 fuzzy, 0 lost of 90. Lost spans count as pymupdf-text parse failures.
- Synthetic questions generated by heuristic; selection used synthetic only, real set held out.
- Heuristic generator is lexically leaky by construction; synthetic numbers favor sparse retrieval. Judge on the real set.
- Generation judge skipped: no Anthropic credentials in this environment.
