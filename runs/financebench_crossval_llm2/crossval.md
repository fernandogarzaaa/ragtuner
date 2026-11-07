Configs: 64. Real questions: 30. Synthetic questions: 51.

| measure | value |
|---|---|
| Spearman, synthetic vs real recall@5 ranking | 0.351 |
| Default, real recall@5 | 0.367 |
| Synthetic-selected winner, real recall@5 | 0.300 |
| Oracle best on full real set (optimistic upper bound) | 0.467 |
| Split-half selection: mean delta vs default on held-out half | -0.016 |
| Split-half selection: share of splits beating the default | 37% |

Splits: 500. Oracle config: `pymupdf-text|structure-600-300|bge-small|dense|minilm@20|k5|5e8d5894`.

Most frequent split-half picks: `pymupdf-text|structure-600-300|bge-small|dense|minilm@20|k5|5e8d5894` (258), `pymupdf-text|recursive-50-512|bge-small|hybrid|norerank|k5|5ee7c8a0` (76), `docling|recursive-50-512|none|sparse|minilm@20|k5|498b07a0` (65), `pymupdf-text|fixed-24-120|bge-small|dense|minilm@20|k5|f63def47` (48), `docling|recursive-50-512|none|better-rag|minilm@20|k5|4209d02d` (12)

- Default embedder 'openai-small' unavailable (unavailable: RuntimeError: neither OPENAI_API_KEY nor OPENROUTER_API_KEY is set); default run with bge-small instead.
