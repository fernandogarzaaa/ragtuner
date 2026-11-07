Configs: 64. Real questions: 30. Synthetic questions: 85.

| measure | value |
|---|---|
| Spearman, synthetic vs real recall@5 ranking | -0.121 |
| Default, real recall@5 | 0.367 |
| Synthetic-selected winner, real recall@5 | 0.267 |
| Oracle best on full real set (optimistic upper bound) | 0.467 |
| Split-half selection: mean delta vs default on held-out half | -0.049 |
| Split-half selection: share of splits beating the default | 21% |

Splits: 500. Oracle config: `pymupdf-text|structure-600-300|bge-small|dense|minilm@20|k5|5e8d5894`.

Most frequent split-half picks: `pymupdf-text|structure-600-300|bge-small|dense|minilm@20|k5|5e8d5894` (158), `pymupdf-text|recursive-50-512|bge-small|hybrid|norerank|k5|5ee7c8a0` (88), `docling|recursive-50-512|bge-small|hybrid|minilm@20|k5|bec37034` (59), `pymupdf-text|fixed-24-120|bge-small|hybrid|minilm@20|k5|22932efa` (57), `pymupdf-text|fixed-24-120|bge-small|dense|minilm@20|k5|f63def47` (43)

- Default embedder 'openai-small' unavailable (unavailable: RuntimeError: OPENAI_API_KEY is not set); default run with bge-small instead.
