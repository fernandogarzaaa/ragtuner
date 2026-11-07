import numpy as np

from ragtuner.embed import EmbeddingCache, get_embedder
from ragtuner.index import BM25Index, DenseIndex, rrf
from ragtuner.index.bm25 import tokenize


def test_bm25_stems_and_ranks_relevant_first():
    idx = BM25Index(["Customers may request refunds within 30 days.", "Shipping takes two days.", "Unrelated text about cats."])
    hits = idx.search("refund policy", k=3)
    assert hits[0][0] == 0
    assert tokenize("refunds refunding") == ["refund", "refund"]


def test_dense_index_returns_nearest_by_cosine():
    vecs = np.array([[1, 0], [0, 1], [0.7, 0.7]], dtype=np.float32)
    idx = DenseIndex(vecs)
    hits = idx.search(np.array([0.9, 0.1]), k=2)
    assert [h[0] for h in hits] == [0, 2]


def test_rrf_prefers_items_ranked_well_in_both_lists():
    fused = rrf([[1, 2, 3], [3, 1, 4]])
    assert fused[0][0] == 1
    assert {i for i, _ in fused} == {1, 2, 3, 4}


def test_embedding_cache_hits_second_time(tmp_path):
    cache = EmbeddingCache(tmp_path)
    emb = get_embedder("hash-64")
    a = cache.embed(emb, ["alpha", "beta"])
    b = cache.embed(emb, ["beta", "gamma"])
    assert a.shape == (2, 64) and np.allclose(a[1], b[0])
    assert (cache.hits, cache.misses) == (1, 3)
    # query and document embeddings are cached under different keys
    cache.embed(emb, ["alpha"], kind="query")
    assert cache.misses == 4
