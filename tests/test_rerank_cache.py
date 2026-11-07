from ragtuner.index.rerank import CrossEncoderReranker


class _FakeModel:
    def __init__(self):
        self.calls = 0

    def predict(self, pairs, **kw):
        self.calls += len(pairs)
        return [float(len(t)) for _, t in pairs]


def test_rerank_cache_scores_each_pair_once():
    rr = CrossEncoderReranker.__new__(CrossEncoderReranker)
    rr._model = _FakeModel()
    rr._cache, rr.scored, rr.cache_hits = {}, 0, 0
    assert rr.rerank("q", ["aa", "bbb"]) == [2.0, 3.0]
    assert rr.rerank("q", ["bbb", "c"]) == [3.0, 1.0]
    assert rr._model.calls == 3 and rr.cache_hits == 1
    assert rr.rerank("other", ["aa"]) == [2.0] and rr._model.calls == 4


def test_rerank_depth_only_reorders_the_head(tmp_path):
    from ragtuner.chunk import Chunk
    from ragtuner.config import RagConfig
    from ragtuner.pipeline import BuildStats, ModelPool, RetrievalIndex

    chunks = [Chunk(id=f"c{i}", doc_id="d", text="x" * (i + 1), start=i * 10, end=i * 10 + i + 1, index_text="x" * (i + 1)) for i in range(6)]
    pool = ModelPool()
    rr = CrossEncoderReranker.__new__(CrossEncoderReranker)
    rr._model, rr._cache, rr.scored, rr.cache_hits = _FakeModel(), {}, 0, 0
    pool._rerankers["fake"] = rr
    cfg = RagConfig(parser="plaintext", embedder=None, retrieval="sparse", reranker="fake", top_k=2, candidates=6, rerank_depth=3)
    idx = RetrievalIndex(cfg, chunks, pool=pool, emb_cache=None, stats=BuildStats())
    idx.candidates = lambda q, n, allowed=None: [(0, 0.9), (1, 0.8), (2, 0.7), (5, 0.6), (4, 0.5)]
    hits = idx.retrieve("q", k=5)
    # head (0,1,2) reranked by text length -> 2,1,0; tail keeps order 5,4
    assert [h.chunk_id for h in hits] == ["c2", "c1", "c0", "c5", "c4"]
    assert rr._model.calls == 3
