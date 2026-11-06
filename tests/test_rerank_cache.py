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
