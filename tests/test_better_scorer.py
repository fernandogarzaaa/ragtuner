import random

from better_rag import BetterRAG, Document
from better_rag import RagConfig as BRConfig

from ragtuner.index.better import better_rag_scores


def test_scores_match_better_rag_search_ranking_at_lambda_one():
    rng = random.Random(0)
    vocab = [f"t{i}" for i in range(60)] + ["refund", "policy", "fee", "shipping", "receipt"]
    docs = [Document(id=f"c{i}", text=" ".join(rng.choices(vocab, k=rng.randint(5, 80)))) for i in range(150)]
    engine = BetterRAG(BRConfig(chunk_size=200, chunk_overlap=0, top_k=50, mmr_lambda=1.0))
    engine.add_documents(docs)
    for q in ["refund policy fee", "shipping receipt t3 t7", "t1 t2 t3 t4 t5", "fee"]:
        expected = [(h.document_id, round(h.score, 9)) for h in engine.search(q, top_k=50)]
        got = [(d, round(s, 9)) for d, s in better_rag_scores(engine, q)[:50]]
        assert got == expected, q
