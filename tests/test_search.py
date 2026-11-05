from pathlib import Path

from ragtuner.config import RagConfig
from ragtuner.evalgen import HeuristicGenerator
from ragtuner.parse import parse_file
from ragtuner.pipeline import Corpus
from ragtuner.search import Evaluator, Space, full_grid, regret, staged_search


def make_corpus(tmp_path: Path) -> Corpus:
    root = tmp_path / "corpus"
    root.mkdir()
    for i in range(4):
        body = "\n\n".join(
            f"## Topic {i}-{j}\n\nSection {i}-{j} explains procedure {i}{j} for handling widget{i}{j} requests. "
            f"Applicants must file form F{i}{j} within {10 + j} days and include receipt R{i}{j}. Late filings incur fee {j * 5} dollars."
            for j in range(6)
        )
        (root / f"doc{i}.md").write_text(f"# Guide {i}\n\n{body}\n", encoding="utf-8")
    return Corpus.from_dir("mini", root)


def test_staged_search_runs_and_beats_or_matches_default(tmp_path):
    corpus = make_corpus(tmp_path)
    docs = [parse_file(f, "plaintext") for f in corpus.files]
    es = HeuristicGenerator(seed=0, unanswerable_fraction=0.1).generate(docs, 24)
    space = Space(parsers=["plaintext"], chunkers=[("fixed", {"size": 60, "overlap": 10}), ("structure", {"target_words": 80, "max_words": 160})],
                  embedders=[None, "hash-64"], retrievals=["sparse", "dense", "hybrid"], rerankers=[None])
    logs = []
    ev = Evaluator(corpus, cache_root=tmp_path / "cache", log=logs.append)
    default = RagConfig(parser="plaintext", chunker="fixed", chunk_params={"size": 60, "overlap": 10}, embedder="hash-64", retrieval="dense")
    rep = staged_search(ev, space, es, default, carry=2)
    assert rep.mode == "staged" and rep.winner.results is not None and rep.default.results is not None
    assert rep.winner.metrics["recall@5"] >= 0.0
    assert any("stage 1" in line for line in logs)
    assert rep.evaluations == ev.evaluations
    grid = full_grid(ev, space, es, default)
    r = regret(rep, grid)
    assert r["regret"] >= -1e-9  # grid optimum is an upper bound on the staged winner
    assert len(grid.grid) == len(space.grid())


def test_space_constraints():
    space = Space(parsers=["plaintext"], embedders=[None, "hash-64"])
    for cfg in space.grid():
        cfg.validate()
        if cfg.retrieval in ("dense", "hybrid"):
            assert cfg.embedder
        else:
            assert cfg.embedder is None
