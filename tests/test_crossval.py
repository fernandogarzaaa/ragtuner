from pathlib import Path

from ragtuner.config import RagConfig
from ragtuner.crossval import run_crossval, spearman
from ragtuner.evalgen import HeuristicGenerator
from ragtuner.evalset import EvalQuestion, EvalSet
from ragtuner.parse import parse_file
from ragtuner.pipeline import Corpus
from ragtuner.search import Evaluator, Space


def test_spearman_basics():
    assert abs(spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1.0) < 1e-12
    assert abs(spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1.0) < 1e-12
    # tie-corrected: ranks [1.5,1.5,3,4] vs [1,2.5,2.5,4] -> 3.75 / 4.5
    assert abs(spearman([1, 1, 2, 3], [1, 2, 2, 3]) - 3.75 / 4.5) < 1e-9


def test_crossval_runs_end_to_end(tmp_path: Path):
    root = tmp_path / "c"
    root.mkdir()
    for i in range(3):
        (root / f"d{i}.md").write_text("# G\n\n" + "\n\n".join(
            f"## S{j}\n\nSection {i}{j} covers widget{i}{j} rules and form W{i}{j}. Applicants send the form to the regional office, "
            f"keep a copy, and wait {j + 3} days for written confirmation before shipping." for j in range(5)), encoding="utf-8")
    corpus = Corpus.from_dir("c", root)
    docs = [parse_file(f) for f in corpus.files]
    synth = HeuristicGenerator(seed=0, unanswerable_fraction=0.0).generate(docs, 20)
    real = EvalSet("r", [EvalQuestion(q.id + "-r", "What must applicants keep and how long do they wait?", q.spans, source="real")
                         for q in synth.questions[:8]])
    es = EvalSet.merge("c", [synth, real])
    es.metadata = {"parser": "plaintext"}
    space = Space(parsers=["plaintext"], chunkers=[("fixed", {"size": 30, "overlap": 5}), ("structure", {"target_words": 60, "max_words": 120})],
                  embedders=[None, "hash-64"], retrievals=["sparse", "dense"], rerankers=[None])
    default = RagConfig(parser="plaintext", chunker="fixed", chunk_params={"size": 30, "overlap": 5}, embedder="hash-64", retrieval="dense")
    rep = run_crossval(Evaluator(corpus, cache_root=tmp_path / "cache"), space, es, default, splits=20)
    assert rep.n_configs == len(space.grid()) and rep.n_real == 8 and rep.splits == 20
    assert -1.0 <= rep.spearman_recall5 <= 1.0 or rep.spearman_recall5 != rep.spearman_recall5
    assert rep.oracle_real >= rep.synthetic_winner_real and sum(rep.picks.values()) == 20
    assert "Spearman" in rep.to_markdown()
