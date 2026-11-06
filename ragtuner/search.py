"""Config space, evaluator, and the staged search with successive halving.

Selection happens on the synthetic set only. The real set is held out and reported
beside it; the Phase 1 hypothesis is judged on the real set. Both are always reported
separately and never averaged.

top_k is not a searched dimension: retrieval metrics are monotone in k, so it is a
generation-context choice. Metrics are reported at k=5 (primary) and k=10.
"""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ragtuner.cache import ParseCache
from ragtuner.chunk import Chunk
from ragtuner.config import RagConfig
from ragtuner.evalset import EvalQuestion, EvalSet
from ragtuner.metrics import QuestionResult, aggregate, score_question
from ragtuner.pipeline import (
    BuildStats,
    Corpus,
    EmbeddingCache,
    ModelPool,
    RetrievalIndex,
    chunk_docs,
    parse_corpus,
)
from ragtuner.schema import ParsedDocument

PRIMARY_METRIC = "recall@5"
CANDIDATES = 50


@dataclass
class Space:
    parsers: list[str] = field(default_factory=lambda: ["pymupdf-text", "docling"])
    chunkers: list[tuple[str, dict[str, Any]]] = field(default_factory=lambda: [
        ("fixed", {"size": 120, "overlap": 24}),
        ("recursive", {"size": 256, "overlap": 30}),
        ("recursive", {"size": 512, "overlap": 50}),
        ("structure", {"target_words": 300, "max_words": 600}),
    ])
    embedders: list[str | None] = field(default_factory=lambda: [None, "bge-small"])
    retrievals: list[str] = field(default_factory=lambda: ["sparse", "better-rag", "dense", "hybrid"])
    rerankers: list[str | None] = field(default_factory=lambda: [None, "minilm"])
    rewrites: list[str | None] = field(default_factory=lambda: [None])
    top_k: int = 5

    def stage1(self) -> list[RagConfig]:
        """parser x chunker x embedder, with retrieval implied (dense when embedded, sparse otherwise)."""
        out = []
        for p, (c, params), e in itertools.product(self.parsers, self.chunkers, self.embedders):
            out.append(RagConfig(parser=p, chunker=c, chunk_params=dict(params), embedder=e,
                                 retrieval="dense" if e else "sparse", reranker=None, top_k=self.top_k, candidates=CANDIDATES))
        return out

    def stage2(self, base: RagConfig) -> list[RagConfig]:
        """Retrieval mode and reranker variants for one (parser, chunker, embedder) triple."""
        modes = [m for m in self.retrievals if (m in ("dense", "hybrid")) == bool(base.embedder)]
        out = []
        for m, r, q in itertools.product(modes, self.rerankers, self.rewrites):
            out.append(RagConfig(parser=base.parser, chunker=base.chunker, chunk_params=dict(base.chunk_params), embedder=base.embedder,
                                 retrieval=m, reranker=r, top_k=self.top_k, candidates=CANDIDATES, query_rewrite=q))
        return out

    def grid(self) -> list[RagConfig]:
        seen: dict[str, RagConfig] = {}
        for base in self.stage1():
            for cfg in self.stage2(base):
                seen[cfg.key] = cfg
        return list(seen.values())


@dataclass
class ConfigResult:
    config: RagConfig
    metrics: dict[str, float]  # on the question subset evaluated
    n_questions: int
    seconds: float
    stage: str
    fraction: float = 1.0
    stats: dict[str, Any] = field(default_factory=dict)
    results: list[QuestionResult] | None = None
    real_metrics: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"config": self.config.to_dict(), "key": self.config.key, "metrics": self.metrics, "real_metrics": self.real_metrics,
                "n_questions": self.n_questions, "seconds": round(self.seconds, 2), "stage": self.stage, "fraction": self.fraction,
                "stats": self.stats}


class Evaluator:
    """Builds indexes with shared caches and scores eval sets. Parsed docs are cached per
    parser and chunk lists per (parser, chunker) within the process."""

    def __init__(self, corpus: Corpus, *, cache_root: str | Path = ".ragtuner-cache", pool: ModelPool | None = None,
                 log: Callable[[str], None] | None = None, anchor_parser: str | None = None) -> None:
        self.corpus = corpus
        # The parser whose text the eval set's gold offsets point into. Configs using any
        # other parser get their questions' spans projected onto that parser's text.
        self.anchor_parser = anchor_parser
        self._projected: dict[tuple[str, str], EvalQuestion] = {}
        self.projection_stats: dict[str, dict[str, int]] = {}
        self.cache_root = Path(cache_root)
        self.pool = pool or ModelPool()
        self.parse_cache = ParseCache(self.cache_root)
        self.emb_cache = EmbeddingCache(self.cache_root)
        self._docs: dict[str, list[ParsedDocument]] = {}
        self._chunks: dict[str, list[Chunk]] = {}
        self._index: tuple[str, RetrievalIndex] | None = None
        self.log = log or (lambda s: None)
        self.evaluations = 0
        self.total_seconds = 0.0
        self.skipped: list[str] = []

    def docs(self, parser: str) -> list[ParsedDocument]:
        if parser not in self._docs:
            stats = BuildStats()
            self._docs[parser] = parse_corpus(self.corpus, parser, self.parse_cache, stats)
            self.log(f"parsed {stats.docs} docs with {parser} in {stats.parse_seconds:.1f}s ({stats.parse_cache_hits} cache hits)")
        return self._docs[parser]

    def chunks(self, config: RagConfig) -> list[Chunk]:
        key = f"{config.parser}|{config.chunker_key}"
        if key not in self._chunks:
            self._chunks[key] = chunk_docs(self.docs(config.parser), config)
        return self._chunks[key]

    def questions_for(self, parser: str, questions: list[EvalQuestion]) -> list[EvalQuestion]:
        """Questions whose gold spans point into `parser`'s text."""
        if self.anchor_parser is None or parser == self.anchor_parser:
            return questions
        missing = [q for q in questions if (parser, q.id) not in self._projected]
        if missing:
            from ragtuner.align import project_questions

            anchor = {d.id: d for d in self.docs(self.anchor_parser)}
            target = {d.id: d for d in self.docs(parser)}
            projected, stats = project_questions(missing, anchor, target)
            for q in projected:
                self._projected[(parser, q.id)] = q
            agg = self.projection_stats.setdefault(parser, {"spans": 0, "exact": 0, "fuzzy": 0, "lost": 0})
            for k, v in stats.items():
                agg[k] += v
            self.log(f"projected {stats['spans']} gold spans {self.anchor_parser} -> {parser}: {stats}")
        return [self._projected[(parser, q.id)] for q in questions]

    def index(self, config: RagConfig) -> tuple[RetrievalIndex, BuildStats]:
        stats = BuildStats()
        idx = RetrievalIndex(config, self.chunks(config), pool=self.pool, emb_cache=self.emb_cache, stats=stats)
        return idx, stats

    def evaluate(self, config: RagConfig, questions: list[EvalQuestion], *, stage: str = "", fraction: float = 1.0,
                 keep_results: bool = False, real: list[EvalQuestion] | None = None) -> ConfigResult:
        t = time.time()
        idx, stats = self.index(config)
        questions = self.questions_for(config.parser, questions)
        if real:
            real = self.questions_for(config.parser, real)
        results = [score_question(q, idx.retrieve(q.question, k=CANDIDATES, candidates=CANDIDATES)) for q in questions]
        metrics = aggregate(results)
        real_metrics = None
        if real:
            real_results = [score_question(q, idx.retrieve(q.question, k=CANDIDATES, candidates=CANDIDATES)) for q in real]
            real_metrics = aggregate(real_results)
            if keep_results:
                results = results + real_results
        secs = time.time() - t
        self.evaluations += 1
        self.total_seconds += secs
        self.log(f"[{stage}] {config.key} n={len(questions)} {PRIMARY_METRIC}={metrics.get(PRIMARY_METRIC, float('nan')):.3f} ({secs:.1f}s)")
        return ConfigResult(config=config, metrics=metrics, n_questions=len(questions), seconds=secs, stage=stage, fraction=fraction,
                            stats=stats.to_dict(), results=results if keep_results else None, real_metrics=real_metrics)


def _score(r: ConfigResult, metric: str) -> float:
    v = r.metrics.get(metric, float("nan"))
    return -1.0 if math.isnan(v) else v


def _try_eval(ev: "Evaluator", config: RagConfig, questions: list[EvalQuestion], **kw: Any) -> "ConfigResult | None":
    """Evaluate, but drop a config whose API provider has no quota left instead of failing the search."""
    from ragtuner.llm import QuotaExhausted

    try:
        return ev.evaluate(config, questions, **kw)
    except QuotaExhausted as e:
        ev.log(f"[{kw.get('stage', '')}] skipped {config.key}: provider quota exhausted ({str(e)[:120]})")
        ev.skipped.append(config.key)
        return None


def successive_halving(ev: Evaluator, configs: list[RagConfig], evalset: EvalSet, *, stage: str, keep: int,
                       fractions: tuple[float, ...] = (0.25, 0.5, 1.0), metric: str = PRIMARY_METRIC, seed: int = 0) -> list[ConfigResult]:
    """Evaluate all configs on a growing question subset, halving the field each round.
    Returns the final-round results (full set) for the survivors, best first."""
    alive = list(configs)
    results: list[ConfigResult] = []
    for i, frac in enumerate(fractions):
        last = i == len(fractions) - 1
        subset = evalset if frac >= 1.0 else evalset.sample(frac, seed=seed + i)
        results = [r for c in alive if (r := _try_eval(ev, c, subset.questions, stage=f"{stage}/r{i + 1}", fraction=frac))]
        results.sort(key=lambda r: -_score(r, metric))
        if last:
            break
        n_keep = max(keep, math.ceil(len(alive) / 2))
        alive = [r.config for r in results[:n_keep]]
        if len(alive) <= keep:
            # nothing left to prune: jump straight to the full set
            results = [r for c in alive if (r := _try_eval(ev, c, evalset.questions, stage=f"{stage}/final", fraction=1.0))]
            results.sort(key=lambda r: -_score(r, metric))
            break
    return results


@dataclass
class SearchReport:
    corpus: str
    space: dict[str, Any]
    stage1: list[ConfigResult]
    stage2: list[ConfigResult]
    winner: ConfigResult
    default: ConfigResult
    evaluations: int
    seconds: float
    mode: str
    grid: list[ConfigResult] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "corpus": self.corpus, "mode": self.mode, "space": self.space, "evaluations": self.evaluations,
            "seconds": round(self.seconds, 1),
            "winner": self.winner.to_dict(), "default": self.default.to_dict(),
            "stage1": [r.to_dict() for r in self.stage1], "stage2": [r.to_dict() for r in self.stage2],
            "grid": [r.to_dict() for r in self.grid] if self.grid else None,
        }


def staged_search(ev: Evaluator, space: Space, evalset: EvalSet, default: RagConfig, *, carry: int = 2,
                  real: EvalSet | None = None, metric: str = PRIMARY_METRIC, seed: int = 0) -> SearchReport:
    t = time.time()
    _set_anchor(ev, evalset, space)
    synthetic = evalset.synthetic if len(evalset.synthetic) else evalset
    real_qs = real.questions if real else (evalset.real.questions or None)
    ev.log(f"stage 1: {len(space.stage1())} configs (parser x chunker x embedder), selecting on {metric} over {len(synthetic)} synthetic questions")
    s1 = successive_halving(ev, space.stage1(), synthetic, stage="s1", keep=carry, metric=metric, seed=seed)
    carried = [r.config for r in s1[:carry]]
    s2_configs: list[RagConfig] = []
    for base in carried:
        s2_configs.extend(space.stage2(base))
    ev.log(f"stage 2: {len(s2_configs)} configs (retrieval x reranker) for {len(carried)} carried triples")
    s2 = successive_halving(ev, s2_configs, synthetic, stage="s2", keep=3, metric=metric, seed=seed + 10)
    # final: re-evaluate top 3 and the default with per-question results and the real set
    finalists = [r.config for r in s2[:3]]
    final = [ev.evaluate(c, synthetic.questions, stage="final", keep_results=True, real=real_qs) for c in finalists]
    final.sort(key=lambda r: -_score(r, metric))
    default_res = ev.evaluate(default, synthetic.questions, stage="default", keep_results=True, real=real_qs)
    return SearchReport(corpus=ev.corpus.name, space=_space_dict(space), stage1=s1, stage2=s2, winner=final[0], default=default_res,
                        evaluations=ev.evaluations, seconds=time.time() - t, mode="staged")


def full_grid(ev: Evaluator, space: Space, evalset: EvalSet, default: RagConfig, *, real: EvalSet | None = None,
              metric: str = PRIMARY_METRIC) -> SearchReport:
    """Every valid config on the full synthetic set. Used to measure the regret of the
    staged search on a small corpus; too expensive for routine use."""
    t = time.time()
    _set_anchor(ev, evalset, space)
    synthetic = evalset.synthetic if len(evalset.synthetic) else evalset
    real_qs = real.questions if real else (evalset.real.questions or None)
    grid = [ev.evaluate(c, synthetic.questions, stage="grid", real=real_qs) for c in space.grid()]
    grid.sort(key=lambda r: -_score(r, metric))
    winner = ev.evaluate(grid[0].config, synthetic.questions, stage="final", keep_results=True, real=real_qs)
    default_res = ev.evaluate(default, synthetic.questions, stage="default", keep_results=True, real=real_qs)
    return SearchReport(corpus=ev.corpus.name, space=_space_dict(space), stage1=[], stage2=[], winner=winner, default=default_res,
                        evaluations=ev.evaluations, seconds=time.time() - t, mode="grid", grid=grid)


def _set_anchor(ev: Evaluator, evalset: EvalSet, space: Space) -> None:
    if ev.anchor_parser is None:
        ev.anchor_parser = evalset.metadata.get("parser")
    if ev.anchor_parser is None and len(space.parsers) > 1:
        raise ValueError("the eval set does not record its anchor parser; spans cannot be compared across parsers")


def regret(staged: SearchReport, grid: SearchReport, metric: str = PRIMARY_METRIC) -> dict[str, float]:
    best = grid.grid[0].metrics.get(metric, float("nan")) if grid.grid else float("nan")
    found = staged.winner.metrics.get(metric, float("nan"))
    return {"grid_best": best, "staged_winner": found, "regret": best - found}


def _space_dict(space: Space) -> dict[str, Any]:
    return {"parsers": space.parsers, "chunkers": [f"{c}:{p}" for c, p in space.chunkers], "embedders": space.embedders,
            "retrievals": space.retrievals, "rerankers": space.rerankers, "rewrites": space.rewrites, "top_k": space.top_k,
            "stage1_configs": len(space.stage1()), "grid_configs": len(space.grid())}


def estimate_cost(ev: Evaluator, space: Space, *, words_per_doc: int | None = None) -> dict[str, Any]:
    """Rough upfront accounting: how many embedding passes and texts the search implies."""
    docs = ev.docs(space.parsers[0])
    words = words_per_doc * len(docs) if words_per_doc else sum(len(d.text.split()) for d in docs)
    texts = 0
    for c, params in space.chunkers:
        size = params.get("size") or params.get("target_words") or 300
        overlap = params.get("overlap", 0)
        texts += int(words / max(size - overlap, 1)) + len(docs)
    passes = len(space.parsers) * len([e for e in space.embedders if e])
    return {"docs": len(docs), "words": words, "chunk_texts_per_parser": texts, "embedding_passes": passes,
            "texts_to_embed": texts * passes, "stage1_configs": len(space.stage1()), "grid_configs": len(space.grid())}
