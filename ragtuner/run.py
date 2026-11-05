"""End-to-end run: eval set generation and the tuning loop, behind the CLI."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from ragtuner.cache import ParseCache
from ragtuner.config import FROZEN_DEFAULT, RagConfig
from ragtuner.diagnose import analyze
from ragtuner.embed import available_embedders
from ragtuner.evalgen import HeuristicGenerator, LLMGenerator, import_financebench, import_qa_pairs
from ragtuner.evalset import EvalSet
from ragtuner.parse.base import available_parsers
from ragtuner.pipeline import Corpus, ModelPool, parse_corpus
from ragtuner.report import write_report
from ragtuner.search import Evaluator, Space, estimate_cost, full_grid, regret, staged_search


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def load_corpus(name: str, root: str | Path) -> Corpus:
    root = Path(root)
    docs_dir = root / "docs" if (root / "docs").is_dir() else root
    corpus = Corpus.from_dir(name, docs_dir)
    ids = root / "doc_ids.json"
    if ids.exists():
        mapping = json.loads(ids.read_text())
        corpus.doc_ids = {str(docs_dir / fname): doc_id for fname, doc_id in mapping.items()}
    return corpus


def have_llm() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def generate_evalset(corpus: Corpus, *, parser: str, n_synthetic: int, cache_root: Path, seed: int = 0,
                     generator: str = "auto", real_financebench: Path | None = None, real_qa: Path | None = None,
                     table_fraction: float = 0.25) -> EvalSet:
    """Synthetic questions come from the parsed block model of `parser` (independent of any
    chunker). Real questions are imported and held out."""
    cache = ParseCache(cache_root)
    docs = parse_corpus(corpus, parser, cache)
    log(f"evalgen: {len(docs)} docs parsed with {parser}")
    gen_name = generator
    if generator == "auto":
        gen_name = "llm" if have_llm() else "heuristic"
    if gen_name == "llm":
        gen = LLMGenerator(seed=seed, table_fraction=table_fraction)
    else:
        gen = HeuristicGenerator(seed=seed, table_fraction=table_fraction)
    synth = gen.generate(docs, n_synthetic, name=f"{corpus.name}-synthetic")
    log(f"evalgen: {len(synth)} synthetic questions via {gen_name} ({synth.summary()})")
    sets = [synth]
    if real_financebench:
        by_name = {}
        for d in docs:
            by_name[d.id] = d
        real = import_financebench(real_financebench, by_name, name=f"{corpus.name}-real")
        log(f"evalgen: {len(real)} real FinanceBench questions located ({len(real.metadata.get('skipped', []))} skipped)")
        sets.append(real)
    if real_qa:
        rows = [json.loads(line) for line in Path(real_qa).read_text().splitlines() if line.strip()]
        real = import_qa_pairs(rows, docs, name=f"{corpus.name}-real")
        log(f"evalgen: {len(real)} real QA questions located with weak gold ({real.metadata.get('skipped')} skipped)")
        sets.append(real)
    es = EvalSet.merge(corpus.name, sets)
    es.metadata = {"corpus": corpus.name, "parser": parser, "generator": gen_name, "seed": seed, "n_synthetic_requested": n_synthetic}
    return es


def space_preset(name: str, *, parsers: list[str] | None = None) -> Space:
    status = available_parsers()
    emb = available_embedders()
    default_parsers = [p for p in ("pymupdf-text", "docling") if status.get(p) == "ok"]
    embedders: list[str | None] = [None] + [e for e in ("bge-small", "openai-small") if emb.get(e) == "ok"]
    if name == "smoke":
        return Space(parsers=parsers or default_parsers[:1], chunkers=[("fixed", {"size": 120, "overlap": 24}), ("structure", {"target_words": 300, "max_words": 600})],
                     embedders=embedders[:2], retrievals=["sparse", "dense", "hybrid"], rerankers=[None])
    if name == "phase1":
        return Space(parsers=parsers or default_parsers, embedders=embedders)
    raise ValueError(f"unknown space preset {name!r}")


def resolve_default(default: RagConfig | None) -> tuple[RagConfig, list[str]]:
    notes = []
    cfg = default or FROZEN_DEFAULT
    emb = available_embedders()
    if cfg.embedder and emb.get(cfg.embedder) != "ok":
        notes.append(f"Default embedder {cfg.embedder!r} unavailable ({emb.get(cfg.embedder)}); default run with bge-small instead.")
        cfg = RagConfig.from_dict({**cfg.to_dict(), "embedder": "bge-small"})
    parsers = available_parsers()
    if parsers.get(cfg.parser) != "ok":
        raise RuntimeError(f"default parser {cfg.parser!r} unavailable: {parsers.get(cfg.parser)}")
    return cfg, notes


def tune(corpus: Corpus, evalset: EvalSet, *, out: Path, cache_root: Path, space: Space, default: RagConfig | None = None,
         mode: str = "staged", with_grid: bool = False, carry: int = 2, gate: dict[str, Any] | None = None, notes: list[str] | None = None) -> dict[str, Path]:
    default_cfg, dnotes = resolve_default(default)
    notes = list(notes or []) + dnotes
    pool = ModelPool()
    ev = Evaluator(corpus, cache_root=cache_root, pool=pool, log=log)
    est = estimate_cost(ev, space)
    log(f"estimate: {est}")
    if mode == "grid":
        rep = full_grid(ev, space, evalset, default_cfg)
        reg = None
    else:
        rep = staged_search(ev, space, evalset, default_cfg, carry=carry)
        reg = None
        if with_grid:
            grid_rep = full_grid(ev, space, evalset, default_cfg)
            reg = regret(rep, grid_rep)
            rep.grid = grid_rep.grid
    assert rep.winner.results is not None and rep.default.results is not None
    all_qs = evalset.questions
    fails = analyze(all_qs, rep.winner.results, ev.chunks(rep.winner.config), k=5)
    dfails = analyze(all_qs, rep.default.results, ev.chunks(rep.default.config), k=5)
    notes.append(f"Synthetic questions generated by {evalset.metadata.get('generator', '?')}; selection used synthetic only, real set held out.")
    if evalset.metadata.get("generator") == "heuristic":
        notes.append("Heuristic generator is lexically leaky by construction; synthetic numbers favor sparse retrieval. Judge on the real set.")
    weak = sum(1 for q in evalset.real.questions if q.gold_quality == "weak")
    if weak:
        notes.append(f"{weak} real questions have weak (automatically located) gold spans.")
    paths = write_report(out, rep, evalset, fails, dfails, gate=gate, notes=notes, regret=reg)
    (out / "search.json").write_text(json.dumps(rep.to_dict(), indent=2, default=str))
    (out / "winner_results.json").write_text(json.dumps([r.to_dict() for r in rep.winner.results], indent=2, default=str))
    (out / "default_results.json").write_text(json.dumps([r.to_dict() for r in rep.default.results], indent=2, default=str))
    log(f"done: {rep.evaluations} evaluations in {rep.seconds / 60:.1f} min; report at {paths['report_md']}")
    return paths
