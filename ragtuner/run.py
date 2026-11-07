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
    acl_path = root / "acl.json"
    if acl_path.exists():
        from ragtuner.access import load_acl
        from ragtuner.parse.base import doc_id_for

        raw = load_acl(acl_path)
        # keys may be file names or doc ids; resolve file names to the doc id the parser assigns
        by_name = {f.name: (corpus.doc_id_for(f) or doc_id_for(f)) for f in corpus.files}
        corpus.acl = {by_name.get(k, k): v for k, v in raw.items()}
    return corpus


def have_llm() -> bool:
    from ragtuner.llm import provider

    return provider() is not None


def generate_evalset(corpus: Corpus, *, parser: str, n_synthetic: int, cache_root: Path, seed: int = 0,
                     generator: str = "auto", real_financebench: Path | None = None, real_qa: Path | None = None,
                     table_fraction: float = 0.25, llm_model: str | None = None, max_calls: int | None = None,
                     unanswerable_fraction: float = 0.1, multihop_fraction: float = 0.1, checkpoint: Path | None = None) -> EvalSet:
    """Synthetic questions come from the parsed block model of `parser` (independent of any
    chunker). Real questions are imported and held out."""
    cache = ParseCache(cache_root)
    docs = parse_corpus(corpus, parser, cache)
    log(f"evalgen: {len(docs)} docs parsed with {parser}")
    gen_name = generator
    if generator == "auto":
        gen_name = "llm" if have_llm() else "heuristic"
    if gen_name == "llm":
        gen = LLMGenerator(seed=seed, table_fraction=table_fraction, model=llm_model, max_calls=max_calls,
                           unanswerable_fraction=unanswerable_fraction, multihop_fraction=multihop_fraction, checkpoint=checkpoint)
    else:
        gen = HeuristicGenerator(seed=seed, table_fraction=table_fraction)
    synth = gen.generate(docs, n_synthetic, name=f"{corpus.name}-synthetic")
    log(f"evalgen: {len(synth)} synthetic questions via {gen_name} ({synth.summary()})")
    if gen_name == "llm":
        from ragtuner.llm import USAGE

        log(f"evalgen: llm model {synth.metadata.get('model')} calls={synth.metadata.get('calls')} dropped={synth.metadata.get('dropped')} usage={USAGE}")
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
    es.metadata = {"corpus": corpus.name, "parser": parser, "generator": gen_name, "seed": seed, "n_synthetic_requested": n_synthetic,
                   "generator_model": synth.metadata.get("model"), "generator_calls": synth.metadata.get("calls")}
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
    if name == "phase2":
        from ragtuner.llm import provider

        rewrites: list[str | None] = [None, "prf"] + (["llm"] if provider() and os.environ.get("RAGTUNER_LLM_REWRITE") == "1" else [])
        return Space(parsers=parsers or default_parsers, embedders=embedders, rewrites=rewrites)
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
            notes.append(f"Staged search alone: {rep.evaluations} evaluations in {rep.seconds / 60:.1f} min. "
                         f"The full grid for the regret check added {ev.evaluations - rep.evaluations} evaluations and {grid_rep.seconds / 60:.1f} min.")
            rep.seconds += grid_rep.seconds
            rep.evaluations = ev.evaluations
    assert rep.winner.results is not None and rep.default.results is not None
    # failure analysis must see spans in the same text as the config's chunks
    fails = analyze(ev.questions_for(rep.winner.config.parser, evalset.questions), rep.winner.results, ev.chunks(rep.winner.config), k=5)
    dfails = analyze(ev.questions_for(rep.default.config.parser, evalset.questions), rep.default.results, ev.chunks(rep.default.config), k=5)
    for parser, st in ev.projection_stats.items():
        notes.append(f"Gold spans anchored in {ev.anchor_parser} text were projected onto {parser} text: {st['exact']} exact, "
                     f"{st['fuzzy']} fuzzy, {st['lost']} lost of {st['spans']}. Lost spans count as {parser} parse failures.")
    notes.append(f"Synthetic questions generated by {evalset.metadata.get('generator', '?')}; selection used synthetic only, real set held out.")
    if evalset.metadata.get("generator") == "heuristic":
        notes.append("Heuristic generator is lexically leaky by construction; synthetic numbers favor sparse retrieval. Judge on the real set.")
    weak = sum(1 for q in evalset.real.questions if q.gold_quality == "weak")
    if weak:
        notes.append(f"{weak} real questions have weak (automatically located) gold spans.")
    generation = None
    if have_llm() and os.environ.get("RAGTUNER_JUDGE") == "1":
        try:
            from ragtuner.judge import Judge

            judge = Judge()
            finalists = [rep.winner, rep.default]
            generation = {}
            for r in finalists:
                assert r.results is not None
                hits = {qr.question_id: qr.hits for qr in r.results}
                g = judge.run(r.config.key, evalset.real.questions or evalset.questions[:30], hits, k=r.config.top_k)
                generation[r.config.key] = g.to_dict()
                log(f"judge: {r.config.key} mean score {g.mean_score:.3f} over {len(g.judged)} questions ({g.errors} errors)")
            (out / "generation.json").write_text(json.dumps(generation, indent=2))
            notes.append("Generation quality (LLM judge) for winner and default is in generation.json.")
        except Exception as e:  # noqa: BLE001
            notes.append(f"Generation judge skipped: {type(e).__name__}: {e}")
    else:
        notes.append("Generation judge skipped: set RAGTUNER_JUDGE=1 with LLM credentials to run it.")
    paths = write_report(out, rep, evalset, fails, dfails, gate=gate, notes=notes, regret=reg)
    # record corpus file hashes so `ragtuner monitor` can tell when the index has gone stale
    from ragtuner.serve.runtime import corpus_state

    man_path = out / "manifest.json"
    man = json.loads(man_path.read_text())
    man["corpus_state"] = corpus_state(corpus.files)
    man_path.write_text(json.dumps(man, indent=2))
    (out / "search.json").write_text(json.dumps(rep.to_dict(), indent=2, default=str))
    (out / "winner_results.json").write_text(json.dumps([r.to_dict() for r in rep.winner.results], indent=2, default=str))
    (out / "default_results.json").write_text(json.dumps([r.to_dict() for r in rep.default.results], indent=2, default=str))
    log(f"done: {rep.evaluations} evaluations in {rep.seconds / 60:.1f} min; report at {paths['report_md']}")
    return paths
