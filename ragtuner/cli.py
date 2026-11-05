from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ragtuner.cache import ParseCache
from ragtuner.parse.base import available_parsers, parse_file


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ragtuner", description="Eval-driven RAG configuration tuner.")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("parsers", help="List parser backends and whether they import")

    pp = sub.add_parser("parse", help="Parse one file into the intermediate document model")
    pp.add_argument("path")
    pp.add_argument("--parser", default=None, help="plaintext | pymupdf-text | docling (default by suffix)")
    pp.add_argument("--out", default=None, help="Write ParsedDocument JSON here")
    pp.add_argument("--cache", default=None, help="Cache directory (enables the parse cache)")
    pp.add_argument("--summary", action="store_true", help="Print block type counts instead of JSON")

    gp = sub.add_parser("gate", help="20-page parser gate: sample, build labeling sheet, score")
    gsub = gp.add_subparsers(dest="gate_command", required=True)
    gs = gsub.add_parser("sample", help="Sample pages and build the labeling sheet")
    gs.add_argument("pdfs", nargs="+", help="PDF files or directories")
    gs.add_argument("--n", type=int, default=20)
    gs.add_argument("--seed", type=int, default=0)
    gs.add_argument("--parsers", default="pymupdf-text,docling")
    gs.add_argument("--out", default="gate_out")
    gc = gsub.add_parser("score", help="Score a filled-in gold file against the sheet")
    gc.add_argument("--sheet", required=True)
    gc.add_argument("--gold", required=True)
    gc.add_argument("--threshold", type=float, default=0.8)
    gc.add_argument("--json", action="store_true")
    eg = sub.add_parser("evalgen", help="Generate the eval set for a corpus (synthetic + imported real queries)")
    eg.add_argument("corpus_dir")
    eg.add_argument("--name", default=None)
    eg.add_argument("--parser", default="pymupdf-text", help="Parser whose block model the generator reads")
    eg.add_argument("--n", type=int, default=100)
    eg.add_argument("--generator", default="auto", help="auto | llm | heuristic")
    eg.add_argument("--seed", type=int, default=0)
    eg.add_argument("--table-fraction", type=float, default=0.25)
    eg.add_argument("--real-financebench", default=None, help="FinanceBench rows JSONL for real questions")
    eg.add_argument("--real-qa", default=None, help="JSONL of {question, answer} rows for real questions (weak gold)")
    eg.add_argument("--cache", default=".ragtuner-cache")
    eg.add_argument("--out", required=True)

    tn = sub.add_parser("tune", help="Run the staged config search and write the report")
    tn.add_argument("corpus_dir")
    tn.add_argument("--name", default=None)
    tn.add_argument("--evalset", required=True)
    tn.add_argument("--out", required=True)
    tn.add_argument("--cache", default=".ragtuner-cache")
    tn.add_argument("--space", default="phase1", help="phase1 | smoke")
    tn.add_argument("--parsers", default=None, help="Comma list to restrict parsers in the space")
    tn.add_argument("--default", default=None, help="YAML of the frozen default config (else the built-in one)")
    tn.add_argument("--mode", default="staged", help="staged | grid")
    tn.add_argument("--with-grid", action="store_true", help="Also run the full grid and report regret")
    tn.add_argument("--carry", type=int, default=2)
    tn.add_argument("--gate", default=None, help="gate report JSON to embed in the report")
    tn.add_argument("--note", action="append", default=[])
    del sp
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "parsers":
            for name, status in available_parsers().items():
                print(f"{name:14} {status}")
            return 0
        if args.command == "parse":
            return _cmd_parse(args)
        if args.command == "gate":
            return _cmd_gate(args)
        if args.command == "evalgen":
            return _cmd_evalgen(args)
        if args.command == "tune":
            return _cmd_tune(args)
    except (ValueError, FileNotFoundError, KeyError) as e:
        print(f"ragtuner: error: {e}", file=sys.stderr)
        return 1
    return 1


def _cmd_parse(args: argparse.Namespace) -> int:
    if args.cache:
        from ragtuner.parse.base import default_parser_for

        name = args.parser or default_parser_for(Path(args.path))
        doc, hit = ParseCache(args.cache).parse(args.path, name)
        print(f"cache {'hit' if hit else 'miss'}", file=sys.stderr)
    else:
        doc = parse_file(args.path, args.parser)
    if args.out:
        doc.save(args.out)
        print(f"wrote {args.out}", file=sys.stderr)
    if args.summary or args.out:
        counts: dict[str, int] = {}
        for b in doc.blocks:
            counts[b.type.value] = counts.get(b.type.value, 0) + 1
        print(json.dumps({"id": doc.id, "parser": doc.parser, "pages": doc.pages, "chars": len(doc.text), "blocks": counts}))
    else:
        print(json.dumps(doc.to_dict(), indent=2, ensure_ascii=False))
    return 0


def _cmd_gate(args: argparse.Namespace) -> int:
    from ragtuner.parse import gate

    if args.gate_command == "sample":
        pdfs: list[Path] = []
        for p in args.pdfs:
            path = Path(p)
            if path.is_dir():
                pdfs.extend(sorted(path.rglob("*.pdf")))
            elif path.suffix.lower() == ".pdf":
                pdfs.append(path)
        if not pdfs:
            raise ValueError("no PDF files found")
        parsers = [x.strip() for x in args.parsers.split(",") if x.strip()]
        samples = gate.sample_pages(pdfs, args.n, seed=args.seed)
        paths = gate.build_sheet(samples, parsers, Path(args.out))
        pools = {}
        for s in samples:
            pools[s.pool] = pools.get(s.pool, 0) + 1
        print(json.dumps({"sampled_pages": len(samples), "pools": pools, **{k: str(v) for k, v in paths.items()}}, indent=2))
        print("Next: fill in 'tables' for each row of gold_template.jsonl, then run `ragtuner gate score`.", file=sys.stderr)
        return 0
    if args.gate_command == "score":
        report = gate.score(Path(args.sheet), Path(args.gold), threshold=args.threshold)
        if args.json:
            print(json.dumps({"threshold": report.threshold, "per_parser": report.per_parser}, indent=2))
        else:
            print(report.to_markdown())
        return 0
    return 1


def _cmd_evalgen(args: argparse.Namespace) -> int:
    from ragtuner.run import generate_evalset, load_corpus

    corpus = load_corpus(args.name or Path(args.corpus_dir).name, args.corpus_dir)
    es = generate_evalset(corpus, parser=args.parser, n_synthetic=args.n, cache_root=Path(args.cache), seed=args.seed,
                          generator=args.generator, real_financebench=Path(args.real_financebench) if args.real_financebench else None,
                          real_qa=Path(args.real_qa) if args.real_qa else None, table_fraction=args.table_fraction)
    es.save(args.out)
    print(json.dumps(es.summary(), indent=2))
    return 0


def _cmd_tune(args: argparse.Namespace) -> int:
    from ragtuner.config import RagConfig
    from ragtuner.evalset import EvalSet
    from ragtuner.run import load_corpus, space_preset, tune

    corpus = load_corpus(args.name or Path(args.corpus_dir).name, args.corpus_dir)
    es = EvalSet.load(args.evalset)
    parsers = [p.strip() for p in args.parsers.split(",")] if args.parsers else None
    space = space_preset(args.space, parsers=parsers)
    default = RagConfig.from_yaml(args.default) if args.default else None
    gate = json.loads(Path(args.gate).read_text()) if args.gate else None
    paths = tune(corpus, es, out=Path(args.out), cache_root=Path(args.cache), space=space, default=default, mode=args.mode,
                 with_grid=args.with_grid, carry=args.carry, gate=gate, notes=args.note)
    print(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
