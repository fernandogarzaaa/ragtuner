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


if __name__ == "__main__":
    raise SystemExit(main())
