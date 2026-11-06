"""Production monitoring: read the API's query log, compare it with the tuning baseline,
check corpus freshness, and propose concrete fixes.

Signals
- low confidence: share of recent queries whose top score is below the 10th percentile of
  the winner's top scores on answerable eval questions (expected about 10%).
- drift: median top score of recent queries against the eval baseline median.
- empty results, and latency p50/p95.
- freshness: corpus files added, removed or changed since the manifest was written.
Proposals are actions with commands: label low-confidence queries as new real eval
questions (written to a candidates file), re-run tuning when confidence drifts, reload
when the corpus changed, check permissions when queries return nothing.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

LOW_SHARE_ALERT = 0.2
DRIFT_ALERT = 0.2


def _pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    if not xs:
        return float("nan")
    i = min(len(xs) - 1, max(0, int(round(p * (len(xs) - 1)))))
    return xs[i]


def monitor(log_path: str | Path, run_dir: str | Path, *, corpus_dir: str | Path | None = None, window: int = 500,
            candidates_out: str | Path | None = None) -> dict[str, Any]:
    run = Path(run_dir)
    winner = json.loads((run / "winner_results.json").read_text())
    base_scores = [r["top_score"] for r in winner if r.get("top_score") is not None and r.get("span_ranks")]
    p10, base_median = _pct(base_scores, 0.10), statistics.median(base_scores) if base_scores else float("nan")
    recs = [json.loads(line) for line in Path(log_path).read_text().splitlines() if line.strip()][-window:]
    scored = [r for r in recs if r.get("top_score") is not None]
    empty = [r for r in recs if not r.get("n_hits")]
    low = [r for r in scored if r["top_score"] < p10]
    lat = [r["latency_ms"] for r in recs if r.get("latency_ms") is not None]
    rec_median = statistics.median([r["top_score"] for r in scored]) if scored else float("nan")
    out: dict[str, Any] = {
        "queries": len(recs), "baseline_p10": p10, "baseline_median": base_median, "recent_median": rec_median,
        "low_confidence_share": len(low) / len(scored) if scored else 0.0, "empty_share": len(empty) / len(recs) if recs else 0.0,
        "latency_p50_ms": _pct(lat, 0.5), "latency_p95_ms": _pct(lat, 0.95), "proposals": [],
    }
    drift = (base_median - rec_median) / abs(base_median) if base_scores and scored and base_median else 0.0
    out["median_drop"] = drift
    if candidates_out and low:
        seen, rows = set(), []
        for r in sorted(low, key=lambda r: r["top_score"]):
            if r["query"] not in seen:
                seen.add(r["query"])
                rows.append({"question": r["query"], "top_score": r["top_score"], "hits": r["hits"][:5], "answer": None})
        Path(candidates_out).write_text("".join(json.dumps(x) + "\n" for x in rows))
        out["candidates_file"] = str(candidates_out)
    if out["low_confidence_share"] > LOW_SHARE_ALERT:
        out["proposals"].append({"signal": "low_confidence", "action": "Label the low-confidence queries (answer + evidence) and add them as real eval questions, then re-run tuning on the enlarged real set.",
                                 "command": "ragtuner monitor ... --candidates candidates.jsonl  # label, then: ragtuner evalgen ... --real-qa candidates.jsonl"})
    if drift > DRIFT_ALERT:
        out["proposals"].append({"signal": "drift", "action": f"Top-score median fell {drift:.0%} below the eval baseline: traffic has moved away from what the config was tuned on. Re-run evalgen and tune with recent queries included.",
                                 "command": "ragtuner evalgen ... && ragtuner tune ..."})
    if recs and out["empty_share"] > 0.05:
        out["proposals"].append({"signal": "empty", "action": "More than 5% of queries returned nothing: check tenant and principal headers against acl.json, and whether the corpus covers these topics.",
                                 "command": "inspect queries.jsonl entries with n_hits=0"})
    if corpus_dir is not None and (run / "manifest.json").exists():
        from ragtuner.run import load_corpus
        from ragtuner.serve.runtime import corpus_state, diff_state

        old = json.loads((run / "manifest.json").read_text()).get("corpus_state") or {}
        new = corpus_state(load_corpus("c", corpus_dir).files)
        old_n = {Path(k).name: v for k, v in old.items()}
        new_n = {Path(k).name: v for k, v in new.items()}
        diff = diff_state(old_n, new_n)
        out["freshness"] = {k: len(v) for k, v in diff.items()}
        if any(diff.values()):
            out["proposals"].append({"signal": "stale_index", "action": f"Corpus changed since tuning ({out['freshness']}): reload the index (incremental), and re-run tuning if many documents changed.",
                                     "command": "curl -X POST -H 'X-Admin-Token: ...' http://host/admin/reload   # or: ragtuner sync"})
    return out


def to_markdown(m: dict[str, Any]) -> str:
    lines = ["| signal | value |", "|---|---|",
             f"| queries in window | {m['queries']} |",
             f"| low-confidence share (expected about 10%) | {m['low_confidence_share']:.1%} |",
             f"| top-score median: baseline / recent | {m['baseline_median']:.3f} / {m['recent_median']:.3f} |",
             f"| empty-result share | {m['empty_share']:.1%} |",
             f"| latency p50 / p95 (ms) | {m['latency_p50_ms']} / {m['latency_p95_ms']} |"]
    if "freshness" in m:
        lines.append(f"| corpus changes since tuning | {m['freshness']} |")
    lines.append("")
    if m["proposals"]:
        lines += [f"- **{p['signal']}**: {p['action']} `{p['command']}`" for p in m["proposals"]]
    else:
        lines.append("No action needed.")
    return "\n".join(lines)
