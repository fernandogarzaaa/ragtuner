"""Report writer and config export."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ragtuner.config import RagConfig
from ragtuner.diagnose import FailureReport
from ragtuner.evalset import EvalSet
from ragtuner.search import PRIMARY_METRIC, ConfigResult, SearchReport

HYPOTHESIS_MARGIN = 0.05
METRICS = ["recall@5", "mrr@10", "recall@10", "ndcg@10", "hit@50", "refusal_rate@p10"]


def _fmt(v: Any) -> str:
    if v is None:
        return "n/a"
    try:
        if v != v:  # nan
            return "n/a"
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return str(v)


def _row(name: str, r: ConfigResult, real: bool) -> str:
    m = r.real_metrics if real else r.metrics
    if m is None:
        return f"| {name} | " + " | ".join("n/a" for _ in METRICS) + " |"
    return f"| {name} | " + " | ".join(_fmt(m.get(k)) for k in METRICS) + " |"


@dataclass
class Verdict:
    metric: str
    default_real: float | None
    winner_real: float | None
    delta_real: float | None
    default_synth: float
    winner_synth: float
    delta_synth: float
    passed: bool | None  # None when there is no real set
    n_real: int
    n_synth: int

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def verdict(rep: SearchReport, n_real: int, n_synth: int, metric: str = PRIMARY_METRIC) -> Verdict:
    ds, ws = rep.default.metrics.get(metric, float("nan")), rep.winner.metrics.get(metric, float("nan"))
    dr = rep.default.real_metrics.get(metric) if rep.default.real_metrics else None
    wr = rep.winner.real_metrics.get(metric) if rep.winner.real_metrics else None
    delta_real = (wr - dr) if (dr is not None and wr is not None) else None
    passed = None if delta_real is None else delta_real >= HYPOTHESIS_MARGIN
    return Verdict(metric=metric, default_real=dr, winner_real=wr, delta_real=delta_real, default_synth=ds, winner_synth=ws,
                   delta_synth=ws - ds, passed=passed, n_real=n_real, n_synth=n_synth)


def write_report(out_dir: str | Path, rep: SearchReport, evalset: EvalSet, failures: FailureReport, default_failures: FailureReport | None,
                 *, gate: dict[str, Any] | None = None, notes: list[str] | None = None, regret: dict[str, float] | None = None) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_real, n_synth = len(evalset.real), len(evalset.synthetic)
    v = verdict(rep, n_real, n_synth)
    md = [f"# ragtuner report: {rep.corpus}", ""]
    md.append(f"Mode: {rep.mode}. Evaluations: {rep.evaluations}. Wall time: {rep.seconds / 60:.1f} min.")
    md.append("")
    md.append("## Verdict")
    md.append("")
    if v.passed is None:
        md.append(f"No real-query set, so the pre-registered hypothesis cannot be judged. Synthetic {v.metric}: default {_fmt(v.default_synth)}, winner {_fmt(v.winner_synth)} (delta {_fmt(v.delta_synth)}).")
    else:
        md.append(f"Hypothesis (winner beats default by >= {HYPOTHESIS_MARGIN:.2f} {v.metric} on the held-out real set): **{'PASS' if v.passed else 'FAIL'}**.")
        md.append(f"Real set (n={n_real}): default {_fmt(v.default_real)}, winner {_fmt(v.winner_real)}, delta {_fmt(v.delta_real)}.")
        md.append(f"Synthetic set (n={n_synth}, used for selection only): default {_fmt(v.default_synth)}, winner {_fmt(v.winner_synth)}, delta {_fmt(v.delta_synth)}.")
    md.append("")
    md.append("## Configs")
    md.append("")
    md.append(f"Default: `{rep.default.config.key}`")
    md.append("")
    md.append(f"Winner: `{rep.winner.config.key}`")
    md.append("")
    md.append("```yaml")
    import yaml

    md.append(yaml.safe_dump(rep.winner.config.to_dict(), sort_keys=False).strip())
    md.append("```")
    md.append("")
    for title, real in (("Real queries (held out)", True), ("Synthetic queries (selection set)", False)):
        md.append(f"## {title}")
        md.append("")
        md.append("| config | " + " | ".join(METRICS) + " |")
        md.append("|" + "---|" * (len(METRICS) + 1))
        md.append(_row("default", rep.default, real))
        md.append(_row("winner", rep.winner, real))
        md.append("")
    md.append("## Eval set")
    md.append("")
    md.append("```json")
    md.append(json.dumps(evalset.summary(), indent=2))
    md.append("```")
    md.append("")
    if gate:
        md.append("## Parser gate")
        md.append("")
        md.append(gate.get("markdown", json.dumps(gate)))
        md.append("")
    md.append("## Failure analysis (winner)")
    md.append("")
    md.append(failures.to_markdown())
    md.append("")
    if default_failures:
        md.append("## Failure analysis (default)")
        md.append("")
        md.append(default_failures.to_markdown())
        md.append("")
    md.append("## Search trace")
    md.append("")
    md.append("| stage | config | n | " + PRIMARY_METRIC + " | seconds | embedded texts |")
    md.append("|---|---|---|---|---|---|")
    for r in rep.stage1 + rep.stage2 + ([] if rep.grid is None else rep.grid):
        md.append(f"| {r.stage} | `{r.config.key}` | {r.n_questions} | {_fmt(r.metrics.get(PRIMARY_METRIC))} | {r.seconds:.1f} | {r.stats.get('embedded_texts', 0)} |")
    md.append("")
    if regret:
        md.append("## Regret vs full grid")
        md.append("")
        md.append(f"Grid best {_fmt(regret['grid_best'])}, staged winner {_fmt(regret['staged_winner'])}, regret {_fmt(regret['regret'])} ({PRIMARY_METRIC}, synthetic).")
        md.append("")
    if notes:
        md.append("## Notes")
        md.append("")
        md.extend(f"- {n}" for n in notes)
        md.append("")
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")
    payload = {"verdict": v.to_dict(), "search": rep.to_dict(), "evalset": evalset.summary(), "failures": failures.to_dict(),
               "default_failures": default_failures.to_dict() if default_failures else None, "gate": gate, "regret": regret, "notes": notes or []}
    (out / "report.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    (out / "diagnoses.json").write_text(json.dumps([d.to_dict() for d in failures.diagnoses], indent=2), encoding="utf-8")
    export(out, rep.winner.config, rep)
    evalset.save(out / "eval_set.jsonl")
    return {"report_md": out / "report.md", "report_json": out / "report.json", "best_config": out / "best_config.yaml"}


def _pkg_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:  # noqa: BLE001
        return None


def export(out_dir: Path, config: RagConfig, rep: SearchReport | None = None) -> Path:
    """best_config.yaml plus a manifest tying it to the exact runtime that evaluated it."""
    config.to_yaml(out_dir / "best_config.yaml")
    cfg_hash = hashlib.sha256(json.dumps(config.to_dict(), sort_keys=True).encode()).hexdigest()
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:  # noqa: BLE001
        commit = None
    manifest = {
        "config_hash": cfg_hash,
        "config_key": config.key,
        "ragtuner_version": _pkg_version("ragtuner"),
        "ragtuner_commit": commit,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {p: _pkg_version(p) for p in ("pymupdf", "docling", "sentence-transformers", "torch", "better-rag", "numpy")},
        "models": {"embedder": config.embedder, "reranker": config.reranker},
        "metrics": {"synthetic": rep.winner.metrics, "real": rep.winner.real_metrics} if rep else None,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out_dir / "best_config.yaml"
