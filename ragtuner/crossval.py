"""Does the synthetic set pick the configs real users would want?

Two checks, both from the Phase 1 spec:
1. Agreement: rank every config in the space on the synthetic set and on the real set,
   report Spearman correlation of the two rankings. Low correlation means the generator
   is biased and its winner cannot be trusted.
2. Honest selection on real queries: repeatedly split the real questions in half,
   select the config with the best recall@5 on one half, and measure it against the
   default on the other half. This estimates what tuning on a small real set buys,
   without selecting and scoring on the same questions.
An oracle number (best config on the full real set) is also reported, labeled as an
optimistic upper bound because it selects on the test questions.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from ragtuner.config import RagConfig
from ragtuner.evalset import EvalSet
from ragtuner.metrics import score_question
from ragtuner.search import CANDIDATES, Evaluator, Space


def _ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(a: list[float], b: list[float]) -> float:
    ra, rb = _ranks(a), _ranks(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra) ** 0.5
    vb = sum((y - mb) ** 2 for y in rb) ** 0.5
    return cov / (va * vb) if va and vb else float("nan")


@dataclass
class CrossvalReport:
    n_configs: int
    n_real: int
    n_synth: int
    spearman_recall5: float
    default_key: str
    default_real: float
    oracle_key: str
    oracle_real: float
    synthetic_winner_key: str
    synthetic_winner_real: float
    splits: int
    mean_delta: float
    win_rate: float
    picks: dict[str, int] = field(default_factory=dict)
    table: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    def to_markdown(self) -> str:
        top_picks = sorted(self.picks.items(), key=lambda kv: -kv[1])[:5]
        lines = [
            f"Configs: {self.n_configs}. Real questions: {self.n_real}. Synthetic questions: {self.n_synth}.",
            "",
            "| measure | value |",
            "|---|---|",
            f"| Spearman, synthetic vs real recall@5 ranking | {self.spearman_recall5:.3f} |",
            f"| Default, real recall@5 | {self.default_real:.3f} |",
            f"| Synthetic-selected winner, real recall@5 | {self.synthetic_winner_real:.3f} |",
            f"| Oracle best on full real set (optimistic upper bound) | {self.oracle_real:.3f} |",
            f"| Split-half selection: mean delta vs default on held-out half | {self.mean_delta:+.3f} |",
            f"| Split-half selection: share of splits beating the default | {self.win_rate:.0%} |",
            "",
            f"Splits: {self.splits}. Oracle config: `{self.oracle_key}`.",
            "",
            "Most frequent split-half picks: " + ", ".join(f"`{k}` ({v})" for k, v in top_picks),
        ]
        return "\n".join(lines)


def run_crossval(ev: Evaluator, space: Space, evalset: EvalSet, default: RagConfig, *, splits: int = 200, seed: int = 0,
                 k: int = 5) -> CrossvalReport:
    if ev.anchor_parser is None:
        ev.anchor_parser = evalset.metadata.get("parser")
    synth = [q for q in evalset.synthetic.questions if q.answerable]
    real = [q for q in evalset.real.questions if q.answerable]
    if len(real) < 4:
        raise ValueError("need at least 4 answerable real questions")
    configs = {c.key: c for c in space.grid()}
    configs[default.key] = default
    hit: dict[str, dict[str, float]] = {}  # config key -> question id -> recall@k
    synth_score: dict[str, float] = {}
    for key, cfg in configs.items():
        idx, _ = ev.index(cfg)
        qs_s = ev.questions_for(cfg.parser, synth)
        qs_r = ev.questions_for(cfg.parser, real)
        rs = [score_question(q, idx.retrieve(q.question, k=CANDIDATES, candidates=CANDIDATES)) for q in qs_s]
        rr = {q.id: score_question(q, idx.retrieve(q.question, k=CANDIDATES, candidates=CANDIDATES)).recall_at(k) for q in qs_r}
        synth_score[key] = sum(r.recall_at(k) for r in rs) / len(rs)
        hit[key] = rr
        ev.log(f"[crossval] {key} synthetic={synth_score[key]:.3f} real={sum(rr.values()) / len(rr):.3f}")
    real_score = {key: sum(v.values()) / len(v) for key, v in hit.items()}
    grid_keys = [key for key in configs if key != default.key or default.key in {c.key for c in space.grid()}]
    rho = spearman([synth_score[x] for x in grid_keys], [real_score[x] for x in grid_keys])
    synth_winner = max(grid_keys, key=lambda x: (synth_score[x], x))
    oracle = max(grid_keys, key=lambda x: (real_score[x], x))
    rng = random.Random(seed)
    ids = [q.id for q in real]
    deltas, wins, picks = [], 0, {}
    for _ in range(splits):
        rng.shuffle(ids)
        half = len(ids) // 2
        sel, test = ids[:half], ids[half:]
        pick = max(grid_keys, key=lambda x: (sum(hit[x][i] for i in sel), synth_score[x]))
        picks[pick] = picks.get(pick, 0) + 1
        d = sum(hit[pick][i] for i in test) / len(test) - sum(hit[default.key][i] for i in test) / len(test)
        deltas.append(d)
        wins += d > 0
    table = sorted(({"key": x, "synthetic": round(synth_score[x], 4), "real": round(real_score[x], 4)} for x in grid_keys),
                   key=lambda r: -r["real"])
    return CrossvalReport(
        n_configs=len(grid_keys), n_real=len(real), n_synth=len(synth), spearman_recall5=rho,
        default_key=default.key, default_real=real_score[default.key], oracle_key=oracle, oracle_real=real_score[oracle],
        synthetic_winner_key=synth_winner, synthetic_winner_real=real_score[synth_winner], splits=splits,
        mean_delta=sum(deltas) / len(deltas), win_rate=wins / splits, picks=picks, table=table,
    )
