import json
from pathlib import Path

from ragtuner.parse import gate


def test_normalize_cell_handles_money_and_negatives():
    assert gate.normalize_cell("$ 1,234") == "1234"
    assert gate.normalize_cell("(12)") == "-12"
    assert gate.normalize_cell("12.5%") == "12.5"
    assert gate.normalize_cell("  North   America ") == "north america"


def test_score_page_aligns_tables_and_counts_cells():
    gold = [[["Region", "Q1"], ["North", "120"]]]
    pred_perfect = [[["Region", "Q1"], ["North", "$120"]]]
    assert gate.score_page(pred_perfect, gold) == (4, 4, 4)
    pred_partial = [[["Region", "Q1"], ["North", "999"]], [["junk"]]]
    tp, pred_n, gold_n = gate.score_page(pred_partial, gold)
    assert (tp, pred_n, gold_n) == (3, 5, 4)
    assert gate.score_page([], gold) == (0, 0, 4)


def test_sample_build_and_score_end_to_end(sample_pdf: Path, tmp_path: Path):
    samples = gate.sample_pages([sample_pdf], n=2, seed=1)
    assert len(samples) == 2
    assert {s.pool for s in samples} <= {"random", "table_candidate"}
    paths = gate.build_sheet(samples, ["pymupdf-text"], tmp_path / "gate")
    assert paths["sheet"].exists() and paths["gold_template"].exists()
    assert sorted(p.name for p in paths["pages"].iterdir()) == ["report_p1.png", "report_p2.png"]

    # a human fills in gold: page 1 has the 3x3 table, page 2 has none
    gold_rows = [json.loads(line) for line in paths["gold_template"].read_text().splitlines()]
    for r in gold_rows:
        r["tables"] = [[["Region", "Q1", "Q2"], ["North", "120", "140"], ["South", "90", "110"]]] if r["page"] == 1 else []
    gold_path = tmp_path / "gold.jsonl"
    gold_path.write_text("\n".join(json.dumps(r) for r in gold_rows))

    report = gate.score(paths["sheet"], gold_path, threshold=0.8)
    m = report.per_parser["pymupdf-text"]
    # text-only backend extracts no tables: recall is 0, so it must fail the gate
    assert m["gold_cells"] == 9
    assert m["cell_recall"] == 0.0
    assert not report.passed("pymupdf-text")
    assert "pymupdf-text" in report.to_markdown()
