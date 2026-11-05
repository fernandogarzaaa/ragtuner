import json

from ragtuner.cli import main


def test_parsers_lists_backends(capsys):
    assert main(["parsers"]) == 0
    out = capsys.readouterr().out
    assert "plaintext" in out and "pymupdf-text" in out and "docling" in out


def test_parse_summary_and_out(sample_pdf, tmp_path, capsys):
    out = tmp_path / "doc.json"
    assert main(["parse", str(sample_pdf), "--out", str(out)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["pages"] == 2 and summary["blocks"]["heading"] >= 2
    assert out.exists()


def test_parse_bad_suffix_for_parser(tmp_path, capsys):
    p = tmp_path / "a.txt"
    p.write_text("x")
    assert main(["parse", str(p), "--parser", "pymupdf-text"]) == 1
    assert "does not handle" in capsys.readouterr().err
