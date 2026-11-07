from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    """A two-page born-digital PDF with a large heading, body paragraphs, a bullet list
    and a ruled table drawn with lines so PyMuPDF's table finder can see it."""
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Annual Report 2025", fontsize=22)
    page.insert_text((72, 110), "Revenue grew in every region. Costs were flat across the year.", fontsize=10)
    page.insert_text((72, 130), "- North America led growth", fontsize=10)
    page.insert_text((72, 145), "- Europe was stable", fontsize=10)
    # ruled 3x3 table
    x0, y0, w, h = 72, 200, 120, 20
    cells = [["Region", "Q1", "Q2"], ["North", "120", "140"], ["South", "90", "110"]]
    for r in range(4):
        page.draw_line((x0, y0 + r * h), (x0 + 3 * w, y0 + r * h))
    for c in range(4):
        page.draw_line((x0 + c * w, y0), (x0 + c * w, y0 + 3 * h))
    for r, row in enumerate(cells):
        for c, val in enumerate(row):
            page.insert_text((x0 + c * w + 6, y0 + r * h + 14), val, fontsize=10)
    page2 = doc.new_page()
    page2.insert_text((72, 72), "Outlook", fontsize=18)
    page2.insert_text((72, 100), "We expect continued growth next year.", fontsize=10)
    out = tmp_path / "report.pdf"
    doc.save(out)
    doc.close()
    return out
