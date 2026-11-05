from pathlib import Path

from ragtuner.parse import parse_file
from ragtuner.schema import BlockType

MD = """# Refund policy

Customers may return items within 30 days.
A receipt is required.

## Fees

| Item | Fee |
|---|---|
| Restocking | $5 |
| Shipping | $12 |

- Fees are waived for defects
- Gift cards are final sale

Closing paragraph.
"""


def test_markdown_blocks_and_offsets(tmp_path: Path):
    p = tmp_path / "policy.md"
    p.write_text(MD, encoding="utf-8")
    doc = parse_file(p)
    types = [b.type for b in doc.blocks]
    assert types == [
        BlockType.HEADING, BlockType.PARAGRAPH, BlockType.HEADING, BlockType.TABLE,
        BlockType.LIST, BlockType.PARAGRAPH,
    ]
    assert doc.blocks[0].level == 1 and doc.blocks[2].level == 2
    assert doc.blocks[1].text == "Customers may return items within 30 days. A receipt is required."
    table = doc.blocks[3].table
    assert table is not None
    assert table.rows == [["Item", "Fee"], ["Restocking", "$5"], ["Shipping", "$12"]]
    assert table.header_rows == 1
    doc.validate()
    assert doc.parser == "plaintext"
    assert doc.id.startswith("policy-")


def test_default_parser_is_chosen_by_suffix(tmp_path: Path):
    p = tmp_path / "a.txt"
    p.write_text("one\n\ntwo", encoding="utf-8")
    doc = parse_file(p)
    assert [b.text for b in doc.blocks] == ["one", "two"]
