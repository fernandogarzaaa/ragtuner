from __future__ import annotations

import difflib
import hashlib
import random
import re

from ragtuner.index.bm25 import tokenize
from ragtuner.schema import Block, BlockType, ParsedDocument

_WS = re.compile(r"\s+")


def _norm_map(text: str) -> tuple[str, list[int]]:
    """Lowercase, collapse whitespace; return normalized text and a map norm_idx -> orig_idx."""
    out: list[str] = []
    idx: list[int] = []
    prev_space = True
    for i, ch in enumerate(text):
        if ch.isspace():
            if not prev_space:
                out.append(" ")
                idx.append(i)
            prev_space = True
        else:
            out.append(ch.lower())
            idx.append(i)
            prev_space = False
    return "".join(out), idx


def locate(doc: ParsedDocument, snippet: str, *, within: tuple[int, int] | None = None, min_ratio: float = 0.6) -> tuple[int, int, float] | None:
    """Find snippet in doc.text. Returns (start, end, quality) where quality is 1.0 for an
    exact or whitespace-normalized match and the difflib ratio for a fuzzy one.
    Fuzzy matching is restricted to a window around the best anchor so it stays cheap."""
    snippet = snippet.strip()
    if not snippet:
        return None
    lo, hi = within or (0, len(doc.text))
    hay = doc.text[lo:hi]
    pos = hay.find(snippet)
    if pos >= 0:
        return lo + pos, lo + pos + len(snippet), 1.0
    nh, hmap = _norm_map(hay)
    ns, _ = _norm_map(snippet)
    pos = nh.find(ns)
    if pos >= 0 and ns:
        return lo + hmap[pos], lo + hmap[pos + len(ns) - 1] + 1, 1.0
    # fuzzy: anchor on the longest matching block, then expand to snippet length
    if len(ns) < 20:
        return None
    sm = difflib.SequenceMatcher(None, nh, ns, autojunk=False)
    m = sm.find_longest_match(0, len(nh), 0, len(ns))
    if m.size < max(15, int(0.25 * len(ns))):
        return None
    start_n = max(0, m.a - m.b)
    end_n = min(len(nh), start_n + len(ns))
    ratio = difflib.SequenceMatcher(None, nh[start_n:end_n], ns, autojunk=False).ratio()
    if ratio < min_ratio:
        return None
    return lo + hmap[start_n], lo + hmap[end_n - 1] + 1, ratio


def lexical_overlap(question: str, span_text: str) -> float:
    q = set(tokenize(question))
    if not q:
        return 0.0
    s = set(tokenize(span_text))
    return len(q & s) / len(q)


def eligible_blocks(doc: ParsedDocument, *, min_words: int = 25) -> list[Block]:
    out = []
    for b in doc.blocks:
        if b.type is BlockType.TABLE:
            if b.table is not None and len(b.table.rows) >= 2 and len(b.table.cells()) >= 4:
                out.append(b)
        elif b.type in (BlockType.PARAGRAPH, BlockType.LIST) and len(b.text.split()) >= min_words:
            out.append(b)
    return out


def sample_blocks(docs: list[ParsedDocument], n: int, *, seed: int = 0, table_fraction: float = 0.25) -> list[tuple[ParsedDocument, Block]]:
    """Stratified sample: at least table_fraction tables when the corpus has them, the rest
    spread across documents proportionally to their eligible block counts."""
    rng = random.Random(seed)
    tables: list[tuple[ParsedDocument, Block]] = []
    prose: list[tuple[ParsedDocument, Block]] = []
    for d in docs:
        for b in eligible_blocks(d):
            (tables if b.type is BlockType.TABLE else prose).append((d, b))
    n_tables = min(len(tables), int(round(n * table_fraction))) if tables else 0
    chosen = rng.sample(tables, n_tables) + rng.sample(prose, min(len(prose), n - n_tables))
    rng.shuffle(chosen)
    return chosen


def heading_path(doc: ParsedDocument, block: Block) -> list[str]:
    path: list[tuple[int, str]] = []
    for b in doc.blocks:
        if b.start >= block.start:
            break
        if b.type is BlockType.HEADING:
            level = b.level or 1
            path = [(lv, t) for lv, t in path if lv < level] + [(level, b.text)]
    return [t for _, t in path]


def context_window(doc: ParsedDocument, block: Block, *, before: int = 1, after: int = 1, max_chars: int = 3000) -> str:
    i = next(k for k, b in enumerate(doc.blocks) if b.id == block.id)
    parts = [b.text for b in doc.blocks[max(0, i - before) : i + after + 1]]
    text = "\n\n".join(parts)
    return text[:max_chars]


def make_id(prefix: str, *parts: str) -> str:
    h = hashlib.sha1("|".join(parts).encode()).hexdigest()[:10]
    return f"{prefix}-{h}"


def sentences(text: str) -> list[tuple[int, int]]:
    """(start, end) of sentences inside text, relative offsets."""
    out = []
    pos = 0
    for m in re.finditer(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])", text):
        out.append((pos, m.start()))
        pos = m.end()
    out.append((pos, len(text)))
    return [(s, e) for s, e in out if text[s:e].strip()]
