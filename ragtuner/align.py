"""Project gold spans from the eval set's anchor parse onto another parser's text.

Gold spans are character offsets into ONE parse of each document (the anchor parser the
eval set was generated from). A config that uses a different parser produces chunks
with offsets into a different text, so spans must be re-anchored before scoring, or
the comparison silently measures nothing.

Projection aligns alphanumeric token sequences, which ignores formatting differences
such as markdown table pipes versus plain reading-order text:
1. exact contiguous token match;
2. otherwise the best window seeded at the span's rarest tokens, scored by multiset
   token overlap, accepted at >= MIN_OVERLAP of the span's tokens, edges trimmed.
A span that cannot be projected is marked lost (zero length): the parser dropped or
garbled that text, which is a real parse-stage failure for that parser and is scored
and diagnosed as one.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import replace

from ragtuner.evalset import EvalQuestion, GoldSpan
from ragtuner.schema import ParsedDocument

_TOK = re.compile(r"[A-Za-z0-9]+")
MIN_OVERLAP = 0.6


def _toks(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(0).lower(), m.start(), m.end()) for m in _TOK.finditer(text)]


class Projector:
    def __init__(self, dst: ParsedDocument) -> None:
        self.doc = dst
        self.toks = _toks(dst.text)
        self.words = [t for t, _, _ in self.toks]
        self.pos: dict[str, list[int]] = defaultdict(list)
        for i, w in enumerate(self.words):
            self.pos[w].append(i)

    def project(self, span_text: str, *, min_overlap: float = MIN_OVERLAP) -> tuple[int, int, float] | None:
        src = [t for t, _, _ in _toks(span_text)]
        m = len(src)
        if m == 0 or not self.words:
            return None
        for p in self.pos.get(src[0], ()):
            if self.words[p : p + m] == src:
                return self.toks[p][1], self.toks[p + m - 1][2], 1.0
        need = Counter(src)
        present = [t for t in need if self.pos.get(t)]
        if not present:
            return None
        seeds = sorted(present, key=lambda t: (len(self.pos[t]), t))[:4]
        width = int(m * 1.25) + 1
        best: tuple[float, int, int] | None = None
        for t in seeds:
            offsets_in_src = [j for j, x in enumerate(src) if x == t][:5]
            for p in self.pos[t][:60]:
                for j in offsets_in_src:
                    s = max(0, p - j)
                    e = min(len(self.words), s + width)
                    window = Counter(self.words[s:e])
                    ov = sum(min(c, window[w]) for w, c in need.items()) / m
                    if best is None or ov > best[0]:
                        best = (ov, s, e)
        if best is None or best[0] < min_overlap:
            return None
        ov, s, e = best
        while s < e and self.words[s] not in need:
            s += 1
        while e > s and self.words[e - 1] not in need:
            e -= 1
        if s >= e:
            return None
        return self.toks[s][1], self.toks[e - 1][2], ov


def lost(span: GoldSpan) -> GoldSpan:
    return GoldSpan(span.doc_id, 0, 0, page=span.page, block_type=span.block_type, block_id=None)


def project_questions(questions: list[EvalQuestion], anchor: dict[str, ParsedDocument], target: dict[str, ParsedDocument]) -> tuple[list[EvalQuestion], dict[str, int]]:
    """Return questions with spans re-anchored onto `target` docs, plus counts."""
    projectors: dict[str, Projector] = {}
    stats = {"spans": 0, "exact": 0, "fuzzy": 0, "lost": 0}
    out = []
    for q in questions:
        new_spans = []
        for sp in q.spans:
            stats["spans"] += 1
            src_doc, dst_doc = anchor.get(sp.doc_id), target.get(sp.doc_id)
            if src_doc is None or dst_doc is None:
                stats["lost"] += 1
                new_spans.append(lost(sp))
                continue
            if sp.doc_id not in projectors:
                projectors[sp.doc_id] = Projector(dst_doc)
            res = projectors[sp.doc_id].project(src_doc.text[sp.start : sp.end])
            if res is None:
                stats["lost"] += 1
                new_spans.append(lost(sp))
            else:
                s, e, quality = res
                stats["exact" if quality >= 1.0 else "fuzzy"] += 1
                new_spans.append(GoldSpan(sp.doc_id, s, e, page=sp.page, block_type=sp.block_type, block_id=None))
        out.append(replace(q, spans=tuple(new_spans)))
    return out, stats
