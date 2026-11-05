"""LLM question generator (Claude). Block-level, span-anchored, with a paraphrase pass
that lowers lexical overlap with the evidence, plus multi-hop and unanswerable questions.

Every generated question is validated: the evidence quote must be located verbatim (or
whitespace-normalized) inside the source block, or the question is dropped. The model
never sees chunk boundaries, only parsed blocks, so no chunker is favored.
"""

from __future__ import annotations

import json
import random
import re
from typing import Any, Callable

from ragtuner.evalgen.common import (
    context_window,
    heading_path,
    lexical_overlap,
    locate,
    make_id,
    sample_blocks,
)
from ragtuner.evalset import EvalQuestion, EvalSet, GoldSpan
from ragtuner.schema import Block, BlockType, ParsedDocument

MODEL = "claude-opus-5-5"

_SYSTEM = (
    "You write evaluation questions for a document retrieval system. You are given one passage "
    "from a longer document, with its heading path. Write a question that a real user who has not "
    "read the passage might ask, and that this passage answers. Rules: the question must be "
    "self-contained (name the entity, period, or table it is about; never say 'this document' or "
    "'the table'); prefer wording a user would type, not the passage's own words; the evidence "
    "must be a verbatim quote copied exactly from the passage, 1 to 3 sentences or 1 to 4 table "
    "rows, that fully answers the question. Reply with JSON only: "
    '{"question": str, "answer": str, "evidence": str, "question_type": "factual"|"numeric"|"procedural"|"definition"}'
)

_PARAPHRASE = (
    "Rewrite the question so it keeps exactly the same meaning and answer but shares as few words "
    "as possible with the evidence passage. Use synonyms and a different sentence shape. Reply with "
    'JSON only: {"question": str}'
)

_MULTIHOP = (
    "You are given two passages from the same document. Write one question that can only be "
    "answered by combining information from both passages, and give one verbatim evidence quote "
    "from each passage. Reply with JSON only: "
    '{"question": str, "answer": str, "evidence_1": str, "evidence_2": str}'
)

_UNANSWERABLE = (
    "You are given a passage and its heading path. Write a question on the same topic that sounds "
    "natural but that this passage does NOT answer and that the surrounding document is unlikely to "
    'answer (asks for a detail that is not there). Reply with JSON only: {"question": str}'
)

Caller = Callable[[str, str], str]  # (system, user) -> text


def default_caller(model: str = MODEL, effort: str = "medium") -> Caller:
    import anthropic

    client = anthropic.Anthropic()

    def call(system: str, user: str) -> str:
        resp = client.beta.messages.create(
            model=model,
            max_tokens=2000,
            system=system,
            output_config={"effort": effort},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": user}],
        )
        if resp.stop_reason == "refusal":
            raise RuntimeError("model refused")
        return "".join(b.text for b in resp.content if b.type == "text")

    return call


def _json(text: str) -> dict[str, Any]:
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError("no JSON in model reply")
    return json.loads(m.group(0))


class LLMGenerator:
    name = "llm"

    def __init__(self, caller: Caller | None = None, *, seed: int = 0, table_fraction: float = 0.25,
                 multihop_fraction: float = 0.1, unanswerable_fraction: float = 0.1, max_overlap: float = 0.5,
                 paraphrase_rounds: int = 2) -> None:
        self.caller = caller or default_caller()
        self.seed = seed
        self.table_fraction = table_fraction
        self.multihop_fraction = multihop_fraction
        self.unanswerable_fraction = unanswerable_fraction
        self.max_overlap = max_overlap
        self.paraphrase_rounds = paraphrase_rounds
        self.calls = 0
        self.dropped: list[dict[str, Any]] = []

    def _call(self, system: str, user: str) -> dict[str, Any]:
        self.calls += 1
        return _json(self.caller(system, user))

    def generate(self, docs: list[ParsedDocument], n: int, *, name: str = "synthetic") -> EvalSet:
        rng = random.Random(self.seed)
        n_un = int(round(n * self.unanswerable_fraction))
        n_mh = int(round(n * self.multihop_fraction))
        n_single = n - n_un - n_mh
        questions: list[EvalQuestion] = []
        blocks = sample_blocks(docs, n_single + n_un, seed=self.seed, table_fraction=self.table_fraction)
        for doc, block in blocks[:n_single]:
            q = self._single(doc, block)
            if q:
                questions.append(q)
        for doc, block in blocks[n_single : n_single + n_un]:
            q = self._unanswerable(doc, block)
            if q:
                questions.append(q)
        for doc, b1, b2 in _pairs(docs, n_mh, rng):
            q = self._multihop(doc, b1, b2)
            if q:
                questions.append(q)
        return EvalSet(name, questions, {"generator": self.name, "model": MODEL, "seed": self.seed, "calls": self.calls,
                                         "dropped": len(self.dropped)})

    def _user(self, doc: ParsedDocument, block: Block) -> str:
        path = " > ".join(heading_path(doc, block)) or "(no heading)"
        kind = "table" if block.type is BlockType.TABLE else "passage"
        return f"Document: {doc.id}\nHeading path: {path}\nKind: {kind}\n\n<passage>\n{block.text[:6000]}\n</passage>"

    def _single(self, doc: ParsedDocument, block: Block) -> EvalQuestion | None:
        try:
            out = self._call(_SYSTEM, self._user(doc, block))
            question, answer, evidence = str(out["question"]).strip(), str(out.get("answer", "")).strip(), str(out["evidence"])
        except Exception as e:  # noqa: BLE001
            self.dropped.append({"block": block.id, "reason": f"generate: {e}"})
            return None
        loc = locate(doc, evidence, within=(block.start, block.end), min_ratio=0.9)
        if loc is None:
            self.dropped.append({"block": block.id, "reason": "evidence not found in block"})
            return None
        start, end, quality = loc
        span_text = doc.text[start:end]
        overlap = lexical_overlap(question, span_text)
        rounds = 0
        while overlap > self.max_overlap and rounds < self.paraphrase_rounds:
            rounds += 1
            try:
                new_q = str(self._call(_PARAPHRASE, f"Question: {question}\n\nEvidence passage:\n{span_text}")["question"]).strip()
            except Exception:  # noqa: BLE001
                break
            new_overlap = lexical_overlap(new_q, span_text)
            if new_overlap < overlap and new_q.endswith("?"):
                question, overlap = new_q, new_overlap
        span = GoldSpan(doc.id, start, end, page=block.page, block_type=block.type.value, block_id=block.id)
        qtype = str(out.get("question_type", "factual"))
        if block.type is BlockType.TABLE and qtype == "factual":
            qtype = "numeric" if any(ch.isdigit() for ch in answer) else "factual"
        return EvalQuestion(id=make_id("llm", doc.id, str(start), question), question=question, spans=(span,), answer=answer,
                            source="synthetic-llm", question_type=qtype, lexical_overlap=round(overlap, 3),
                            gold_quality="strong" if quality >= 0.99 else "weak",
                            metadata={"generator": self.name, "paraphrase_rounds": rounds, "heading_path": heading_path(doc, block)})

    def _unanswerable(self, doc: ParsedDocument, block: Block) -> EvalQuestion | None:
        try:
            question = str(self._call(_UNANSWERABLE, self._user(doc, block))["question"]).strip()
        except Exception as e:  # noqa: BLE001
            self.dropped.append({"block": block.id, "reason": f"unanswerable: {e}"})
            return None
        return EvalQuestion(id=make_id("llm-un", doc.id, question), question=question, spans=(), source="synthetic-llm",
                            question_type="unanswerable", metadata={"generator": self.name, "near_block": block.id})

    def _multihop(self, doc: ParsedDocument, b1: Block, b2: Block) -> EvalQuestion | None:
        user = (f"Document: {doc.id}\n\n<passage_1 heading='{' > '.join(heading_path(doc, b1))}'>\n{context_window(doc, b1, before=0, after=0)}\n</passage_1>\n\n"
                f"<passage_2 heading='{' > '.join(heading_path(doc, b2))}'>\n{context_window(doc, b2, before=0, after=0)}\n</passage_2>")
        try:
            out = self._call(_MULTIHOP, user)
            question = str(out["question"]).strip()
            ev = [str(out["evidence_1"]), str(out["evidence_2"])]
        except Exception as e:  # noqa: BLE001
            self.dropped.append({"block": b1.id, "reason": f"multihop: {e}"})
            return None
        spans = []
        for b, e in zip((b1, b2), ev):
            loc = locate(doc, e, within=(b.start, b.end), min_ratio=0.9)
            if loc is None:
                self.dropped.append({"block": b.id, "reason": "multihop evidence not found"})
                return None
            spans.append(GoldSpan(doc.id, loc[0], loc[1], page=b.page, block_type=b.type.value, block_id=b.id))
        overlap = max(lexical_overlap(question, doc.text[s.start : s.end]) for s in spans)
        return EvalQuestion(id=make_id("llm-mh", doc.id, question), question=question, spans=tuple(spans), answer=str(out.get("answer", "")),
                            source="synthetic-llm", question_type="multi-hop", lexical_overlap=round(overlap, 3), metadata={"generator": self.name})


def _pairs(docs: list[ParsedDocument], n: int, rng: random.Random) -> list[tuple[ParsedDocument, Block, Block]]:
    from ragtuner.evalgen.common import eligible_blocks

    out = []
    cands = [(d, eligible_blocks(d)) for d in docs]
    cands = [(d, bs) for d, bs in cands if len(bs) >= 2]
    for _ in range(n):
        if not cands:
            break
        d, bs = rng.choice(cands)
        b1, b2 = rng.sample(bs, 2)
        out.append((d, b1, b2))
    return out
