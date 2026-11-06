"""The serving runtime: one exported config over one corpus, shared by the HTTP API, the
MCP server and `ragtuner sync`.

It uses the same build path as the tuner (parse cache, chunkers, embedding cache, index),
so what serves is exactly what was evaluated. reload() is incremental: unchanged files hit
the parse cache and unchanged chunk texts hit the embedding cache, so only new or edited
content costs parsing or embedding. Every query can be appended to a JSONL log that
`ragtuner monitor` reads.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

from ragtuner.access import Access
from ragtuner.config import RagConfig
from ragtuner.index.bm25 import tokenize
from ragtuner.pipeline import ModelPool, RetrievalIndex, build_index
from ragtuner.schema import hash_file

_SENT = re.compile(r"(?<=[.!?])\s+")


def corpus_state(files: list[Path]) -> dict[str, str]:
    return {str(f): hash_file(f) for f in files}


def diff_state(old: dict[str, str], new: dict[str, str]) -> dict[str, list[str]]:
    return {
        "added": sorted(set(new) - set(old)),
        "removed": sorted(set(old) - set(new)),
        "changed": sorted(f for f in set(old) & set(new) if old[f] != new[f]),
    }


class Runtime:
    def __init__(self, corpus_dir: str | Path, config: RagConfig | str | Path, *, cache_root: str | Path = ".ragtuner-cache",
                 query_log: str | Path | None = None, name: str | None = None, previous_state: dict[str, str] | None = None) -> None:
        from ragtuner.run import load_corpus

        self.corpus_dir = Path(corpus_dir)
        self.name = name or self.corpus_dir.name
        self.config = config if isinstance(config, RagConfig) else RagConfig.from_yaml(config)
        self.cache_root = Path(cache_root)
        self.query_log = Path(query_log) if query_log else None
        self.pool = ModelPool()
        self._load_corpus = load_corpus
        self._lock = threading.RLock()
        self.index: RetrievalIndex | None = None
        self.state: dict[str, str] = dict(previous_state or {})
        self.last_reload: dict[str, Any] = {}
        self.reload()

    @property
    def has_acl(self) -> bool:
        return bool(self.corpus.acl)

    def reload(self) -> dict[str, Any]:
        """(Re)build from the corpus directory. Returns what changed and what it cost."""
        t = time.time()
        corpus = self._load_corpus(self.name, self.corpus_dir)
        new_state = corpus_state(corpus.files)
        diff = diff_state(self.state, new_state)
        index, stats = build_index(corpus, self.config, cache_root=self.cache_root, pool=self.pool)
        with self._lock:
            self.corpus, self.index, self.state = corpus, index, new_state
        self.last_reload = {
            "docs": stats.docs, "chunks": stats.chunks, "parsed": stats.docs - stats.parse_cache_hits,
            "parse_cache_hits": stats.parse_cache_hits, "embedded_texts": stats.embedded_texts,
            "embed_cache_hits": stats.embed_cache_hits, "seconds": round(time.time() - t, 2), **diff,
        }
        return self.last_reload

    def access_for(self, tenant: str | None, principal: str | None) -> Access | None:
        # Fail closed: when the corpus has permissions, a request without identity is anonymous.
        if not self.has_acl:
            return None
        return Access(tenant=tenant, principal=principal)

    def search(self, query: str, *, k: int | None = None, tenant: str | None = None, principal: str | None = None) -> list[dict[str, Any]]:
        if not query or not query.strip():
            raise ValueError("query must not be empty")
        t = time.time()
        with self._lock:
            assert self.index is not None
            hits = self.index.retrieve(query, k=k, access=self.access_for(tenant, principal))
        out = [{**h.to_dict(), "text": h.text} for h in hits]
        self._log("search", query, out, time.time() - t, tenant, principal)
        return out

    def ask(self, query: str, *, tenant: str | None = None, principal: str | None = None) -> dict[str, Any]:
        """Extractive answer with citations by default; an LLM answer grounded in the same
        hits when RAGTUNER_LLM_ANSWERS=1 and LLM credentials are present."""
        t = time.time()
        with self._lock:
            assert self.index is not None
            hits = self.index.retrieve(query, access=self.access_for(tenant, principal))
        citations = [{"n": i + 1, "doc_id": h.doc_id, "page": h.page, "chunk_id": h.chunk_id, "text": h.text[:600]} for i, h in enumerate(hits)]
        if not hits:
            answer, mode = "Nothing in the index you can access answers that.", "none"
        elif os.environ.get("RAGTUNER_LLM_ANSWERS") == "1":
            answer, mode = self._llm_answer(query, hits), "llm"
        else:
            answer, mode = self._extractive(query, hits), "extractive"
        hit_dicts = [h.to_dict() for h in hits]
        self._log("ask", query, hit_dicts, time.time() - t, tenant, principal)
        return {"answer": answer, "mode": mode, "citations": citations}

    @staticmethod
    def _extractive(query: str, hits: list[Any], max_sentences: int = 3) -> str:
        q = set(tokenize(query))
        ranked = []
        for rank, h in enumerate(hits):
            for s in _SENT.split(h.text):
                s = s.strip()
                if len(s.split()) < 4:
                    continue
                overlap = len(q & set(tokenize(s))) / max(len(q), 1)
                ranked.append((overlap, -rank, s, rank + 1))
        ranked.sort(reverse=True)
        picked = [f"{s} [{n}]" for ov, _, s, n in ranked[:max_sentences] if ov > 0]
        return " ".join(picked) if picked else f"{hits[0].text[:400]} [1]"

    @staticmethod
    def _llm_answer(query: str, hits: list[Any]) -> str:
        from ragtuner.llm import default_caller

        ctx = "\n\n".join(f"[{i + 1}] {h.text}" for i, h in enumerate(hits))
        system = ("Answer only from the numbered context passages and cite them like [1]. If they do not contain the "
                  "answer, say so. Quote numbers exactly.")
        try:
            return default_caller(effort="low")(system, f"Question: {query}\n\nContext:\n{ctx}").strip()
        except Exception as e:  # noqa: BLE001
            return f"(LLM answer unavailable: {type(e).__name__}) " + Runtime._extractive(query, hits)

    def _log(self, kind: str, query: str, hits: list[dict[str, Any]], secs: float, tenant: str | None, principal: str | None) -> None:
        if not self.query_log:
            return
        rec = {"ts": time.time(), "kind": kind, "query": query, "n_hits": len(hits),
               "top_score": hits[0]["score"] if hits else None, "hits": [h["chunk_id"] for h in hits[:10]],
               "latency_ms": round(secs * 1000, 1), "tenant": tenant, "principal": principal, "config": self.config.key}
        self.query_log.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.query_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def manifest(self) -> dict[str, Any]:
        return {"name": self.name, "config": self.config.to_dict(), "config_key": self.config.key, "corpus_state": self.state,
                "last_reload": self.last_reload, "acl": self.has_acl}
