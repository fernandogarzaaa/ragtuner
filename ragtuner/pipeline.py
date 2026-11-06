"""Build a retrieval index for (corpus, config) and run queries against it.

Everything expensive is cached: parsed documents by (parser, file hash), embeddings by
(model, text hash). Models stay loaded in a ModelPool across configs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ragtuner.cache import ParseCache
from ragtuner.chunk import Chunk, get_chunker
from ragtuner.config import RagConfig
from ragtuner.embed import Embedder, EmbeddingCache, get_embedder
from ragtuner.index import BM25Index, DenseIndex, get_reranker, rrf
from ragtuner.index.rerank import CrossEncoderReranker
from ragtuner.schema import ParsedDocument


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    doc_id: str
    start: int
    end: int
    score: float
    rank: int
    text: str
    page: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"chunk_id": self.chunk_id, "doc_id": self.doc_id, "start": self.start, "end": self.end,
                "score": round(self.score, 6), "rank": self.rank, "page": self.page}


@dataclass
class BuildStats:
    docs: int = 0
    chunks: int = 0
    parse_seconds: float = 0.0
    parse_cache_hits: int = 0
    chunk_seconds: float = 0.0
    embed_seconds: float = 0.0
    embedded_texts: int = 0  # cache misses, i.e. what actually cost compute or money
    embed_cache_hits: int = 0
    index_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class ModelPool:
    """Keeps embedders and rerankers loaded across configs within one run."""

    def __init__(self) -> None:
        self._embedders: dict[str, Embedder] = {}
        self._rerankers: dict[str, CrossEncoderReranker] = {}

    def embedder(self, name: str) -> Embedder:
        if name not in self._embedders:
            self._embedders[name] = get_embedder(name)
        return self._embedders[name]

    def reranker(self, name: str) -> CrossEncoderReranker:
        if name not in self._rerankers:
            self._rerankers[name] = get_reranker(name)
        return self._rerankers[name]


@dataclass
class Corpus:
    name: str
    files: list[Path]
    doc_ids: dict[str, str] = field(default_factory=dict)  # file path -> doc id override
    acl: dict[str, Any] = field(default_factory=dict)  # doc id -> DocACL (see ragtuner.access)

    @classmethod
    def from_dir(cls, name: str, root: str | Path, suffixes: Iterable[str] = (".pdf", ".md", ".txt")) -> "Corpus":
        root = Path(root)
        files = sorted(p for p in root.rglob("*") if p.suffix.lower() in set(suffixes))
        return cls(name=name, files=files)

    def doc_id_for(self, path: Path) -> str | None:
        return self.doc_ids.get(str(path))


def parse_corpus(corpus: Corpus, parser: str, cache: ParseCache, stats: BuildStats | None = None) -> list[ParsedDocument]:
    t = time.time()
    docs = []
    hits = 0
    for f in corpus.files:
        doc, hit = cache.parse(f, parser, doc_id=corpus.doc_id_for(f))
        hits += int(hit)
        docs.append(doc)
    if stats is not None:
        stats.docs = len(docs)
        stats.parse_seconds += time.time() - t
        stats.parse_cache_hits += hits
    return docs


class RetrievalIndex:
    def __init__(self, config: RagConfig, chunks: list[Chunk], *, pool: ModelPool, emb_cache: EmbeddingCache | None, stats: BuildStats) -> None:
        config.validate()
        self.config = config
        self.chunks = chunks
        self.pool = pool
        self.emb_cache = emb_cache
        self.stats = stats
        self._bm25: BM25Index | None = None
        self._dense: DenseIndex | None = None
        self._better: Any = None
        self._embedder: Embedder | None = None
        t = time.time()
        texts = [c.index_text for c in chunks]
        if config.retrieval in ("sparse", "hybrid"):
            self._bm25 = BM25Index(texts)
        if config.retrieval in ("dense", "hybrid"):
            assert config.embedder
            self._embedder = pool.embedder(config.embedder)
            te = time.time()
            if emb_cache is not None:
                before = (emb_cache.hits, emb_cache.misses)
                vecs = emb_cache.embed(self._embedder, texts)
                stats.embed_cache_hits += emb_cache.hits - before[0]
                stats.embedded_texts += emb_cache.misses - before[1]
            else:
                vecs = self._embedder.embed(texts) if texts else np.zeros((0, self._embedder.dim), dtype=np.float32)
                stats.embedded_texts += len(texts)
            stats.embed_seconds += time.time() - te
            self._dense = DenseIndex(vecs)
        if config.retrieval == "better-rag":
            from better_rag import BetterRAG, Document
            from better_rag import RagConfig as BRConfig

            max_words = max((c.words for c in chunks), default=1) + 1
            engine = BetterRAG(BRConfig(chunk_size=max_words, chunk_overlap=0, top_k=config.candidates, mmr_lambda=1.0))
            engine.add_documents([Document(id=c.id, text=c.index_text) for c in chunks])
            self._better = engine
        self._reranker = pool.reranker(config.reranker) if config.reranker else None
        self._by_id = {c.id: i for i, c in enumerate(chunks)}
        stats.index_seconds += time.time() - t
        stats.chunks = len(chunks)

    def candidates(self, query: str, n: int, allowed: np.ndarray | None = None) -> list[tuple[int, float]]:
        """Top-n candidate chunk indices. `allowed` is a pre-filter mask applied inside each
        index before ranking, so access control never starves the result list."""
        mode = self.config.retrieval
        if mode == "sparse":
            assert self._bm25
            return self._bm25.search(query, n, allowed)
        if mode == "dense":
            assert self._dense and self._embedder
            return self._dense.search(self._query_vec(query), n, allowed)
        if mode == "hybrid":
            assert self._bm25 and self._dense
            s = [i for i, _ in self._bm25.search(query, n, allowed)]
            d = [i for i, _ in self._dense.search(self._query_vec(query), n, allowed)]
            return rrf([s, d], limit=n)
        if mode == "better-rag":
            assert self._better
            # BETTER-RAG has no native filter: rank everything, then filter before cutting to n.
            depth = len(self.chunks) if allowed is not None else n
            try:
                hits = self._better.search(query, top_k=max(depth, 1))
            except ValueError:
                return []
            ranked = [(self._by_id[h.document_id], h.score) for h in hits]
            if allowed is not None:
                ranked = [(i, sc) for i, sc in ranked if allowed[i]]
            return ranked[:n]
        raise ValueError(mode)

    def _query_vec(self, query: str) -> np.ndarray:
        assert self._embedder
        if self.emb_cache is not None:
            return self.emb_cache.embed(self._embedder, [query], kind="query")[0]
        return self._embedder.embed([query], kind="query")[0]

    def retrieve(self, query: str, *, k: int | None = None, candidates: int | None = None, access: Any = None) -> list[Hit]:
        """access: a ragtuner.access.Access, or None for unrestricted (evaluation) retrieval."""
        from ragtuner.access import allowed_mask

        k = k or self.config.top_k
        n = max(candidates or self.config.candidates, k)
        cands = self.candidates(query, n, allowed_mask(self.chunks, access))
        if self._reranker and cands:
            scores = self._reranker.rerank(query, [self.chunks[i].index_text for i, _ in cands])
            cands = sorted(zip((i for i, _ in cands), scores), key=lambda x: -x[1])
        out = []
        for rank, (i, score) in enumerate(cands[:k], start=1):
            c = self.chunks[i]
            out.append(Hit(chunk_id=c.id, doc_id=c.doc_id, start=c.start, end=c.end, score=float(score), rank=rank, text=c.text, page=c.page))
        return out


def chunk_docs(docs: list[ParsedDocument], config: RagConfig, stats: BuildStats | None = None, acl: dict[str, Any] | None = None) -> list[Chunk]:
    t = time.time()
    chunker = get_chunker(config.chunker, **config.chunk_params)
    out: list[Chunk] = []
    for d in docs:
        out.extend(chunker.chunk(d))
    if acl:
        from ragtuner.access import apply_acl

        out = apply_acl(out, acl)
    if stats is not None:
        stats.chunk_seconds += time.time() - t
    return out


def build_index(corpus: Corpus, config: RagConfig, *, cache_root: str | Path = ".ragtuner-cache", pool: ModelPool | None = None,
                docs: list[ParsedDocument] | None = None) -> tuple[RetrievalIndex, BuildStats]:
    stats = BuildStats()
    pool = pool or ModelPool()
    parse_cache = ParseCache(cache_root)
    emb_cache = EmbeddingCache(cache_root)
    if docs is None:
        docs = parse_corpus(corpus, config.parser, parse_cache, stats)
    else:
        stats.docs = len(docs)
    chunks = chunk_docs(docs, config, stats, acl=corpus.acl)
    index = RetrievalIndex(config, chunks, pool=pool, emb_cache=emb_cache, stats=stats)
    return index, stats
