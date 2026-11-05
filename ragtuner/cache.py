"""Content-addressed on-disk cache for parsed documents.

Key = sha256(parser name | parser version | file content hash). Same bytes through the
same parser version never parse twice, across configs and across runs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ragtuner.parse.base import get_parser
from ragtuner.schema import ParsedDocument, hash_file


class ParseCache:
    def __init__(self, root: str | Path = ".ragtuner-cache") -> None:
        self.root = Path(root) / "parsed"
        self.root.mkdir(parents=True, exist_ok=True)

    def key(self, parser: str, version: str, content_hash: str) -> str:
        return hashlib.sha256(f"{parser}|{version}|{content_hash}".encode()).hexdigest()

    def path_for(self, parser: str, version: str, content_hash: str) -> Path:
        return self.root / f"{self.key(parser, version, content_hash)}.json"

    def get(self, parser: str, version: str, content_hash: str) -> ParsedDocument | None:
        p = self.path_for(parser, version, content_hash)
        if not p.exists():
            return None
        return ParsedDocument.from_dict(json.loads(p.read_text(encoding="utf-8")))

    def put(self, doc: ParsedDocument) -> Path:
        p = self.path_for(doc.parser, doc.parser_version, doc.content_hash)
        doc.save(p)
        return p

    def parse(self, path: str | Path, parser: str, *, doc_id: str | None = None) -> tuple[ParsedDocument, bool]:
        """Return (document, was_cache_hit)."""
        p = Path(path)
        backend = get_parser(parser)
        content_hash = hash_file(p)
        cached = self.get(backend.name, backend.version, content_hash)
        if cached is not None:
            return cached, True
        doc = backend.parse(p, doc_id=doc_id)
        self.put(doc)
        return doc, False
