from __future__ import annotations

import importlib
from pathlib import Path
from typing import Callable, Protocol, runtime_checkable

from ragtuner.schema import ParsedDocument, hash_file


@runtime_checkable
class Parser(Protocol):
    name: str
    version: str
    suffixes: frozenset[str]

    def parse(self, path: Path, *, doc_id: str | None = None) -> ParsedDocument: ...


# name -> (module path, class name). Imported lazily so optional heavy deps stay optional.
_REGISTRY: dict[str, tuple[str, str]] = {
    "plaintext": ("ragtuner.parse.plaintext", "PlainTextParser"),
    "pymupdf-text": ("ragtuner.parse.pymupdf_text", "PyMuPDFTextParser"),
    "docling": ("ragtuner.parse.docling_backend", "DoclingParser"),
}


def register(name: str, module: str, cls: str) -> None:
    _REGISTRY[name] = (module, cls)


def get_parser(name: str) -> Parser:
    if name not in _REGISTRY:
        raise KeyError(f"unknown parser {name!r}; known: {sorted(_REGISTRY)}")
    module, cls = _REGISTRY[name]
    factory: Callable[[], Parser] = getattr(importlib.import_module(module), cls)
    return factory()


def available_parsers() -> dict[str, str]:
    """Map parser name -> 'ok' or the import error that makes it unavailable."""
    status: dict[str, str] = {}
    for name in sorted(_REGISTRY):
        try:
            get_parser(name)
            status[name] = "ok"
        except Exception as e:  # noqa: BLE001 - we want to report any failure
            status[name] = f"unavailable: {type(e).__name__}: {e}"
    return status


def default_parser_for(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".markdown"}:
        return "plaintext"
    if suffix == ".pdf":
        return "pymupdf-text"
    raise ValueError(f"no default parser for {suffix!r}")


def parse_file(path: str | Path, parser: str | None = None, *, doc_id: str | None = None) -> ParsedDocument:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    name = parser or default_parser_for(p)
    backend = get_parser(name)
    if p.suffix.lower() not in backend.suffixes:
        raise ValueError(f"parser {name!r} does not handle {p.suffix!r} (handles {sorted(backend.suffixes)})")
    return backend.parse(p, doc_id=doc_id)


def doc_id_for(path: Path) -> str:
    """Stable default document id: stem plus a short content hash, so renames and
    re-uploads of the same bytes resolve to the same id."""
    return f"{path.stem}-{hash_file(path)[:8]}"
