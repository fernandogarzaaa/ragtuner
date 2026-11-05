"""Chunkers. Every chunker maps a ParsedDocument to Chunks whose offsets point back into
the document's canonical text, so one span-anchored eval set scores every chunker."""

from ragtuner.chunk.base import Chunk, Chunker, available_chunkers, get_chunker, word_count
from ragtuner.chunk.fixed import FixedWordChunker
from ragtuner.chunk.recursive import RecursiveChunker
from ragtuner.chunk.structure import StructureChunker

__all__ = [
    "Chunk", "Chunker", "FixedWordChunker", "RecursiveChunker", "StructureChunker",
    "available_chunkers", "get_chunker", "word_count",
]
