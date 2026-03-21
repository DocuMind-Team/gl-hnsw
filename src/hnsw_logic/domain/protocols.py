"""Lightweight structural protocols shared across runtime and tests.

These interfaces decouple service constructors from concrete storage and
provider implementations. The production code still wires concrete classes,
but tests and alternate runtimes can satisfy the same contracts without
subclassing internal implementations.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol

import numpy as np

from hnsw_logic.domain.models import (
    DocBrief,
    DocRecord,
    GlobalSemanticMemory,
    LogicEdge,
)


class EmbeddingProvider(Protocol):
    """Provider capability required by encoder and retrieval scoring."""

    def embed_texts(self, texts: Iterable[str]) -> np.ndarray: ...


class SearchNeighbor(Protocol):
    """Minimal HNSW neighbor surface required by retrieval."""

    doc_id: str
    score: float
    rank: int


class Searcher(Protocol):
    """Nearest-neighbor search contract used by retrieval."""

    def search(self, query_vector: np.ndarray, top_k: int) -> Sequence[SearchNeighbor]: ...


class BriefStore(Protocol):
    """Read model for persisted document briefs."""

    def all(self) -> list[DocBrief]: ...

    def read(self, doc_id: str) -> DocBrief | None: ...


class GraphStore(Protocol):
    """Read contract for persisted logic edges."""

    def has_edges(self) -> bool: ...

    def get_out_edges(self, doc_id: str) -> list[LogicEdge]: ...

    def all_edges(self) -> list[LogicEdge]: ...


class CorpusStore(Protocol):
    """Read contract for normalized corpus records."""

    def read_processed(self) -> list[DocRecord]: ...


class SemanticMemoryStore(Protocol):
    """Read contract for semantic memory used at query time."""

    def read(self) -> GlobalSemanticMemory: ...
