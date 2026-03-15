from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class DocRecord:
    doc_id: str
    title: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DocBrief:
    doc_id: str
    title: str
    summary: str
    entities: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    claims: list[str] = field(default_factory=list)
    relation_hints: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LogicEdge:
    src_doc_id: str
    dst_doc_id: str
    relation_type: str
    confidence: float
    evidence_spans: list[str]
    discovery_path: list[str]
    edge_card_text: str
    created_at: str
    last_validated_at: str | None = None
    utility_score: float = 0.0


@dataclass(slots=True)
class AnchorMemory:
    anchor_doc_id: str
    explored_docs: list[str] = field(default_factory=list)
    rejected_docs: list[str] = field(default_factory=list)
    accepted_edge_ids: list[str] = field(default_factory=list)
    active_hypotheses: list[str] = field(default_factory=list)
    successful_queries: list[str] = field(default_factory=list)
    failed_queries: list[str] = field(default_factory=list)
    rejection_reasons: dict[str, str] = field(default_factory=dict)
    top_candidate_scores: dict[str, float] = field(default_factory=dict)
    accepted_edge_scores: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class GlobalSemanticMemory:
    canonical_entities: dict[str, str] = field(default_factory=dict)
    aliases: dict[str, list[str]] = field(default_factory=dict)
    relation_patterns: dict[str, list[str]] = field(default_factory=dict)
    rejection_patterns: dict[str, list[str]] = field(default_factory=dict)


@dataclass(slots=True)
class SearchHit:
    doc_id: str
    title: str
    final_score: float
    geometric_score: float
    logical_score: float
    source_kind: str
    via_edge: str | None
    summary: str
    rank: int = 0


@dataclass(slots=True)
class SearchResponse:
    query: str
    hits: list[SearchHit]


@dataclass(slots=True)
class JobStatus:
    job_id: str
    job_type: str
    state: str
    payload: dict[str, Any]
    message: str
    created_at: str
    updated_at: str
