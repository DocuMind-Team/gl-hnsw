from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class AnchorPlan:
    doc_id: str
    priority: float
    batch_id: str
    bridge_potential: float = 0.0
    coverage_pressure: float = 0.0
    reason: str = ""


@dataclass(slots=True)
class IndexingPlan:
    generated_at: str
    dataset_hint: str
    graph_potential: float
    anchors: list[AnchorPlan] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AnchorDossier:
    anchor_doc_id: str
    dataset_hint: str
    brief: dict[str, Any]
    full_doc: dict[str, Any]
    anchor_memory: dict[str, Any]
    semantic_memory: dict[str, Any]
    graph_stats: dict[str, Any]
    surrogate_query_terms: list[str] = field(default_factory=list)
    active_hypotheses: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CandidateBundleItem:
    candidate_doc_id: str
    proposal_reason: str
    query: str
    score_hint: float
    signals: dict[str, Any]
    candidate_brief: dict[str, Any]


@dataclass(slots=True)
class CandidateBundle:
    anchor_doc_id: str
    generated_at: str
    candidates: list[CandidateBundleItem] = field(default_factory=list)


@dataclass(slots=True)
class JudgmentBundleItem:
    candidate_doc_id: str
    verdict: dict[str, Any]
    signals: dict[str, Any]


@dataclass(slots=True)
class JudgmentBundle:
    anchor_doc_id: str
    generated_at: str
    judgments: list[JudgmentBundleItem] = field(default_factory=list)


@dataclass(slots=True)
class CounterevidenceBundleItem:
    candidate_doc_id: str
    keep: bool
    risk_flags: list[str] = field(default_factory=list)
    counterevidence: list[str] = field(default_factory=list)
    decision_reason: str = ""
    risk_penalty: float = 0.0


@dataclass(slots=True)
class CounterevidenceBundle:
    anchor_doc_id: str
    generated_at: str
    checks: list[CounterevidenceBundleItem] = field(default_factory=list)


@dataclass(slots=True)
class ReviewBundleItem:
    candidate_doc_id: str
    keep: bool
    reviewed_utility_score: float
    reviewed_confidence: float
    relation_type: str
    decision_reason: str
    final_verdict: dict[str, Any]
    risk_flags: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ReviewBundle:
    anchor_doc_id: str
    generated_at: str
    reviews: list[ReviewBundleItem] = field(default_factory=list)


@dataclass(slots=True)
class MemoryLearningBundle:
    anchor_doc_id: str
    generated_at: str
    learned_patterns: list[str] = field(default_factory=list)
    failure_patterns: list[str] = field(default_factory=list)
    reference_updates: dict[str, list[str]] = field(default_factory=dict)
