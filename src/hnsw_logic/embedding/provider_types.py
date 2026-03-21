from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class CandidateProposal:
    doc_id: str
    reason: str
    query: str
    score_hint: float


@dataclass(slots=True)
class JudgeResult:
    accepted: bool
    relation_type: str
    confidence: float
    evidence_spans: list[str]
    rationale: str
    support_score: float = 0.0
    contradiction_flags: list[str] | None = None
    decision_reason: str = ""
    semantic_relation_label: str = ""
    canonical_relation: str = ""
    utility_score: float = 0.0
    uncertainty: float = 0.0


@dataclass(slots=True)
class JudgeSignals:
    dense_score: float
    sparse_score: float
    overlap_score: float
    content_overlap_score: float
    mention_score: float
    role_listing_score: float
    forward_reference_score: float
    reverse_reference_score: float
    direction_score: float
    local_support: float
    utility_score: float
    best_relation: str
    stage_pair: str
    risk_flags: list[str]
    relation_fit_scores: dict[str, float]
    topic_family_match: float = 0.0
    topic_cluster_match: float = 0.0
    stance_contrast: float = 0.0
    bridge_gain: float = 0.0
    duplicate_penalty: float = 0.0
    contrastive_bridge_score: float = 0.0
    topic_consistency: float = 0.0
    duplicate_risk: float = 0.0
    query_surface_match: float = 0.0
    uncertainty_hint: float = 0.0
    drift_risk: float = 0.0
