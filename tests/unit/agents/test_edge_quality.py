from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from hnsw_logic.agents.orchestration.orchestrator import LogicOrchestrator
from hnsw_logic.config.schema import ProviderConfig, RetrievalConfig
from hnsw_logic.domain.models import DocBrief
from hnsw_logic.domain.tokens import deterministic_vector
from hnsw_logic.embedding.providers.stub import StubProvider
from hnsw_logic.embedding.providers.types import JudgeResult


class FakeProvider(StubProvider):
    def __init__(self):
        super().__init__(ProviderConfig(kind="stub"))
        self.relation_priors = {
            "supporting_evidence": 1.0,
            "implementation_detail": 1.02,
            "same_concept": 0.9,
            "comparison": 0.92,
            "prerequisite": 0.98,
        }

    def embed_texts(self, texts):
        return np.vstack([deterministic_vector(text, self.embedding_dim) for text in texts]).astype(np.float32)


class FakeJudge:
    def __init__(self, provider, verdicts):
        self.provider = provider
        self._verdicts = verdicts

    def run_many(self, anchor, candidates):
        return {candidate.doc_id: self._verdicts[candidate.doc_id] for candidate in candidates}

    def run(self, anchor, candidate):
        return self._verdicts[candidate.doc_id]


class FakeReviewer:
    def __init__(self, provider, verdicts):
        self.provider = provider
        self._verdicts = verdicts

    def run_many_with_signals(self, anchor, candidates):
        return {candidate.doc_id: self._verdicts[candidate.doc_id] for candidate, _, _ in candidates}

    def run_with_signals(self, anchor, candidate, signals, verdict):
        return self._verdicts[candidate.doc_id]


def _brief(
    doc_id: str,
    title: str,
    summary: str,
    keywords: list[str],
    *,
    entities: list[str] | None = None,
    relation_hints: list[str] | None = None,
    metadata: dict | None = None,
) -> DocBrief:
    return DocBrief(
        doc_id=doc_id,
        title=title,
        summary=summary,
        keywords=keywords,
        entities=entities or [],
        relation_hints=relation_hints or [],
        claims=[summary],
        metadata=metadata or {},
    )


def _orchestrator(provider=None, judge_verdicts=None, reviewer_verdicts=None) -> LogicOrchestrator:
    provider = provider or FakeProvider()
    return LogicOrchestrator(
        doc_profiler=SimpleNamespace(provider=provider),
        corpus_scout=SimpleNamespace(provider=provider),
        relation_judge=FakeJudge(provider, judge_verdicts or {}),
        edge_reviewer=FakeReviewer(provider, reviewer_verdicts or {}) if reviewer_verdicts else None,
        memory_curator=SimpleNamespace(provider=provider),
        retrieval_config=RetrievalConfig(),
    )


def test_signal_bundle_exposes_skill_runtime_reports():
    provider = FakeProvider()
    orchestrator = _orchestrator(provider=provider)
    anchor = _brief(
        "arg-1",
        "Public transit should replace highway expansion",
        "The argument says transit investment is better than highway expansion for city mobility.",
        ["transit", "highway", "mobility", "policy"],
        relation_hints=["debate", "comparison"],
        metadata={"source_dataset": "arguana", "topic_cluster": "transit-mobility", "topic_family": "policy-transit", "stance": "pro"},
    )
    candidate = _brief(
        "arg-2",
        "Highway expansion is better than transit investment",
        "The argument says highway expansion is preferable to transit investment for city mobility.",
        ["transit", "highway", "mobility", "policy"],
        relation_hints=["debate", "comparison"],
        metadata={"source_dataset": "arguana", "topic_cluster": "transit-mobility", "topic_family": "policy-transit", "stance": "con"},
    )

    metrics = orchestrator._candidate_metrics(anchor, candidate)
    _, best_relation, fit_scores = orchestrator._pair_rerank(anchor, candidate, metrics)
    signals = orchestrator._signal_bundle(anchor, candidate, metrics, fit_scores, best_relation)

    assert signals.topic_consistency > 0.0
    assert signals.bridge_gain >= 0.0
    assert signals.query_surface_match >= 0.0
    assert signals.duplicate_risk >= 0.0
    assert signals.drift_risk >= 0.0


def test_same_topic_argumentative_contrast_accepts_comparison_edge():
    provider = type("OpenAICompatibleProvider", (FakeProvider,), {})()
    anchor = _brief(
        "arg-1",
        "Public transit should replace highway expansion",
        "Transit investment is better than highway expansion for city mobility.",
        ["transit", "highway", "mobility", "policy", "transit investment"],
        relation_hints=["debate", "comparison"],
        metadata={"source_dataset": "arguana", "topic_cluster": "transit-mobility", "topic_family": "policy-transit", "stance": "pro"},
    )
    candidate = _brief(
        "arg-2",
        "Highway expansion remains the best mobility strategy",
        "Highway expansion is preferable to transit investment for city mobility and congestion.",
        ["transit", "highway", "mobility", "policy", "highway expansion"],
        relation_hints=["debate", "comparison"],
        metadata={"source_dataset": "arguana", "topic_cluster": "transit-mobility", "topic_family": "policy-transit", "stance": "con"},
    )
    judge = JudgeResult(
        accepted=True,
        relation_type="comparison",
        confidence=0.9,
        evidence_spans=[
            "Transit investment is better than highway expansion for city mobility.",
            "Highway expansion is preferable to transit investment for city mobility and congestion.",
        ],
        rationale="The documents argue opposite positions on the same transport policy topic.",
        support_score=0.82,
        contradiction_flags=["contrastive_argument", "alternative_position"],
        decision_reason="Same-topic contrast bridge with durable retrieval value.",
        utility_score=0.86,
        uncertainty=0.12,
        canonical_relation="comparison",
        semantic_relation_label="contrastive_bridge",
    )
    review = JudgeResult(
        accepted=True,
        relation_type="comparison",
        confidence=0.92,
        evidence_spans=judge.evidence_spans,
        rationale="The contrast is reusable across many transportation-policy queries.",
        support_score=0.84,
        contradiction_flags=["contrastive_argument"],
        decision_reason="Reviewer confirms same-topic contrast bridge.",
        utility_score=0.9,
        uncertainty=0.08,
        canonical_relation="comparison",
        semantic_relation_label="contrastive_bridge",
    )
    orchestrator = _orchestrator(provider=provider)

    assessment = orchestrator._assessment_for(anchor, candidate, judge, review)

    assert assessment.accepted is True
    assert assessment.relation_type == "comparison"
    assert assessment.edge is not None
    assert assessment.edge.activation_profile["activation_prior"] > 0.0


def test_cross_topic_argumentative_pair_is_rejected():
    provider = type("OpenAICompatibleProvider", (FakeProvider,), {})()
    anchor = _brief(
        "arg-1",
        "Public transit should replace highway expansion",
        "Transit investment is better than highway expansion for city mobility.",
        ["transit", "highway", "mobility", "policy"],
        relation_hints=["debate", "comparison"],
        metadata={"source_dataset": "arguana", "topic_cluster": "transit-mobility", "topic_family": "policy-transit", "stance": "pro"},
    )
    candidate = _brief(
        "arg-2",
        "Public schools need stronger art funding",
        "Art funding should rise because cultural programs help students flourish.",
        ["schools", "art", "funding", "students", "culture"],
        relation_hints=["debate", "comparison"],
        metadata={"source_dataset": "arguana", "topic_cluster": "culture-funding", "topic_family": "culture-funding", "stance": "con"},
    )
    judge = JudgeResult(
        accepted=True,
        relation_type="comparison",
        confidence=0.88,
        evidence_spans=["Both documents are policy arguments."],
        rationale="The pair shares public-policy framing.",
        support_score=0.78,
        contradiction_flags=[],
        decision_reason="Policy overlap only.",
        utility_score=0.42,
        uncertainty=0.22,
        canonical_relation="comparison",
        semantic_relation_label="comparison",
    )
    orchestrator = _orchestrator(provider=provider)

    assessment = orchestrator._assessment_for(anchor, candidate, judge)

    assert assessment.accepted is False
    assert assessment.reject_reason in {
        "wrong_relation_type",
        "topic_drift",
        "low_utility",
        "weak_evidence",
        "graph_hygiene_failure",
        "low_confidence",
    }


def test_topic_drift_pair_is_rejected_even_with_positive_verdict():
    provider = type("OpenAICompatibleProvider", (FakeProvider,), {})()
    anchor = _brief(
        "doc-08",
        "Logic Overlay Graph",
        "The logic overlay graph adds one-hop logical expansion after HNSW recall.",
        ["logic", "overlay", "graph", "hnsw", "jump", "policy"],
        relation_hints=["one-hop", "expansion"],
        metadata={"topic": "logic"},
    )
    candidate = _brief(
        "doc-22",
        "SQLite Job Registry",
        "Workers persist job state in SQLite for build and admin tasks.",
        ["sqlite", "jobs", "workers", "build", "admin"],
        relation_hints=["persist", "registry"],
        metadata={"topic": "ops"},
    )
    judge = JudgeResult(
        accepted=True,
        relation_type="implementation_detail",
        confidence=0.95,
        evidence_spans=["Workers track build jobs in a SQLite registry."],
        rationale="The registry is implementation detail for job workers.",
        support_score=0.62,
        contradiction_flags=[],
        decision_reason="Topical relation only.",
        utility_score=0.34,
        uncertainty=0.18,
        canonical_relation="implementation_detail",
        semantic_relation_label="implementation_detail",
    )
    orchestrator = _orchestrator(provider=provider)

    assessment = orchestrator._assessment_for(anchor, candidate, judge)

    assert assessment.accepted is False
    assert assessment.reject_reason in {
        "topic_drift",
        "low_utility",
        "wrong_relation_type",
        "weak_evidence",
        "graph_hygiene_failure",
        "low_support",
    }


def test_activation_profile_is_attached_to_scientific_bridge():
    provider = type("OpenAICompatibleProvider", (FakeProvider,), {})()
    anchor = _brief(
        "med-1",
        "Obesity increases chronic kidney disease progression risk",
        "The passage describes obesity and metabolic syndrome as risk factors for chronic kidney disease progression.",
        ["obesity", "metabolic", "syndrome", "chronic", "kidney", "disease", "risk"],
        relation_hints=["clinical", "risk", "evidence"],
        metadata={"source_dataset": "nfcorpus", "topic": "clinical_retrieval", "topic_cluster": "kidney-risk", "topic_family": "ckd-risk"},
    )
    candidate = _brief(
        "med-2",
        "Metabolic syndrome worsens chronic kidney disease outcomes",
        "The candidate explains how metabolic syndrome worsens chronic kidney disease outcomes and patient risk.",
        ["metabolic", "syndrome", "chronic", "kidney", "disease", "outcomes", "risk"],
        relation_hints=["clinical", "outcome", "evidence"],
        metadata={"source_dataset": "nfcorpus", "topic": "clinical_retrieval", "topic_cluster": "kidney-risk", "topic_family": "ckd-risk"},
    )
    judge = JudgeResult(
        accepted=True,
        relation_type="supporting_evidence",
        confidence=0.84,
        evidence_spans=[
            "Obesity and metabolic syndrome increase chronic kidney disease progression risk.",
            "Metabolic syndrome worsens chronic kidney disease outcomes and patient risk.",
        ],
        rationale="The candidate adds aligned clinical evidence for the same risk family.",
        support_score=0.78,
        contradiction_flags=[],
        decision_reason="Clinically aligned evidence is retrieval-useful.",
        utility_score=0.76,
        uncertainty=0.14,
        canonical_relation="supporting_evidence",
        semantic_relation_label="supporting_evidence",
    )
    review = JudgeResult(
        accepted=True,
        relation_type="supporting_evidence",
        confidence=0.87,
        evidence_spans=judge.evidence_spans,
        rationale="Reviewer confirms high bridge gain for renal-risk queries.",
        support_score=0.8,
        contradiction_flags=[],
        decision_reason="Reviewer confirms clinically aligned support.",
        utility_score=0.8,
        uncertainty=0.1,
        canonical_relation="supporting_evidence",
        semantic_relation_label="supporting_evidence",
    )
    orchestrator = _orchestrator(provider=provider)

    assessment = orchestrator._assessment_for(anchor, candidate, judge, review)

    assert assessment.accepted is True
    assert assessment.edge is not None
    assert assessment.edge.activation_profile["activation_prior"] > 0.0
    assert "query_surface_terms" in assessment.edge.activation_profile


def test_reviewer_soft_acceptance_is_not_overridden_when_hard_constraints_clear():
    provider = type("OpenAICompatibleProvider", (FakeProvider,), {})()
    anchor = _brief(
        "doc-a",
        "DeepAgents Memory Overview",
        "The system persists semantic and anchor memory through a durable backend.",
        ["deepagents", "memory", "semantic", "anchor", "backend"],
        relation_hints=["memory", "backend"],
        metadata={"topic": "deepagents", "stage": "agent_overview"},
    )
    candidate = _brief(
        "doc-b",
        "Persistent Memory Backend",
        "The backend stores learned patterns and long-term retrieval memory.",
        ["persistent", "memory", "backend", "patterns", "retrieval"],
        relation_hints=["memory", "backend"],
        metadata={"topic": "deepagents", "stage": "agent_memory"},
    )
    judge = JudgeResult(
        accepted=True,
        relation_type="implementation_detail",
        confidence=0.8,
        evidence_spans=[
            "The system persists semantic and anchor memory through a durable backend.",
            "The backend stores learned patterns and long-term retrieval memory.",
        ],
        rationale="The backend operationalizes the overview's memory subsystem.",
        support_score=0.74,
        contradiction_flags=[],
        decision_reason="Implementation detail accepted by judge.",
        utility_score=0.7,
        uncertainty=0.18,
        canonical_relation="implementation_detail",
        semantic_relation_label="implementation_detail",
    )
    review = JudgeResult(
        accepted=True,
        relation_type="implementation_detail",
        confidence=0.86,
        evidence_spans=judge.evidence_spans,
        rationale="Reviewer confirms durable retrieval value.",
        support_score=0.8,
        contradiction_flags=[],
        decision_reason="Reviewer confirms high utility without hard blockers.",
        utility_score=0.82,
        uncertainty=0.1,
        canonical_relation="implementation_detail",
        semantic_relation_label="implementation_detail",
    )
    orchestrator = _orchestrator(provider=provider)

    assessment = orchestrator._assessment_for(anchor, candidate, judge, review)

    assert assessment.accepted is True
    assert assessment.relation_type == "implementation_detail"
    assert assessment.reject_reason == ""
