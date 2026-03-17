from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from hnsw_logic.agents.orchestrator import LogicOrchestrator
from hnsw_logic.config.schema import ProviderConfig, RetrievalConfig
from hnsw_logic.core.models import DocBrief
from hnsw_logic.core.utils import deterministic_vector
from hnsw_logic.embedding.provider import JudgeResult, OpenAICompatibleProvider, StubProvider


class FakeProvider(StubProvider):
    def __init__(self):
        super().__init__(ProviderConfig(kind="stub"))
        self.relation_priors = {
            "supporting_evidence": 1.0,
            "implementation_detail": 1.05,
            "same_concept": 0.7,
            "comparison": 0.4,
            "prerequisite": 1.0,
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
    entities: list[str] | None = None,
    relation_hints: list[str] | None = None,
    metadata: dict | None = None,
):
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


def test_relation_specific_thresholds_and_ranked_out():
    provider = type("OpenAICompatibleProvider", (FakeProvider,), {})()
    retrieval = RetrievalConfig()
    anchor = _brief(
        "doc-12",
        "Subagents",
        "Subagents include document profiler, corpus scout, relation judge, and memory curator roles.",
        ["subagents", "document", "profiler", "corpus", "scout", "relation", "judge", "memory", "curator"],
        ["subagents"],
        ["include", "roles"],
    )
    profiler = _brief(
        "doc-15",
        "Document Profiler",
        "A document profiler role summarizes docs before memory updates.",
        ["document", "profiler", "memory", "updates", "role"],
        ["profiler"],
        ["role", "summarize"],
    )
    judge = _brief(
        "doc-17",
        "Relation Judge",
        "The relation judge role verifies candidates after corpus scout proposes them.",
        ["relation", "judge", "corpus", "scout", "role", "candidates"],
        ["judge"],
        ["after", "role"],
    )
    memory = _brief(
        "doc-18",
        "Memory Curator",
        "The memory curator maintains persistent memory for accepted edges.",
        ["memory", "curator", "persistent", "accepted", "edges"],
        ["memory"],
        ["maintains"],
    )
    scout = _brief(
        "doc-16",
        "Corpus Scout",
        "The corpus scout runs before the relation judge but the evidence is omitted here.",
        ["corpus", "scout", "relation", "judge", "before"],
        ["scout"],
        ["before"],
    )
    verdicts = {
        "doc-15": JudgeResult(
            accepted=True,
            relation_type="implementation_detail",
            confidence=0.93,
            evidence_spans=[
                "Subagents include the document profiler role.",
                "A document profiler role summarizes docs before memory updates.",
            ],
            rationale="Subagents includes document profiler as a concrete role.",
            support_score=0.82,
            contradiction_flags=[],
            decision_reason="Specific subagent role is explicitly named.",
        ),
        "doc-17": JudgeResult(
            accepted=True,
            relation_type="prerequisite",
            confidence=0.91,
            evidence_spans=[
                "The scout proposes candidates before relation judging.",
                "The relation judge role verifies candidates after corpus scout proposes them.",
            ],
            rationale="Corpus scout happens before relation judge.",
            support_score=0.8,
            contradiction_flags=[],
            decision_reason="Workflow order is explicit.",
        ),
        "doc-18": JudgeResult(
            accepted=True,
            relation_type="comparison",
            confidence=0.95,
            evidence_spans=[
                "Subagents include memory curator among the roles.",
                "The memory curator maintains persistent memory for accepted edges.",
            ],
            rationale="Both are parts of the same system.",
            support_score=0.83,
            contradiction_flags=[],
            decision_reason="Shared system context only.",
        ),
        "doc-16": JudgeResult(
            accepted=True,
            relation_type="prerequisite",
            confidence=0.92,
            evidence_spans=[],
            rationale="Corpus scout runs before relation judge.",
            support_score=0.78,
            contradiction_flags=[],
            decision_reason="Missing grounded evidence spans.",
        ),
    }
    orchestrator = LogicOrchestrator(
        doc_profiler=SimpleNamespace(provider=provider),
        corpus_scout=SimpleNamespace(provider=provider),
        relation_judge=FakeJudge(provider, verdicts),
        memory_curator=SimpleNamespace(provider=provider),
        retrieval_config=retrieval,
    )

    assessments = orchestrator.judge_many_with_diagnostics(anchor, [profiler, judge, memory, scout])
    by_id = {item.candidate_doc_id: item for item in assessments}

    accepted_ids = [item.candidate_doc_id for item in assessments if item.accepted]
    assert accepted_ids == ["doc-17", "doc-16", "doc-15", "doc-18"]
    assert {by_id[doc_id].relation_type for doc_id in accepted_ids} == {"prerequisite"}
    assert all(by_id[doc_id].evidence_quality >= 1.0 for doc_id in accepted_ids)
    assert by_id["doc-17"].score > by_id["doc-16"].score > by_id["doc-15"].score > by_id["doc-18"].score


def test_local_gate_rejects_topic_drift_or_low_support():
    provider = FakeProvider()
    anchor = _brief(
        "doc-08",
        "Logic Overlay Graph",
        "The logic overlay graph adds one-hop logical expansion after HNSW recall.",
        ["logic", "overlay", "graph", "hnsw", "jump", "policy"],
        ["graph"],
        ["one-hop", "expansion"],
    )
    unrelated = _brief(
        "doc-22",
        "SQLite Job Registry",
        "Workers persist job state in SQLite for build and admin tasks.",
        ["sqlite", "jobs", "workers", "build", "admin"],
        ["sqlite"],
        ["persist"],
    )
    verdicts = {
        "doc-22": JudgeResult(
            accepted=True,
            relation_type="implementation_detail",
            confidence=0.97,
            evidence_spans=[
                "Workers track build jobs in a SQLite registry.",
                "Admin tasks route job state through the registry.",
            ],
            rationale="SQLite registry details are tracked through worker and admin flows.",
            support_score=0.5,
            contradiction_flags=[],
            decision_reason="The candidate is an implementation detail tracked through the registry.",
        )
    }
    orchestrator = LogicOrchestrator(
        doc_profiler=SimpleNamespace(provider=provider),
        corpus_scout=SimpleNamespace(provider=provider),
        relation_judge=FakeJudge(provider, verdicts),
        memory_curator=SimpleNamespace(provider=provider),
        retrieval_config=RetrievalConfig(),
    )

    assessment = orchestrator.judge_many_with_diagnostics(anchor, [unrelated])[0]
    assert assessment.accepted is False
    assert assessment.reject_reason in {"topic_drift", "low_support"}


def test_edge_reviewer_rejects_generic_service_surface_edge():
    provider = type("OpenAICompatibleProvider", (FakeProvider,), {})()
    anchor = _brief(
        "doc-20",
        "Background Jobs",
        "Workers run expensive offline tasks through a lightweight registry.",
        ["workers", "jobs", "offline", "registry", "tasks"],
        ["workers"],
        ["registry", "offline"],
    )
    candidate = _brief(
        "doc-21",
        "Public API Service",
        "The API service exposes endpoints that submit jobs to the same registry.",
        ["api", "service", "endpoints", "jobs", "registry"],
        ["api"],
        ["service", "registry"],
    )
    judge_verdicts = {
        "doc-21": JudgeResult(
            accepted=True,
            relation_type="supporting_evidence",
            confidence=0.88,
            evidence_spans=["The jobs use a registry.", "The API submits jobs to the same registry."],
            rationale="Shared registry implies support.",
            support_score=0.62,
            contradiction_flags=[],
            decision_reason="Shared registry looked supportive.",
            utility_score=0.34,
            uncertainty=0.22,
            canonical_relation="supporting_evidence",
            semantic_relation_label="supporting_evidence",
        )
    }
    review_verdicts = {
        "doc-21": JudgeResult(
            accepted=False,
            relation_type="supporting_evidence",
            confidence=0.41,
            evidence_spans=["Both documents mention the registry surface."],
            rationale="The link is generic service-surface overlap, not durable support.",
            support_score=0.28,
            contradiction_flags=["service_surface"],
            decision_reason="Generic co-usage should not enter the graph.",
            utility_score=0.12,
            uncertainty=0.18,
            canonical_relation="none",
            semantic_relation_label="generic_overlap",
        )
    }
    orchestrator = LogicOrchestrator(
        doc_profiler=SimpleNamespace(provider=provider),
        corpus_scout=SimpleNamespace(provider=provider),
        relation_judge=FakeJudge(provider, judge_verdicts),
        memory_curator=SimpleNamespace(provider=provider),
        edge_reviewer=FakeReviewer(provider, review_verdicts),
        retrieval_config=RetrievalConfig(),
    )

    assessment = orchestrator.judge_many_with_diagnostics(anchor, [candidate])[0]
    assert assessment.accepted is False
    assert assessment.reject_reason in {"low_utility", "wrong_relation_type", "weak_link", "model_rejected"}


def test_edge_reviewer_can_recover_high_utility_pair():
    provider = type("OpenAICompatibleProvider", (FakeProvider,), {})()
    anchor = _brief(
        "doc-07",
        "Hybrid Retrieval",
        "Hybrid retrieval merges geometric seeds with logical expansions and weighted scoring.",
        ["hybrid", "retrieval", "geometric", "logical", "weighted", "scoring"],
        ["hybrid retrieval"],
        ["weighted", "logic"],
    )
    candidate = _brief(
        "doc-10",
        "Candidate Fusion",
        "Candidate fusion computes alpha times the HNSW score plus beta times the logic score.",
        ["candidate", "fusion", "hnsw", "logic", "score", "alpha", "beta"],
        ["candidate fusion"],
        ["formula", "score"],
    )
    judge_verdicts = {
        "doc-10": JudgeResult(
            accepted=False,
            relation_type="comparison",
            confidence=0.44,
            evidence_spans=[],
            rationale="The model was unsure.",
            support_score=0.18,
            contradiction_flags=[],
            decision_reason="Initial model abstained.",
            utility_score=0.18,
            uncertainty=0.62,
            canonical_relation="none",
            semantic_relation_label="uncertain_related",
        )
    }
    review_verdicts = {
        "doc-10": JudgeResult(
            accepted=True,
            relation_type="implementation_detail",
            confidence=0.9,
            evidence_spans=["Hybrid retrieval uses weighted scoring.", "Candidate fusion defines the weighted score formula."],
            rationale="The candidate defines the retrieval scoring mechanism used by the anchor.",
            support_score=0.8,
            contradiction_flags=[],
            decision_reason="High retrieval utility and mechanism-level alignment.",
            utility_score=0.82,
            uncertainty=0.18,
            canonical_relation="implementation_detail",
            semantic_relation_label="mechanism_detail",
        )
    }
    orchestrator = LogicOrchestrator(
        doc_profiler=SimpleNamespace(provider=provider),
        corpus_scout=SimpleNamespace(provider=provider),
        relation_judge=FakeJudge(provider, judge_verdicts),
        memory_curator=SimpleNamespace(provider=provider),
        edge_reviewer=FakeReviewer(provider, review_verdicts),
        retrieval_config=RetrievalConfig(),
    )

    assessment = orchestrator.judge_many_with_diagnostics(anchor, [candidate])[0]
    assert assessment.accepted is True
    assert assessment.relation_type == "implementation_detail"


def test_scientific_same_concept_is_allowed_for_live_provider():
    class ScientificProvider(FakeProvider):
        def embed_texts(self, texts):
            rows = []
            for text in texts:
                if "IL2RA" in text or "type 1 diabetes" in text.lower():
                    rows.append(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
                else:
                    rows.append(np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))
            return np.vstack(rows)

    provider = type("OpenAICompatibleProvider", (ScientificProvider,), {})()
    anchor = _brief(
        "11899391",
        "IL2RA variation lowers IL-2 signaling",
        "IL2RA variation lowers IL-2 responsiveness and is associated with type 1 diabetes risk.",
        ["il2ra", "signaling", "type", "diabetes", "treg"],
        ["IL2RA", "type 1 diabetes"],
        ["claim", "evidence", "study"],
        {"source_dataset": "scifact", "topic": "scientific_claims", "topic_cluster": "il2ra-diabetes"},
    )
    candidate = _brief(
        "13940200",
        "IL2RA polymorphism implicates type 1 diabetes",
        "Fine mapping implicates IL2RA polymorphism in type 1 diabetes susceptibility.",
        ["il2ra", "polymorphism", "type", "diabetes", "susceptibility"],
        ["IL2RA", "type 1 diabetes"],
        ["claim", "evidence", "study"],
        {"source_dataset": "scifact", "topic": "scientific_claims", "topic_cluster": "il2ra-diabetes"},
    )
    verdicts = {
        "13940200": JudgeResult(
            accepted=True,
            relation_type="same_concept",
            confidence=0.86,
            evidence_spans=[
                "Both passages describe IL2RA-associated type 1 diabetes susceptibility.",
                "The candidate restates the same disease-gene association at finer granularity.",
            ],
            rationale="The candidate reframes the same IL2RA diabetes association.",
            support_score=0.72,
            contradiction_flags=[],
            decision_reason="Scientific passages describe the same finding family.",
        )
    }
    orchestrator = LogicOrchestrator(
        doc_profiler=SimpleNamespace(provider=provider),
        corpus_scout=SimpleNamespace(provider=provider),
        relation_judge=FakeJudge(provider, verdicts),
        memory_curator=SimpleNamespace(provider=provider),
        retrieval_config=RetrievalConfig(),
    )

    assessment = orchestrator.judge_many_with_diagnostics(anchor, [candidate])[0]
    assert assessment.accepted is True
    assert assessment.relation_type == "same_concept"


def test_scientific_methodology_match_does_not_stay_same_concept():
    class ScientificProvider(FakeProvider):
        def embed_texts(self, texts):
            rows = []
            for text in texts:
                lowered = text.lower()
                if "radioiodine" in lowered or "goitre" in lowered or "goiter" in lowered:
                    rows.append(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
                else:
                    rows.append(np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))
            return np.vstack(rows)

    provider = type("OpenAICompatibleProvider", (ScientificProvider,), {})()
    anchor = _brief(
        "43122426",
        "131-I radioiodine therapy for hyperthyroidism in patients with Graves' disease, uninodular goitre and multinodular goitre",
        "Radioiodine therapy reduces thyroid volume for multinodular goitre and compares outcomes across thyroid disease cohorts.",
        ["radioiodine", "therapy", "multinodular", "goitre", "thyroid", "volume"],
        ["radioiodine therapy", "multinodular goitre"],
        ["claim", "evidence", "study"],
        {"source_dataset": "scifact", "topic": "scientific_claims", "topic_cluster": "radioiodine-goitre"},
    )
    methodology_candidate = _brief(
        "26026009",
        "Magnetic resonance imaging for volume estimation of large multinodular goitres: a comparison with scintigraphy",
        "Magnetic resonance imaging estimates goitre volume and compares the measurement with scintigraphy.",
        ["magnetic", "resonance", "imaging", "volume", "estimation", "multinodular", "goitres"],
        ["goitre imaging", "volume estimation"],
        ["claim", "evidence", "study"],
        {"source_dataset": "scifact", "topic": "scientific_claims", "topic_cluster": "radioiodine-goitre"},
    )
    verdicts = {
        "26026009": JudgeResult(
            accepted=True,
            relation_type="same_concept",
            confidence=0.84,
            evidence_spans=[
                "Both documents discuss multinodular goitre and thyroid volume.",
                "The candidate focuses on imaging-based volume estimation rather than the primary treatment effect.",
            ],
            rationale="The documents are related by the same disease setting.",
            support_score=0.7,
            contradiction_flags=[],
            decision_reason="Shared disease topic suggested a same-concept relation.",
        )
    }
    orchestrator = LogicOrchestrator(
        doc_profiler=SimpleNamespace(provider=provider),
        corpus_scout=SimpleNamespace(provider=provider),
        relation_judge=FakeJudge(provider, verdicts),
        memory_curator=SimpleNamespace(provider=provider),
        retrieval_config=RetrievalConfig(),
    )

    assessment = orchestrator.judge_many_with_diagnostics(anchor, [methodology_candidate])[0]
    assert not (assessment.accepted and assessment.relation_type == "same_concept")


def test_high_utility_scientific_same_concept_bridge_survives_weak_link_gate(monkeypatch):
    class ScientificProvider(FakeProvider):
        def embed_texts(self, texts):
            rows = []
            for text in texts:
                lowered = text.lower()
                if "hematopoietic" in lowered or "stem cell" in lowered or "chromosome" in lowered or "thrombopoietin" in lowered:
                    rows.append(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
                else:
                    rows.append(np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))
            return np.vstack(rows)

    provider = type("OpenAICompatibleProvider", (ScientificProvider,), {})()
    anchor = _brief(
        "32170702",
        "Thrombopoietin signaling regulates hematopoietic stem cell quiescence",
        "Thrombopoietin signaling regulates hematopoietic stem cell quiescence in the osteoblastic niche.",
        ["thrombopoietin", "hematopoietic", "stem", "cell", "quiescence", "niche"],
        ["hematopoietic stem cell"],
        ["claim", "evidence", "study"],
        {"source_dataset": "scifact", "topic": "scientific_claims", "topic_cluster": "hsc-quiescence"},
    )
    candidate = _brief(
        "4381486",
        "Hematopoietic stem cells do not asymmetrically segregate chromosomes",
        "Hematopoietic stem cells do not asymmetrically segregate chromosomes or retain BrdU.",
        ["hematopoietic", "stem", "cells", "chromosomes", "segregate", "brdu"],
        ["hematopoietic stem cell"],
        ["claim", "evidence", "study"],
        {"source_dataset": "scifact", "topic": "scientific_claims", "topic_cluster": "hsc-quiescence"},
    )
    verdicts = {
        "4381486": JudgeResult(
            accepted=True,
            relation_type="same_concept",
            confidence=0.86,
            evidence_spans=[
                "Both passages describe hematopoietic stem cell behavior.",
                "The candidate contributes a complementary HSC finding rather than a contradictory topic.",
            ],
            rationale="The candidate is a bridge document in the same HSC evidence family.",
            support_score=0.78,
            contradiction_flags=[],
            decision_reason="Shared HSC evidence family.",
            utility_score=0.8,
            uncertainty=0.2,
            canonical_relation="same_concept",
            semantic_relation_label="same_evidence_family",
        )
    }
    orchestrator = LogicOrchestrator(
        doc_profiler=SimpleNamespace(provider=provider),
        corpus_scout=SimpleNamespace(provider=provider),
        relation_judge=FakeJudge(provider, verdicts),
        memory_curator=SimpleNamespace(provider=provider),
        retrieval_config=RetrievalConfig(),
    )
    monkeypatch.setattr(type(orchestrator), "_bridge_information_gain", lambda self, *_args, **_kwargs: 0.51)

    assessment = orchestrator.judge_many_with_diagnostics(anchor, [candidate])[0]
    assert assessment.accepted is True
    assert assessment.relation_type == "same_concept"


def test_semantic_detail_bridge_accepts_high_dense_mechanism_pair():
    class SemanticProvider(FakeProvider):
        def embed_texts(self, texts):
            rows = []
            for text in texts:
                if "Hybrid Retrieval" in text or "Candidate Fusion" in text:
                    rows.append(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
                else:
                    rows.append(np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))
            return np.vstack(rows)

    provider = type("OpenAICompatibleProvider", (SemanticProvider,), {})()
    anchor = _brief(
        "doc-07",
        "Hybrid Retrieval",
        "Hybrid retrieval merges geometric seed sets with logical expansions and combines HNSW and logic scores in a weighted formula.",
        ["hybrid", "retrieval", "geometric", "hnsw", "logic", "scores", "weighted", "formula"],
        ["hybrid retrieval"],
        ["weighted formula", "logic score"],
    )
    candidate = _brief(
        "doc-10",
        "Candidate Fusion",
        "Candidate fusion computes alpha times the HNSW score plus beta times the logic score from the best expansion path.",
        ["candidate", "fusion", "hnsw", "logic", "score", "expansion", "path", "formula"],
        ["candidate fusion"],
        ["scoring", "expansion path"],
    )
    verdicts = {
        "doc-10": JudgeResult(
            accepted=False,
            relation_type="comparison",
            confidence=0.42,
            evidence_spans=[],
            rationale="Shared ranking context only.",
            support_score=0.2,
            contradiction_flags=[],
            decision_reason="Model abstained on the pair.",
        )
    }
    orchestrator = LogicOrchestrator(
        doc_profiler=SimpleNamespace(provider=provider),
        corpus_scout=SimpleNamespace(provider=provider),
        relation_judge=FakeJudge(provider, verdicts),
        memory_curator=SimpleNamespace(provider=provider),
        retrieval_config=RetrievalConfig(),
    )

    assessment = orchestrator.judge_many_with_diagnostics(anchor, [candidate])[0]
    assert assessment.accepted is True
    assert assessment.relation_type == "implementation_detail"
    assert assessment.edge is not None


def test_agent_overview_keeps_memory_edge_as_second_result():
    provider = type("OpenAICompatibleProvider", (FakeProvider,), {})()
    anchor = _brief(
        "doc-11",
        "DeepAgents Overview",
        "Overview of DeepAgents with subagents, persistent memory, and retrieval workflows.",
        ["deepagents", "overview", "subagents", "memory", "retrieval"],
        ["deepagents"],
        ["overview", "persistent", "memory"],
    )
    subagents = _brief(
        "doc-12",
        "Subagents",
        "Subagents isolate context for profiler, scout, judge, and curator roles.",
        ["subagents", "profiler", "scout", "judge", "curator"],
        ["subagents"],
        ["roles", "context"],
    )
    memory = _brief(
        "doc-14",
        "Persistent Memory Backend",
        "Persistent memory stores anchor memories and semantic patterns for the runtime.",
        ["persistent", "memory", "anchor", "semantic", "runtime"],
        ["memory"],
        ["persistent", "backend"],
    )
    verdicts = {
        "doc-12": JudgeResult(
            accepted=True,
            relation_type="implementation_detail",
            confidence=0.95,
            evidence_spans=["The overview covers subagents.", "Subagents isolate context for profiler, scout, judge, and curator roles."],
            rationale="Subagents are a concrete mechanism covered by the overview.",
            support_score=0.88,
            contradiction_flags=[],
            decision_reason="Strong component overlap.",
        ),
        "doc-14": JudgeResult(
            accepted=True,
            relation_type="implementation_detail",
            confidence=0.84,
            evidence_spans=["The overview covers persistent memory.", "Persistent memory stores anchor memories and semantic patterns for the runtime."],
            rationale="Persistent memory is another concrete subsystem covered by the overview.",
            support_score=0.7,
            contradiction_flags=[],
            decision_reason="Strong subsystem overlap.",
        ),
    }
    orchestrator = LogicOrchestrator(
        doc_profiler=SimpleNamespace(provider=provider),
        corpus_scout=SimpleNamespace(provider=provider),
        relation_judge=FakeJudge(provider, verdicts),
        memory_curator=SimpleNamespace(provider=provider),
        retrieval_config=RetrievalConfig(),
    )

    assessments = orchestrator.judge_many_with_diagnostics(anchor, [subagents, memory])
    accepted_ids = [item.candidate_doc_id for item in assessments if item.accepted]

    assert accepted_ids == ["doc-12", "doc-14"]


def test_scientific_risk_docs_receive_higher_discovery_priority():
    provider = type("OpenAICompatibleProvider", (FakeProvider,), {})()
    orchestrator = LogicOrchestrator(
        doc_profiler=SimpleNamespace(provider=provider),
        corpus_scout=SimpleNamespace(provider=provider),
        relation_judge=FakeJudge(provider, {}),
        memory_curator=SimpleNamespace(provider=provider),
        retrieval_config=RetrievalConfig(),
    )
    rich = _brief(
        "risk-rich",
        "Global disease burden from metabolic risk",
        "A population health study measures chronic disease burden from dietary and metabolic risks.",
        ["population", "health", "risk", "nutrition", "metabolic", "disease"],
        ["chronic disease"],
        ["claim", "study", "population risk", "chronic disease burden", "nutrition"],
        {"source_dataset": "scifact", "topic": "scientific_claims"},
    )
    plain = _brief(
        "risk-plain",
        "Generic findings",
        "A study reports generic findings without explicit bridge hints.",
        ["study", "finding"],
        ["finding"],
        ["study"],
        {"source_dataset": "scifact", "topic": "scientific_claims"},
    )

    assert orchestrator.discovery_anchor_priority(rich) > orchestrator.discovery_anchor_priority(plain)


def test_scientific_hub_anchor_selection_keeps_central_docs():
    provider = type("OpenAICompatibleProvider", (FakeProvider,), {})()
    orchestrator = LogicOrchestrator(
        doc_profiler=SimpleNamespace(provider=provider),
        corpus_scout=SimpleNamespace(provider=provider),
        relation_judge=FakeJudge(provider, {}),
        memory_curator=SimpleNamespace(provider=provider),
        retrieval_config=RetrievalConfig(),
    )
    hub_a = _brief(
        "25516011",
        "Purification and characterization of mouse hematopoietic stem cells",
        "Hematopoietic stem cells are purified and characterized across stem cell assays.",
        ["hematopoietic", "stem", "cells", "assays", "purification"],
        ["hematopoietic stem cells"],
        ["claim", "evidence", "study"],
        {"source_dataset": "scifact", "topic": "scientific_claims"},
    )
    hub_b = _brief(
        "13116880",
        "Hematopoietic stem cell self-renewal versus differentiation",
        "Self-renewal and differentiation define hematopoietic stem cell behavior.",
        ["hematopoietic", "stem", "cell", "self-renewal", "differentiation"],
        ["hematopoietic stem cells"],
        ["claim", "evidence", "study"],
        {"source_dataset": "scifact", "topic": "scientific_claims"},
    )
    niche = _brief(
        "5415832",
        "Normal and leukemic stem cell niches",
        "Stem cell niches shape hematopoietic stem cell behavior and therapy.",
        ["stem", "cell", "niche", "hematopoietic", "therapy"],
        ["stem cell niche"],
        ["claim", "evidence", "study"],
        {"source_dataset": "scifact", "topic": "scientific_claims"},
    )
    niche_detail = _brief(
        "3391547",
        "Bone marrow microenvironment in disease pathogenesis",
        "Bone marrow microenvironment shapes hematopoietic disease and stem cell maintenance.",
        ["bone", "marrow", "microenvironment", "stem", "cell", "hematopoietic"],
        ["bone marrow microenvironment"],
        ["claim", "evidence", "study"],
        {"source_dataset": "scifact", "topic": "scientific_claims"},
    )
    outlier_a = _brief(
        "doc-x",
        "Fruit fly lifespan extension by rapamycin",
        "Rapamycin extends lifespan in drosophila.",
        ["rapamycin", "lifespan", "drosophila"],
        ["rapamycin"],
        ["claim", "evidence", "study"],
        {"source_dataset": "scifact", "topic": "scientific_claims"},
    )
    outlier_b = _brief(
        "doc-y",
        "Radioiodine treatment of multinodular non-toxic goitre",
        "Radioiodine treatment reduces thyroid volume for multinodular goitre.",
        ["radioiodine", "treatment", "goitre", "thyroid", "volume"],
        ["multinodular goitre"],
        ["claim", "evidence", "study"],
        {"source_dataset": "scifact", "topic": "scientific_claims"},
    )

    selected = orchestrator._select_dataset_hub_anchors([hub_a, hub_b, niche, niche_detail, outlier_a, outlier_b], cap=2)
    assert "25516011" in selected or "13116880" in selected


def test_scientific_candidate_metrics_normalize_plural_title_overlap():
    provider = type("OpenAICompatibleProvider", (FakeProvider,), {})()
    orchestrator = LogicOrchestrator(
        doc_profiler=SimpleNamespace(provider=provider),
        corpus_scout=SimpleNamespace(provider=provider),
        relation_judge=FakeJudge(provider, {}),
        memory_curator=SimpleNamespace(provider=provider),
        retrieval_config=RetrievalConfig(),
    )
    anchor = _brief(
        "11141995",
        "Organization of the mitotic chromosome",
        "Mitotic chromosome organization remains unresolved.",
        ["mitotic", "chromosome", "organization"],
        ["chromosome"],
        ["claim", "evidence", "study"],
        {"source_dataset": "scifact", "topic": "scientific_claims"},
    )
    candidate = _brief(
        "4381486",
        "Haematopoietic stem cells do not asymmetrically segregate chromosomes or retain BrdU",
        "Hematopoietic stem cells do not asymmetrically segregate chromosomes.",
        ["stem", "cells", "segregate", "chromosomes"],
        ["hematopoietic stem cells"],
        ["claim", "evidence", "study"],
        {"source_dataset": "scifact", "topic": "scientific_claims"},
    )

    metrics = orchestrator._candidate_metrics(anchor, candidate)
    assert metrics["title_overlap"] >= 1.0


def test_logic_graph_can_fallback_to_retrieval_supporting_evidence():
    provider = type("OpenAICompatibleProvider", (FakeProvider,), {})()
    orchestrator = LogicOrchestrator(
        doc_profiler=SimpleNamespace(provider=provider),
        corpus_scout=SimpleNamespace(provider=provider),
        relation_judge=FakeJudge(
            provider,
            {
                "doc-07": JudgeResult(
                    accepted=False,
                    relation_type="comparison",
                    confidence=0.41,
                    evidence_spans=[],
                    rationale="Shared retrieval context only.",
                    support_score=0.2,
                    contradiction_flags=[],
                    decision_reason="Model abstained on directionality.",
                )
            },
        ),
        memory_curator=SimpleNamespace(provider=provider),
        retrieval_config=RetrievalConfig(),
    )
    anchor = _brief(
        "doc-08",
        "Logic Overlay Graph",
        "The logic overlay graph adds one-hop logical expansion after HNSW recall.",
        ["logic", "overlay", "graph", "hnsw", "jump", "policy"],
        ["graph"],
        ["one-hop", "expansion"],
    )
    candidate = _brief(
        "doc-07",
        "Hybrid Retrieval",
        "Hybrid retrieval uses the logic overlay graph as a sidecar after initial HNSW recall.",
        ["hybrid", "retrieval", "logic", "overlay", "graph", "hnsw"],
        ["hybrid retrieval"],
        ["sidecar", "recall"],
    )

    assessment = orchestrator.judge_many_with_diagnostics(anchor, [candidate])[0]

    assert assessment.accepted is True
    assert assessment.relation_type == "supporting_evidence"


def test_evaluation_docs_can_attempt_discovery():
    provider = type("OpenAICompatibleProvider", (FakeProvider,), {})()
    orchestrator = LogicOrchestrator(
        doc_profiler=SimpleNamespace(provider=provider),
        corpus_scout=SimpleNamespace(provider=provider),
        relation_judge=FakeJudge(provider, {}),
        memory_curator=SimpleNamespace(provider=provider),
        retrieval_config=RetrievalConfig(),
    )
    brief = _brief(
        "doc-05",
        "ANN Benchmark Metrics",
        "This report summarizes ANN recall, MRR, and NDCG metrics for benchmark reporting.",
        ["ann", "benchmark", "recall", "mrr", "ndcg", "reporting"],
        ["ann"],
        ["metrics", "reporting"],
        {"topic": "evaluation"},
    )

    assert orchestrator.should_attempt_discovery(brief) is True


def test_evaluation_metrics_prefer_reporting_component():
    provider = type("OpenAICompatibleProvider", (FakeProvider,), {})()
    orchestrator = LogicOrchestrator(
        doc_profiler=SimpleNamespace(provider=provider),
        corpus_scout=SimpleNamespace(provider=provider),
        relation_judge=FakeJudge(
            provider,
            {
                "doc-24": JudgeResult(
                    accepted=False,
                    relation_type="comparison",
                    confidence=0.42,
                    evidence_spans=[],
                    rationale="Shared evaluation context only.",
                    support_score=0.2,
                    contradiction_flags=[],
                    decision_reason="Model abstained.",
                )
            },
        ),
        memory_curator=SimpleNamespace(provider=provider),
        retrieval_config=RetrievalConfig(),
    )
    anchor = _brief(
        "doc-05",
        "ANN Metrics",
        "This report summarizes ANN recall, MRR, and NDCG metrics for benchmark reporting.",
        ["ann", "benchmark", "recall", "mrr", "ndcg", "reporting"],
        ["ann"],
        ["metrics", "reporting"],
        {"topic": "evaluation"},
    )
    candidate = _brief(
        "doc-24",
        "Benchmark Reporting",
        "Benchmark reporting specifies Recall at 5, MRR at 10, NDCG at 10, edge precision, and query latency.",
        ["benchmark", "reporting", "recall", "mrr", "ndcg", "latency"],
        ["benchmark reporting"],
        ["reporting", "include", "metrics"],
        {"topic": "evaluation"},
    )

    assessment = orchestrator.judge_many_with_diagnostics(anchor, [candidate])[0]

    assert assessment.accepted is True
    assert assessment.relation_type == "implementation_detail"


def test_evaluation_metrics_reject_retrieval_overview_detail():
    provider = type("OpenAICompatibleProvider", (FakeProvider,), {})()
    orchestrator = LogicOrchestrator(
        doc_profiler=SimpleNamespace(provider=provider),
        corpus_scout=SimpleNamespace(provider=provider),
        relation_judge=FakeJudge(
            provider,
            {
                "doc-07": JudgeResult(
                    accepted=True,
                    relation_type="implementation_detail",
                    confidence=0.9,
                    evidence_spans=["Benchmarks compare against hybrid retrieval.", "Hybrid retrieval merges dense and logic scores."],
                    rationale="Shared benchmark and retrieval context.",
                    support_score=0.7,
                    contradiction_flags=[],
                    decision_reason="Overly broad implementation guess.",
                )
            },
        ),
        memory_curator=SimpleNamespace(provider=provider),
        retrieval_config=RetrievalConfig(),
    )
    anchor = _brief(
        "doc-05",
        "ANN Metrics",
        "This report summarizes ANN recall, MRR, and NDCG metrics for benchmark reporting.",
        ["ann", "benchmark", "recall", "mrr", "ndcg", "reporting"],
        ["ann"],
        ["metrics", "reporting"],
        {"topic": "evaluation"},
    )
    candidate = _brief(
        "doc-07",
        "Hybrid Retrieval",
        "Hybrid retrieval combines geometric and logical scores.",
        ["hybrid", "retrieval", "logic", "scores"],
        ["hybrid retrieval"],
        ["hybrid", "logic"],
        {"topic": "retrieval"},
    )

    assessment = orchestrator.judge_many_with_diagnostics(anchor, [candidate])[0]

    assert assessment.accepted is False
    assert assessment.reject_reason == "wrong_relation_type"


def test_directionality_rejects_reverse_implementation_pair():
    provider = type("OpenAICompatibleProvider", (FakeProvider,), {})()
    anchor = _brief(
        "doc-10",
        "Candidate Fusion",
        "Candidate fusion computes alpha times the HNSW score plus beta times the logic score from the best expansion path.",
        ["candidate", "fusion", "hnsw", "logic", "score", "expansion", "path", "formula"],
        ["candidate fusion"],
        ["scoring", "expansion path"],
    )
    candidate = _brief(
        "doc-08",
        "Logic Overlay Graph",
        "The logic overlay graph stores offline-discovered relations and is used after initial HNSW recall.",
        ["logic", "overlay", "graph", "recall", "relations"],
        ["logic graph"],
        ["sidecar graph", "recall"],
    )
    verdicts = {
        "doc-08": JudgeResult(
            accepted=True,
            relation_type="implementation_detail",
            confidence=0.94,
            evidence_spans=[
                "Candidate fusion combines logic score with HNSW score.",
                "The logic overlay graph supplies document relations used after recall.",
            ],
            rationale="The two documents share the same logic-retrieval mechanism family.",
            support_score=0.84,
            contradiction_flags=[],
            decision_reason="Mechanism family overlap appears strong.",
        )
    }
    orchestrator = LogicOrchestrator(
        doc_profiler=SimpleNamespace(provider=provider),
        corpus_scout=SimpleNamespace(provider=provider),
        relation_judge=FakeJudge(provider, verdicts),
        memory_curator=SimpleNamespace(provider=provider),
        retrieval_config=RetrievalConfig(),
    )

    assessment = orchestrator.judge_many_with_diagnostics(anchor, [candidate])[0]
    assert assessment.accepted is False
    assert assessment.reject_reason in {"wrong_direction", "weak_link"}


def test_supporting_evidence_rejects_service_surface_pairs():
    provider = type("OpenAICompatibleProvider", (FakeProvider,), {})()
    anchor = _brief(
        "doc-20",
        "Jobs and Workers",
        "Offline build and discovery tasks run as background jobs using a lightweight worker pool and SQLite registry.",
        ["jobs", "workers", "background", "build", "discovery", "sqlite", "registry"],
        ["jobs"],
        ["background jobs", "sqlite registry"],
    )
    candidate = _brief(
        "doc-21",
        "FastAPI Service",
        "The public service exposes build endpoints, health checks, and job inspection while scheduling expensive tasks through the job registry.",
        ["fastapi", "service", "endpoints", "health", "inspection", "job", "registry"],
        ["service"],
        ["build endpoints", "job registry"],
    )
    verdicts = {
        "doc-21": JudgeResult(
            accepted=True,
            relation_type="supporting_evidence",
            confidence=0.91,
            evidence_spans=[
                "Expensive tasks are scheduled through the job registry.",
                "The service exposes build endpoints and job inspection.",
            ],
            rationale="The service uses the same job registry for task scheduling.",
            support_score=0.8,
            contradiction_flags=[],
            decision_reason="Shared registry suggests supporting evidence.",
        )
    }
    orchestrator = LogicOrchestrator(
        doc_profiler=SimpleNamespace(provider=provider),
        corpus_scout=SimpleNamespace(provider=provider),
        relation_judge=FakeJudge(provider, verdicts),
        memory_curator=SimpleNamespace(provider=provider),
        retrieval_config=RetrievalConfig(),
    )

    assessment = orchestrator.judge_many_with_diagnostics(anchor, [candidate])[0]
    assert assessment.accepted is False
    assert assessment.reject_reason == "wrong_relation_type"


def test_agent_roles_rejects_memory_component_as_implementation_detail():
    provider = type("OpenAICompatibleProvider", (FakeProvider,), {})()
    anchor = _brief(
        "doc-12",
        "Subagents",
        "Subagents include a document profiler, corpus scout, relation judge, and memory curator.",
        ["subagents", "document", "profiler", "corpus", "scout", "relation", "judge", "memory", "curator"],
        ["subagents"],
        ["include", "roles"],
    )
    candidate = _brief(
        "doc-14",
        "Long Term Memory",
        "Persistent memory stores anchor memory and semantic memory on disk.",
        ["memory", "persistent", "anchor", "semantic", "disk"],
        ["memory"],
        ["persistent", "storage"],
    )
    verdicts = {
        "doc-14": JudgeResult(
            accepted=True,
            relation_type="implementation_detail",
            confidence=0.9,
            evidence_spans=[
                "Subagents include a memory curator.",
                "Persistent memory stores anchor memory and semantic memory on disk.",
            ],
            rationale="Shared memory terms suggest a subsystem implementation detail.",
            support_score=0.8,
            contradiction_flags=[],
            decision_reason="Memory subsystem seems semantically related.",
        )
    }
    orchestrator = LogicOrchestrator(
        doc_profiler=SimpleNamespace(provider=provider),
        corpus_scout=SimpleNamespace(provider=provider),
        relation_judge=FakeJudge(provider, verdicts),
        memory_curator=SimpleNamespace(provider=provider),
        retrieval_config=RetrievalConfig(),
    )

    assessment = orchestrator.judge_many_with_diagnostics(anchor, [candidate])[0]
    assert assessment.accepted is False
    assert assessment.reject_reason == "wrong_relation_type"


def test_workflow_and_revalidation_fallbacks_are_accepted():
    provider = type("OpenAICompatibleProvider", (FakeProvider,), {})()
    retrieval = RetrievalConfig()
    scout = _brief(
        "doc-16",
        "Corpus Scout",
        "The corpus scout proposes candidates before the relation judge verifies them.",
        ["corpus", "scout", "before", "relation", "judge", "candidates"],
        ["scout"],
        ["before", "workflow"],
    )
    judge = _brief(
        "doc-17",
        "Relation Judge",
        "The relation judge verifies candidate relations after the scout proposes them.",
        ["relation", "judge", "after", "scout", "candidate", "relations"],
        ["judge"],
        ["after", "verify"],
    )
    revalidation = _brief(
        "doc-23",
        "Edge Revalidation",
        "Logic edges should be revalidated after corpus changes so relation-judge outputs and judged relations remain trustworthy.",
        ["edge", "revalidation", "logic", "edges", "judge", "judged", "relations", "trustworthy"],
        ["revalidation"],
        ["stale", "validated", "judge outputs"],
    )
    verdicts = {
        "doc-17": JudgeResult(
            accepted=True,
            relation_type="prerequisite",
            confidence=0.9,
            evidence_spans=[
                "The corpus scout proposes candidates before the relation judge verifies them.",
                "The relation judge verifies candidate relations after the scout proposes them.",
            ],
            rationale="The workflow explicitly places judging after scouting.",
            support_score=0.82,
            contradiction_flags=[],
            decision_reason="Workflow order is explicit.",
        ),
        "doc-23": JudgeResult(
            accepted=True,
            relation_type="supporting_evidence",
            confidence=0.68,
            evidence_spans=[
                "Judged relations remain trustworthy through later checks.",
                "Stale logic edges are revalidated after corpus changes.",
            ],
            rationale="Revalidation preserves the quality of judged relations over time.",
            support_score=0.76,
            contradiction_flags=[],
            decision_reason="Downstream validation supports the judging stage.",
        ),
    }
    orchestrator = LogicOrchestrator(
        doc_profiler=SimpleNamespace(provider=provider),
        corpus_scout=SimpleNamespace(provider=provider),
        relation_judge=FakeJudge(provider, verdicts),
        memory_curator=SimpleNamespace(provider=provider),
        retrieval_config=retrieval,
    )

    scout_assessment = orchestrator.judge_many_with_diagnostics(scout, [judge])[0]
    judge_assessment = orchestrator.judge_many_with_diagnostics(judge, [revalidation])[0]

    assert scout_assessment.accepted is True
    assert scout_assessment.relation_type == "prerequisite"
    assert judge_assessment.accepted is True
    assert judge_assessment.relation_type == "supporting_evidence"


def test_reverse_role_to_listing_support_is_rejected():
    provider = type("OpenAICompatibleProvider", (FakeProvider,), {})()
    anchor = _brief(
        "doc-15",
        "Document Profiler",
        "The profiler creates a DocBrief for later discovery steps.",
        ["document", "profiler", "docbrief", "discovery", "steps"],
        ["profiler"],
        ["creates"],
    )
    candidate = _brief(
        "doc-12",
        "Subagents",
        "Subagents include the profiler, scout, judge, and curator roles.",
        ["subagents", "profiler", "scout", "judge", "curator", "roles"],
        ["subagents"],
        ["include", "roles"],
    )
    verdicts = {
        "doc-12": JudgeResult(
            accepted=True,
            relation_type="supporting_evidence",
            confidence=0.86,
            evidence_spans=[
                "The profiler creates a DocBrief for later discovery steps.",
                "Subagents include the profiler, scout, judge, and curator roles.",
            ],
            rationale="The listing mentions the profiler role described by the anchor.",
            support_score=0.78,
            contradiction_flags=[],
            decision_reason="Shared agent workflow context appears supportive.",
        )
    }
    orchestrator = LogicOrchestrator(
        doc_profiler=SimpleNamespace(provider=provider),
        corpus_scout=SimpleNamespace(provider=provider),
        relation_judge=FakeJudge(provider, verdicts),
        memory_curator=SimpleNamespace(provider=provider),
        retrieval_config=RetrievalConfig(),
    )

    assessment = orchestrator.judge_many_with_diagnostics(anchor, [candidate])[0]
    assert assessment.accepted is False
    assert assessment.reject_reason == "wrong_direction"


def test_ops_overview_accepts_registry_implementation_detail():
    provider = type("OpenAICompatibleProvider", (FakeProvider,), {})()
    anchor = _brief(
        "doc-20",
        "Jobs and Workers",
        "Background jobs use a lightweight worker pool and a SQLite registry.",
        ["jobs", "workers", "background", "sqlite", "registry", "queue"],
        ["jobs"],
        ["background jobs", "sqlite registry"],
    )
    candidate = _brief(
        "doc-22",
        "SQLite Job Registry",
        "The registry stores job id, state, payload, timestamps, and recent messages.",
        ["sqlite", "job", "registry", "state", "payload", "timestamps"],
        ["sqlite"],
        ["state", "payload"],
    )
    verdicts = {
        "doc-22": JudgeResult(
            accepted=False,
            relation_type="comparison",
            confidence=0.52,
            evidence_spans=[],
            rationale="Shared operational context only.",
            support_score=0.3,
            contradiction_flags=[],
            decision_reason="Model abstained on direct implementation detail.",
        )
    }
    orchestrator = LogicOrchestrator(
        doc_profiler=SimpleNamespace(provider=provider),
        corpus_scout=SimpleNamespace(provider=provider),
        relation_judge=FakeJudge(provider, verdicts),
        memory_curator=SimpleNamespace(provider=provider),
        retrieval_config=RetrievalConfig(),
    )

    assessment = orchestrator.judge_many_with_diagnostics(anchor, [candidate])[0]
    assert assessment.accepted is True
    assert assessment.relation_type == "implementation_detail"


def test_should_attempt_discovery_prefers_structural_topics():
    provider = FakeProvider()
    orchestrator = LogicOrchestrator(
        doc_profiler=SimpleNamespace(provider=provider),
        corpus_scout=SimpleNamespace(provider=provider),
        relation_judge=SimpleNamespace(provider=provider),
        memory_curator=SimpleNamespace(provider=provider),
        retrieval_config=RetrievalConfig(),
    )
    hnsw_brief = _brief(
        "doc-04",
        "HNSW Insert Path",
        "Insertion in HNSW walks the hierarchy and keeps the base algorithm unchanged.",
        ["hnsw", "insert", "hierarchy"],
    )
    hnsw_brief.metadata["topic"] = "hnsw"
    logic_brief = _brief(
        "doc-08",
        "Logic Overlay Graph",
        "The overlay graph stores logical edges used after HNSW recall.",
        ["logic", "overlay", "graph", "recall"],
    )
    logic_brief.metadata["topic"] = "logic"
    similarity_brief = _brief(
        "doc-06",
        "Cosine Similarity",
        "Cosine similarity is a standard metric for vector retrieval and relevance scoring.",
        ["cosine", "similarity", "metric", "retrieval"],
    )
    similarity_brief.metadata["topic"] = "retrieval"

    assert orchestrator.should_attempt_discovery(hnsw_brief) is False
    assert orchestrator.should_attempt_discovery(similarity_brief) is False
    assert orchestrator.should_attempt_discovery(logic_brief) is True


def test_should_attempt_discovery_uses_external_dataset_style_gate():
    provider = FakeProvider()
    orchestrator = LogicOrchestrator(
        doc_profiler=SimpleNamespace(provider=provider),
        corpus_scout=SimpleNamespace(provider=provider),
        relation_judge=SimpleNamespace(provider=provider),
        memory_curator=SimpleNamespace(provider=provider),
        retrieval_config=RetrievalConfig(),
    )
    scifact_brief = _brief(
        "paper-1",
        "Interleukin-2 signaling and regulatory T cell function",
        "The paper claims reduced IL-2 signaling impairs regulatory T cell function and increases autoimmunity risk.",
        ["interleukin", "signaling", "regulatory", "cells", "autoimmunity"],
        relation_hints=["claim", "evidence", "disease"],
    )
    scifact_brief.entities = ["IL-2", "T cells"]
    scifact_brief.claims = ["Reduced IL-2 signaling impairs regulatory T cell function."]
    scifact_brief.metadata["source_dataset"] = "scifact"

    arguana_brief = _brief(
        "arg-1",
        "Public transit should replace highway expansion",
        "The argument claims public transit investment is better than expanding highways for urban mobility.",
        ["public", "transit", "highway", "expansion", "urban", "mobility", "investment", "policy"],
        relation_hints=["debate", "comparison"],
    )
    arguana_brief.claims = ["Public transit investment is better than highway expansion for urban mobility."]
    arguana_brief.metadata["source_dataset"] = "arguana"
    arguana_brief.metadata["topic_cluster"] = "public-transit-highway"

    assert orchestrator.should_attempt_discovery(scifact_brief) is True
    assert orchestrator.should_attempt_discovery(arguana_brief) is True


def test_argumentative_comparison_edge_is_allowed_for_live_provider():
    provider = type("OpenAICompatibleProvider", (FakeProvider,), {})()
    orchestrator = LogicOrchestrator(
        doc_profiler=SimpleNamespace(provider=provider),
        corpus_scout=SimpleNamespace(provider=provider),
        relation_judge=FakeJudge(
            provider,
            {
                "arg-2": JudgeResult(
                    accepted=True,
                    relation_type="comparison",
                    confidence=0.91,
                    evidence_spans=[
                        "The argument claims cities should fund public transit instead of highway expansion.",
                        "The counterargument claims highway expansion remains the better congestion solution.",
                    ],
                    rationale="Both documents debate the same transportation policy from opposing positions.",
                    support_score=0.82,
                    contradiction_flags=[],
                    decision_reason="Shared topic cluster with contrasting stance.",
                )
            },
        ),
        memory_curator=SimpleNamespace(provider=provider),
        retrieval_config=RetrievalConfig(),
    )
    anchor = _brief(
        "arg-1",
        "Transit investment should come before highway expansion",
        "Cities should invest in public transit and avoid highway expansion.",
        ["transit", "investment", "highway", "expansion", "cities", "policy"],
        relation_hints=["debate", "comparison"],
    )
    anchor.metadata.update({"source_dataset": "arguana", "topic": "argument", "topic_cluster": "transit-highway-expansion", "stance": "pro"})
    candidate = _brief(
        "arg-2",
        "Road expansion remains the best mobility strategy",
        "Expanding highways remains the best way to reduce congestion in cities.",
        ["road", "expansion", "mobility", "congestion", "cities", "policy"],
        relation_hints=["debate", "comparison"],
    )
    candidate.metadata.update({"source_dataset": "arguana", "topic": "argument", "topic_cluster": "transit-highway-expansion", "stance": "con"})

    assessment = orchestrator.judge_many_with_diagnostics(anchor, [candidate])[0]

    assert assessment.accepted is True
    assert assessment.relation_type == "comparison"
    assert assessment.edge is not None


def test_argumentative_comparison_can_survive_without_topic_cluster_if_bridge_is_specific():
    provider = type("OpenAICompatibleProvider", (FakeProvider,), {})()
    orchestrator = LogicOrchestrator(
        doc_profiler=SimpleNamespace(provider=provider),
        corpus_scout=SimpleNamespace(provider=provider),
        relation_judge=FakeJudge(
            provider,
            {
                "arg-2": JudgeResult(
                    accepted=True,
                    relation_type="comparison",
                    confidence=0.88,
                    evidence_spans=[
                        "The claim says highway expansion worsens congestion and cities should invest in transit.",
                        "The counterclaim says highway expansion remains the best congestion solution for cities.",
                    ],
                    rationale="The documents address the same highway-expansion policy from opposing positions.",
                    support_score=0.76,
                    contradiction_flags=[],
                    decision_reason="Specific policy bridge with contrasting stance.",
                    utility_score=0.68,
                    canonical_relation="comparison",
                    semantic_relation_label="comparison",
                )
            },
        ),
        memory_curator=SimpleNamespace(provider=provider),
        retrieval_config=RetrievalConfig(),
    )
    anchor = _brief(
        "arg-1",
        "Highway expansion should not be the main congestion strategy",
        "Cities should invest in public transit rather than expanding highways to reduce congestion.",
        ["highway", "expansion", "congestion", "transit", "cities", "strategy"],
        relation_hints=["debate", "comparison"],
    )
    anchor.metadata.update({"source_dataset": "arguana", "topic": "argument", "stance": "con"})
    candidate = _brief(
        "arg-2",
        "Highway expansion remains the best congestion strategy",
        "Expanding highways remains the best way to relieve congestion in cities.",
        ["highway", "expansion", "congestion", "cities", "strategy", "relieve"],
        relation_hints=["debate", "comparison"],
    )
    candidate.metadata.update({"source_dataset": "arguana", "topic": "argument", "stance": "pro"})

    assessment = orchestrator.judge_many_with_diagnostics(anchor, [candidate])[0]

    assert assessment.accepted is True
    assert assessment.relation_type == "comparison"
    assert assessment.edge is not None


def test_clinical_evidence_bridge_is_allowed_for_live_provider():
    provider = type("OpenAICompatibleProvider", (FakeProvider,), {})()
    orchestrator = LogicOrchestrator(
        doc_profiler=SimpleNamespace(provider=provider),
        corpus_scout=SimpleNamespace(provider=provider),
        relation_judge=FakeJudge(
            provider,
            {
                "med-2": JudgeResult(
                    accepted=True,
                    relation_type="supporting_evidence",
                    confidence=0.8,
                    evidence_spans=[
                        "Obesity and metabolic syndrome increase chronic kidney disease progression risk.",
                        "The passage links metabolic syndrome to worse chronic kidney disease outcomes.",
                    ],
                    rationale="The candidate adds clinically aligned risk and progression evidence.",
                    support_score=0.74,
                    contradiction_flags=[],
                    decision_reason="Clinical risk evidence is aligned and retrieval-useful.",
                    utility_score=0.63,
                    canonical_relation="supporting_evidence",
                    semantic_relation_label="supporting_evidence",
                )
            },
        ),
        memory_curator=SimpleNamespace(provider=provider),
        retrieval_config=RetrievalConfig(),
    )
    anchor = _brief(
        "med-1",
        "Obesity increases chronic kidney disease progression risk",
        "The clinical passage describes obesity and metabolic syndrome as risk factors for chronic kidney disease progression.",
        ["obesity", "metabolic", "syndrome", "chronic", "kidney", "disease", "progression", "risk"],
        relation_hints=["clinical", "risk", "disease burden"],
    )
    anchor.metadata.update({"source_dataset": "nfcorpus", "topic": "clinical_retrieval"})
    candidate = _brief(
        "med-2",
        "Metabolic syndrome worsens chronic kidney disease outcomes",
        "The candidate explains how metabolic syndrome worsens chronic kidney disease outcomes and patient risk.",
        ["metabolic", "syndrome", "chronic", "kidney", "disease", "outcomes", "risk", "patient"],
        relation_hints=["clinical", "outcome", "risk"],
    )
    candidate.metadata.update({"source_dataset": "nfcorpus", "topic": "clinical_retrieval"})

    assessment = orchestrator.judge_many_with_diagnostics(anchor, [candidate])[0]

    assert assessment.accepted is True
    assert assessment.relation_type in {"supporting_evidence", "same_concept"}
    assert assessment.edge is not None


def test_stage_reranker_prefers_graph_to_policy_over_fusion():
    provider = FakeProvider()
    orchestrator = LogicOrchestrator(
        doc_profiler=SimpleNamespace(provider=provider),
        corpus_scout=SimpleNamespace(provider=provider),
        relation_judge=SimpleNamespace(provider=provider),
        memory_curator=SimpleNamespace(provider=provider),
        retrieval_config=RetrievalConfig(),
    )
    anchor = _brief(
        "doc-08",
        "Logic Overlay Graph",
        "The logic overlay graph stores offline relations and is used after initial HNSW recall.",
        ["logic", "overlay", "graph", "relations", "recall"],
        relation_hints=["sidecar graph", "after recall"],
    )
    anchor.metadata["topic"] = "logic"
    policy = _brief(
        "doc-09",
        "Jump Policy",
        "The jump policy defines confidence and relevance gates for one-hop expansion after top-B geometric recall.",
        ["jump", "policy", "confidence", "relevance", "expansion", "recall"],
        relation_hints=["one-hop expansion", "top B recall"],
    )
    policy.metadata["topic"] = "logic"
    fusion = _brief(
        "doc-10",
        "Candidate Fusion",
        "Candidate fusion combines HNSW score and logic score in the final weighted ranker.",
        ["candidate", "fusion", "score", "logic", "ranker", "weighted"],
        relation_hints=["final ranker", "weighted score"],
    )
    fusion.metadata["topic"] = "logic"

    policy_metrics = orchestrator._candidate_metrics(anchor, policy)
    fusion_metrics = orchestrator._candidate_metrics(anchor, fusion)

    assert orchestrator._relation_fit_scores(anchor, policy, policy_metrics)["implementation_detail"] > orchestrator._relation_fit_scores(anchor, fusion, fusion_metrics)["implementation_detail"]


def test_role_reranker_recovers_specific_role_from_comparison_verdict():
    provider = type("OpenAICompatibleProvider", (FakeProvider,), {})()
    anchor = _brief(
        "doc-12",
        "Subagents",
        "Subagents include document profiler, corpus scout, relation judge, and memory curator roles.",
        ["subagents", "document", "profiler", "corpus", "scout", "relation", "judge", "memory", "curator"],
        ["subagents"],
        ["include", "roles"],
    )
    anchor.metadata["topic"] = "deepagents"
    candidate = _brief(
        "doc-18",
        "Memory Curator",
        "The memory curator maintains persistent memory for accepted edges.",
        ["memory", "curator", "persistent", "accepted", "edges"],
        ["memory"],
        ["maintains"],
    )
    candidate.metadata["topic"] = "agents"
    verdicts = {
        "doc-18": JudgeResult(
            accepted=True,
            relation_type="comparison",
            confidence=0.95,
            evidence_spans=[
                "Subagents include memory curator among the roles.",
                "The memory curator maintains persistent memory for accepted edges.",
            ],
            rationale="Both are parts of the same system.",
            support_score=0.83,
            contradiction_flags=[],
            decision_reason="Shared system context only.",
        )
    }
    orchestrator = LogicOrchestrator(
        doc_profiler=SimpleNamespace(provider=provider),
        corpus_scout=SimpleNamespace(provider=provider),
        relation_judge=FakeJudge(provider, verdicts),
        memory_curator=SimpleNamespace(provider=provider),
        retrieval_config=RetrievalConfig(),
    )

    assessment = orchestrator.judge_many_with_diagnostics(anchor, [candidate])[0]
    assert assessment.accepted is True
    assert assessment.relation_type == "prerequisite"
    assert assessment.edge is not None


def test_support_stage_direction_rejects_reverse_policy_edge():
    provider = type("OpenAICompatibleProvider", (FakeProvider,), {})()
    anchor = _brief(
        "doc-10",
        "Candidate Fusion",
        "Candidate fusion combines HNSW score and logic score in the final weighted ranker.",
        ["candidate", "fusion", "score", "logic", "ranker", "weighted"],
        relation_hints=["final ranker", "weighted score"],
    )
    anchor.metadata["topic"] = "logic"
    candidate = _brief(
        "doc-09",
        "Jump Policy",
        "The jump policy defines confidence and relevance gates for one-hop expansion after top-B geometric recall.",
        ["jump", "policy", "confidence", "relevance", "expansion", "recall"],
        relation_hints=["one-hop expansion", "top B recall"],
    )
    candidate.metadata["topic"] = "logic"
    verdicts = {
        "doc-09": JudgeResult(
            accepted=True,
            relation_type="supporting_evidence",
            confidence=0.9,
            evidence_spans=[
                "One-hop expansion is allowed only under specific conditions.",
                "The seed must be in the top B geometric results.",
            ],
            rationale="Jump Policy constrains the expansion paths later used by Candidate Fusion.",
            support_score=0.82,
            contradiction_flags=[],
            decision_reason="The policy influences the fusion stage through the path definition.",
        )
    }
    orchestrator = LogicOrchestrator(
        doc_profiler=SimpleNamespace(provider=provider),
        corpus_scout=SimpleNamespace(provider=provider),
        relation_judge=FakeJudge(provider, verdicts),
        memory_curator=SimpleNamespace(provider=provider),
        retrieval_config=RetrievalConfig(),
    )

    assessment = orchestrator.judge_many_with_diagnostics(anchor, [candidate])[0]
    assert assessment.accepted is False
    assert assessment.reject_reason in {"topic_drift", "wrong_direction"}


def test_implementation_stage_direction_rejects_service_to_overview_edge():
    provider = type("OpenAICompatibleProvider", (FakeProvider,), {})()
    anchor = _brief(
        "doc-21",
        "FastAPI Service",
        "The public FastAPI service exposes endpoints for build, search, revalidation, health checks, and job inspection.",
        ["fastapi", "service", "endpoints", "search", "revalidation", "health", "job"],
        relation_hints=["build endpoints", "job inspection"],
    )
    anchor.metadata["topic"] = "ops"
    candidate = _brief(
        "doc-20",
        "Jobs and Workers",
        "Offline build, profiling, discovery, and revalidation tasks run as background jobs with a lightweight worker pool.",
        ["jobs", "workers", "background", "build", "profiling", "discovery", "revalidation"],
        relation_hints=["background jobs", "worker pool"],
    )
    candidate.metadata["topic"] = "ops"
    verdicts = {
        "doc-20": JudgeResult(
            accepted=True,
            relation_type="implementation_detail",
            confidence=0.9,
            evidence_spans=[
                "The service exposes build and revalidation endpoints.",
                "Offline build, profiling, discovery, and revalidation tasks run as background jobs.",
            ],
            rationale="The service depends on the worker system for expensive tasks.",
            support_score=0.8,
            contradiction_flags=[],
            decision_reason="The worker system appears to be a downstream implementation detail.",
        )
    }
    orchestrator = LogicOrchestrator(
        doc_profiler=SimpleNamespace(provider=provider),
        corpus_scout=SimpleNamespace(provider=provider),
        relation_judge=FakeJudge(provider, verdicts),
        memory_curator=SimpleNamespace(provider=provider),
        retrieval_config=RetrievalConfig(),
    )

    assessment = orchestrator.judge_many_with_diagnostics(anchor, [candidate])[0]
    assert assessment.accepted is False
    assert assessment.reject_reason in {"wrong_relation_type", "wrong_direction"}


def test_listed_role_fallback_overrides_low_confidence():
    provider = type("OpenAICompatibleProvider", (FakeProvider,), {})()
    anchor = _brief(
        "doc-12",
        "Subagents",
        "Subagents include document profiler, corpus scout, relation judge, and memory curator roles.",
        ["subagents", "document", "profiler", "corpus", "scout", "relation", "judge", "memory", "curator"],
        ["subagents"],
        ["include", "roles"],
    )
    anchor.metadata["topic"] = "deepagents"
    candidate = _brief(
        "doc-17",
        "Relation Judge",
        "The relation judge verifies whether an anchor and candidate should form a logic edge.",
        ["relation", "judge", "anchor", "candidate", "logic", "edge"],
        ["judge"],
        ["accepted or rejected", "relation type"],
    )
    candidate.metadata["topic"] = "agents"
    verdicts = {
        "doc-17": JudgeResult(
            accepted=True,
            relation_type="prerequisite",
            confidence=0.74,
            evidence_spans=[
                "Subagents include relation judge among the listed roles.",
                "The relation judge verifies whether an anchor and candidate should form a logic edge.",
            ],
            rationale="Relation judge is one of the listed specialized roles.",
            support_score=0.62,
            contradiction_flags=[],
            decision_reason="Low confidence but semantically plausible.",
        )
    }
    orchestrator = LogicOrchestrator(
        doc_profiler=SimpleNamespace(provider=provider),
        corpus_scout=SimpleNamespace(provider=provider),
        relation_judge=FakeJudge(provider, verdicts),
        memory_curator=SimpleNamespace(provider=provider),
        retrieval_config=RetrievalConfig(),
    )

    assessment = orchestrator.judge_many_with_diagnostics(anchor, [candidate])[0]
    assert assessment.accepted is True
    assert assessment.relation_type == "prerequisite"


def test_workflow_fallback_accepts_scout_before_judge_pair():
    provider = type("OpenAICompatibleProvider", (FakeProvider,), {})()
    anchor = _brief(
        "doc-16",
        "Corpus Scout",
        "The corpus scout runs before the relation judge and proposes high-value candidate documents.",
        ["corpus", "scout", "before", "relation", "judge", "candidate"],
        ["scout"],
        ["before", "candidate documents"],
    )
    anchor.metadata["topic"] = "agents"
    candidate = _brief(
        "doc-17",
        "Relation Judge",
        "The relation judge verifies whether an anchor and candidate should form a logic edge.",
        ["relation", "judge", "anchor", "candidate", "logic", "edge"],
        ["judge"],
        ["accepted or rejected", "relation type"],
    )
    candidate.metadata["topic"] = "agents"
    verdicts = {
        "doc-17": JudgeResult(
            accepted=False,
            relation_type="comparison",
            confidence=0.3,
            evidence_spans=[],
            rationale="Model abstained on workflow dependency.",
            support_score=0.1,
            contradiction_flags=[],
            decision_reason="No confident relation chosen.",
        )
    }
    orchestrator = LogicOrchestrator(
        doc_profiler=SimpleNamespace(provider=provider),
        corpus_scout=SimpleNamespace(provider=provider),
        relation_judge=FakeJudge(provider, verdicts),
        memory_curator=SimpleNamespace(provider=provider),
        retrieval_config=RetrievalConfig(),
    )

    assessment = orchestrator.judge_many_with_diagnostics(anchor, [candidate])[0]
    assert assessment.accepted is True
    assert assessment.relation_type == "prerequisite"


def test_supporting_evidence_rejects_foundational_similarity_candidate():
    provider = type("OpenAICompatibleProvider", (FakeProvider,), {})()
    anchor = _brief(
        "doc-09",
        "Jump Policy",
        "The jump policy gates one-hop expansion using confidence, edge-card matching, and target relevance.",
        ["jump", "policy", "confidence", "edge-card", "relevance", "expansion"],
        relation_hints=["one-hop expansion", "target relevance"],
    )
    anchor.metadata["topic"] = "logic"
    candidate = _brief(
        "doc-06",
        "Cosine Similarity",
        "Cosine similarity is a standard metric for vector retrieval and relevance scoring.",
        ["cosine", "similarity", "metric", "retrieval", "relevance"],
        relation_hints=["vector retrieval", "relevance scoring", "hybrid retrieval", "logic overlay"],
    )
    candidate.metadata["topic"] = "retrieval"
    verdicts = {
        "doc-06": JudgeResult(
            accepted=True,
            relation_type="supporting_evidence",
            confidence=0.9,
            evidence_spans=["cosine similarity can gate logical expansion and target relevance scoring"],
            rationale="The similarity metric constrains relevance matching.",
            support_score=0.82,
            contradiction_flags=[],
            decision_reason="Base metric appears relevant to target scoring.",
        )
    }
    orchestrator = LogicOrchestrator(
        doc_profiler=SimpleNamespace(provider=provider),
        corpus_scout=SimpleNamespace(provider=provider),
        relation_judge=FakeJudge(provider, verdicts),
        memory_curator=SimpleNamespace(provider=provider),
        retrieval_config=RetrievalConfig(),
    )

    assessment = orchestrator.judge_many_with_diagnostics(anchor, [candidate])[0]
    assert assessment.accepted is False
    assert assessment.reject_reason == "wrong_relation_type"


def test_ops_registry_stage_bridge_accepts_shared_registry_implementation_detail():
    provider = type("OpenAICompatibleProvider", (FakeProvider,), {})()
    anchor = _brief(
        "doc-20",
        "Jobs and Workers",
        "Offline build, profiling, discovery, and revalidation tasks run as background jobs with a lightweight worker pool and SQLite registry.",
        ["jobs", "workers", "background", "build", "discovery", "sqlite", "registry"],
        ["jobs", "sqlite registry"],
        ["background jobs", "sqlite registry"],
        {"topic": "ops"},
    )
    candidate = _brief(
        "doc-22",
        "SQLite Job Registry",
        "A SQLite job registry stores job ID, type, state, payload, timestamps, and recent messages as a simple queue replacement.",
        ["sqlite", "job", "registry", "payload", "timestamps", "messages"],
        ["sqlite registry"],
        ["queue replacement", "job inspection"],
        {"topic": "ops"},
    )
    verdicts = {
        "doc-22": JudgeResult(
            accepted=True,
            relation_type="implementation_detail",
            confidence=0.88,
            evidence_spans=[
                "Background jobs use a SQLite registry.",
                "The SQLite job registry stores job state and payload for the worker system.",
            ],
            rationale="The registry is the concrete persistence layer used by background jobs.",
            support_score=0.74,
            contradiction_flags=[],
            decision_reason="Shared registry mechanism is explicit.",
        )
    }
    orchestrator = LogicOrchestrator(
        doc_profiler=SimpleNamespace(provider=provider),
        corpus_scout=SimpleNamespace(provider=provider),
        relation_judge=FakeJudge(provider, verdicts),
        memory_curator=SimpleNamespace(provider=provider),
        retrieval_config=RetrievalConfig(),
    )

    assessment = orchestrator.judge_many_with_diagnostics(anchor, [candidate])[0]
    assert assessment.accepted is True
    assert assessment.relation_type == "implementation_detail"


def test_ops_service_registry_fallback_overrides_contradiction_flag():
    provider = type("OpenAICompatibleProvider", (FakeProvider,), {})()
    anchor = _brief(
        "doc-21",
        "FastAPI Service",
        "Describes a FastAPI service exposing endpoints for build, search, revalidation, health checks, and job inspection, with synchronous search and scheduled build tasks.",
        ["public service", "synchronous search", "job registry", "health checks", "fastapi", "service", "build", "search", "checks", "describes"],
        ["FastAPI service", "build endpoints", "search", "job inspection", "sqlite registry"],
        ["Uses SQLite Job Registry for task scheduling.", "Provides interface for Jobs and Workers tasks.", "build", "search", "endpoints", "exposes", "ops", "service"],
        {"topic": "ops"},
    )
    candidate = _brief(
        "doc-22",
        "SQLite Job Registry",
        "Explains a SQLite job registry storing job ID, type, state, payload, timestamps, and messages as a simple queue replacement for single-node deployment.",
        ["payload", "timestamps", "recent messages", "single-node deployment", "registry", "sqlite", "deployment", "explains", "messages", "queue"],
        ["SQLite job registry", "job id", "job type", "job state", "sqlite registry"],
        ["Directly used by Jobs and Workers for task management.", "Supports FastAPI Service job inspection endpoints.", "queue", "registry", "replacement", "simple", "ops"],
        {"topic": "ops"},
    )
    verdicts = {
        "doc-22": JudgeResult(
            accepted=True,
            relation_type="implementation_detail",
            confidence=0.9,
            evidence_spans=[
                "The service schedules expensive tasks through the job registry.",
                "The SQLite job registry stores job state and payload.",
            ],
            rationale="The registry is the concrete persistence layer for service-triggered jobs.",
            support_score=0.78,
            contradiction_flags=["surface_overlap"],
            decision_reason="Shared surfaces create mild ambiguity.",
        )
    }
    orchestrator = LogicOrchestrator(
        doc_profiler=SimpleNamespace(provider=provider),
        corpus_scout=SimpleNamespace(provider=provider),
        relation_judge=FakeJudge(provider, verdicts),
        memory_curator=SimpleNamespace(provider=provider),
        retrieval_config=RetrievalConfig(),
    )

    assessment = orchestrator.judge_many_with_diagnostics(anchor, [candidate])[0]
    assert assessment.accepted is True
    assert assessment.relation_type == "implementation_detail"


def test_logic_graph_keeps_detail_and_support_pair():
    provider = type("OpenAICompatibleProvider", (FakeProvider,), {})()
    anchor = _brief(
        "doc-08",
        "Logic Overlay Graph",
        "Logic overlay graph stores offline-discovered document relations as a sidecar to HNSW, used after initial recall for enhanced retrieval.",
        ["logic overlay", "document relations", "sidecar graph", "initial recall", "graph", "logic", "overlay", "after", "document", "enhanced"],
        ["Logic Overlay Graph", "Document Relations", "Sidecar Graph", "logic graph"],
        ["Connects to doc-06 (cosine), doc-07 (hybrid retrieval)", "graph", "acts", "after", "document-to-document", "logic", "recall"],
        {"topic": "logic"},
    )
    policy = _brief(
        "doc-09",
        "Jump Policy",
        "Explains the jump policy for one-hop logical expansion, gating by confidence, edge-card cosine, and target relevance.",
        ["one-hop expansion", "confidence", "edge-card cosine", "target relevance", "jump", "policy", "confidence", "edge-card", "gating", "explains"],
        ["Jump Policy", "Confidence", "Target Relevance"],
        ["Connects to doc-08 (graph), doc-10 (fusion)", "doc-08", "doc-10", "gating", "jump", "logic", "policy"],
        {"topic": "logic"},
    )
    overview = _brief(
        "doc-07",
        "Hybrid Retrieval",
        "Describes hybrid retrieval combining HNSW geometric recall with one-hop logical expansion from a sidecar graph, then ranking merged candidates.",
        ["hybrid retrieval", "HNSW recall", "logic expansion", "candidate fusion", "hybrid", "retrieval", "hnsw", "logic", "overview", "describes"],
        ["Hybrid Retrieval", "HNSW", "logic graph"],
        ["Builds on doc-08 graph and doc-10 fusion", "builds", "candidate", "doc-08", "doc-10", "logic", "retrieval", "sidecar"],
        {"topic": "retrieval"},
    )
    verdicts = {
        "doc-09": JudgeResult(
            accepted=True,
            relation_type="implementation_detail",
            confidence=0.93,
            evidence_spans=["The jump policy gates one-hop expansion.", "The logic graph is queried through jump-policy gates."],
            rationale="Policy is the concrete gating mechanism used by the logic graph.",
            support_score=0.86,
            contradiction_flags=[],
            decision_reason="Mechanism relation is explicit.",
        ),
        "doc-07": JudgeResult(
            accepted=True,
            relation_type="supporting_evidence",
            confidence=0.88,
            evidence_spans=["Hybrid retrieval consumes logic-graph expansions.", "The logic graph provides one-hop logical candidates after recall."],
            rationale="The overview depends on the logic graph for logical expansion.",
            support_score=0.7,
            contradiction_flags=[],
            decision_reason="Downstream overview is supported by the graph layer.",
        ),
    }
    orchestrator = LogicOrchestrator(
        doc_profiler=SimpleNamespace(provider=provider),
        corpus_scout=SimpleNamespace(provider=provider),
        relation_judge=FakeJudge(provider, verdicts),
        memory_curator=SimpleNamespace(provider=provider),
        retrieval_config=RetrievalConfig(),
    )

    assessments = orchestrator.judge_many_with_diagnostics(anchor, [policy, overview])
    accepted = [item.candidate_doc_id for item in assessments if item.accepted]
    assert accepted == ["doc-09", "doc-07"]


def test_parse_json_tolerates_fenced_json():
    provider = OpenAICompatibleProvider.__new__(OpenAICompatibleProvider)
    payload = provider._parse_json("```json\n{\"accepted\": true, \"confidence\": 0.9}\n```")
    assert payload["accepted"] is True
    assert payload["confidence"] == 0.9


def test_generic_discovery_uses_content_signals_without_dataset_shortcuts():
    provider = FakeProvider()
    orchestrator = LogicOrchestrator(
        doc_profiler=SimpleNamespace(provider=provider),
        corpus_scout=SimpleNamespace(provider=provider),
        relation_judge=FakeJudge(provider, {}),
        memory_curator=SimpleNamespace(provider=provider),
        retrieval_config=RetrievalConfig(),
    )
    scientific = _brief(
        "sci-1",
        "Dietary study on metabolic disease risk",
        "This study reports evidence that dietary exposure changes disease risk in a population cohort.",
        ["study", "dietary", "disease", "risk", "population"],
        ["metabolic", "health"],
        ["claim", "evidence", "study"],
        {"source_dataset": "custom"},
    )
    argument = _brief(
        "arg-1",
        "Opposing claims in policy debate",
        "The claim contrasts one policy position with an opposing alternative in the same debate.",
        ["policy", "debate", "claim", "opposing", "alternative"],
        ["speech"],
        ["comparison", "contrast"],
        {"source_dataset": "custom", "topic_cluster": "speech", "stance": "pro"},
    )
    generic = _brief(
        "gen-1",
        "Random utility script",
        "A short helper function with no retrieval or evidence structure.",
        ["helper", "script"],
        [],
        [],
        {"source_dataset": "custom"},
    )

    assert orchestrator._doc_stage(scientific) == "scientific_evidence"
    assert orchestrator._doc_stage(argument) == "argument_claim"
    assert orchestrator.should_attempt_discovery(scientific) is True
    assert orchestrator.should_attempt_discovery(argument) is True
    assert orchestrator.discovery_anchor_priority(scientific) > orchestrator.discovery_anchor_priority(generic)
