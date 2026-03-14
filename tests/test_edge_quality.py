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


def _brief(doc_id: str, title: str, summary: str, keywords: list[str], entities: list[str] | None = None, relation_hints: list[str] | None = None):
    return DocBrief(
        doc_id=doc_id,
        title=title,
        summary=summary,
        keywords=keywords,
        entities=entities or [],
        relation_hints=relation_hints or [],
        claims=[summary],
        metadata={},
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
        relation_hints=["vector retrieval", "relevance scoring"],
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


def test_parse_json_tolerates_fenced_json():
    provider = OpenAICompatibleProvider.__new__(OpenAICompatibleProvider)
    payload = provider._parse_json("```json\n{\"accepted\": true, \"confidence\": 0.9}\n```")
    assert payload["accepted"] is True
    assert payload["confidence"] == 0.9
