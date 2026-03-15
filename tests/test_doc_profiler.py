from __future__ import annotations

import json

from hnsw_logic.config.schema import ProviderConfig
from hnsw_logic.core.models import DocBrief, DocRecord
from hnsw_logic.embedding.provider import JudgeSignals, OpenAICompatibleProvider, ProviderBase


def test_doc_profiler_returns_expected_fields(app_container):
    profiler = app_container.agent_factory.create_doc_profiler()
    brief = profiler.run(
        DocRecord(doc_id="x", title="Test Doc", text="DeepAgents memory and HNSW logic overlay retrieval.", metadata={})
    )
    assert brief.doc_id == "x"
    assert brief.summary
    assert "deepagents" in brief.entities
    assert brief.keywords


def test_doc_profiler_enriches_search_facets(app_container):
    profiler = app_container.agent_factory.create_doc_profiler()
    brief = profiler.run(
        DocRecord(
            doc_id="svc",
            title="FastAPI Service",
            text="The public service exposes search endpoints and build endpoints through FastAPI. Expensive jobs go through a SQLite registry.",
            metadata={"topic": "ops"},
        )
    )
    assert brief.metadata["stage"] == "ops_service"
    assert brief.metadata["doc_kind"] == "service"
    assert "registry" in brief.keywords
    assert "service" in brief.relation_hints


def test_openai_profile_postprocess_enriches_argumentative_docs():
    provider = OpenAICompatibleProvider.__new__(OpenAICompatibleProvider)
    ProviderBase.__init__(provider, ProviderConfig(kind="openai_compatible"))
    doc = DocRecord(
        doc_id="transport-pro-01",
        title="",
        text="Cities should invest in public transit instead of expanding highways. Transit improves access, lowers congestion, and benefits the public.",
        metadata={"source_dataset": "arguana"},
    )

    brief = provider._postprocess_profile(doc, {"summary": "", "entities": [], "keywords": [], "claims": [], "relation_hints": []})

    assert brief.title.startswith("Cities should invest in public transit")
    assert brief.metadata["source_dataset"] == "arguana"
    assert brief.metadata["topic"] == "argument"
    assert brief.metadata["topic_cluster"]
    assert brief.metadata["stance"] == "pro"
    assert "debate" in brief.relation_hints


def test_openai_profile_postprocess_adds_scientific_risk_facets():
    provider = OpenAICompatibleProvider.__new__(OpenAICompatibleProvider)
    ProviderBase.__init__(provider, ProviderConfig(kind="openai_compatible"))
    doc = DocRecord(
        doc_id="risk-01",
        title="Global risk assessment",
        text=(
            "A global public health study measured dietary and metabolic risk factors, mortality, and chronic disease burden. "
            "Nutrition, BMI, and fasting glucose contributed to deaths and disability outcomes."
        ),
        metadata={"source_dataset": "scifact"},
    )

    brief = provider._postprocess_profile(doc, {"summary": "", "entities": [], "keywords": [], "claims": [], "relation_hints": []})

    assert "population risk" in brief.relation_hints
    assert "chronic disease burden" in brief.relation_hints
    assert "nutrition" in brief.relation_hints
    assert "nutrition" in brief.keywords
    assert brief.metadata["topic"] == "scientific_claims"


def test_profile_docs_content_filter_falls_back_to_local_profiles():
    provider = OpenAICompatibleProvider.__new__(OpenAICompatibleProvider)
    ProviderBase.__init__(provider, ProviderConfig(kind="openai_compatible"))
    provider.require_remote = True
    provider.trace_path = None
    provider._invoke_json = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("Moderation Block content_filter"))

    docs = [
        DocRecord(doc_id="a", title="IL-2 signaling", text="Reduced IL-2 signaling impairs Treg function.", metadata={"source_dataset": "scifact"}),
        DocRecord(doc_id="b", title="Clinical outcomes", text="Patients improved after targeted therapy.", metadata={"source_dataset": "nfcorpus"}),
    ]

    briefs = provider.profile_docs(docs)

    assert [brief.doc_id for brief in briefs] == ["a", "b"]
    assert all(brief.summary for brief in briefs)
    assert briefs[0].metadata["source_dataset"] == "scifact"
    assert briefs[1].metadata["source_dataset"] == "nfcorpus"


def test_judge_relations_content_filter_falls_back_to_local_heuristic():
    provider = OpenAICompatibleProvider.__new__(OpenAICompatibleProvider)
    ProviderBase.__init__(provider, ProviderConfig(kind="openai_compatible"))
    provider.require_remote = True
    provider.trace_path = None
    provider._invoke_json = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("Moderation Block content_filter"))

    anchor = DocBrief(
        doc_id="anchor",
        title="Memory Curator",
        summary="Curates memory artifacts for retrieval.",
        entities=["memory", "retrieval"],
        keywords=["memory", "curation", "retrieval"],
        claims=["Memory curation supports retrieval."],
        relation_hints=["memory", "retrieval"],
        metadata={},
    )
    candidate = DocBrief(
        doc_id="candidate",
        title="Graph Memory",
        summary="Stores edge stats and retrieval memory state.",
        entities=["memory", "retrieval"],
        keywords=["memory", "graph", "retrieval"],
        claims=["Graph memory stores edge state for retrieval."],
        relation_hints=["memory", "retrieval"],
        metadata={},
    )
    signals = JudgeSignals(
        dense_score=0.7,
        sparse_score=0.0,
        overlap_score=0.6,
        content_overlap_score=0.4,
        mention_score=0.4,
        role_listing_score=0.0,
        forward_reference_score=0.0,
        reverse_reference_score=0.0,
        direction_score=0.0,
        local_support=0.7,
        utility_score=0.6,
        best_relation="implementation_detail",
        stage_pair="memory->memory",
        risk_flags=[],
        relation_fit_scores={"implementation_detail": 0.7},
    )

    verdicts = provider.judge_relations_with_signals(anchor, [(candidate, signals)])

    assert verdicts["candidate"].accepted is True
    assert verdicts["candidate"].relation_type == "implementation_detail"


def test_profile_docs_malformed_batch_retries_single_remote_calls():
    provider = OpenAICompatibleProvider.__new__(OpenAICompatibleProvider)
    ProviderBase.__init__(provider, ProviderConfig(kind="openai_compatible"))
    provider.require_remote = True
    provider.trace_path = None

    calls: list[tuple[str, str]] = []

    def fake_invoke_json(system_prompt, user_prompt, **kwargs):
        stage = kwargs.get("stage", "")
        calls.append((stage, user_prompt))
        if stage == "profile_docs_batch":
            raise json.JSONDecodeError("Expecting ',' delimiter", '{"broken": true', 14)
        title_line = next(line for line in user_prompt.splitlines() if line.startswith("title: "))
        doc_id = "a" if title_line.endswith("Alpha") else "b"
        return {
            "summary": f"summary for {doc_id}",
            "entities": [],
            "keywords": [doc_id],
            "claims": [f"claim {doc_id}"],
            "relation_hints": ["hint"],
        }

    provider._invoke_json = fake_invoke_json

    docs = [
        DocRecord(doc_id="a", title="Alpha", text="Alpha text.", metadata={"source_dataset": "scifact"}),
        DocRecord(doc_id="b", title="Beta", text="Beta text.", metadata={"source_dataset": "scifact"}),
    ]

    briefs = provider.profile_docs(docs)

    assert [brief.doc_id for brief in briefs] == ["a", "b"]
    assert any(stage == "profile_docs_batch" for stage, _ in calls)
    assert sum(1 for stage, _ in calls if stage == "profile_doc") == 2


def test_judge_relations_malformed_batch_retries_single_remote_calls():
    provider = OpenAICompatibleProvider.__new__(OpenAICompatibleProvider)
    ProviderBase.__init__(provider, ProviderConfig(kind="openai_compatible"))
    provider.require_remote = True
    provider.trace_path = None

    anchor = DocBrief(
        doc_id="anchor",
        title="Hybrid Retrieval",
        summary="Hybrid retrieval combines dense and sparse scores.",
        entities=["retrieval"],
        keywords=["hybrid", "retrieval"],
        claims=["Hybrid retrieval combines dense and sparse scores."],
        relation_hints=["fusion"],
        metadata={},
    )
    candidate = DocBrief(
        doc_id="candidate",
        title="Candidate Fusion",
        summary="Candidate fusion merges dense and sparse ranking signals.",
        entities=["fusion"],
        keywords=["fusion", "ranking"],
        claims=["Candidate fusion merges dense and sparse ranking signals."],
        relation_hints=["fusion"],
        metadata={},
    )
    signals = JudgeSignals(
        dense_score=0.75,
        sparse_score=0.2,
        overlap_score=0.55,
        content_overlap_score=0.4,
        mention_score=0.3,
        role_listing_score=0.0,
        forward_reference_score=0.0,
        reverse_reference_score=0.0,
        direction_score=0.2,
        local_support=0.7,
        utility_score=0.7,
        best_relation="implementation_detail",
        stage_pair="retrieval->fusion",
        risk_flags=[],
        relation_fit_scores={"implementation_detail": 0.78},
    )

    calls: list[str] = []

    def fake_invoke_json(system_prompt, user_prompt, **kwargs):
        stage = kwargs.get("stage", "")
        calls.append(stage)
        if stage == "judge_relations_batch":
            raise json.JSONDecodeError("Expecting ',' delimiter", '{"broken": true', 14)
        return {
            "accepted": True,
            "canonical_relation": "implementation_detail",
            "semantic_relation_label": "mechanism_detail",
            "confidence": 0.91,
            "utility_score": 0.83,
            "uncertainty": 0.08,
            "evidence_spans": ["dense and sparse ranking signals"],
            "rationale": "The candidate describes the fusion mechanism used by the anchor.",
            "support_score": 0.82,
            "contradiction_flags": [],
            "decision_reason": "Strong mechanism overlap.",
        }

    provider._invoke_json = fake_invoke_json

    verdicts = provider.judge_relations_with_signals(anchor, [(candidate, signals)])

    assert verdicts["candidate"].accepted is True
    assert verdicts["candidate"].relation_type == "implementation_detail"
    assert "judge_relations_batch" in calls
    assert "judge_relation" in calls
