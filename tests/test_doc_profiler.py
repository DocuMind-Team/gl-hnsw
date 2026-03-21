from __future__ import annotations

import json
from types import SimpleNamespace

from hnsw_logic.config.schema import ProviderConfig
from hnsw_logic.core.models import DocBrief, DocRecord
from hnsw_logic.embedding.provider import JudgeResult, JudgeSignals, OpenAICompatibleProvider, ProviderBase, StubProvider


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


def test_openai_profile_postprocess_derives_structured_topic_family():
    provider = OpenAICompatibleProvider.__new__(OpenAICompatibleProvider)
    ProviderBase.__init__(provider, ProviderConfig(kind="openai_compatible"))
    doc = DocRecord(
        doc_id="test-culture-ahrtsdlgra-con01a",
        title="",
        text="Artists should be allowed to provoke disgust when challenging taboos and social norms.",
        metadata={"source_dataset": "arguana"},
    )

    brief = provider._postprocess_profile(doc, {"summary": "", "entities": [], "keywords": [], "claims": [], "relation_hints": []})

    assert brief.metadata["topic_family"] == "test-culture-ahrtsdlgra"
    assert brief.metadata["topic_cluster"]


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


def test_profile_source_payload_uses_excerpt_and_top_terms():
    provider = OpenAICompatibleProvider.__new__(OpenAICompatibleProvider)
    ProviderBase.__init__(provider, ProviderConfig(kind="openai_compatible"))
    doc = DocRecord(
        doc_id="arg-1",
        title="Debate over provocative art",
        text=(
            "Art that provokes disgust can still drive social change. "
            "Some readers find the language offensive. "
            "The document argues that taboo-breaking expression matters. "
            "A later sentence adds retrieval utility terms for comparison. "
            "This fifth sentence should be dropped from the excerpt."
        ),
        metadata={"source_dataset": "arguana", "topic": "argument"},
    )

    payload = provider._profile_source_payload(doc)

    assert payload["doc_id"] == "arg-1"
    assert payload["title"] == "Debate over provocative art"
    assert "This fifth sentence should be dropped" not in payload["text_excerpt"]
    assert payload["source_dataset"] == "arguana"
    assert payload["topic_hint"] == "argument"
    assert "comparison" in payload["top_terms"]


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


def test_counterevidence_content_filter_falls_back_to_local_checker():
    provider = OpenAICompatibleProvider.__new__(OpenAICompatibleProvider)
    ProviderBase.__init__(provider, ProviderConfig(kind="openai_compatible"))
    provider.require_remote = True
    provider.trace_path = None
    provider._invoke_json = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("Moderation Block content_filter"))

    anchor = DocBrief(
        doc_id="anchor",
        title="Transit investment should replace highway expansion",
        summary="The anchor argues for transit-first congestion policy.",
        entities=["transit", "congestion"],
        keywords=["transit", "congestion", "policy"],
        claims=["Transit investment is a better congestion strategy."],
        relation_hints=["debate", "comparison"],
        metadata={},
    )
    candidate = DocBrief(
        doc_id="candidate",
        title="Highway expansion remains the best congestion strategy",
        summary="The candidate advocates the opposite highway-first policy.",
        entities=["highway expansion", "congestion"],
        keywords=["highway", "expansion", "congestion", "policy"],
        claims=["Highway expansion is the strongest congestion strategy."],
        relation_hints=["debate", "comparison"],
        metadata={},
    )
    signals = JudgeSignals(
        dense_score=0.42,
        sparse_score=0.35,
        overlap_score=0.38,
        content_overlap_score=0.31,
        mention_score=0.18,
        role_listing_score=0.0,
        forward_reference_score=0.0,
        reverse_reference_score=0.0,
        direction_score=0.0,
        local_support=0.58,
        utility_score=0.66,
        best_relation="comparison",
        stage_pair="argument_claim->argument_claim",
        risk_flags=["near_duplicate"],
        relation_fit_scores={"comparison": 0.71},
    )
    verdict = JudgeResult(
        accepted=True,
        relation_type="comparison",
        confidence=0.84,
        evidence_spans=["The pair offers a reusable policy contrast."],
        rationale="The candidate creates a contrasting stance bridge.",
        support_score=0.68,
        contradiction_flags=[],
        decision_reason="Argumentative contrast bridge.",
    )

    checks = provider.check_counterevidence_many(anchor, [(candidate, signals, verdict)])

    assert checks["candidate"]["keep"] is True
    assert checks["candidate"]["decision_reason"] == "base checker local decision"


def test_counterevidence_many_does_not_run_local_checker_when_batch_succeeds(monkeypatch):
    provider = OpenAICompatibleProvider.__new__(OpenAICompatibleProvider)
    ProviderBase.__init__(provider, ProviderConfig(kind="openai_compatible"))
    provider.require_remote = True
    provider.trace_path = None

    anchor = DocBrief(
        doc_id="anchor",
        title="Transit investment should replace highway expansion",
        summary="The anchor argues for transit-first congestion policy.",
        entities=["transit", "congestion"],
        keywords=["transit", "congestion", "policy"],
        claims=["Transit investment is a better congestion strategy."],
        relation_hints=["debate", "comparison"],
        metadata={},
    )
    candidate = DocBrief(
        doc_id="candidate",
        title="Highway expansion remains the best congestion strategy",
        summary="The candidate advocates the opposite highway-first policy.",
        entities=["highway expansion", "congestion"],
        keywords=["highway", "expansion", "congestion", "policy"],
        claims=["Highway expansion is the strongest congestion strategy."],
        relation_hints=["debate", "comparison"],
        metadata={},
    )
    signals = JudgeSignals(
        dense_score=0.42,
        sparse_score=0.35,
        overlap_score=0.38,
        content_overlap_score=0.31,
        mention_score=0.18,
        role_listing_score=0.0,
        forward_reference_score=0.0,
        reverse_reference_score=0.0,
        direction_score=0.0,
        local_support=0.58,
        utility_score=0.66,
        best_relation="comparison",
        stage_pair="argument_claim->argument_claim",
        risk_flags=["near_duplicate"],
        relation_fit_scores={"comparison": 0.71},
    )
    verdict = JudgeResult(
        accepted=True,
        relation_type="comparison",
        confidence=0.84,
        evidence_spans=["The pair offers a reusable policy contrast."],
        rationale="The candidate creates a contrasting stance bridge.",
        support_score=0.68,
        contradiction_flags=[],
        decision_reason="Argumentative contrast bridge.",
    )

    provider._invoke_json = lambda *args, **kwargs: [
        {
            "candidate_doc_id": "candidate",
            "keep": True,
            "risk_flags": [],
            "counterevidence": [],
            "decision_reason": "remote checker decision",
            "risk_penalty": 0.0,
        }
    ]

    calls = {"count": 0}
    original = ProviderBase.check_counterevidence

    def counted_check(self, *_args, **_kwargs):
        calls["count"] += 1
        return original(self, *_args, **_kwargs)

    monkeypatch.setattr(ProviderBase, "check_counterevidence", counted_check)

    checks = provider.check_counterevidence_many(anchor, [(candidate, signals, verdict)])

    assert checks["candidate"]["decision_reason"] == "remote checker decision"
    assert calls["count"] == 0


def test_counterevidence_normalizes_duplicate_only_comparison_bridge():
    provider = StubProvider(ProviderConfig(kind="stub"))
    anchor = DocBrief(
        doc_id="anchor",
        title="Campuses should restrict hate speech",
        summary="Restricting hate speech protects students in a campus setting.",
        entities=["campus", "speech"],
        keywords=["campus", "speech", "restrict"],
        claims=[],
        relation_hints=["debate", "comparison"],
        metadata={"source_dataset": "arguana", "topic_cluster": "campus-hate-speech", "stance": "pro"},
    )
    candidate = DocBrief(
        doc_id="candidate",
        title="Campuses should not restrict hate speech",
        summary="Open expression matters more than campus restrictions on speech.",
        entities=["campus", "speech"],
        keywords=["campus", "speech", "restrict"],
        claims=[],
        relation_hints=["debate", "comparison"],
        metadata={"source_dataset": "arguana", "topic_cluster": "campus-hate-speech", "stance": "con"},
    )
    signals = JudgeSignals(
        dense_score=0.7,
        sparse_score=0.65,
        overlap_score=0.58,
        content_overlap_score=0.52,
        mention_score=0.33,
        role_listing_score=0.0,
        forward_reference_score=0.0,
        reverse_reference_score=0.0,
        direction_score=0.0,
        local_support=0.61,
        utility_score=0.72,
        best_relation="comparison",
        stage_pair="argument_claim->argument_claim",
        risk_flags=["near_duplicate", "near_duplicate_bridge"],
        relation_fit_scores={"comparison": 0.81},
        topic_cluster_match=1.0,
        stance_contrast=1.0,
        bridge_gain=0.49,
        duplicate_penalty=0.44,
        contrastive_bridge_score=0.84,
    )
    verdict = JudgeResult(
        accepted=True,
        relation_type="comparison",
        confidence=0.88,
        evidence_spans=["The pair offers directly opposed campus-speech policies."],
        rationale="Reusable contrast bridge on the same debate topic.",
        support_score=0.7,
        contradiction_flags=[],
        decision_reason="contrast bridge",
        utility_score=0.74,
        canonical_relation="comparison",
        semantic_relation_label="comparison",
    )

    normalized = provider._normalize_counterevidence_result(
        anchor,
        candidate,
        signals,
        verdict,
        {
            "keep": False,
            "risk_flags": ["near_duplicate", "near_duplicate_bridge", "contradicts the anchor claim"],
            "counterevidence": [],
            "decision_reason": "duplicate only",
            "risk_penalty": 0.8,
        },
    )

    assert normalized["keep"] is True
    assert "near_duplicate" not in normalized["risk_flags"]
    assert normalized["risk_penalty"] <= 0.22


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
        payload = json.loads(user_prompt)
        title = payload["document"]["title"]
        doc_id = "a" if title == "Alpha" else "b"
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


def test_profile_docs_output_limit_batch_splits_then_succeeds():
    provider = OpenAICompatibleProvider.__new__(OpenAICompatibleProvider)
    ProviderBase.__init__(provider, ProviderConfig(kind="openai_compatible"))
    provider.require_remote = True
    provider.trace_path = None

    calls: list[tuple[str, int]] = []

    def fake_invoke_json(system_prompt, user_prompt, **kwargs):
        stage = kwargs.get("stage", "")
        if stage != "profile_docs_batch":
            raise AssertionError(f"unexpected stage {stage}")
        payload = json.loads(user_prompt)
        batch_size = len(payload["documents"])
        calls.append((stage, batch_size))
        if batch_size > 1:
            raise RuntimeError("router_output_limitation: output token rate limit exceeded")
        doc = payload["documents"][0]
        return [
            {
                "doc_id": doc["doc_id"],
                "summary": f"summary for {doc['doc_id']}",
                "entities": [],
                "keywords": [doc["doc_id"]],
                "claims": [f"claim {doc['doc_id']}"],
                "relation_hints": ["hint"],
            }
        ]

    provider._invoke_json = fake_invoke_json

    docs = [
        DocRecord(doc_id="a", title="Alpha", text="Alpha text.", metadata={"source_dataset": "scifact"}),
        DocRecord(doc_id="b", title="Beta", text="Beta text.", metadata={"source_dataset": "scifact"}),
    ]

    briefs = provider.profile_docs(docs)

    assert [brief.doc_id for brief in briefs] == ["a", "b"]
    assert calls[0] == ("profile_docs_batch", 2)
    assert calls[1:] == [("profile_docs_batch", 1), ("profile_docs_batch", 1)]


def test_profile_docs_emits_successful_briefs_via_callback():
    provider = OpenAICompatibleProvider.__new__(OpenAICompatibleProvider)
    ProviderBase.__init__(provider, ProviderConfig(kind="openai_compatible"))
    provider.require_remote = True
    provider.trace_path = None

    def fake_invoke_json(system_prompt, user_prompt, **kwargs):
        payload = json.loads(user_prompt)
        return [
            {
                "doc_id": doc["doc_id"],
                "summary": f"summary for {doc['doc_id']}",
                "entities": [],
                "keywords": [doc["doc_id"]],
                "claims": [f"claim {doc['doc_id']}"],
                "relation_hints": ["hint"],
            }
            for doc in payload["documents"]
        ]

    provider._invoke_json = fake_invoke_json
    seen: list[str] = []
    docs = [
        DocRecord(doc_id="a", title="Alpha", text="Alpha text.", metadata={"source_dataset": "scifact"}),
        DocRecord(doc_id="b", title="Beta", text="Beta text.", metadata={"source_dataset": "nfcorpus"}),
    ]

    briefs = provider.profile_docs(docs, on_brief=lambda brief: seen.append(brief.doc_id))

    assert [brief.doc_id for brief in briefs] == ["a", "b"]
    assert seen == ["a", "b"]


def test_profile_doc_empty_response_falls_back_to_local_profile():
    provider = OpenAICompatibleProvider.__new__(OpenAICompatibleProvider)
    ProviderBase.__init__(provider, ProviderConfig(kind="openai_compatible"))
    provider.require_remote = True
    provider.trace_path = None
    provider._invoke_json = lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("empty response body"))

    doc = DocRecord(
        doc_id="empty",
        title="Hematopoietic stem cells",
        text="Hematopoietic stem cells do not segregate chromosomes asymmetrically.",
        metadata={"source_dataset": "scifact"},
    )

    brief = provider.profile_doc(doc)

    assert brief.doc_id == "empty"
    assert brief.summary
    assert brief.metadata["source_dataset"] == "scifact"


def test_invoke_json_retries_transient_connection_errors(monkeypatch):
    provider = OpenAICompatibleProvider.__new__(OpenAICompatibleProvider)
    ProviderBase.__init__(provider, ProviderConfig(kind="openai_compatible"))
    provider.require_remote = True
    provider.trace_path = None

    calls = {"count": 0}

    class FakeChat:
        def invoke(self, *_args, **_kwargs):
            calls["count"] += 1
            if calls["count"] < 3:
                raise RuntimeError("Connection error: unexpected eof while reading")
            return SimpleNamespace(content='{"ok": true}')

    provider._chat = FakeChat()
    monkeypatch.setattr("hnsw_logic.embedding.provider.time.sleep", lambda *_args, **_kwargs: None)

    payload = provider._invoke_json("system", "user", stage="generic")

    assert payload == {"ok": True}
    assert calls["count"] == 3


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


def test_judge_relations_with_signals_returns_empty_for_empty_candidate_list():
    provider = OpenAICompatibleProvider.__new__(OpenAICompatibleProvider)
    ProviderBase.__init__(provider, ProviderConfig(kind="openai_compatible"))

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

    assert provider.judge_relations_with_signals(anchor, []) == {}


def test_invoke_json_retries_after_empty_remote_response():
    provider = OpenAICompatibleProvider.__new__(OpenAICompatibleProvider)
    ProviderBase.__init__(provider, ProviderConfig(kind="openai_compatible"))
    provider.trace_path = None

    class FakeChat:
        def __init__(self):
            self.calls = 0

        def invoke(self, *_args, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return type("Response", (), {"content": ""})()
            return type("Response", (), {"content": '{"ok": true}'})()

    provider._chat = FakeChat()
    payload = provider._invoke_json("system", "user", stage="generic")

    assert payload == {"ok": True}
    assert provider._chat.calls == 2
