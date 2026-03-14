from __future__ import annotations

from hnsw_logic.config.schema import ProviderConfig
from hnsw_logic.core.models import DocRecord
from hnsw_logic.embedding.provider import OpenAICompatibleProvider, ProviderBase


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
