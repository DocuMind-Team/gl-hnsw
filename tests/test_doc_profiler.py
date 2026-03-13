from __future__ import annotations

from hnsw_logic.core.models import DocRecord


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
