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
