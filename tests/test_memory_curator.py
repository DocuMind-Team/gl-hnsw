from __future__ import annotations

from hnsw_logic.core.models import AnchorMemory, DocBrief, GlobalSemanticMemory
from hnsw_logic.memory.curator import MemoryCuratorService


def test_memory_curator_normalizes_provider_payload_lists():
    service = MemoryCuratorService()
    anchor = DocBrief(
        doc_id="anchor",
        title="Hybrid Retrieval",
        summary="Hybrid retrieval uses logical overlays.",
        entities=["retrieval"],
        keywords=["hybrid", "retrieval"],
        claims=["Hybrid retrieval uses logical overlays."],
        relation_hints=["hybrid", "retrieval"],
        metadata={},
    )
    anchor_memory = AnchorMemory(anchor_doc_id="anchor")
    semantic_memory = GlobalSemanticMemory()

    next_anchor_memory, next_semantic_memory = service.merge(
        anchor_brief=anchor,
        anchor_memory=anchor_memory,
        semantic_memory=semantic_memory,
        accepted_edges=[],
        rejected_docs=[],
        provider_payload={
            "active_hypotheses": ["Hybrid", "hybrid", "", "Retrieval", "retrieval", "Overlay", "Memory", "Graph"],
            "successful_queries": ["query a", "query a", "query b", " ", "query c"],
            "failed_queries": ["bad a", "bad a", "bad b"],
            "aliases": {"retrieval": ["Hybrid Retrieval", "hybrid retrieval", "Retrieval Logic"]},
            "relation_patterns": {"same_concept": ["a->b", "A->B", "c->d"]},
        },
    )

    assert next_anchor_memory.active_hypotheses == ["Hybrid", "Retrieval", "Overlay", "Memory", "Graph"]
    assert next_anchor_memory.successful_queries == ["query a", "query b", "query c"]
    assert next_anchor_memory.failed_queries == ["bad a", "bad b"]
    assert next_semantic_memory.aliases["retrieval"] == ["Hybrid Retrieval", "Retrieval Logic"]
    assert next_semantic_memory.relation_patterns["same_concept"] == ["a->b", "c->d"]
