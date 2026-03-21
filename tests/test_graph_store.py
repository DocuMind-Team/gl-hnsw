from __future__ import annotations

from hnsw_logic.core.models import LogicEdge


def test_graph_store_roundtrip(test_root):
    from hnsw_logic.graph.store import GraphStore

    store = GraphStore(test_root / "data" / "graph" / "accepted_edges.jsonl")
    edge = LogicEdge(
        src_doc_id="a",
        dst_doc_id="b",
        relation_type="supporting_evidence",
        confidence=0.9,
        evidence_spans=["x"],
        discovery_path=["unit"],
        edge_card_text="edge",
        created_at="2026-03-10T00:00:00Z",
        last_validated_at="2026-03-10T00:00:00Z",
        activation_profile={"activation_prior": 0.72, "topic_signature": ["bridge"]},
    )
    store.add_edges([edge])
    persisted = store.get_out_edges("a")[0]
    assert persisted.dst_doc_id == "b"
    assert persisted.activation_profile["activation_prior"] == 0.72


def test_graph_store_deduplicates_edges_by_relation_key(test_root):
    from hnsw_logic.graph.store import GraphStore

    store = GraphStore(test_root / "data" / "graph" / "accepted_edges.jsonl")
    low = LogicEdge(
        src_doc_id="a",
        dst_doc_id="b",
        relation_type="comparison",
        confidence=0.7,
        evidence_spans=["x"],
        discovery_path=["unit"],
        edge_card_text="low",
        created_at="2026-03-10T00:00:00Z",
        last_validated_at="2026-03-10T00:00:00Z",
        utility_score=0.6,
    )
    high = LogicEdge(
        src_doc_id="a",
        dst_doc_id="b",
        relation_type="comparison",
        confidence=0.92,
        evidence_spans=["y"],
        discovery_path=["unit"],
        edge_card_text="high",
        created_at="2026-03-10T00:00:00Z",
        last_validated_at="2026-03-10T00:00:00Z",
        utility_score=0.9,
    )
    store.add_edges([low, high, low])
    store.reload()
    edges = store.get_out_edges("a")
    assert len(edges) == 1
    assert edges[0].edge_card_text == "high"


def test_graph_store_exposes_has_edges_and_edge_count(test_root):
    from hnsw_logic.graph.store import GraphStore

    store = GraphStore(test_root / "data" / "graph" / "accepted_edges.jsonl")

    assert store.has_edges() is False
    assert store.edge_count() == 0

    store.add_edges(
        [
            LogicEdge(
                src_doc_id="a",
                dst_doc_id="b",
                relation_type="comparison",
                confidence=0.9,
                evidence_spans=["x"],
                discovery_path=["unit"],
                edge_card_text="edge",
                created_at="2026-03-10T00:00:00Z",
                last_validated_at="2026-03-10T00:00:00Z",
            )
        ]
    )
    store.reload()

    assert store.has_edges() is True
    assert store.edge_count() == 1


def test_discovery_service_adds_mirror_edges_for_symmetric_relations(test_root):
    from hnsw_logic.graph.store import GraphStore
    from hnsw_logic.services.discovery import LogicDiscoveryService

    store = GraphStore(test_root / "data" / "graph" / "accepted_edges.jsonl")
    service = LogicDiscoveryService(
        orchestrator=None,
        brief_store=None,
        graph_store=store,
        anchor_memory_store=None,
        semantic_memory_store=None,
        graph_memory_store=None,
        curator_service=None,
    )
    edge = LogicEdge(
        src_doc_id="a",
        dst_doc_id="b",
        relation_type="same_concept",
        confidence=0.9,
        evidence_spans=["x"],
        discovery_path=["unit"],
        edge_card_text="edge",
        created_at="2026-03-10T00:00:00Z",
        last_validated_at="2026-03-10T00:00:00Z",
        utility_score=0.8,
    )
    augmented = service._augment_with_mirror_edges([edge])
    assert {(item.src_doc_id, item.dst_doc_id, item.relation_type) for item in augmented} == {
        ("a", "b", "same_concept"),
        ("b", "a", "same_concept"),
    }
