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
    )
    store.add_edges([edge])
    assert store.get_out_edges("a")[0].dst_doc_id == "b"
