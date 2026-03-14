from __future__ import annotations

from hnsw_logic.agents.subagents.query_strategy import QueryStrategyAgent
from hnsw_logic.core.models import DocBrief
from hnsw_logic.retrieval.sparse import SparseHit


def _brief(doc_id: str, title: str) -> DocBrief:
    return DocBrief(doc_id=doc_id, title=title, summary=title, claims=[title], metadata={})


def test_query_strategy_limits_sparse_for_argumentative_queries():
    agent = QueryStrategyAgent()
    briefs = {
        "d1": _brief("d1", "General Topic"),
        "d2": _brief("d2", "Another Topic"),
        "d3": _brief("d3", "Debate Evidence"),
    }
    decision = agent.run(
        query="culture debate argument about policy and society",
        dense_rows=[("d1", 0.82, "geometric"), ("d2", 0.77, "geometric")],
        sparse_hits=[SparseHit(doc_id="d3", score=1.0)],
        briefs=briefs,
        dataset_hint="arguana",
        graph_available=False,
    )

    assert decision.sparse_gate < 0.3
    assert decision.allow_sparse_only is False
    assert decision.graph_gate == 0.0


def test_query_strategy_keeps_sparse_and_graph_for_technical_queries():
    agent = QueryStrategyAgent()
    briefs = {
        "d1": _brief("d1", "Hybrid Retrieval"),
        "d2": _brief("d2", "Candidate Fusion"),
    }
    decision = agent.run(
        query="How are hybrid retrieval scores fused with graph expansion?",
        dense_rows=[("d1", 0.84, "geometric"), ("d2", 0.8, "geometric")],
        sparse_hits=[SparseHit(doc_id="d2", score=1.0), SparseHit(doc_id="d1", score=0.93)],
        briefs=briefs,
        dataset_hint="gl_hnsw_demo",
        graph_available=True,
    )

    assert decision.sparse_gate > 0.5
    assert decision.graph_gate > 0.5
