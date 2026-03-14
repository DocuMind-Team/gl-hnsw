from __future__ import annotations

from pathlib import Path

from hnsw_logic.agents.subagents.query_strategy import QueryStrategyAgent
from hnsw_logic.core.models import DocBrief
from hnsw_logic.core.utils import read_jsonl
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


def test_query_strategy_uses_provider_decision_and_writes_trace(tmp_path: Path):
    class FakeProvider:
        def plan_query_strategy(self, payload):
            assert payload["signals"]["dataset_hint"] == "scifact"
            return {
                "mode": "dense_plus_sparse",
                "sparse_gate": 0.88,
                "allow_sparse_only": True,
                "graph_gate": 0.0,
                "sparse_boost": 1.1,
                "novelty_bias": 0.9,
                "reason": "remote agent enabled sparse terminology support",
                "uncertainty": 0.12,
            }

    agent = QueryStrategyAgent(provider=FakeProvider(), trace_path=tmp_path / "query_strategy_traces.jsonl")
    briefs = {
        "d1": _brief("d1", "Study Claim"),
        "d2": _brief("d2", "Evidence Summary"),
    }

    decision = agent.run(
        query="What evidence supports the study claim?",
        dense_rows=[("d1", 0.82, "geometric")],
        sparse_hits=[SparseHit(doc_id="d2", score=0.94)],
        briefs=briefs,
        dataset_hint="scifact",
        graph_available=False,
    )

    traces = read_jsonl(tmp_path / "query_strategy_traces.jsonl")

    assert decision.mode == "dense_plus_sparse"
    assert decision.sparse_boost == 1.1
    assert traces[0]["source"] == "remote"


def test_query_strategy_skips_remote_for_argumentative_dense_only():
    class FakeProvider:
        def __init__(self):
            self.calls = 0

        def plan_query_strategy(self, payload):
            self.calls += 1
            return {"mode": "dense_plus_sparse"}

    provider = FakeProvider()
    agent = QueryStrategyAgent(provider=provider)
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

    assert decision.mode == "dense_only"
    assert provider.calls == 0
