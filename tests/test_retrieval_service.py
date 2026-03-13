from __future__ import annotations

from pathlib import Path

from hnsw_logic.config.schema import ProviderConfig, RetrievalConfig
from hnsw_logic.core.models import DocBrief
from hnsw_logic.embedding.provider import StubProvider
from hnsw_logic.graph.store import GraphStore
from hnsw_logic.hnsw.searcher import Neighbor
from hnsw_logic.retrieval.jump_policy import JumpPolicy
from hnsw_logic.retrieval.scorer import RetrievalScorer
from hnsw_logic.retrieval.service import HybridRetrievalService


class FakeSearcher:
    def __init__(self, neighbors):
        self._neighbors = neighbors

    def search(self, query_vector, top_k: int):
        return self._neighbors[:top_k]


class FakeBriefStore:
    def __init__(self, briefs: list[DocBrief]):
        self._briefs = {brief.doc_id: brief for brief in briefs}

    def all(self) -> list[DocBrief]:
        return list(self._briefs.values())

    def read(self, doc_id: str) -> DocBrief | None:
        return self._briefs.get(doc_id)


def _brief(
    doc_id: str,
    title: str,
    summary: str,
    *,
    claims: list[str] | None = None,
    keywords: list[str] | None = None,
    relation_hints: list[str] | None = None,
    metadata: dict | None = None,
) -> DocBrief:
    return DocBrief(
        doc_id=doc_id,
        title=title,
        summary=summary,
        claims=claims or [summary],
        keywords=keywords or [],
        relation_hints=relation_hints or [],
        metadata=metadata or {},
    )


def test_search_adds_supplemental_seed_for_structured_memory_doc(tmp_path: Path):
    provider = StubProvider(ProviderConfig(kind="stub"))
    retrieval_config = RetrievalConfig()
    briefs = [
        _brief(
            "doc-11",
            "DeepAgents Overview",
            "DeepAgents powers subagents and shell execution.",
            keywords=["deepagents", "subagents", "shell"],
            relation_hints=["overview"],
            metadata={"topic": "deepagents"},
        ),
        _brief(
            "doc-14",
            "Long Term Memory",
            "Persistent storage routes /memories paths through a composite backend on disk.",
            claims=["The system persists anchor memory and semantic memory through a composite backend."],
            keywords=["memory", "persistent", "storage", "backend"],
            relation_hints=["memory", "persistent"],
            metadata={"topic": "deepagents", "stage": "agent_memory", "doc_kind": "memory"},
        ),
    ]
    service = HybridRetrievalService(
        searcher=FakeSearcher([Neighbor(doc_id="doc-11", score=0.62, rank=1)]),
        brief_store=FakeBriefStore(briefs),
        graph_store=GraphStore(tmp_path / "accepted_edges.jsonl"),
        scorer=RetrievalScorer(provider, retrieval_config),
        jump_policy=JumpPolicy(retrieval_config),
        semantic_memory_store=None,
    )

    response = service.search("How is memory persisted for the agent system?", top_k=3, use_memory_bias=False)
    by_id = {hit.doc_id: hit for hit in response.hits}

    assert "doc-14" in by_id
    assert by_id["doc-14"].source_kind == "supplemental"


def test_search_uses_claim_view_for_profiler_query(tmp_path: Path):
    provider = StubProvider(ProviderConfig(kind="stub"))
    retrieval_config = RetrievalConfig()
    briefs = [
        _brief(
            "doc-04",
            "Evaluation Metrics",
            "The benchmark reports recall, MRR, and NDCG.",
            keywords=["metrics", "benchmark", "evaluation"],
            relation_hints=["report"],
            metadata={"topic": "evaluation"},
        ),
        _brief(
            "doc-15",
            "Document Profiler",
            "A profiling role compresses documents for discovery.",
            claims=["The document profiler creates a DocBrief with title, summary, entities, keywords, claims, and relation hints."],
            keywords=["profiler", "docbrief", "claims", "entities"],
            relation_hints=["docbrief", "summary"],
            metadata={"topic": "agents", "stage": "agent_roles", "doc_kind": "role", "role_type": "profiler"},
        ),
    ]
    service = HybridRetrievalService(
        searcher=FakeSearcher([Neighbor(doc_id="doc-04", score=0.64, rank=1)]),
        brief_store=FakeBriefStore(briefs),
        graph_store=GraphStore(tmp_path / "accepted_edges.jsonl"),
        scorer=RetrievalScorer(provider, retrieval_config),
        jump_policy=JumpPolicy(retrieval_config),
        semantic_memory_store=None,
    )

    response = service.search("What does the document profiler produce?", top_k=3, use_memory_bias=False)
    by_id = {hit.doc_id: hit for hit in response.hits}

    assert "doc-15" in by_id
    assert by_id["doc-15"].source_kind == "supplemental"
    assert by_id["doc-15"].final_score > 0.35
