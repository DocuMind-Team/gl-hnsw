from __future__ import annotations

from pathlib import Path

from hnsw_logic.config.schema import ProviderConfig, RetrievalConfig
from hnsw_logic.core.models import DocBrief, DocRecord, LogicEdge
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


class FakeCorpusStore:
    def __init__(self, docs: list[DocRecord]):
        self._docs = docs
        self._fail_first = False

    def read_processed(self) -> list[DocRecord]:
        if self._fail_first:
            self._fail_first = False
            raise FileNotFoundError
        return list(self._docs)


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
        corpus_store=FakeCorpusStore(
            [
                DocRecord(doc_id="doc-11", title="DeepAgents Overview", text="DeepAgents powers subagents and shell execution."),
                DocRecord(doc_id="doc-14", title="Long Term Memory", text="Persistent storage routes memories through a composite backend on disk."),
            ]
        ),
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
        corpus_store=FakeCorpusStore(
            [
                DocRecord(doc_id="doc-04", title="Evaluation Metrics", text="The benchmark reports recall, MRR, and NDCG."),
                DocRecord(doc_id="doc-15", title="Document Profiler", text="The profiler produces a DocBrief with title, summary, entities, keywords, claims, and relation hints."),
            ]
        ),
    )

    response = service.search("What does the document profiler produce?", top_k=3, use_memory_bias=False)
    by_id = {hit.doc_id: hit for hit in response.hits}

    assert "doc-15" in by_id
    assert by_id["doc-15"].source_kind == "supplemental"
    assert by_id["doc-15"].final_score > 0.35


def test_search_can_boost_dense_result_with_sparse_raw_text(tmp_path: Path):
    provider = StubProvider(ProviderConfig(kind="stub"))
    retrieval_config = RetrievalConfig()
    briefs = [
        _brief(
            "doc-a",
            "General System Notes",
            "A broad overview of system behavior.",
            metadata={"topic": "general"},
        ),
        _brief(
            "doc-b",
            "Latency Tuning",
            "A short note about optimizations.",
            metadata={"topic": "general"},
        ),
    ]
    service = HybridRetrievalService(
        searcher=FakeSearcher(
            [
                Neighbor(doc_id="doc-a", score=0.66, rank=1),
                Neighbor(doc_id="doc-b", score=0.61, rank=2),
            ]
        ),
        brief_store=FakeBriefStore(briefs),
        graph_store=GraphStore(tmp_path / "accepted_edges.jsonl"),
        scorer=RetrievalScorer(provider, retrieval_config),
        jump_policy=JumpPolicy(retrieval_config),
        semantic_memory_store=None,
        corpus_store=FakeCorpusStore(
            [
                DocRecord(doc_id="doc-a", title="General System Notes", text="A broad overview of system behavior."),
                DocRecord(doc_id="doc-b", title="Latency Tuning", text="Vector search latency tuning uses ef_search caching and query expansion guards."),
            ]
        ),
    )

    response = service.search("How do we tune vector search latency?", top_k=2, use_memory_bias=False)
    by_id = {hit.doc_id: hit for hit in response.hits}

    assert "doc-b" in by_id
    assert by_id["doc-b"].source_kind == "geometric"
    assert by_id["doc-b"].final_score > 0.61


def test_search_rejects_sparse_only_candidate_when_dense_sparse_disagree(tmp_path: Path):
    provider = StubProvider(ProviderConfig(kind="stub"))
    retrieval_config = RetrievalConfig()
    briefs = [
        _brief(
            "doc-a",
            "Vector Index Notes",
            "Notes about dense retrieval tuning.",
            claims=["Dense retrieval tuning notes."],
            metadata={"topic": "general"},
        ),
        _brief(
            "doc-b",
            "Candidate Ranking",
            "Notes about final ranking.",
            claims=["Final ranking notes."],
            metadata={"topic": "general"},
        ),
        _brief(
            "doc-c",
            "Culture Debate",
            "A generic cultural discussion.",
            claims=["A generic cultural discussion."],
            metadata={"topic": "general"},
        ),
    ]
    service = HybridRetrievalService(
        searcher=FakeSearcher(
            [
                Neighbor(doc_id="doc-a", score=0.66, rank=1),
                Neighbor(doc_id="doc-b", score=0.62, rank=2),
            ]
        ),
        brief_store=FakeBriefStore(briefs),
        graph_store=GraphStore(tmp_path / "accepted_edges.jsonl"),
        scorer=RetrievalScorer(provider, retrieval_config),
        jump_policy=JumpPolicy(retrieval_config),
        semantic_memory_store=None,
        corpus_store=FakeCorpusStore(
            [
                DocRecord(doc_id="doc-a", title="Vector Index Notes", text="Dense retrieval tuning for vector indexes."),
                DocRecord(doc_id="doc-b", title="Candidate Ranking", text="Final ranking notes for retrieval systems."),
                DocRecord(doc_id="doc-c", title="Culture Debate", text="Culture debate about tradition."),
            ]
        ),
    )

    response = service.search("culture debate evidence preserving tradition society policy", top_k=3, use_memory_bias=False)
    hit_ids = [hit.doc_id for hit in response.hits]

    assert "doc-c" not in hit_ids


def test_search_retains_dense_top_hits_against_sparse_overwrite(tmp_path: Path):
    provider = StubProvider(ProviderConfig(kind="stub"))
    retrieval_config = RetrievalConfig()
    briefs = [
        _brief("doc-03", "HNSW Parameters", "Parameters control build quality and search latency.", metadata={"topic": "hnsw"}),
        _brief("doc-02", "HNSW Layers", "Layers improve long-range navigation in HNSW.", metadata={"topic": "hnsw"}),
        _brief("doc-01", "HNSW Overview", "Overview of HNSW search.", metadata={"topic": "hnsw"}),
        _brief("doc-04", "HNSW Insert Path", "Insertion walks the hierarchy and preserves the base algorithm.", metadata={"topic": "hnsw"}),
        _brief("doc-24", "Benchmark Reporting", "Reports include latency, recall, and MRR metrics for the benchmark.", metadata={"topic": "evaluation"}),
    ]
    service = HybridRetrievalService(
        searcher=FakeSearcher(
            [
                Neighbor(doc_id="doc-03", score=0.84, rank=1),
                Neighbor(doc_id="doc-02", score=0.81, rank=2),
                Neighbor(doc_id="doc-01", score=0.79, rank=3),
                Neighbor(doc_id="doc-04", score=0.77, rank=4),
                Neighbor(doc_id="doc-24", score=0.74, rank=5),
            ]
        ),
        brief_store=FakeBriefStore(briefs),
        graph_store=GraphStore(tmp_path / "accepted_edges.jsonl"),
        scorer=RetrievalScorer(provider, retrieval_config),
        jump_policy=JumpPolicy(retrieval_config),
        semantic_memory_store=None,
        corpus_store=FakeCorpusStore(
            [
                DocRecord(doc_id="doc-03", title="HNSW Parameters", text="Parameters control HNSW build quality and search latency."),
                DocRecord(doc_id="doc-02", title="HNSW Layers", text="Layers improve long-range navigation in HNSW."),
                DocRecord(doc_id="doc-01", title="HNSW Overview", text="Overview of HNSW search."),
                DocRecord(doc_id="doc-04", title="HNSW Insert Path", text="Insertion walks the hierarchy and preserves the base algorithm."),
                DocRecord(doc_id="doc-24", title="Benchmark Reporting", text="Reports include latency, recall, and MRR metrics for the benchmark."),
            ]
        ),
    )

    response = service.search("Which parameters control HNSW build quality and search latency?", top_k=5, use_memory_bias=False)
    hit_ids = [hit.doc_id for hit in response.hits]

    assert "doc-04" in hit_ids
    assert hit_ids.index("doc-04") < hit_ids.index("doc-24")


def test_search_applies_graph_neighborhood_bonus_to_relevant_seed(tmp_path: Path):
    provider = StubProvider(ProviderConfig(kind="stub"))
    retrieval_config = RetrievalConfig()
    graph_store = GraphStore(tmp_path / "accepted_edges.jsonl")
    graph_store.add_edges(
        [
            LogicEdge(
                src_doc_id="doc-07",
                dst_doc_id="doc-10",
                relation_type="implementation_detail",
                confidence=0.9,
                evidence_spans=[],
                discovery_path=["test"],
                edge_card_text="[REL=implementation_detail] Hybrid Retrieval -> Candidate Fusion",
                created_at="2026-03-10T00:00:00Z",
                last_validated_at="2026-03-10T00:00:00Z",
            )
        ]
    )
    graph_store.reload()
    briefs = [
        _brief(
            "doc-07",
            "Hybrid Retrieval",
            "Hybrid retrieval merges geometric seeds with logical overlay candidates.",
            claims=["Hybrid retrieval uses a logical overlay sidecar."],
            keywords=["hybrid", "retrieval", "logical", "overlay"],
            relation_hints=["overlay", "sidecar"],
            metadata={"topic": "retrieval"},
        ),
        _brief(
            "doc-10",
            "Candidate Fusion",
            "Candidate fusion combines HNSW score and logic score.",
            claims=["Candidate fusion computes weighted ranking from logic paths."],
            keywords=["candidate", "fusion", "logic", "score"],
            relation_hints=["weighted", "logic"],
            metadata={"topic": "logic"},
        ),
        _brief(
            "doc-24",
            "Benchmark Reporting",
            "Benchmark reporting summarizes latency and recall.",
            claims=["Benchmark reports compare HNSW and hybrid retrieval."],
            keywords=["benchmark", "latency", "recall", "hybrid"],
            relation_hints=["report"],
            metadata={"topic": "evaluation"},
        ),
    ]
    service = HybridRetrievalService(
        searcher=FakeSearcher(
            [
                Neighbor(doc_id="doc-24", score=0.74, rank=1),
                Neighbor(doc_id="doc-07", score=0.70, rank=2),
                Neighbor(doc_id="doc-10", score=0.69, rank=3),
            ]
        ),
        brief_store=FakeBriefStore(briefs),
        graph_store=graph_store,
        scorer=RetrievalScorer(provider, retrieval_config),
        jump_policy=JumpPolicy(retrieval_config),
        semantic_memory_store=None,
        corpus_store=FakeCorpusStore(
            [
                DocRecord(doc_id="doc-07", title="Hybrid Retrieval", text="Hybrid retrieval merges geometric seeds with logical overlay candidates."),
                DocRecord(doc_id="doc-10", title="Candidate Fusion", text="Candidate fusion combines HNSW score and logic score."),
                DocRecord(doc_id="doc-24", title="Benchmark Reporting", text="Benchmark reporting summarizes latency and recall."),
            ]
        ),
    )

    response = service.search("What is the role of the logic overlay graph?", top_k=3, use_memory_bias=False)
    hit_ids = [hit.doc_id for hit in response.hits]
    by_id = {hit.doc_id: hit for hit in response.hits}

    assert hit_ids.index("doc-07") < hit_ids.index("doc-10")
    assert by_id["doc-07"].final_score > 0.70


def test_search_matches_baseline_when_strategy_abstains(tmp_path: Path):
    provider = StubProvider(ProviderConfig(kind="stub"))
    retrieval_config = RetrievalConfig()
    briefs = [
        _brief("doc-a", "Dense Winner", "Dense winner summary.", metadata={"topic": "general"}),
        _brief("doc-b", "Sparse Distraction", "Sparse distraction summary.", metadata={"topic": "general"}),
    ]

    class AbstainStrategy:
        def run(self, **kwargs):
            return type(
                "Strategy",
                (),
                {"sparse_gate": 0.0, "allow_sparse_only": False, "graph_gate": 0.0, "rationale": "abstain"},
            )()

    service = HybridRetrievalService(
        searcher=FakeSearcher([Neighbor(doc_id="doc-a", score=0.72, rank=1)]),
        brief_store=FakeBriefStore(briefs),
        graph_store=GraphStore(tmp_path / "accepted_edges.jsonl"),
        scorer=RetrievalScorer(provider, retrieval_config),
        jump_policy=JumpPolicy(retrieval_config),
        semantic_memory_store=None,
        corpus_store=FakeCorpusStore(
            [
                DocRecord(doc_id="doc-a", title="Dense Winner", text="Dense winner summary."),
                DocRecord(doc_id="doc-b", title="Sparse Distraction", text="sparse distraction keyword keyword keyword"),
            ]
        ),
        query_strategy_agent=AbstainStrategy(),
    )

    baseline = service.search_baseline("keyword", top_k=2)
    hybrid = service.search("keyword", top_k=2, use_memory_bias=False)

    assert [hit.doc_id for hit in hybrid.hits] == [hit.doc_id for hit in baseline.hits]


def test_search_refreshes_corpus_cache_after_initial_missing_processed_docs(tmp_path: Path):
    provider = StubProvider(ProviderConfig(kind="stub"))
    retrieval_config = RetrievalConfig()
    briefs = [
        _brief("doc-a", "Dense Winner", "Dense winner summary.", metadata={"topic": "general"}),
        _brief("doc-b", "Sparse Match", "Sparse match summary.", metadata={"topic": "general"}),
    ]
    corpus = FakeCorpusStore(
        [
            DocRecord(doc_id="doc-a", title="Dense Winner", text="Dense winner summary.", metadata={"source_dataset": "scifact"}),
            DocRecord(doc_id="doc-b", title="Sparse Match", text="Protein evidence supports the study claim.", metadata={"source_dataset": "scifact"}),
        ]
    )
    corpus._fail_first = True
    service = HybridRetrievalService(
        searcher=FakeSearcher(
            [
                Neighbor(doc_id="doc-a", score=0.66, rank=1),
                Neighbor(doc_id="doc-b", score=0.61, rank=2),
            ]
        ),
        brief_store=FakeBriefStore(briefs),
        graph_store=GraphStore(tmp_path / "accepted_edges.jsonl"),
        scorer=RetrievalScorer(provider, retrieval_config),
        jump_policy=JumpPolicy(retrieval_config),
        semantic_memory_store=None,
        corpus_store=corpus,
    )

    service.search("Which protein evidence supports the study claim?", top_k=2, use_memory_bias=False)

    assert service._dataset_hint == "scifact"
