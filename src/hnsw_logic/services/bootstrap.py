from __future__ import annotations

from dataclasses import dataclass

from hnsw_logic.agents.factory import AgentFactory
from hnsw_logic.agents.tools.registry import build_agent_tools
from hnsw_logic.config.settings import AppSettings, load_settings
from hnsw_logic.docs.brief_store import BriefStore
from hnsw_logic.embedding.encoder import EmbeddingEncoder
from hnsw_logic.embedding.provider import build_provider
from hnsw_logic.graph.store import GraphStore
from hnsw_logic.hnsw.index_builder import HnswIndexBuilder
from hnsw_logic.hnsw.searcher import HnswSearcher
from hnsw_logic.jobs.registry import JobRegistry
from hnsw_logic.memory.anchor_memory import AnchorMemoryStore
from hnsw_logic.memory.curator import MemoryCuratorService
from hnsw_logic.memory.graph_memory import GraphMemoryStore
from hnsw_logic.memory.semantic_memory import SemanticMemoryStore
from hnsw_logic.memory.self_update import ControlledSelfUpdateManager
from hnsw_logic.retrieval.jump_policy import JumpPolicy
from hnsw_logic.retrieval.scorer import RetrievalScorer
from hnsw_logic.retrieval.service import HybridRetrievalService
from hnsw_logic.services.corpus import CorpusStore
from hnsw_logic.services.discovery import LogicDiscoveryService
from hnsw_logic.services.evaluation import EvaluationService
from hnsw_logic.services.offline_supervisor import OfflineIndexingSupervisor
from hnsw_logic.services.pipeline import BuildPipeline


@dataclass(slots=True)
class AppContainer:
    settings: AppSettings
    provider: object
    corpus_store: CorpusStore
    brief_store: BriefStore
    graph_store: GraphStore
    anchor_memory_store: AnchorMemoryStore
    semantic_memory_store: SemanticMemoryStore
    graph_memory_store: GraphMemoryStore
    searcher: HnswSearcher
    discovery_service: LogicDiscoveryService
    retrieval: HybridRetrievalService
    pipeline: BuildPipeline
    evaluation: EvaluationService
    jobs: JobRegistry
    agent_factory: AgentFactory
    offline_supervisor: OfflineIndexingSupervisor


def build_app(root_dir=None) -> AppContainer:
    settings = load_settings(root_dir)
    paths = settings.app.paths
    provider = build_provider(settings.app.provider, settings.root_dir)
    corpus_store = CorpusStore(settings.root_dir / paths.raw_dir, settings.root_dir / paths.processed_dir / "docs.jsonl")
    brief_store = BriefStore(settings.root_dir / paths.memories_dir / "doc_briefs")
    graph_store = GraphStore(settings.root_dir / paths.graph_dir / "accepted_edges.jsonl")
    anchor_memory_store = AnchorMemoryStore(settings.root_dir / paths.memories_dir / "anchor_memory")
    semantic_memory_store = SemanticMemoryStore(
        settings.root_dir / paths.memories_dir / "entity_memory" / "entities.json",
        settings.root_dir / paths.memories_dir / "relation_memory" / "relation_patterns.json",
        settings.root_dir / paths.memories_dir / "relation_memory" / "rejection_patterns.json",
    )
    graph_memory_store = GraphMemoryStore(settings.root_dir / paths.memories_dir / "graph_memory" / "edge_stats.json")
    encoder = EmbeddingEncoder(provider, settings.root_dir / paths.processed_dir / "embeddings.json")
    hnsw_builder = HnswIndexBuilder(
        settings.hnsw,
        settings.root_dir / paths.indices_dir / "docs.bin",
        settings.root_dir / paths.indices_dir / "docs_meta.json",
    )
    searcher = HnswSearcher(
        settings.hnsw,
        settings.root_dir / paths.indices_dir / "docs.bin",
        settings.root_dir / paths.indices_dir / "docs_meta.json",
    )
    tools = build_agent_tools(corpus_store, brief_store, graph_store, anchor_memory_store, semantic_memory_store, searcher)
    agent_factory = AgentFactory(
        agents_config=settings.agents,
        provider_config=settings.app.provider,
        retrieval_config=settings.retrieval,
        provider=provider,
        tools=tools,
        skills_root=settings.root_dir / settings.agents.skills_root,
        workspace_root=settings.root_dir / paths.workspace_dir,
        memories_root=settings.root_dir / paths.memories_dir,
        repo_root=settings.root_dir,
        corpus_store=corpus_store,
        brief_store=brief_store,
        graph_store=graph_store,
        anchor_memory_store=anchor_memory_store,
        semantic_memory_store=semantic_memory_store,
        graph_memory_store=graph_memory_store,
        searcher=searcher,
    )
    orchestrator = agent_factory.create_orchestrator()
    discovery = LogicDiscoveryService(
        orchestrator=orchestrator,
        brief_store=brief_store,
        graph_store=graph_store,
        anchor_memory_store=anchor_memory_store,
        semantic_memory_store=semantic_memory_store,
        graph_memory_store=graph_memory_store,
        curator_service=MemoryCuratorService(),
    )
    self_update_manager = ControlledSelfUpdateManager(settings.root_dir, settings.agents.self_update_allowlist)
    offline_supervisor = OfflineIndexingSupervisor(
        orchestrator=orchestrator,
        discovery_service=discovery,
        deepagent=orchestrator.deepagent,
        runtime_toolsets=agent_factory.runtime_toolsets,
        workspace_root=settings.root_dir / paths.workspace_dir,
        agents_config=settings.agents,
        self_update_manager=self_update_manager,
        agents_memory_path=settings.root_dir / ".deepagents" / "AGENTS.md",
    )
    scorer = RetrievalScorer(provider, settings.retrieval)
    retrieval = HybridRetrievalService(
        searcher=searcher,
        brief_store=brief_store,
        graph_store=graph_store,
        scorer=scorer,
        jump_policy=JumpPolicy(settings.retrieval),
        semantic_memory_store=semantic_memory_store,
        corpus_store=corpus_store,
    )
    pipeline = BuildPipeline(
        corpus_store=corpus_store,
        encoder=encoder,
        hnsw_builder=hnsw_builder,
        discovery_service=discovery,
        brief_store=brief_store,
        provider=provider,
        settings=settings,
        graph_store=graph_store,
        graph_memory_store=graph_memory_store,
        offline_supervisor=offline_supervisor,
    )
    evaluation = EvaluationService(
        retrieval_service=retrieval,
        baseline_search_fn=retrieval.search_baseline,
        settings=settings,
        graph_store=graph_store,
    )
    jobs = JobRegistry(settings.root_dir / paths.jobs_db)
    return AppContainer(
        settings=settings,
        provider=provider,
        corpus_store=corpus_store,
        brief_store=brief_store,
        graph_store=graph_store,
        anchor_memory_store=anchor_memory_store,
        semantic_memory_store=semantic_memory_store,
        graph_memory_store=graph_memory_store,
        searcher=searcher,
        discovery_service=discovery,
        retrieval=retrieval,
        pipeline=pipeline,
        evaluation=evaluation,
        jobs=jobs,
        agent_factory=agent_factory,
        offline_supervisor=offline_supervisor,
    )
