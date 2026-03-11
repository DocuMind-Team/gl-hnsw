from __future__ import annotations

from hnsw_logic.core.models import DocBrief, DocRecord, LogicEdge
from hnsw_logic.docs.brief_store import BriefStore
from hnsw_logic.graph.store import GraphStore
from hnsw_logic.memory.anchor_memory import AnchorMemoryStore
from hnsw_logic.memory.curator import MemoryCuratorService
from hnsw_logic.memory.graph_memory import GraphMemoryStore
from hnsw_logic.memory.semantic_memory import SemanticMemoryStore


class LogicDiscoveryService:
    def __init__(
        self,
        orchestrator,
        brief_store: BriefStore,
        graph_store: GraphStore,
        anchor_memory_store: AnchorMemoryStore,
        semantic_memory_store: SemanticMemoryStore,
        graph_memory_store: GraphMemoryStore,
        curator_service: MemoryCuratorService,
    ):
        self.orchestrator = orchestrator
        self.brief_store = brief_store
        self.graph_store = graph_store
        self.anchor_memory_store = anchor_memory_store
        self.semantic_memory_store = semantic_memory_store
        self.graph_memory_store = graph_memory_store
        self.curator_service = curator_service

    def ensure_briefs(self, docs: list[DocRecord]) -> list[DocBrief]:
        briefs: list[DocBrief] = []
        missing_docs: list[DocRecord] = []
        for doc in docs:
            brief = self.brief_store.read(doc.doc_id)
            if brief is None:
                missing_docs.append(doc)
            else:
                briefs.append(brief)
        if missing_docs:
            profiled = self.orchestrator.profile_many(missing_docs)
            for brief in profiled:
                self.brief_store.write(brief)
                briefs.append(brief)
        briefs.sort(key=lambda brief: brief.doc_id)
        return briefs

    def discover_for_anchor(self, doc_id: str, briefs: list[DocBrief]) -> list[LogicEdge]:
        brief_map = {brief.doc_id: brief for brief in briefs}
        anchor = brief_map[doc_id]
        proposals = self.orchestrator.scout(anchor, briefs)
        candidate_docs = [brief_map[proposal.doc_id] for proposal in proposals if proposal.doc_id in brief_map]
        assessments = self.orchestrator.judge_many_with_diagnostics(anchor, candidate_docs)
        accepted = [item.edge for item in assessments if item.accepted and item.edge is not None]
        accepted_ids = {edge.dst_doc_id for edge in accepted}
        rejected = [item.candidate_doc_id for item in assessments if not item.accepted]
        rejection_reasons = {item.candidate_doc_id: item.reject_reason for item in assessments if not item.accepted}
        top_candidate_scores = {item.candidate_doc_id: round(item.score, 6) for item in assessments}
        accepted_edge_scores = {item.candidate_doc_id: round(item.score, 6) for item in assessments if item.accepted}
        self.graph_store.add_edges(accepted)
        anchor_memory = self.anchor_memory_store.read(anchor.doc_id)
        semantic_memory = self.semantic_memory_store.read()
        provider_payload = self.orchestrator.curate(anchor, accepted, rejected)
        anchor_memory, semantic_memory = self.curator_service.merge(
            anchor_brief=anchor,
            anchor_memory=anchor_memory,
            semantic_memory=semantic_memory,
            accepted_edges=accepted,
            rejected_docs=rejected,
            provider_payload=provider_payload,
            rejection_reasons=rejection_reasons,
            top_candidate_scores=top_candidate_scores,
            accepted_edge_scores=accepted_edge_scores,
        )
        self.anchor_memory_store.write(anchor_memory)
        self.semantic_memory_store.write(semantic_memory)
        stats = self.graph_memory_store.read()
        stats["accepted_edges"] = len(self.graph_store.all_edges())
        self.graph_memory_store.write(stats)
        return accepted
