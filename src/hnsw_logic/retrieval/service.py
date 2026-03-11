from __future__ import annotations

from hnsw_logic.core.models import SearchHit, SearchResponse
from hnsw_logic.docs.brief_store import BriefStore
from hnsw_logic.graph.store import GraphStore
from hnsw_logic.hnsw.searcher import HnswSearcher
from hnsw_logic.memory.semantic_memory import SemanticMemoryStore
from hnsw_logic.retrieval.jump_policy import JumpPolicy
from hnsw_logic.retrieval.scorer import ExpandedCandidate, RetrievalScorer


class HybridRetrievalService:
    def __init__(
        self,
        searcher: HnswSearcher,
        brief_store: BriefStore,
        graph_store: GraphStore,
        scorer: RetrievalScorer,
        jump_policy: JumpPolicy,
        semantic_memory_store: SemanticMemoryStore | None = None,
    ):
        self.searcher = searcher
        self.brief_store = brief_store
        self.graph_store = graph_store
        self.scorer = scorer
        self.jump_policy = jump_policy
        self.semantic_memory_store = semantic_memory_store

    def _hits_from_ranked(self, query: str, ranked: list[dict], top_k: int) -> SearchResponse:
        hits = [
            SearchHit(
                doc_id=row["doc_id"],
                title=row["title"],
                final_score=row["final_score"],
                geometric_score=row["geometric_score"],
                logical_score=row["logical_score"],
                source_kind=row["source_kind"],
                via_edge=row["via_edge"],
                summary=row["summary"],
                rank=row["rank"],
            )
            for row in ranked[:top_k]
        ]
        return SearchResponse(query=query, hits=hits)

    def _apply_memory_bias(self, query: str, rows: list[dict]) -> list[dict]:
        if self.semantic_memory_store is None:
            return rows
        memory = self.semantic_memory_store.read()
        query_terms = set(query.lower().split())
        for row in rows:
            bias = 0.0
            brief = self.brief_store.read(row["doc_id"])
            if brief is None:
                continue
            for entity in brief.entities:
                alias_text = " ".join(memory.aliases.get(entity, []))
                alias_tokens = set(alias_text.lower().split()) | {memory.canonical_entities.get(entity, entity).lower()}
                if query_terms & alias_tokens:
                    bias += 0.05
            row["final_score"] += bias
        rows.sort(key=lambda item: (-item["final_score"], item["doc_id"]))
        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank
        return rows

    def search_baseline(self, query: str, top_k: int = 10) -> SearchResponse:
        query_emb = self.scorer.encode_query(query)
        seed_neighbors = self.searcher.search(query_emb, top_k=50)
        briefs = {brief.doc_id: brief for brief in self.brief_store.all()}
        ranked = []
        for neighbor in seed_neighbors[:top_k]:
            brief = briefs.get(neighbor.doc_id)
            if brief is None:
                continue
            ranked.append(
                {
                    "doc_id": neighbor.doc_id,
                    "title": brief.title,
                    "final_score": neighbor.score,
                    "geometric_score": neighbor.score,
                    "logical_score": 0.0,
                    "source_kind": "geometric",
                    "via_edge": None,
                    "summary": brief.summary,
                    "rank": len(ranked) + 1,
                }
            )
        return self._hits_from_ranked(query, ranked, top_k)

    def search(self, query: str, top_k: int = 10, use_memory_bias: bool = True) -> SearchResponse:
        query_emb = self.scorer.encode_query(query)
        seed_neighbors = self.searcher.search(query_emb, top_k=50)
        briefs = {brief.doc_id: brief for brief in self.brief_store.all()}
        seeds = {neighbor.doc_id: (neighbor.score, "geometric") for neighbor in seed_neighbors}
        expanded: list[ExpandedCandidate] = []

        for neighbor in seed_neighbors[: self.jump_policy.max_seeds]:
            for edge in self.graph_store.get_out_edges(neighbor.doc_id)[: self.jump_policy.max_expansions_per_seed]:
                brief = briefs.get(edge.dst_doc_id)
                if brief is None:
                    continue
                edge_emb = self.scorer.edge_embedding(edge)
                target_rel_score = self.scorer.score_target(query_emb, brief)
                if self.jump_policy.allow_jump(query_emb, edge_emb, edge, target_rel_score):
                    expanded.append(
                        ExpandedCandidate(
                            doc_id=edge.dst_doc_id,
                            source_doc_id=neighbor.doc_id,
                            edge=edge,
                            seed_score=neighbor.score,
                            edge_match=float(query_emb.dot(edge_emb)),
                            target_rel_score=target_rel_score,
                        )
                    )
        ranked = self.scorer.rank(query, query_emb, seeds, expanded, briefs, top_k)
        if use_memory_bias:
            ranked = self._apply_memory_bias(query, ranked)
        return self._hits_from_ranked(query, ranked, top_k)
