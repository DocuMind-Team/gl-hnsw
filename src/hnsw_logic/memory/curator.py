from __future__ import annotations

from hnsw_logic.core.models import AnchorMemory, DocBrief, GlobalSemanticMemory, LogicEdge


class MemoryCuratorService:
    def merge(
        self,
        anchor_brief: DocBrief,
        anchor_memory: AnchorMemory,
        semantic_memory: GlobalSemanticMemory,
        accepted_edges: list[LogicEdge],
        rejected_docs: list[str],
        provider_payload: dict,
        rejection_reasons: dict[str, str] | None = None,
        top_candidate_scores: dict[str, float] | None = None,
        accepted_edge_scores: dict[str, float] | None = None,
    ) -> tuple[AnchorMemory, GlobalSemanticMemory]:
        anchor_memory.explored_docs = sorted(set(anchor_memory.explored_docs + [edge.dst_doc_id for edge in accepted_edges] + rejected_docs))
        anchor_memory.rejected_docs = sorted(set(anchor_memory.rejected_docs + rejected_docs))
        anchor_memory.accepted_edge_ids = sorted(set(anchor_memory.accepted_edge_ids + [f"{edge.src_doc_id}->{edge.dst_doc_id}" for edge in accepted_edges]))
        anchor_memory.active_hypotheses = provider_payload.get("active_hypotheses", anchor_brief.relation_hints[:3])
        anchor_memory.successful_queries = provider_payload.get("successful_queries", [])
        anchor_memory.failed_queries = provider_payload.get("failed_queries", [])
        anchor_memory.rejection_reasons.update(rejection_reasons or {})
        for doc_id, score in (top_candidate_scores or {}).items():
            anchor_memory.top_candidate_scores[doc_id] = max(anchor_memory.top_candidate_scores.get(doc_id, 0.0), score)
        for doc_id, score in (accepted_edge_scores or {}).items():
            anchor_memory.accepted_edge_scores[doc_id] = max(anchor_memory.accepted_edge_scores.get(doc_id, 0.0), score)

        aliases = provider_payload.get("aliases", {})
        for entity, items in aliases.items():
            semantic_memory.aliases[entity] = sorted(set(semantic_memory.aliases.get(entity, []) + items))
            semantic_memory.canonical_entities.setdefault(entity, entity)

        for relation_type, patterns in provider_payload.get("relation_patterns", {}).items():
            semantic_memory.relation_patterns[relation_type] = sorted(set(semantic_memory.relation_patterns.get(relation_type, []) + patterns))

        if rejected_docs:
            semantic_memory.rejection_patterns.setdefault(anchor_brief.doc_id, [])
            rejected_entries = [
                f"{doc_id}:{(rejection_reasons or {}).get(doc_id, 'rejected')}"
                for doc_id in rejected_docs
            ]
            semantic_memory.rejection_patterns[anchor_brief.doc_id] = sorted(
                set(semantic_memory.rejection_patterns[anchor_brief.doc_id] + rejected_entries)
            )
        return anchor_memory, semantic_memory
