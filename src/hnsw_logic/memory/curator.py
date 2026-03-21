from __future__ import annotations

from hnsw_logic.core.models import AnchorMemory, DocBrief, GlobalSemanticMemory, LogicEdge


class MemoryCuratorService:
    @staticmethod
    def _normalized_items(values, *, limit: int) -> list[str]:
        items: list[str] = []
        seen: set[str] = set()
        for value in values or []:
            text = str(value).strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            items.append(text)
            if len(items) >= limit:
                break
        return items

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
        anchor_memory.active_hypotheses = self._normalized_items(
            provider_payload.get("active_hypotheses", anchor_brief.relation_hints[:3]),
            limit=6,
        )
        anchor_memory.successful_queries = self._normalized_items(provider_payload.get("successful_queries", []), limit=8)
        anchor_memory.failed_queries = self._normalized_items(provider_payload.get("failed_queries", []), limit=8)
        anchor_memory.rejection_reasons.update(rejection_reasons or {})
        for doc_id, score in (top_candidate_scores or {}).items():
            anchor_memory.top_candidate_scores[doc_id] = max(anchor_memory.top_candidate_scores.get(doc_id, 0.0), score)
        for doc_id, score in (accepted_edge_scores or {}).items():
            anchor_memory.accepted_edge_scores[doc_id] = max(anchor_memory.accepted_edge_scores.get(doc_id, 0.0), score)

        aliases = provider_payload.get("aliases", {})
        for entity, items in aliases.items():
            semantic_memory.aliases[entity] = self._normalized_items(
                [*semantic_memory.aliases.get(entity, []), *items],
                limit=8,
            )
            semantic_memory.canonical_entities.setdefault(entity, entity)

        for relation_type, patterns in provider_payload.get("relation_patterns", {}).items():
            semantic_memory.relation_patterns[relation_type] = self._normalized_items(
                [*semantic_memory.relation_patterns.get(relation_type, []), *patterns],
                limit=12,
            )

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
