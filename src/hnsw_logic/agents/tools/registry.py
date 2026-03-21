from __future__ import annotations

from typing import Callable

from hnsw_logic.docs.brief_store import BriefStore
from hnsw_logic.graph.store import GraphStore
from hnsw_logic.hnsw.searcher import HnswSearcher
from hnsw_logic.memory.anchor_memory import AnchorMemoryStore
from hnsw_logic.memory.semantic_memory import SemanticMemoryStore
from hnsw_logic.core.utils import to_jsonable
from hnsw_logic.services.corpus import CorpusStore


def build_agent_tools(
    corpus_store: CorpusStore,
    brief_store: BriefStore,
    graph_store: GraphStore,
    anchor_memory_store: AnchorMemoryStore,
    semantic_memory_store: SemanticMemoryStore,
    hnsw_searcher: HnswSearcher | None = None,
) -> dict[str, Callable]:
    def search_summaries(query: str, topn: int = 5) -> list[dict]:
        """Search doc briefs by summary token overlap and return the top matches."""
        query_terms = set(query.lower().split())
        rows = []
        for brief in brief_store.all():
            overlap = len(query_terms & set(brief.summary.lower().split()))
            if overlap:
                rows.append({"doc_id": brief.doc_id, "title": brief.title, "overlap": overlap, "summary": brief.summary})
        rows.sort(key=lambda item: (-item["overlap"], item["doc_id"]))
        return rows[:topn]

    def lookup_entities(entities: list[str], topn: int = 5) -> list[dict]:
        """Lookup briefs that mention the requested entities."""
        wanted = set(item.lower() for item in entities)
        rows = []
        for brief in brief_store.all():
            overlap = sorted(wanted & set(entity.lower() for entity in brief.entities))
            if overlap:
                rows.append({"doc_id": brief.doc_id, "title": brief.title, "entities": overlap})
        rows.sort(key=lambda item: (-len(item["entities"]), item["doc_id"]))
        return rows[:topn]

    def get_hnsw_neighbors(doc_id: str, k: int = 5) -> list[dict]:
        """Return approximate nearest neighbors for a document using the HNSW index."""
        if hnsw_searcher is None:
            return []
        doc_map = {doc.doc_id: doc for doc in corpus_store.read_processed()}
        doc = doc_map.get(doc_id)
        if doc is None:
            return []
        from hnsw_logic.embedding.provider_stub import StubProvider
        from hnsw_logic.config.schema import ProviderConfig

        provider = StubProvider(ProviderConfig(embedding_dim=hnsw_searcher.config.vector_dim))
        query_vector = provider.embed_texts([f"{doc.title}\n{doc.text}"])[0]
        return [to_jsonable(neighbor) for neighbor in hnsw_searcher.search(query_vector, top_k=k + 1) if neighbor.doc_id != doc_id][:k]

    def read_doc_brief(doc_id: str) -> dict | None:
        """Read the stored DocBrief for a document id."""
        brief = brief_store.read(doc_id)
        return to_jsonable(brief) if brief else None

    def read_doc_full(doc_id: str) -> dict | None:
        """Read the full normalized document payload for a document id."""
        for doc in corpus_store.read_processed():
            if doc.doc_id == doc_id:
                return to_jsonable(doc)
        return None

    def load_anchor_memory(doc_id: str) -> dict:
        """Load anchor memory state for a specific document id."""
        return to_jsonable(anchor_memory_store.read(doc_id))

    return {
        "search_summaries": search_summaries,
        "lookup_entities": lookup_entities,
        "get_hnsw_neighbors": get_hnsw_neighbors,
        "read_doc_brief": read_doc_brief,
        "read_doc_full": read_doc_full,
        "load_anchor_memory": load_anchor_memory,
    }
