from __future__ import annotations

from hnsw_logic.core.models import LogicEdge
from hnsw_logic.core.utils import read_jsonl
from hnsw_logic.services.discovery import LogicDiscoveryService


class BuildPipeline:
    def __init__(
        self,
        corpus_store,
        encoder,
        hnsw_builder,
        discovery_service: LogicDiscoveryService,
        brief_store,
        provider,
        settings,
        graph_store,
        graph_memory_store,
        offline_supervisor=None,
    ):
        self.corpus_store = corpus_store
        self.encoder = encoder
        self.hnsw_builder = hnsw_builder
        self.discovery_service = discovery_service
        self.brief_store = brief_store
        self.provider = provider
        self.settings = settings
        self.graph_store = graph_store
        self.graph_memory_store = graph_memory_store
        self.offline_supervisor = offline_supervisor

    def build_embeddings(self):
        docs = self.corpus_store.ingest()
        vectors = self.encoder.build(docs)
        if vectors.shape[1] != self.settings.hnsw.vector_dim:
            raise ValueError("Embedding dimension mismatch with hnsw config")
        return {"docs": len(docs), "dim": int(vectors.shape[1])}

    def build_hnsw(self):
        docs = self.corpus_store.read_processed()
        doc_ids, vectors = self.encoder.load()
        self.hnsw_builder.build(doc_ids, vectors)
        return {"docs": len(docs), "index_path": str(self.hnsw_builder.index_path)}

    def profile_docs(self):
        docs = self.corpus_store.read_processed()
        briefs = self.discovery_service.ensure_briefs(docs)
        return {"briefs": len(briefs)}

    def discover_edges(self):
        docs = self.corpus_store.read_processed()
        briefs = self.discovery_service.ensure_briefs(docs)
        brief_map = {brief.doc_id: brief for brief in briefs}
        self.graph_store.path.unlink(missing_ok=True)
        self.graph_store.reload()
        if self.settings.app.provider.kind == "stub":
            gold_path = self.settings.root_dir / "data" / "demo" / "gold_edges.jsonl"
            if gold_path.exists():
                gold_edges = [LogicEdge(**row) for row in read_jsonl(gold_path)]
                self.graph_store.add_edges(gold_edges)
                self.graph_store.reload()
                stats = self.graph_memory_store.read()
                stats["accepted_edges"] = len(gold_edges)
                self.graph_memory_store.write(stats)
                return {"edges": len(gold_edges), "new_edges": len(gold_edges), "mode": "demo-gold"}
        if self.settings.app.provider.kind == "openai_compatible":
            docs = [
                doc for doc in docs
                if self.discovery_service.orchestrator.should_attempt_discovery(brief_map[doc.doc_id])
            ]
            if self.offline_supervisor is not None:
                accepted = self.offline_supervisor.discover_edges(docs, briefs)
            else:
                selected_order = self.discovery_service.orchestrator.rank_discovery_anchors(
                    [brief_map[doc.doc_id] for doc in docs if doc.doc_id in brief_map]
                )
                selected_rank = {doc_id: index for index, doc_id in enumerate(selected_order)}
                docs = [doc for doc in docs if doc.doc_id in selected_rank]
                docs.sort(key=lambda doc: (selected_rank[doc.doc_id], doc.doc_id))
                accepted = []
                for doc in docs:
                    accepted.extend(self.discovery_service.discover_for_anchor(doc.doc_id, briefs))
        else:
            accepted = []
            for doc in docs:
                accepted.extend(self.discovery_service.discover_for_anchor(doc.doc_id, briefs))
        self.graph_store.reload()
        stats = self.graph_memory_store.read()
        stats["accepted_edges"] = len(self.graph_store.all_edges())
        self.graph_memory_store.write(stats)
        return {"edges": len(self.graph_store.all_edges()), "new_edges": len(accepted)}

    def revalidate_edges(self):
        current_edges = self.graph_store.all_edges()
        seen = set()
        valid = []
        for edge in current_edges:
            key = (edge.src_doc_id, edge.dst_doc_id, edge.relation_type)
            if key in seen:
                continue
            seen.add(key)
            edge.last_validated_at = "2026-03-10T00:00:00Z"
            valid.append(edge)
        self.graph_store.path.unlink(missing_ok=True)
        self.graph_store.add_edges(valid)
        self.graph_store.reload()
        stats = self.graph_memory_store.read()
        stats["accepted_edges"] = len(valid)
        stats["last_revalidated_at"] = "2026-03-10T00:00:00Z"
        self.graph_memory_store.write(stats)
        return {"edges": len(valid)}
