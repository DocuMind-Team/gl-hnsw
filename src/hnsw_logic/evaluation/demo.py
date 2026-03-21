from __future__ import annotations

import time
from math import log2

from pydantic import BaseModel

from hnsw_logic.domain.serialization import read_json, read_jsonl, write_json


class EvaluationMetrics(BaseModel):
    name: str
    recall_at_5: float
    mrr_at_10: float
    ndcg_at_10: float
    avg_latency_ms: float


class EdgeQualityMetrics(BaseModel):
    relation_type: str
    accepted_count: int
    precision: float


class EdgeQualityCase(BaseModel):
    src_doc_id: str
    dst_doc_id: str
    relation_type: str
    edge_card_text: str = ""


class EdgeQualityReport(BaseModel):
    by_relation: list[EdgeQualityMetrics]
    false_positives: list[EdgeQualityCase]
    false_negatives: list[EdgeQualityCase]
    avg_edges_per_anchor: float


class EvaluationReport(BaseModel):
    baseline: EvaluationMetrics
    hybrid: EvaluationMetrics
    hybrid_no_memory_bias: EvaluationMetrics
    accepted_edge_count: int
    edge_precision: float
    edge_quality: EdgeQualityReport


class EvaluationService:
    def __init__(self, retrieval_service, baseline_search_fn, settings, graph_store):
        self.retrieval_service = retrieval_service
        self.baseline_search_fn = baseline_search_fn
        self.settings = settings
        self.graph_store = graph_store

    def _metric_bundle(self, name: str, rows: list[dict], qrels: dict[str, dict[str, int]]) -> EvaluationMetrics:
        recalls = []
        mrrs = []
        ndcgs = []
        latencies = []
        for row in rows:
            query_id = row["query_id"]
            hits = row["hits"]
            rels = qrels[query_id]
            latencies.append(row["latency_ms"])
            retrieved_ids = [hit.doc_id for hit in hits[:5]]
            recalls.append(len(set(retrieved_ids) & set(rels.keys())) / max(1, len(rels)))
            rr = 0.0
            for rank, hit in enumerate(hits[:10], start=1):
                if hit.doc_id in rels:
                    rr = 1.0 / rank
                    break
            mrrs.append(rr)
            dcg = 0.0
            for rank, hit in enumerate(hits[:10], start=1):
                gain = rels.get(hit.doc_id, 0)
                if gain:
                    dcg += (2**gain - 1) / log2(rank + 1)
            ideal = sorted(rels.values(), reverse=True)
            idcg = 0.0
            for rank, gain in enumerate(ideal[:10], start=1):
                idcg += (2**gain - 1) / log2(rank + 1)
            ndcgs.append(dcg / idcg if idcg else 0.0)
        return EvaluationMetrics(
            name=name,
            recall_at_5=sum(recalls) / len(recalls),
            mrr_at_10=sum(mrrs) / len(mrrs),
            ndcg_at_10=sum(ndcgs) / len(ndcgs),
            avg_latency_ms=sum(latencies) / len(latencies),
        )

    def evaluate(self) -> EvaluationReport:
        queries = read_json(self.settings.root_dir / "data" / "demo" / "queries.json", default=[])
        qrels = read_json(self.settings.root_dir / "data" / "demo" / "qrels.json", default={})
        gold_edge_rows = read_jsonl(self.settings.root_dir / "data" / "demo" / "gold_edges.jsonl")
        gold_edges = {(row["src_doc_id"], row["dst_doc_id"], row["relation_type"]) for row in gold_edge_rows}

        baseline_rows = []
        hybrid_rows = []
        hybrid_no_memory_rows = []
        for item in queries:
            start = time.perf_counter()
            baseline = self.baseline_search_fn(item["text"], top_k=10)
            baseline_rows.append({"query_id": item["query_id"], "hits": baseline.hits, "latency_ms": (time.perf_counter() - start) * 1000.0})

            start = time.perf_counter()
            hybrid = self.retrieval_service.search(item["text"], top_k=10)
            hybrid_rows.append({"query_id": item["query_id"], "hits": hybrid.hits, "latency_ms": (time.perf_counter() - start) * 1000.0})

            start = time.perf_counter()
            no_memory = self.retrieval_service.search(item["text"], top_k=10, use_memory_bias=False)
            hybrid_no_memory_rows.append({"query_id": item["query_id"], "hits": no_memory.hits, "latency_ms": (time.perf_counter() - start) * 1000.0})

        live_edge_rows = self.graph_store.all_edges()
        live_edges = {(edge.src_doc_id, edge.dst_doc_id, edge.relation_type) for edge in live_edge_rows}
        precision = len(live_edges & gold_edges) / len(live_edges) if live_edges else 0.0

        relation_metrics: list[EdgeQualityMetrics] = []
        relation_types = sorted({edge.relation_type for edge in live_edge_rows} | {row["relation_type"] for row in gold_edge_rows})
        for relation_type in relation_types:
            accepted = [edge for edge in live_edge_rows if edge.relation_type == relation_type]
            accepted_keys = {(edge.src_doc_id, edge.dst_doc_id, edge.relation_type) for edge in accepted}
            relation_precision = len(accepted_keys & gold_edges) / len(accepted_keys) if accepted_keys else 0.0
            relation_metrics.append(
                EdgeQualityMetrics(
                    relation_type=relation_type,
                    accepted_count=len(accepted_keys),
                    precision=relation_precision,
                )
            )

        false_positives = [
            EdgeQualityCase(
                src_doc_id=edge.src_doc_id,
                dst_doc_id=edge.dst_doc_id,
                relation_type=edge.relation_type,
                edge_card_text=edge.edge_card_text,
            )
            for edge in live_edge_rows
            if (edge.src_doc_id, edge.dst_doc_id, edge.relation_type) not in gold_edges
        ][:5]
        false_negatives = [
            EdgeQualityCase(
                src_doc_id=row["src_doc_id"],
                dst_doc_id=row["dst_doc_id"],
                relation_type=row["relation_type"],
                edge_card_text=row.get("edge_card_text", ""),
            )
            for row in gold_edge_rows
            if (row["src_doc_id"], row["dst_doc_id"], row["relation_type"]) not in live_edges
        ][:5]
        unique_anchors = {edge.src_doc_id for edge in live_edge_rows}
        avg_edges_per_anchor = len(live_edge_rows) / max(1, len(unique_anchors))
        report = EvaluationReport(
            baseline=self._metric_bundle("baseline_hnsw", baseline_rows, qrels),
            hybrid=self._metric_bundle("hybrid_overlay", hybrid_rows, qrels),
            hybrid_no_memory_bias=self._metric_bundle("hybrid_no_memory_bias", hybrid_no_memory_rows, qrels),
            accepted_edge_count=len(live_edges),
            edge_precision=precision,
            edge_quality=EdgeQualityReport(
                by_relation=relation_metrics,
                false_positives=false_positives,
                false_negatives=false_negatives,
                avg_edges_per_anchor=avg_edges_per_anchor,
            ),
        )
        write_json(self.settings.root_dir / "data" / "results" / "evaluation_report.json", report.model_dump(mode="json"))
        return report
