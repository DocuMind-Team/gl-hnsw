from __future__ import annotations

import json
import shutil
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from math import log2
from pathlib import Path

from pydantic import BaseModel

from hnsw_logic.app.container import build_app
from hnsw_logic.config.schema import ProviderConfig
from hnsw_logic.config.settings import load_settings
from hnsw_logic.domain.models import DocRecord
from hnsw_logic.domain.serialization import append_jsonl, read_jsonl, write_json
from hnsw_logic.embedding.providers.stub import StubProvider
from hnsw_logic.evaluation.demo import EvaluationMetrics

BEIR_URL_TEMPLATE = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"


class BeirEvalReport(BaseModel):
    dataset: str
    split: str
    corpus_size: int
    query_count: int
    baseline: EvaluationMetrics
    supplemental: EvaluationMetrics
    improved_recall_queries: list[str]
    degraded_recall_queries: list[str]
    improved_rr_queries: list[str]
    degraded_rr_queries: list[str]
    work_root: str


@dataclass(slots=True)
class BeirDataset:
    corpus: list[DocRecord]
    queries: list[dict]
    qrels: dict[str, dict[str, int]]


def _download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, target.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def download_and_extract_beir_dataset(dataset: str, cache_root: Path) -> Path:
    cache_root.mkdir(parents=True, exist_ok=True)
    dataset_root = cache_root / dataset
    if dataset_root.exists():
        return dataset_root

    zip_path = cache_root / f"{dataset}.zip"
    if not zip_path.exists():
        _download(BEIR_URL_TEMPLATE.format(dataset=dataset), zip_path)

    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(cache_root)
    return dataset_root


def _token_set(text: str) -> set[str]:
    return {token.lower() for token in text.split() if len(token) > 2}


def _sample_corpus(
    corpus: list[DocRecord],
    queries: list[dict],
    qrels: dict[str, dict[str, int]],
    corpus_limit: int,
) -> list[DocRecord]:
    positive_doc_ids = {doc_id for rels in qrels.values() for doc_id in rels}
    if len(positive_doc_ids) >= corpus_limit:
        selected_ids = sorted(positive_doc_ids)
    else:
        query_terms = [_token_set(item["text"]) for item in queries]
        scored = []
        for doc in corpus:
            if doc.doc_id in positive_doc_ids:
                continue
            doc_terms = _token_set(f"{doc.title} {doc.text}")
            max_overlap = max((len(doc_terms & terms) for terms in query_terms), default=0)
            hit_queries = sum(1 for terms in query_terms if doc_terms & terms)
            scored.append((max_overlap, hit_queries, doc.doc_id))
        scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
        remaining = max(corpus_limit - len(positive_doc_ids), 0)
        selected_ids = sorted(positive_doc_ids) + [doc_id for _, _, doc_id in scored[:remaining]]
    selected_set = set(selected_ids)
    return [doc for doc in corpus if doc.doc_id in selected_set]


def load_beir_dataset(
    dataset_root: Path,
    split: str = "test",
    query_limit: int | None = None,
    corpus_limit: int | None = None,
) -> BeirDataset:
    corpus_rows = read_jsonl(dataset_root / "corpus.jsonl")
    corpus = [
        DocRecord(
            doc_id=row["_id"],
            title=str(row.get("title", "")).strip(),
            text=str(row.get("text", "")).strip(),
            metadata={"source_dataset": dataset_root.name},
        )
        for row in corpus_rows
    ]

    qrels: dict[str, dict[str, int]] = {}
    with (dataset_root / "qrels" / f"{split}.tsv").open("r", encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            query_id, corpus_id, score = line.rstrip("\n").split("\t")
            qrels.setdefault(query_id, {})[corpus_id] = int(score)

    queries: list[dict] = []
    with (dataset_root / "queries.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            query_id = row["_id"]
            if query_id not in qrels:
                continue
            queries.append({"query_id": query_id, "text": str(row.get("text", "")).strip()})

    queries.sort(key=lambda item: item["query_id"])
    if query_limit is not None:
        queries = queries[:query_limit]
        qrels = {item["query_id"]: qrels[item["query_id"]] for item in queries}
    if corpus_limit is not None and len(corpus) > corpus_limit:
        corpus = _sample_corpus(corpus, queries, qrels, corpus_limit)
    return BeirDataset(corpus=corpus, queries=queries, qrels=qrels)


def prepare_beir_work_root(repo_root: Path, work_root: Path, corpus: list[DocRecord]) -> None:
    if work_root.exists():
        shutil.rmtree(work_root)
    shutil.copytree(repo_root / "configs", work_root / "configs")
    shutil.copytree(repo_root / ".deepagents", work_root / ".deepagents")
    (work_root / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (work_root / "data" / "demo").mkdir(parents=True, exist_ok=True)
    append_jsonl(work_root / "data" / "raw" / "beir.jsonl", corpus)


def build_stub_briefs(app) -> None:
    docs = app.corpus_store.read_processed()
    provider = StubProvider(ProviderConfig(kind="stub", embedding_dim=app.settings.hnsw.vector_dim))
    for doc in docs:
        app.brief_store.write(provider.profile_doc(doc))


def _should_build_offline_graph(dataset: str) -> bool:
    return dataset.lower() in {"scifact", "nfcorpus", "arguana"}


def _metric_bundle(name: str, rows: list[dict], qrels: dict[str, dict[str, int]]) -> EvaluationMetrics:
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


def _query_metrics(rows: list[dict], qrels: dict[str, dict[str, int]]) -> dict[str, tuple[float, float]]:
    metrics: dict[str, tuple[float, float]] = {}
    for row in rows:
        rels = qrels[row["query_id"]]
        hits = row["hits"]
        recall = len({hit.doc_id for hit in hits[:5]} & set(rels)) / max(1, len(rels))
        rr = 0.0
        for rank, hit in enumerate(hits[:10], start=1):
            if hit.doc_id in rels:
                rr = 1.0 / rank
                break
        metrics[row["query_id"]] = (recall, rr)
    return metrics


def evaluate_beir_dataset(
    repo_root: Path,
    dataset: str,
    *,
    split: str = "test",
    query_limit: int | None = None,
    corpus_limit: int | None = None,
    cache_root: Path | None = None,
    work_root: Path | None = None,
) -> BeirEvalReport:
    cache_root = cache_root or repo_root / "data" / "external" / "beir"
    dataset_root = download_and_extract_beir_dataset(dataset, cache_root)
    beir = load_beir_dataset(dataset_root, split=split, query_limit=query_limit, corpus_limit=corpus_limit)

    work_root = work_root or repo_root / "data" / "benchmarks" / f"beir_{dataset}_{split}"
    prepare_beir_work_root(repo_root, work_root, beir.corpus)

    load_settings.cache_clear()
    app = build_app(work_root)
    app.pipeline.build_embeddings()
    app.pipeline.build_hnsw()
    if app.settings.app.provider.kind == "openai_compatible":
        app.pipeline.profile_docs()
        if _should_build_offline_graph(dataset):
            app.pipeline.discover_edges()
    else:
        build_stub_briefs(app)

    briefs = list(app.brief_store.all())
    if briefs:
        app.retrieval.scorer.preload_views(briefs, ("title", "summary", "claims", "relation", "full"))

    baseline_rows = []
    supplemental_rows = []
    for item in beir.queries:
        start = time.perf_counter()
        baseline = app.retrieval.search_baseline(item["text"], top_k=10)
        baseline_rows.append(
            {
                "query_id": item["query_id"],
                "hits": baseline.hits,
                "latency_ms": (time.perf_counter() - start) * 1000.0,
            }
        )

        start = time.perf_counter()
        supplemental = app.retrieval.search(item["text"], top_k=10, use_memory_bias=False)
        supplemental_rows.append(
            {
                "query_id": item["query_id"],
                "hits": supplemental.hits,
                "latency_ms": (time.perf_counter() - start) * 1000.0,
            }
        )

    baseline_metrics = _metric_bundle(f"{dataset}_baseline", baseline_rows, beir.qrels)
    supplemental_metrics = _metric_bundle(f"{dataset}_supplemental", supplemental_rows, beir.qrels)

    baseline_query_metrics = _query_metrics(baseline_rows, beir.qrels)
    supplemental_query_metrics = _query_metrics(supplemental_rows, beir.qrels)
    improved_recall_queries = sorted(
        query_id for query_id, (base_recall, _) in baseline_query_metrics.items()
        if supplemental_query_metrics[query_id][0] > base_recall
    )
    degraded_recall_queries = sorted(
        query_id for query_id, (base_recall, _) in baseline_query_metrics.items()
        if supplemental_query_metrics[query_id][0] < base_recall
    )
    improved_rr_queries = sorted(
        query_id for query_id, (_, base_rr) in baseline_query_metrics.items()
        if supplemental_query_metrics[query_id][1] > base_rr
    )
    degraded_rr_queries = sorted(
        query_id for query_id, (_, base_rr) in baseline_query_metrics.items()
        if supplemental_query_metrics[query_id][1] < base_rr
    )

    report = BeirEvalReport(
        dataset=dataset,
        split=split,
        corpus_size=len(beir.corpus),
        query_count=len(beir.queries),
        baseline=baseline_metrics,
        supplemental=supplemental_metrics,
        improved_recall_queries=improved_recall_queries,
        degraded_recall_queries=degraded_recall_queries,
        improved_rr_queries=improved_rr_queries,
        degraded_rr_queries=degraded_rr_queries,
        work_root=str(work_root),
    )
    write_json(work_root / "data" / "results" / "benchmark_report.json", report.model_dump(mode="json"))
    return report
