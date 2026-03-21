from __future__ import annotations

import json
from pathlib import Path

from hnsw_logic.domain.models import DocRecord
from hnsw_logic.domain.serialization import read_json, read_jsonl
from hnsw_logic.evaluation.beir import (
    _should_build_offline_graph,
    evaluate_beir_dataset,
    load_beir_dataset,
    prepare_beir_work_root,
)


def test_load_beir_dataset_parses_tsv_qrels(tmp_path: Path):
    dataset_root = tmp_path / "scifact"
    (dataset_root / "qrels").mkdir(parents=True)
    (dataset_root / "corpus.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"_id": "d1", "title": "Doc 1", "text": "Alpha"}),
                json.dumps({"_id": "d2", "title": "Doc 2", "text": "Beta"}),
            ]
        ),
        encoding="utf-8",
    )
    (dataset_root / "queries.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"_id": "q1", "text": "alpha"}),
                json.dumps({"_id": "q2", "text": "beta"}),
            ]
        ),
        encoding="utf-8",
    )
    (dataset_root / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq1\td1\t1\nq2\td2\t2\n",
        encoding="utf-8",
    )

    dataset = load_beir_dataset(dataset_root)

    assert len(dataset.corpus) == 2
    assert dataset.qrels["q1"] == {"d1": 1}
    assert dataset.qrels["q2"] == {"d2": 2}
    assert dataset.queries == [
        {"query_id": "q1", "text": "alpha"},
        {"query_id": "q2", "text": "beta"},
    ]


def test_prepare_beir_work_root_writes_raw_jsonl(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[2]
    work_root = tmp_path / "work"
    corpus = [
        DocRecord(doc_id="d1", title="Doc 1", text="Alpha", metadata={"source_dataset": "demo"}),
        DocRecord(doc_id="d2", title="Doc 2", text="Beta", metadata={"source_dataset": "demo"}),
    ]

    prepare_beir_work_root(repo_root, work_root, corpus)
    rows = read_jsonl(work_root / "data" / "raw" / "beir.jsonl")

    assert (work_root / "configs").exists()
    assert [row["doc_id"] for row in rows] == ["d1", "d2"]


def test_load_beir_dataset_keeps_all_positive_docs_when_sampling(tmp_path: Path):
    dataset_root = tmp_path / "scifact"
    (dataset_root / "qrels").mkdir(parents=True)
    corpus_rows = [
        {"_id": "d1", "title": "Alpha", "text": "query alpha fact"},
        {"_id": "d2", "title": "Beta", "text": "query beta fact"},
        {"_id": "d3", "title": "Gamma", "text": "query alpha beta hard negative"},
        {"_id": "d4", "title": "Delta", "text": "irrelevant filler"},
    ]
    (dataset_root / "corpus.jsonl").write_text("\n".join(json.dumps(row) for row in corpus_rows), encoding="utf-8")
    (dataset_root / "queries.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"_id": "q1", "text": "query alpha"}),
                json.dumps({"_id": "q2", "text": "query beta"}),
            ]
        ),
        encoding="utf-8",
    )
    (dataset_root / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq1\td1\t1\nq2\td2\t1\n",
        encoding="utf-8",
    )

    dataset = load_beir_dataset(dataset_root, corpus_limit=3)

    kept_ids = {doc.doc_id for doc in dataset.corpus}
    assert {"d1", "d2"} <= kept_ids
    assert len(kept_ids) == 3


def test_should_build_offline_graph_only_for_structured_beir_sets():
    assert _should_build_offline_graph("scifact") is True
    assert _should_build_offline_graph("nfcorpus") is True
    assert _should_build_offline_graph("arguana") is True


def test_evaluate_beir_dataset_persists_report(tmp_path: Path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[2]
    dataset_root = tmp_path / "scifact"
    (dataset_root / "qrels").mkdir(parents=True)
    (dataset_root / "corpus.jsonl").write_text(
        json.dumps({"_id": "d1", "title": "Doc 1", "text": "Alpha"}) + "\n",
        encoding="utf-8",
    )
    (dataset_root / "queries.jsonl").write_text(
        json.dumps({"_id": "q1", "text": "alpha"}) + "\n",
        encoding="utf-8",
    )
    (dataset_root / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq1\td1\t1\n",
        encoding="utf-8",
    )

    class FakeHit:
        def __init__(self, doc_id: str):
            self.doc_id = doc_id

    class FakeResponse:
        def __init__(self, doc_id: str):
            self.hits = [FakeHit(doc_id)]

    class FakePipeline:
        def build_embeddings(self):
            return {"docs": 1}

        def build_hnsw(self):
            return {"docs": 1}

        def profile_docs(self):
            return {"briefs": 1}

        def discover_edges(self):
            return {"edges": 0, "new_edges": 0}

    class FakeBriefStore:
        def __init__(self):
            self._briefs = []

        def all(self):
            return list(self._briefs)

        def write(self, brief):
            self._briefs.append(brief)

    class FakeScorer:
        def preload_views(self, *_args, **_kwargs):
            return None

    class FakeRetrieval:
        def __init__(self):
            self.scorer = FakeScorer()

        def search_baseline(self, _query: str, top_k: int = 10):
            return FakeResponse("d1")

        def search(self, _query: str, top_k: int = 10, use_memory_bias: bool = False):
            return FakeResponse("d1")

    class FakeApp:
        def __init__(self):
            self.settings = type(
                "Settings",
                (),
                {
                    "app": type("AppCfg", (), {"provider": type("ProviderCfg", (), {"kind": "stub"})()})(),
                    "hnsw": type("HnswCfg", (), {"vector_dim": 1024})(),
                },
            )()
            self.corpus_store = type(
                "CorpusStore",
                (),
                {
                    "read_processed": lambda _self: [
                        DocRecord(doc_id="d1", title="Doc 1", text="Alpha", metadata={"source_dataset": "scifact"})
                    ]
                },
            )()
            self.pipeline = FakePipeline()
            self.brief_store = FakeBriefStore()
            self.retrieval = FakeRetrieval()

    monkeypatch.setattr("hnsw_logic.evaluation.beir.download_and_extract_beir_dataset", lambda *_args, **_kwargs: dataset_root)
    monkeypatch.setattr("hnsw_logic.evaluation.beir.build_app", lambda _work_root: FakeApp())

    work_root = tmp_path / "work"
    report = evaluate_beir_dataset(
        repo_root,
        "scifact",
        query_limit=1,
        corpus_limit=1,
        cache_root=tmp_path / "cache",
        work_root=work_root,
    )

    persisted = read_json(work_root / "data" / "results" / "benchmark_report.json", default={})

    assert report.dataset == "scifact"
    assert persisted["dataset"] == "scifact"
    assert persisted["query_count"] == 1
