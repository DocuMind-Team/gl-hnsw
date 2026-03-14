from __future__ import annotations

import json
from pathlib import Path

from hnsw_logic.core.models import DocRecord
from hnsw_logic.services.beir_eval import _should_build_offline_graph, load_beir_dataset, prepare_beir_work_root
from hnsw_logic.core.utils import read_jsonl


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
    repo_root = Path(__file__).resolve().parents[1]
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
