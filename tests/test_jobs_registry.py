from __future__ import annotations

from hnsw_logic.jobs.registry import JobRegistry


def test_job_registry_roundtrips_structured_payload(test_root):
    registry = JobRegistry(test_root / "data" / "jobs.sqlite3")

    created = registry.create("build_embeddings", {"dataset": "demo", "top_k": 5})
    loaded = registry.get(created.job_id)

    assert loaded is not None
    assert loaded.payload == {"dataset": "demo", "top_k": 5}


def test_job_registry_preserves_raw_string_payload(test_root):
    registry = JobRegistry(test_root / "data" / "jobs.sqlite3")

    created = registry.create("custom", "not-json")
    loaded = registry.get(created.job_id)

    assert loaded is not None
    assert loaded.payload == {"raw": "not-json"}
