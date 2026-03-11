from __future__ import annotations

from fastapi.testclient import TestClient

from hnsw_logic.api.app import create_app


def test_search_api_returns_hits(test_root):
    from hnsw_logic.config.settings import load_settings
    from hnsw_logic.services.bootstrap import build_app

    load_settings.cache_clear()
    container = build_app(test_root)
    container.pipeline.build_embeddings()
    container.pipeline.build_hnsw()
    container.pipeline.profile_docs()
    container.pipeline.discover_edges()

    app = create_app(root_dir=test_root)
    client = TestClient(app)
    response = client.post("/api/v1/search", json={"query": "How does jump policy work?", "top_k": 5})
    assert response.status_code == 200
    payload = response.json()
    assert payload["query"] == "How does jump policy work?"
    assert payload["hits"]
    assert "doc_id" in payload["hits"][0]
