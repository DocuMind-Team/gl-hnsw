from __future__ import annotations

from hnsw_logic.config.settings import load_settings


def test_load_settings_from_temp_root(test_root):
    load_settings.cache_clear()
    settings = load_settings(test_root)
    assert settings.app.provider.kind == "stub"
    assert settings.hnsw.vector_dim == 64
    assert settings.retrieval.seed_top_b == 5
