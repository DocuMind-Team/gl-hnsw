from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel

from .schema import AgentsConfig, AppConfig, HnswConfig, RetrievalConfig


class AppSettings(BaseModel):
    root_dir: Path
    app: AppConfig
    hnsw: HnswConfig
    agents: AgentsConfig
    retrieval: RetrievalConfig


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _apply_env_overrides(app_cfg: dict) -> None:
    provider = app_cfg.setdefault("provider", {})
    mapping = {
        "GL_HNSW_PROVIDER_KIND": "kind",
        "GL_HNSW_BASE_URL": "base_url",
        "GL_HNSW_CHAT_MODEL": "chat_model",
        "GL_HNSW_EMBEDDING_MODEL": "embedding_model",
    }
    for env_name, key in mapping.items():
        if os.getenv(env_name):
            provider[key] = os.environ[env_name]
    if os.getenv("GL_HNSW_EMBEDDING_DIM"):
        provider["embedding_dim"] = int(os.environ["GL_HNSW_EMBEDDING_DIM"])


def _apply_hnsw_env_overrides(hnsw_cfg: dict) -> None:
    if os.getenv("GL_HNSW_VECTOR_DIM"):
        hnsw_cfg["vector_dim"] = int(os.environ["GL_HNSW_VECTOR_DIM"])


@lru_cache(maxsize=1)
def load_settings(root_dir: str | Path | None = None) -> AppSettings:
    repo_root = Path(root_dir or Path(__file__).resolve().parents[3]).resolve()
    app_cfg = _load_yaml(repo_root / "configs" / "app.yaml")
    hnsw_cfg = _load_yaml(repo_root / "configs" / "hnsw.yaml")
    _apply_env_overrides(app_cfg)
    _apply_hnsw_env_overrides(hnsw_cfg)
    settings = AppSettings(
        root_dir=repo_root,
        app=AppConfig.model_validate(app_cfg),
        hnsw=HnswConfig.model_validate(hnsw_cfg),
        agents=AgentsConfig.model_validate(_load_yaml(repo_root / "configs" / "agents.yaml")),
        retrieval=RetrievalConfig.model_validate(_load_yaml(repo_root / "configs" / "retrieval.yaml")),
    )
    return settings
