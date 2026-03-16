from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from hnsw_logic.config.settings import load_settings
from hnsw_logic.services.bootstrap import build_app


@pytest.fixture()
def test_root(tmp_path: Path) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    shutil.copytree(repo_root / "configs", tmp_path / "configs")
    shutil.copytree(repo_root / "data" / "raw", tmp_path / "data" / "raw")
    shutil.copytree(repo_root / "data" / "demo", tmp_path / "data" / "demo")
    shutil.copytree(repo_root / ".deepagents", tmp_path / ".deepagents")
    return tmp_path


@pytest.fixture()
def app_container(test_root: Path):
    load_settings.cache_clear()
    return build_app(test_root)
