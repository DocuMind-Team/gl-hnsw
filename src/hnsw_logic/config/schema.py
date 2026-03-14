from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class ProviderConfig(BaseModel):
    kind: str = "stub"
    base_url: str = "https://api.xiaomimimo.com/v1"
    api_key_env: str = "GL_HNSW_API_KEY"
    chat_model: str = "stub-chat"
    embedding_model: str = "stub-embedding"
    embedding_dim: int = 64


class PathConfig(BaseModel):
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    indices_dir: Path = Path("data/indices")
    graph_dir: Path = Path("data/graph")
    memories_dir: Path = Path("data/memories")
    workspace_dir: Path = Path("data/workspace")
    jobs_db: Path = Path("data/jobs.sqlite3")


class RuntimeConfig(BaseModel):
    log_level: str = "INFO"
    deterministic_seed: int = 7


class AppConfig(BaseModel):
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    paths: PathConfig = Field(default_factory=PathConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)


class HnswConfig(BaseModel):
    m: int = 16
    ef_construction: int = 100
    ef_search: int = 40
    vector_dim: int = 64
    metric: str = "cosine"


class SubagentConfig(BaseModel):
    enabled: bool = True
    skills: list[str] = Field(default_factory=list)


class AgentsConfig(BaseModel):
    runtime_mode: str = "deepagents"
    sandbox_enabled: bool = False
    subagents: dict[str, SubagentConfig] = Field(default_factory=dict)
    live_reasoning: "LiveReasoningConfig" = Field(default_factory=lambda: LiveReasoningConfig())


class LiveReasoningConfig(BaseModel):
    enable_scout_thinking: bool = True
    enable_judge_thinking: bool = True
    enable_curator_thinking: bool = True


class JumpPolicyConfig(BaseModel):
    tau_conf: float = 0.6
    tau_edge: float = 0.2
    tau_target: float = 0.15


class FusionConfig(BaseModel):
    alpha: float = 1.0
    beta: float = 0.7


class RelationQualityConfig(BaseModel):
    enabled: bool = True
    min_confidence: float = 0.8
    min_support: float = 0.3
    min_evidence_quality: float = 0.35


def _default_relation_thresholds() -> dict[str, RelationQualityConfig]:
    return {
        "supporting_evidence": RelationQualityConfig(min_confidence=0.82, min_support=0.3, min_evidence_quality=0.35),
        "implementation_detail": RelationQualityConfig(min_confidence=0.84, min_support=0.34, min_evidence_quality=0.4),
        "same_concept": RelationQualityConfig(enabled=False, min_confidence=0.9, min_support=0.5, min_evidence_quality=0.4),
        "comparison": RelationQualityConfig(enabled=False, min_confidence=0.92, min_support=0.55, min_evidence_quality=0.45),
        "prerequisite": RelationQualityConfig(min_confidence=0.88, min_support=0.38, min_evidence_quality=0.4),
    }


class EdgeQualityConfig(BaseModel):
    max_edges_per_anchor_live: int = 1
    max_judge_candidates_live: int = 4
    second_edge_margin: float = 0.015
    relation_thresholds: dict[str, RelationQualityConfig] = Field(default_factory=_default_relation_thresholds)


class RetrievalConfig(BaseModel):
    initial_top_k: int = 50
    seed_top_b: int = 5
    max_expansions_per_seed: int = 2
    supplemental_seed_top_k: int = 6
    supplemental_seed_min_score: float = 0.3
    supplemental_seed_weight: float = 0.92
    sparse_top_k: int = 12
    sparse_seed_weight: float = 0.78
    sparse_min_score: float = 0.18
    novelty_dense_top_k: int = 12
    sparse_agreement_top_k: int = 8
    sparse_agreement_floor: float = 0.2
    sparse_only_min_agreement: float = 0.35
    sparse_only_min_raw_coverage: float = 0.45
    jump_policy: JumpPolicyConfig = Field(default_factory=JumpPolicyConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    edge_quality: EdgeQualityConfig = Field(default_factory=EdgeQualityConfig)
