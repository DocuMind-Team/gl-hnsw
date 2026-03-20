from __future__ import annotations

import numpy as np

from hnsw_logic.config.schema import RetrievalConfig
from hnsw_logic.core.models import LogicEdge


class JumpPolicy:
    def __init__(self, config: RetrievalConfig):
        self.config = config
        self.tau_conf = config.jump_policy.tau_conf
        self.tau_edge = config.jump_policy.tau_edge
        self.tau_target = config.jump_policy.tau_target
        self.max_seeds = config.seed_top_b
        self.max_expansions_per_seed = config.max_expansions_per_seed

    def allow_jump(
        self,
        query_emb: np.ndarray,
        edge_emb: np.ndarray,
        edge: LogicEdge,
        target_rel_score: float,
        *,
        activation_match: float = 0.0,
    ) -> bool:
        edge_match = float(np.dot(query_emb, edge_emb))
        profile = getattr(edge, "activation_profile", {}) or {}
        activation_prior = max(0.0, min(float(profile.get("activation_prior", getattr(edge, "utility_score", edge.confidence)) or 0.0), 1.0))
        drift_risk = max(0.0, min(float(profile.get("drift_risk", 0.0) or 0.0), 1.0))
        effective_conf = (
            0.5 * edge.confidence
            + 0.25 * getattr(edge, "utility_score", edge.confidence)
            + 0.15 * activation_prior
            + 0.1 * max(0.0, min(activation_match, 1.0))
        )
        if effective_conf < self.tau_conf:
            return False
        edge_threshold = max(0.05, self.tau_edge - 0.06 * activation_prior - 0.04 * max(0.0, min(activation_match, 1.0)))
        if edge_match < edge_threshold:
            return False
        target_threshold = max(0.05, self.tau_target - 0.05 * activation_prior)
        if target_rel_score < target_threshold:
            return False
        if drift_risk >= 0.8 and activation_prior < 0.6 and activation_match < 0.45:
            return False
        return True
