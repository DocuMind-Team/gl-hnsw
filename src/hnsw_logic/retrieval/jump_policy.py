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

    def allow_jump(self, query_emb: np.ndarray, edge_emb: np.ndarray, edge: LogicEdge, target_rel_score: float) -> bool:
        edge_match = float(np.dot(query_emb, edge_emb))
        if edge.confidence < self.tau_conf:
            return False
        if edge_match < self.tau_edge:
            return False
        if target_rel_score < self.tau_target:
            return False
        return True
