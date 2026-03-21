from __future__ import annotations

import numpy as np

from hnsw_logic.domain.models import LogicEdge
from hnsw_logic.retrieval.jump_policy import JumpPolicy


def test_jump_policy_gates_low_confidence(app_container):
    policy = JumpPolicy(app_container.settings.retrieval)
    edge = LogicEdge(
        src_doc_id="a",
        dst_doc_id="b",
        relation_type="supporting_evidence",
        confidence=0.1,
        evidence_spans=[],
        discovery_path=[],
        edge_card_text="edge",
        created_at="2026-03-10T00:00:00Z",
        last_validated_at="2026-03-10T00:00:00Z",
    )
    assert not policy.allow_jump(np.ones(4), np.ones(4), edge, target_rel_score=1.0)
