from __future__ import annotations

from typing import Any


def _value(mapping: dict[str, Any], key: str, default: float = 0.0) -> float:
    return float(mapping.get(key, default) or default)


def main(payload: dict[str, Any]) -> dict[str, Any]:
    features = dict(payload.get("features", {}) or {})
    corpus_profile = dict(payload.get("corpus_profile", {}) or {})

    claim_score = _value(features, "claim_score")
    keyword_score = _value(features, "keyword_score")
    entity_score = _value(features, "entity_score")
    cue_score = _value(features, "cue_score")
    content_score = _value(features, "content_score")
    dataset_signal = _value(features, "dataset_edge_signal")
    centrality = _value(features, "centrality")
    bridge_potential = _value(features, "bridge_potential")
    specificity = _value(features, "title_specificity")
    coverage_gain = _value(features, "coverage_gain")
    cluster_novelty = _value(features, "cluster_novelty")
    family_novelty = _value(features, "family_novelty")

    graph_potential = _value(corpus_profile, "graph_potential")
    bridge_pressure = _value(corpus_profile, "bridge_pressure")
    argument_ratio = _value(corpus_profile, "argument_ratio")

    base_priority = (
        0.18 * claim_score
        + 0.14 * keyword_score
        + 0.12 * entity_score
        + 0.12 * cue_score
        + 0.10 * content_score
        + 0.10 * dataset_signal
        + 0.10 * bridge_potential
        + 0.08 * centrality
        + 0.06 * specificity
    )
    dynamic_priority = (
        0.20 * coverage_gain
        + 0.07 * cluster_novelty
        + 0.09 * family_novelty
        + 0.06 * bridge_pressure
        + 0.04 * argument_ratio
    )
    priority_score = min(2.0, max(0.0, base_priority + dynamic_priority + 0.08 * graph_potential))
    return {
        "priority_score": round(priority_score, 6),
        "base_priority": round(base_priority, 6),
        "dynamic_priority": round(dynamic_priority, 6),
        "notes": [
            "anchor priority blends content richness, graph potential, and coverage gain",
            "family and cluster novelty are bonuses, not hard requirements",
        ],
    }
