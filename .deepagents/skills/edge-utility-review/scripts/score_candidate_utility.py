from __future__ import annotations

from typing import Any


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def main(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = payload.get("metrics", {}) or {}
    signal_report = payload.get("signal_report", {}) or {}
    relation_type = str(payload.get("relation_type", "")).strip() or "supporting_evidence"
    fit_scores = payload.get("fit_scores", {}) or {}

    dense = float(metrics.get("dense_score", 0.0) or 0.0)
    local_support = float(metrics.get("local_support", 0.0) or 0.0)
    mention = float(metrics.get("mention_score", 0.0) or 0.0)
    overlap = max(
        float(metrics.get("overlap_score", 0.0) or 0.0),
        float(metrics.get("content_overlap_score", 0.0) or 0.0),
    )
    service_surface = float(metrics.get("service_surface_score", 0.0) or 0.0)
    bridge_gain = float(signal_report.get("bridge_information_gain", 0.0) or 0.0)
    topic_consistency = float(signal_report.get("topic_consistency", 0.0) or 0.0)
    duplicate_risk = float(signal_report.get("duplicate_risk", 0.0) or 0.0)
    contrast_evidence = float(signal_report.get("contrast_evidence", 0.0) or 0.0)
    query_surface_match = float(signal_report.get("query_surface_match", 0.0) or 0.0)
    drift_risk = float(signal_report.get("drift_risk", 0.0) or 0.0)
    fit = float(fit_scores.get(relation_type, 0.0) or 0.0)

    utility = (
        0.24 * local_support
        + 0.16 * dense
        + 0.14 * bridge_gain
        + 0.14 * topic_consistency
        + 0.1 * query_surface_match
        + 0.1 * fit
        + 0.06 * mention
        + 0.06 * overlap
        + 0.06 * contrast_evidence
        - 0.12 * duplicate_risk
        - 0.14 * drift_risk
        - 0.08 * service_surface
    )

    if relation_type == "comparison":
        utility += 0.06 * contrast_evidence + 0.04 * topic_consistency
    elif relation_type == "same_concept":
        utility += 0.04 * dense + 0.04 * overlap
    elif relation_type == "prerequisite":
        utility += 0.04 * float(metrics.get("specific_role_score", 0.0) or 0.0)

    return {"utility_score": round(_clamp(utility), 6)}
