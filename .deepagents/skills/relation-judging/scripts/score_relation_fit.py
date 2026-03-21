from __future__ import annotations

from typing import Any


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _topic(value: dict[str, Any], field: str) -> str:
    return str((value.get("metadata", {}) or {}).get(field, "")).strip().lower()


def main(payload: dict[str, Any]) -> dict[str, Any]:
    anchor = payload.get("anchor", {}) or {}
    candidate = payload.get("candidate", {}) or {}
    metrics = payload.get("metrics", {}) or {}
    signal_report = payload.get("signal_report", {}) or {}

    dense = float(metrics.get("dense_score", 0.0) or 0.0)
    overlap = max(
        float(metrics.get("overlap_score", 0.0) or 0.0),
        float(metrics.get("content_overlap_score", 0.0) or 0.0),
    )
    mention = float(metrics.get("mention_score", 0.0) or 0.0)
    role_listing = float(metrics.get("role_listing_score", 0.0) or 0.0)
    forward_reference = float(metrics.get("forward_reference_score", 0.0) or 0.0)
    reverse_reference = float(metrics.get("reverse_reference_score", 0.0) or 0.0)
    topic_alignment = float(metrics.get("topic_alignment", 0.0) or 0.0)
    family_match = float(metrics.get("topic_family_match", 0.0) or 0.0)
    cluster_match = float(metrics.get("topic_cluster_match", 0.0) or 0.0)
    stance_contrast = float(metrics.get("stance_contrast", 0.0) or 0.0)
    specific_role = float(metrics.get("specific_role_score", 0.0) or 0.0)
    service_surface = float(metrics.get("service_surface_score", 0.0) or 0.0)
    topic_consistency = float(signal_report.get("topic_consistency", 0.0) or 0.0)
    duplicate_risk = float(signal_report.get("duplicate_risk", 0.0) or 0.0)
    bridge_gain = float(signal_report.get("bridge_information_gain", 0.0) or 0.0)
    contrast_evidence = float(signal_report.get("contrast_evidence", 0.0) or 0.0)
    query_surface_match = float(signal_report.get("query_surface_match", 0.0) or 0.0)
    drift_risk = float(signal_report.get("drift_risk", 0.0) or 0.0)

    anchor_topic = _topic(anchor, "topic")
    candidate_topic = _topic(candidate, "topic")
    argument_like = anchor_topic == "argument" or candidate_topic == "argument" or stance_contrast > 0.0
    evidence_like = anchor_topic in {"scientific_claims", "clinical_retrieval"} or candidate_topic in {"scientific_claims", "clinical_retrieval"}

    implementation_detail = (
        0.32 * dense
        + 0.2 * max(forward_reference - 0.5 * reverse_reference, 0.0)
        + 0.16 * bridge_gain
        + 0.12 * query_surface_match
        + 0.1 * overlap
        + 0.1 * mention
        - 0.14 * service_surface
        - 0.08 * duplicate_risk
        - 0.08 * drift_risk
    )

    supporting_evidence = (
        0.22 * dense
        + 0.22 * mention
        + 0.18 * overlap
        + 0.14 * bridge_gain
        + 0.12 * topic_consistency
        + 0.08 * query_surface_match
        + 0.04 * topic_alignment
        - 0.12 * service_surface
        - 0.08 * duplicate_risk
        - 0.12 * drift_risk
    )

    prerequisite = (
        0.34 * specific_role
        + 0.26 * role_listing
        + 0.16 * mention
        + 0.12 * topic_consistency
        + 0.08 * bridge_gain
        - 0.08 * service_surface
        - 0.08 * drift_risk
    )

    comparison = (
        0.22 * topic_consistency
        + 0.22 * contrast_evidence
        + 0.16 * bridge_gain
        + 0.12 * query_surface_match
        + 0.1 * overlap
        + 0.08 * family_match
        + 0.06 * cluster_match
        + 0.08 * stance_contrast
        - 0.12 * duplicate_risk
        - 0.14 * drift_risk
    )

    same_concept = (
        0.28 * dense
        + 0.22 * overlap
        + 0.16 * bridge_gain
        + 0.14 * topic_consistency
        + 0.1 * topic_alignment
        + 0.1 * query_surface_match
        - 0.14 * duplicate_risk
        - 0.14 * drift_risk
    )

    if argument_like:
        comparison += 0.1 * max(family_match, cluster_match)
        same_concept -= 0.1 * max(stance_contrast, contrast_evidence)
    if evidence_like:
        same_concept += 0.08 * max(topic_alignment, family_match)
        supporting_evidence += 0.04 * max(topic_alignment, bridge_gain)
    if specific_role > 0.0:
        prerequisite += 0.08
        implementation_detail += 0.04

    fit_scores = {
        "implementation_detail": round(_clamp(implementation_detail, 0.0, 1.25), 6),
        "supporting_evidence": round(_clamp(supporting_evidence, 0.0, 1.15), 6),
        "prerequisite": round(_clamp(prerequisite, 0.0, 1.15), 6),
        "comparison": round(_clamp(comparison, 0.0, 1.2), 6),
        "same_concept": round(_clamp(same_concept, 0.0, 1.15), 6),
    }
    best_relation = max(fit_scores.items(), key=lambda item: (item[1], item[0]))[0]
    return {"fit_scores": fit_scores, "best_relation": best_relation}
