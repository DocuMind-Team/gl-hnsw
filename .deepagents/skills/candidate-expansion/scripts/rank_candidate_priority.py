from __future__ import annotations

from typing import Any


def _value(mapping: dict[str, Any], key: str, default: float = 0.0) -> float:
    return float(mapping.get(key, default) or default)


def main(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = dict(payload.get("metrics", {}) or {})
    fit_scores = dict(payload.get("fit_scores", {}) or {})
    signal_report = dict(payload.get("signal_report", {}) or {})
    base_score = float(payload.get("base_score", 0.0) or 0.0)

    priority_score = (
        base_score
        + 0.18 * _value(metrics, "local_support")
        + 0.12 * max((float(value or 0.0) for value in fit_scores.values()), default=0.0)
        + 0.18 * _value(signal_report, "bridge_information_gain")
        + 0.12 * _value(signal_report, "topic_consistency")
        + 0.10 * _value(signal_report, "query_surface_match")
        + 0.08 * _value(signal_report, "contrast_evidence")
        - 0.14 * _value(signal_report, "duplicate_risk")
        - 0.12 * _value(signal_report, "drift_risk")
        - 0.06 * _value(signal_report, "uncertainty_hint")
    )
    return {
        "priority_score": round(priority_score, 6),
        "notes": [
            "candidate priority rewards bridge gain, topic consistency, and query-facing evidence",
            "duplicate and drift risks are penalties, not direct vetoes",
        ],
    }
