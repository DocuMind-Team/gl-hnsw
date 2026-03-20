from __future__ import annotations


def main(payload: dict[str, float]) -> dict[str, float]:
    score = float(payload.get("score", 0.0) or 0.0)
    utility_score = float(payload.get("utility_score", 0.0) or 0.0)
    activation_prior = float(payload.get("activation_prior", 0.0) or 0.0)
    novelty = float(payload.get("novelty", 0.0) or 0.0)
    specific_novelty = float(payload.get("specific_novelty", 0.0) or 0.0)
    drift_risk = float(payload.get("drift_risk", 0.0) or 0.0)
    duplicate_risk = float(payload.get("duplicate_risk", 0.0) or 0.0)

    selection_score = (
        score * (0.9 + 0.1 * utility_score)
        + 0.12 * novelty
        + 0.08 * specific_novelty
        + 0.12 * utility_score
        + 0.08 * activation_prior
        - 0.08 * drift_risk
        - 0.06 * duplicate_risk
    )
    return {
        "selection_score": round(selection_score, 6),
    }
