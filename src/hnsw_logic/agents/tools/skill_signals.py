from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path
from typing import Any

from hnsw_logic.core.utils import to_jsonable


class SkillSignalRuntime:
    def __init__(self, repo_root: Path | None = None):
        self.repo_root = repo_root or Path(__file__).resolve().parents[4]
        self.skills_root = self.repo_root / ".deepagents" / "skills"

    @lru_cache(maxsize=64)
    def _load_script(self, skill_name: str, script_name: str):
        path = self.skills_root / skill_name / "scripts" / f"{script_name}.py"
        if not path.exists():
            raise FileNotFoundError(f"missing skill script: {path}")
        spec = importlib.util.spec_from_file_location(f"gl_hnsw_{skill_name}_{script_name}", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"unable to load skill script: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, "main"):
            raise RuntimeError(f"skill script does not expose main(payload): {path}")
        return module.main

    def _execute(self, skill_name: str, script_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        main = self._load_script(skill_name, script_name)
        result = main(payload)
        if not isinstance(result, dict):
            raise TypeError(f"skill script `{skill_name}/{script_name}` must return dict")
        return result

    @staticmethod
    def _jsonable(value: Any) -> Any:
        return to_jsonable(value)

    def compute_topic_consistency(self, anchor: Any, candidate: Any, local_signals: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._execute(
            "signal-fusion",
            "compute_topic_consistency",
            {
                "anchor": self._jsonable(anchor),
                "candidate": self._jsonable(candidate),
                "local_signals": local_signals or {},
            },
        )

    def compute_duplicate_risk(self, anchor: Any, candidate: Any, local_signals: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._execute(
            "counterevidence-check",
            "compute_duplicate_risk",
            {
                "anchor": self._jsonable(anchor),
                "candidate": self._jsonable(candidate),
                "local_signals": local_signals or {},
            },
        )

    def compute_bridge_gain(self, anchor: Any, candidate: Any, local_signals: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._execute(
            "edge-utility-review",
            "compute_bridge_gain",
            {
                "anchor": self._jsonable(anchor),
                "candidate": self._jsonable(candidate),
                "local_signals": local_signals or {},
            },
        )

    def compute_contrast_evidence(
        self,
        anchor: Any,
        candidate: Any,
        local_signals: dict[str, Any] | None = None,
        verdict: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._execute(
            "relation-judging",
            "compute_contrast_evidence",
            {
                "anchor": self._jsonable(anchor),
                "candidate": self._jsonable(candidate),
                "local_signals": local_signals or {},
                "verdict": verdict or {},
            },
        )

    def compute_query_activation_profile(
        self,
        anchor: Any,
        candidate: Any,
        relation_type: str,
        local_signals: dict[str, Any] | None = None,
        verdict: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._execute(
            "metric-evaluation",
            "compute_query_activation_profile",
            {
                "anchor": self._jsonable(anchor),
                "candidate": self._jsonable(candidate),
                "relation_type": relation_type,
                "local_signals": local_signals or {},
                "verdict": verdict or {},
            },
        )

    def score_relation_fit(
        self,
        anchor: Any,
        candidate: Any,
        metrics: dict[str, Any],
        signal_report: dict[str, Any],
    ) -> dict[str, Any]:
        return self._execute(
            "relation-judging",
            "score_relation_fit",
            {
                "anchor": self._jsonable(anchor),
                "candidate": self._jsonable(candidate),
                "metrics": metrics,
                "signal_report": signal_report,
            },
        )

    def score_candidate_utility(
        self,
        *,
        relation_type: str,
        metrics: dict[str, Any],
        fit_scores: dict[str, Any],
        signal_report: dict[str, Any],
    ) -> dict[str, Any]:
        return self._execute(
            "edge-utility-review",
            "score_candidate_utility",
            {
                "relation_type": relation_type,
                "metrics": metrics,
                "fit_scores": fit_scores,
                "signal_report": signal_report,
            },
        )

    def compute_anchor_priority(
        self,
        brief: Any,
        *,
        features: dict[str, Any] | None = None,
        corpus_profile: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._execute(
            "anchor-planning",
            "rank_anchor_priority",
            {
                "brief": self._jsonable(brief),
                "features": features or {},
                "corpus_profile": corpus_profile or {},
            },
        )

    def compute_candidate_priority(
        self,
        *,
        base_score: float,
        metrics: dict[str, Any],
        fit_scores: dict[str, Any],
        signal_report: dict[str, Any],
    ) -> dict[str, Any]:
        return self._execute(
            "candidate-expansion",
            "rank_candidate_priority",
            {
                "base_score": float(base_score),
                "metrics": metrics,
                "fit_scores": fit_scores,
                "signal_report": signal_report,
            },
        )

    def compute_edge_budget_score(
        self,
        *,
        score: float,
        utility_score: float,
        activation_prior: float,
        novelty: float,
        specific_novelty: float,
        drift_risk: float = 0.0,
        duplicate_risk: float = 0.0,
    ) -> dict[str, Any]:
        return self._execute(
            "edge-utility-review",
            "select_edge_budget",
            {
                "score": float(score),
                "utility_score": float(utility_score),
                "activation_prior": float(activation_prior),
                "novelty": float(novelty),
                "specific_novelty": float(specific_novelty),
                "drift_risk": float(drift_risk),
                "duplicate_risk": float(duplicate_risk),
            },
        )

    def build_signal_report(
        self,
        anchor: Any,
        candidate: Any,
        local_signals: dict[str, Any] | None = None,
        verdict: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        topic_report = self.compute_topic_consistency(anchor, candidate, local_signals=local_signals)
        duplicate_report = self.compute_duplicate_risk(anchor, candidate, local_signals=local_signals)
        bridge_report = self.compute_bridge_gain(anchor, candidate, local_signals=local_signals)
        contrast_report = self.compute_contrast_evidence(anchor, candidate, local_signals=local_signals, verdict=verdict)
        drift_risk = max(
            float(topic_report.get("drift_risk", 0.0) or 0.0),
            float(duplicate_report.get("drift_risk", 0.0) or 0.0),
            float(bridge_report.get("drift_risk", 0.0) or 0.0),
        )
        query_surface_match = max(
            float(topic_report.get("query_surface_match", 0.0) or 0.0),
            float(bridge_report.get("query_surface_match", 0.0) or 0.0),
            float(contrast_report.get("query_surface_match", 0.0) or 0.0),
        )
        uncertainty_hint = max(
            float(topic_report.get("uncertainty_hint", 0.0) or 0.0),
            float(duplicate_report.get("uncertainty_hint", 0.0) or 0.0),
            float(contrast_report.get("uncertainty_hint", 0.0) or 0.0),
        )
        notes = [
            *[str(item) for item in topic_report.get("notes", [])],
            *[str(item) for item in duplicate_report.get("notes", [])],
            *[str(item) for item in bridge_report.get("notes", [])],
            *[str(item) for item in contrast_report.get("notes", [])],
        ]
        return {
            "topic_consistency": float(topic_report.get("topic_consistency", 0.0) or 0.0),
            "duplicate_risk": float(duplicate_report.get("duplicate_risk", 0.0) or 0.0),
            "bridge_information_gain": float(bridge_report.get("bridge_information_gain", 0.0) or 0.0),
            "contrast_evidence": float(contrast_report.get("contrast_evidence", 0.0) or 0.0),
            "query_surface_match": query_surface_match,
            "uncertainty_hint": uncertainty_hint,
            "drift_risk": drift_risk,
            "notes": notes[:12],
            "topic_report": topic_report,
            "duplicate_report": duplicate_report,
            "bridge_report": bridge_report,
            "contrast_report": contrast_report,
        }
