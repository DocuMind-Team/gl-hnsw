from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from hnsw_logic.agents.orchestrator import LogicOrchestrator
from hnsw_logic.config.schema import ProviderConfig, RetrievalConfig
from hnsw_logic.core.models import DocBrief
from hnsw_logic.core.utils import deterministic_vector
from hnsw_logic.embedding.provider import JudgeResult, OpenAICompatibleProvider, StubProvider


class FakeProvider(StubProvider):
    def __init__(self):
        super().__init__(ProviderConfig(kind="stub"))
        self.relation_priors = {
            "supporting_evidence": 1.0,
            "implementation_detail": 1.05,
            "same_concept": 0.7,
            "comparison": 0.4,
            "prerequisite": 1.0,
        }

    def embed_texts(self, texts):
        return np.vstack([deterministic_vector(text, self.embedding_dim) for text in texts]).astype(np.float32)


class FakeJudge:
    def __init__(self, provider, verdicts):
        self.provider = provider
        self._verdicts = verdicts

    def run_many(self, anchor, candidates):
        return {candidate.doc_id: self._verdicts[candidate.doc_id] for candidate in candidates}

    def run(self, anchor, candidate):
        return self._verdicts[candidate.doc_id]


def _brief(doc_id: str, title: str, summary: str, keywords: list[str], entities: list[str] | None = None, relation_hints: list[str] | None = None):
    return DocBrief(
        doc_id=doc_id,
        title=title,
        summary=summary,
        keywords=keywords,
        entities=entities or [],
        relation_hints=relation_hints or [],
        claims=[summary],
        metadata={},
    )


def test_relation_specific_thresholds_and_ranked_out():
    provider = type("OpenAICompatibleProvider", (FakeProvider,), {})()
    retrieval = RetrievalConfig()
    anchor = _brief(
        "doc-12",
        "Subagents",
        "Subagents include document profiler, corpus scout, relation judge, and memory curator roles.",
        ["subagents", "document", "profiler", "corpus", "scout", "relation", "judge", "memory", "curator"],
        ["subagents"],
        ["include", "roles"],
    )
    profiler = _brief(
        "doc-15",
        "Document Profiler",
        "A document profiler role summarizes docs before memory updates.",
        ["document", "profiler", "memory", "updates", "role"],
        ["profiler"],
        ["role", "summarize"],
    )
    judge = _brief(
        "doc-17",
        "Relation Judge",
        "The relation judge role verifies candidates after corpus scout proposes them.",
        ["relation", "judge", "corpus", "scout", "role", "candidates"],
        ["judge"],
        ["after", "role"],
    )
    memory = _brief(
        "doc-18",
        "Memory Curator",
        "The memory curator maintains persistent memory for accepted edges.",
        ["memory", "curator", "persistent", "accepted", "edges"],
        ["memory"],
        ["maintains"],
    )
    scout = _brief(
        "doc-16",
        "Corpus Scout",
        "The corpus scout runs before the relation judge but the evidence is omitted here.",
        ["corpus", "scout", "relation", "judge", "before"],
        ["scout"],
        ["before"],
    )
    verdicts = {
        "doc-15": JudgeResult(
            accepted=True,
            relation_type="implementation_detail",
            confidence=0.93,
            evidence_spans=[
                "Subagents include the document profiler role.",
                "A document profiler role summarizes docs before memory updates.",
            ],
            rationale="Subagents includes document profiler as a concrete role.",
            support_score=0.82,
            contradiction_flags=[],
            decision_reason="Specific subagent role is explicitly named.",
        ),
        "doc-17": JudgeResult(
            accepted=True,
            relation_type="prerequisite",
            confidence=0.91,
            evidence_spans=[
                "The scout proposes candidates before relation judging.",
                "The relation judge role verifies candidates after corpus scout proposes them.",
            ],
            rationale="Corpus scout happens before relation judge.",
            support_score=0.8,
            contradiction_flags=[],
            decision_reason="Workflow order is explicit.",
        ),
        "doc-18": JudgeResult(
            accepted=True,
            relation_type="comparison",
            confidence=0.95,
            evidence_spans=[
                "Subagents include memory curator among the roles.",
                "The memory curator maintains persistent memory for accepted edges.",
            ],
            rationale="Both are parts of the same system.",
            support_score=0.83,
            contradiction_flags=[],
            decision_reason="Shared system context only.",
        ),
        "doc-16": JudgeResult(
            accepted=True,
            relation_type="prerequisite",
            confidence=0.92,
            evidence_spans=[],
            rationale="Corpus scout runs before relation judge.",
            support_score=0.78,
            contradiction_flags=[],
            decision_reason="Missing grounded evidence spans.",
        ),
    }
    orchestrator = LogicOrchestrator(
        doc_profiler=SimpleNamespace(provider=provider),
        corpus_scout=SimpleNamespace(provider=provider),
        relation_judge=FakeJudge(provider, verdicts),
        memory_curator=SimpleNamespace(provider=provider),
        retrieval_config=retrieval,
    )

    assessments = orchestrator.judge_many_with_diagnostics(anchor, [profiler, judge, memory, scout])
    by_id = {item.candidate_doc_id: item for item in assessments}

    accepted_ids = {item.candidate_doc_id for item in assessments if item.accepted}
    assert len(accepted_ids) == 1
    assert {"doc-15", "doc-17"} & accepted_ids
    assert by_id[({"doc-15", "doc-17"} - accepted_ids).pop()].reject_reason == "ranked_out"
    assert by_id["doc-18"].reject_reason == "wrong_relation_type"
    assert by_id["doc-16"].reject_reason == "weak_evidence"


def test_local_gate_rejects_topic_drift_or_low_support():
    provider = FakeProvider()
    anchor = _brief(
        "doc-08",
        "Logic Overlay Graph",
        "The logic overlay graph adds one-hop logical expansion after HNSW recall.",
        ["logic", "overlay", "graph", "hnsw", "jump", "policy"],
        ["graph"],
        ["one-hop", "expansion"],
    )
    unrelated = _brief(
        "doc-22",
        "SQLite Job Registry",
        "Workers persist job state in SQLite for build and admin tasks.",
        ["sqlite", "jobs", "workers", "build", "admin"],
        ["sqlite"],
        ["persist"],
    )
    verdicts = {
        "doc-22": JudgeResult(
            accepted=True,
            relation_type="implementation_detail",
            confidence=0.97,
            evidence_spans=[
                "Workers track build jobs in a SQLite registry.",
                "Admin tasks route job state through the registry.",
            ],
            rationale="SQLite registry details are tracked through worker and admin flows.",
            support_score=0.5,
            contradiction_flags=[],
            decision_reason="The candidate is an implementation detail tracked through the registry.",
        )
    }
    orchestrator = LogicOrchestrator(
        doc_profiler=SimpleNamespace(provider=provider),
        corpus_scout=SimpleNamespace(provider=provider),
        relation_judge=FakeJudge(provider, verdicts),
        memory_curator=SimpleNamespace(provider=provider),
        retrieval_config=RetrievalConfig(),
    )

    assessment = orchestrator.judge_many_with_diagnostics(anchor, [unrelated])[0]
    assert assessment.accepted is False
    assert assessment.reject_reason in {"topic_drift", "low_support"}


def test_parse_json_tolerates_fenced_json():
    provider = OpenAICompatibleProvider.__new__(OpenAICompatibleProvider)
    payload = provider._parse_json("```json\n{\"accepted\": true, \"confidence\": 0.9}\n```")
    assert payload["accepted"] is True
    assert payload["confidence"] == 0.9
