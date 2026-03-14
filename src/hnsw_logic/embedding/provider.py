from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from hnsw_logic.config.schema import ProviderConfig
from hnsw_logic.core.facets import enrich_brief
from hnsw_logic.core.models import DocBrief, DocRecord, LogicEdge
from hnsw_logic.core.utils import append_jsonl, deterministic_vector, to_jsonable, tokenize, top_terms, utc_now


@dataclass(slots=True)
class CandidateProposal:
    doc_id: str
    reason: str
    query: str
    score_hint: float


@dataclass(slots=True)
class JudgeResult:
    accepted: bool
    relation_type: str
    confidence: float
    evidence_spans: list[str]
    rationale: str
    support_score: float = 0.0
    contradiction_flags: list[str] | None = None
    decision_reason: str = ""
    semantic_relation_label: str = ""
    canonical_relation: str = ""
    utility_score: float = 0.0
    uncertainty: float = 0.0


@dataclass(slots=True)
class JudgeSignals:
    dense_score: float
    sparse_score: float
    overlap_score: float
    content_overlap_score: float
    mention_score: float
    role_listing_score: float
    forward_reference_score: float
    reverse_reference_score: float
    direction_score: float
    local_support: float
    utility_score: float
    best_relation: str
    stage_pair: str
    risk_flags: list[str]
    relation_fit_scores: dict[str, float]


class ProviderBase:
    def __init__(self, config: ProviderConfig, root_dir: Path | None = None):
        self.config = config
        self.root_dir = Path(root_dir) if root_dir else None
        self.live_reasoning = {
            "scout": True,
            "judge": True,
            "curator": True,
            "query_strategy": True,
        }
        self.relation_priors = {
            "supporting_evidence": 1.02,
            "implementation_detail": 1.0,
            "same_concept": 0.82,
            "comparison": 0.9,
            "prerequisite": 0.98,
        }
        self.judge_few_shot_text = self._build_generic_judge_examples()
        self.query_strategy_few_shot_text = self._build_query_strategy_examples()

    @property
    def embedding_dim(self) -> int:
        return self.config.embedding_dim

    def configure_live_reasoning(self, live_reasoning_config) -> None:
        self.live_reasoning = {
            "scout": live_reasoning_config.enable_scout_thinking,
            "judge": live_reasoning_config.enable_judge_thinking,
            "curator": live_reasoning_config.enable_curator_thinking,
            "query_strategy": live_reasoning_config.enable_query_strategy_thinking,
        }

    def embed_texts(self, texts: Iterable[str]) -> np.ndarray:
        raise NotImplementedError

    def profile_doc(self, doc: DocRecord) -> DocBrief:
        raise NotImplementedError

    def profile_docs(self, docs: list[DocRecord]) -> list[DocBrief]:
        return [self.profile_doc(doc) for doc in docs]

    def propose_candidates(self, anchor: DocBrief, corpus: list[DocBrief]) -> list[CandidateProposal]:
        raise NotImplementedError

    def judge_relation(self, anchor: DocBrief, candidate: DocBrief) -> JudgeResult:
        raise NotImplementedError

    def judge_relations(self, anchor: DocBrief, candidates: list[DocBrief]) -> dict[str, JudgeResult]:
        return {candidate.doc_id: self.judge_relation(anchor, candidate) for candidate in candidates}

    def judge_relation_with_signals(self, anchor: DocBrief, candidate: DocBrief, signals: JudgeSignals) -> JudgeResult:
        return self.judge_relation(anchor, candidate)

    def judge_relations_with_signals(self, anchor: DocBrief, candidates: list[tuple[DocBrief, JudgeSignals]]) -> dict[str, JudgeResult]:
        return {candidate.doc_id: self.judge_relation_with_signals(anchor, candidate, signals) for candidate, signals in candidates}

    def plan_query_strategy(self, payload: dict) -> dict:
        return {}

    def curate_memory(self, anchor: DocBrief, accepted: list[LogicEdge], rejected: list[str]) -> dict:
        raise NotImplementedError

    def _build_generic_judge_examples(self) -> str:
        examples = [
            {
                "label": "positive",
                "anchor_title": "Hybrid Retrieval",
                "anchor_text": "The retriever combines geometric recall with logical expansions and then ranks the merged candidates.",
                "candidate_title": "Candidate Fusion",
                "candidate_text": "The fusion stage computes a weighted score over the geometric score and the logic score.",
                "expected_relation_type": "implementation_detail",
                "why": "The candidate defines the scoring mechanism used by the anchor.",
            },
            {
                "label": "positive",
                "anchor_title": "Subagents",
                "anchor_text": "The system includes a profiler role, a scout role, a judge role, and a curator role.",
                "candidate_title": "Relation Judge",
                "candidate_text": "The judge role verifies whether an anchor and a candidate should form a durable edge.",
                "expected_relation_type": "prerequisite",
                "why": "The candidate is one of the explicitly listed roles in the anchor.",
            },
            {
                "label": "positive",
                "anchor_title": "Jump Policy",
                "anchor_text": "The policy decides whether a logical candidate may enter the final ranker.",
                "candidate_title": "Candidate Fusion",
                "candidate_text": "Only approved logical candidates contribute logic score to the final ranking stage.",
                "expected_relation_type": "supporting_evidence",
                "why": "The candidate explains the downstream effect of the policy's approval.",
            },
            {
                "label": "positive",
                "anchor_title": "Public policy should prioritize public transit",
                "anchor_text": "The argument claims cities should invest in transit instead of adding highway capacity.",
                "candidate_title": "Road expansion remains the best congestion solution",
                "candidate_text": "The counterargument claims adding lanes improves mobility more reliably than transit spending.",
                "expected_relation_type": "comparison",
                "why": "Both documents discuss the same policy topic from contrasting positions.",
            },
            {
                "label": "negative",
                "anchor_title": "Logic Overlay Graph",
                "anchor_text": "The overlay stores durable document-to-document relations used after initial recall.",
                "candidate_title": "ANN Metrics",
                "candidate_text": "Metrics such as recall and MRR summarize retrieval quality.",
                "expected_relation_type": "none",
                "why": "The topics belong to the same project but do not form a durable document edge.",
            },
            {
                "label": "negative",
                "anchor_title": "Document Profiler",
                "anchor_text": "The profiler produces a structured brief for an input document.",
                "candidate_title": "SQLite Job Registry",
                "candidate_text": "A registry stores job ids, states, and timestamps for background workers.",
                "expected_relation_type": "none",
                "why": "There is no direct semantic dependency between profiling and job persistence.",
            },
            {
                "label": "negative",
                "anchor_title": "Background Jobs",
                "anchor_text": "Workers run expensive offline tasks through a lightweight registry.",
                "candidate_title": "Public API Service",
                "candidate_text": "The service exposes endpoints that can submit jobs to the registry.",
                "expected_relation_type": "none",
                "why": "A service using the same registry is not by itself durable supporting evidence for the worker design.",
            },
        ]
        return "\n".join(json.dumps(example, ensure_ascii=False) for example in examples)

    def _build_query_strategy_examples(self) -> str:
        examples = [
            {
                "query": "culture debate about tradition and public policy",
                "signals": {
                    "dataset_hint": "arguana",
                    "agreement_ratio": 0.0,
                    "query_specificity": 0.16,
                    "graph_available": False,
                },
                "expected": {
                    "mode": "dense_only",
                    "sparse_gate": 0.0,
                    "allow_sparse_only": False,
                    "graph_gate": 0.0,
                    "sparse_boost": 0.0,
                    "novelty_bias": 0.0,
                    "reason": "Argumentative query with disagreement between dense and sparse signals.",
                },
            },
            {
                "query": "Which evidence supports the study claim about disease?",
                "signals": {
                    "dataset_hint": "scifact",
                    "agreement_ratio": 0.5,
                    "query_specificity": 0.48,
                    "graph_available": False,
                },
                "expected": {
                    "mode": "dense_plus_sparse",
                    "sparse_gate": 0.9,
                    "allow_sparse_only": True,
                    "graph_gate": 0.0,
                    "sparse_boost": 1.1,
                    "novelty_bias": 0.95,
                    "reason": "Scientific claim query with exact terminology support.",
                },
            },
            {
                "query": "How does jump policy affect hybrid retrieval scoring?",
                "signals": {
                    "dataset_hint": "gl_hnsw_demo",
                    "agreement_ratio": 0.5,
                    "query_specificity": 0.34,
                    "graph_available": True,
                },
                "expected": {
                    "mode": "dense_sparse_graph",
                    "sparse_gate": 0.95,
                    "allow_sparse_only": True,
                    "graph_gate": 0.95,
                    "sparse_boost": 1.0,
                    "novelty_bias": 1.0,
                    "reason": "Technical query over a structured corpus with durable graph edges.",
                },
            },
        ]
        return "\n".join(json.dumps(example, ensure_ascii=False) for example in examples)


class StubProvider(ProviderBase):
    def embed_texts(self, texts: Iterable[str]) -> np.ndarray:
        return np.vstack([deterministic_vector(text, self.embedding_dim) for text in texts]).astype(np.float32)

    def profile_doc(self, doc: DocRecord) -> DocBrief:
        tokens = tokenize(doc.text)
        entities = sorted({token for token in tokens if token in {"hnsw", "deepagents", "fastapi", "sqlite", "memory", "subagents", "skills"}})
        claims = [sentence.strip() for sentence in doc.text.split(". ")[:2] if sentence.strip()]
        relation_hints = top_terms(doc.text, limit=3)
        return enrich_brief(
            DocBrief(
            doc_id=doc.doc_id,
            title=doc.title,
            summary=" ".join(claims)[:320],
            entities=entities,
            keywords=top_terms(doc.text, limit=6),
            claims=claims[:3],
            relation_hints=relation_hints,
            metadata=doc.metadata,
            )
        )

    def propose_candidates(self, anchor: DocBrief, corpus: list[DocBrief]) -> list[CandidateProposal]:
        anchor_terms = set(anchor.keywords + anchor.entities + anchor.relation_hints)
        proposals: list[CandidateProposal] = []
        for candidate in corpus:
            if candidate.doc_id == anchor.doc_id:
                continue
            candidate_terms = set(candidate.keywords + candidate.entities)
            overlap = sorted(anchor_terms & candidate_terms)
            if not overlap:
                continue
            score_hint = min(0.99, 0.2 + 0.15 * len(overlap))
            proposals.append(
                CandidateProposal(
                    doc_id=candidate.doc_id,
                    reason=f"shared terms: {', '.join(overlap[:4])}",
                    query=" ".join(overlap[:4]),
                    score_hint=score_hint,
                )
            )
        proposals.sort(key=lambda item: (-item.score_hint, item.doc_id))
        return proposals[:6]

    def judge_relation(self, anchor: DocBrief, candidate: DocBrief) -> JudgeResult:
        shared = sorted(set(anchor.keywords + anchor.entities) & set(candidate.keywords + candidate.entities))
        if not shared:
            return JudgeResult(
                accepted=False,
                relation_type="comparison",
                confidence=0.0,
                evidence_spans=[],
                rationale="no meaningful overlap",
                support_score=0.0,
                contradiction_flags=["no_overlap"],
                decision_reason="local overlap check failed",
            )
        relation = "same_concept"
        if "memory" in shared:
            relation = "implementation_detail"
        elif "subagents" in shared:
            relation = "prerequisite"
        elif "retrieval" in shared or "logic" in shared:
            relation = "supporting_evidence"
        confidence = min(0.95, 0.45 + 0.12 * len(shared))
        return JudgeResult(
            accepted=confidence >= 0.6,
            relation_type=relation,
            confidence=confidence,
            evidence_spans=[anchor.summary[:160], candidate.summary[:160]],
            rationale=f"shared terms: {', '.join(shared[:5])}",
            support_score=min(1.0, 0.25 * len(shared)),
            contradiction_flags=[],
            decision_reason="accepted by deterministic overlap heuristic" if confidence >= 0.6 else "confidence below heuristic threshold",
            semantic_relation_label=relation,
            canonical_relation=relation,
            utility_score=min(1.0, confidence * 0.9),
            uncertainty=max(0.0, 1.0 - confidence),
        )

    def curate_memory(self, anchor: DocBrief, accepted: list[LogicEdge], rejected: list[str]) -> dict:
        aliases: dict[str, list[str]] = {}
        for entity in anchor.entities:
            aliases[entity] = sorted({entity, entity.replace("_", " ")})
        return {
            "active_hypotheses": anchor.relation_hints[:3],
            "successful_queries": [edge.edge_card_text for edge in accepted[:3]],
            "failed_queries": rejected[:3],
            "aliases": aliases,
            "relation_patterns": {
                edge.relation_type: sorted({edge.src_doc_id, edge.dst_doc_id}) for edge in accepted
            },
        }


class OpenAICompatibleProvider(StubProvider):
    def __init__(self, config: ProviderConfig, root_dir: Path | None = None):
        super().__init__(config, root_dir)
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key in environment variable {config.api_key_env}")
        self._chat = ChatOpenAI(
            base_url=config.base_url,
            api_key=api_key,
            model=config.chat_model,
            temperature=0,
            timeout=30,
            max_retries=1,
        )
        self._embeddings = None
        self._local_embedding_model = None
        self._local_embedding_tokenizer = None
        self.require_remote = os.getenv("GL_HNSW_REQUIRE_REMOTE", "0") == "1"
        self.trace_path = self.root_dir / "data" / "workspace" / "remote_provider_traces.jsonl" if self.root_dir else None
        if config.embedding_model in {"bge-m3", "BAAI/bge-m3"}:
            self._init_local_bge_m3()
        else:
            self._embeddings = OpenAIEmbeddings(
                base_url=config.base_url,
                api_key=api_key,
                model=config.embedding_model,
            )

    def _trace_remote(self, stage: str, status: str, detail: str = "") -> None:
        if self.trace_path is None:
            return
        append_jsonl(
            self.trace_path,
            [
                {
                    "timestamp": utc_now(),
                    "stage": stage,
                    "status": status,
                    "detail": detail[:240],
                    "model": self.config.chat_model,
                }
            ],
        )

    def _handle_remote_failure(self, stage: str, exc: Exception) -> None:
        self._trace_remote(stage, "fallback", str(exc))
        if self.require_remote:
            raise RuntimeError(f"Remote provider call failed during {stage}: {exc}") from exc

    def _init_local_bge_m3(self) -> None:
        from huggingface_hub import snapshot_download
        from transformers import AutoModel, AutoTokenizer

        candidate_paths = [
            os.environ.get("GL_HNSW_LOCAL_BGE_M3_PATH"),
            str((os.path.expanduser("~/ag_hnsw/data/cache/models/bge-m3"))),
        ]
        model_path = None
        for candidate in candidate_paths:
            if candidate and os.path.exists(candidate):
                model_path = candidate
                break
        if model_path is None:
            try:
                model_path = snapshot_download("BAAI/bge-m3", local_files_only=True)
            except Exception:
                original = os.environ.get("HF_ENDPOINT")
                os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
                model_path = snapshot_download("BAAI/bge-m3")
                if original is None:
                    os.environ.pop("HF_ENDPOINT", None)
                else:
                    os.environ["HF_ENDPOINT"] = original
        self._local_embedding_tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        self._local_embedding_model = AutoModel.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        self._local_embedding_model.eval()

    def _parse_json(self, content: str):
        text = content.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
            if not match:
                raise
            return json.loads(match.group(1))

    def _invoke_json(self, system_prompt: str, user_prompt: str, *, thinking: bool = False, stage: str = "generic"):
        from langchain_core.messages import HumanMessage, SystemMessage

        kwargs = {}
        if thinking:
            kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
        else:
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        try:
            response = self._chat.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)], **kwargs)
            content = response.content if isinstance(response.content, str) else "".join(part.get("text", "") for part in response.content)
            payload = self._parse_json(content)
            self._trace_remote(stage, "success")
            return payload
        except Exception as exc:
            self._trace_remote(stage, "error", str(exc))
            raise

    def _chunk(self, items: list, size: int) -> list[list]:
        return [items[i : i + size] for i in range(0, len(items), size)]

    def _candidate_shortlist(self, anchor: DocBrief, corpus: list[DocBrief], limit: int = 8) -> list[DocBrief]:
        if not corpus:
            return []
        anchor_terms = set(anchor.keywords + anchor.entities + anchor.relation_hints)
        texts = [f"{anchor.title}\n{anchor.summary}"] + [f"{brief.title}\n{brief.summary}" for brief in corpus]
        vectors = self.embed_texts(texts)
        anchor_vec = vectors[0]
        ranked: list[tuple[float, DocBrief]] = []
        for idx, brief in enumerate(corpus, start=1):
            overlap = len(anchor_terms & set(brief.keywords + brief.entities))
            dense_score = float(np.dot(anchor_vec, vectors[idx]))
            topic_cluster_bonus = 0.0
            anchor_cluster = str(anchor.metadata.get("topic_cluster", ""))
            candidate_cluster = str(brief.metadata.get("topic_cluster", ""))
            if anchor_cluster and anchor_cluster == candidate_cluster:
                topic_cluster_bonus += 0.14
            anchor_stance = str(anchor.metadata.get("stance", ""))
            candidate_stance = str(brief.metadata.get("stance", ""))
            if anchor_stance and candidate_stance and anchor_stance != candidate_stance:
                topic_cluster_bonus += 0.08
            score = dense_score + 0.08 * min(overlap, 4) + topic_cluster_bonus
            ranked.append((score, brief))
        ranked.sort(key=lambda item: (-item[0], item[1].doc_id))
        return [brief for _, brief in ranked[:limit]]

    def _judge_instruction(self) -> str:
        return (
            "You are a relation judge. Return JSON only. Prefer precision over recall. "
            "Use relation types: supporting_evidence, implementation_detail, same_concept, comparison, prerequisite. "
            "You may also use canonical_relation='none' when the pair is related but not useful as a durable retrieval edge. "
            "implementation_detail means the candidate defines a mechanism, formula, backend, or component directly used by the anchor. "
            "supporting_evidence means the candidate explains, constrains, or gates a claim in the anchor; do not use it for mere co-usage, API exposure, or two components that share the same registry or service surface. "
            "prerequisite means the candidate is an explicitly named role or earlier step required by the anchor. "
            "comparison is appropriate for debate or argument corpora when two documents address the same topic from contrasting or alternative positions. "
            "Use the supplied signals to judge edge utility, and abstain when utility is low or risk flags dominate."
        )

    def _query_strategy_instruction(self) -> str:
        return (
            "You are a query strategy agent for retrieval. Return JSON only. "
            "You receive local retrieval signals and must decide whether to use dense only, dense plus sparse, "
            "or dense plus sparse plus graph expansion. Prefer stable recall gains over aggressive lexical drift. "
            "When signals look argumentative, opinion-oriented, or dense and sparse strongly disagree, choose dense_only. "
            "When terminology is exact and supported by both query and candidate evidence, choose dense_plus_sparse. "
            "Only enable graph expansion for structured technical corpora with durable graph edges. "
            "You are not ranking documents directly; you are selecting a safe retrieval strategy."
        )

    def _verdict_from_payload(self, payload: dict) -> JudgeResult:
        canonical_relation = str(payload.get("canonical_relation", payload.get("relation_type", "comparison")))
        accepted = bool(payload.get("accepted", False))
        if canonical_relation == "none":
            accepted = False
            relation_type = "comparison"
        else:
            relation_type = canonical_relation
        return JudgeResult(
            accepted=accepted,
            relation_type=relation_type,
            confidence=float(payload.get("confidence", 0.0)),
            evidence_spans=[str(item) for item in payload.get("evidence_spans", [])][:4],
            rationale=str(payload.get("rationale", ""))[:200],
            support_score=float(payload.get("support_score", payload.get("confidence", 0.0))),
            contradiction_flags=[str(item) for item in payload.get("contradiction_flags", [])][:4],
            decision_reason=str(payload.get("decision_reason", ""))[:200],
            semantic_relation_label=str(payload.get("semantic_relation_label", relation_type))[:80],
            canonical_relation=canonical_relation,
            utility_score=float(payload.get("utility_score", 0.0)),
            uncertainty=float(payload.get("uncertainty", max(0.0, 1.0 - float(payload.get("confidence", 0.0))))),
        )

    def _first_sentences(self, text: str, limit: int = 3) -> list[str]:
        return [piece.strip() for piece in re.split(r"(?<=[.!?])\s+|\n+", text) if piece.strip()][:limit]

    def _merge_unique(self, *groups: list[str], limit: int) -> list[str]:
        merged: list[str] = []
        seen: set[str] = set()
        for group in groups:
            for item in group:
                value = str(item).strip()
                if not value:
                    continue
                key = value.lower()
                if key in seen:
                    continue
                seen.add(key)
                merged.append(value)
                if len(merged) >= limit:
                    return merged
        return merged

    def _derive_title(self, doc: DocRecord, claims: list[str]) -> str:
        if doc.title.strip():
            return doc.title.strip()
        basis = next((claim.strip() for claim in claims if claim.strip()), "")
        if not basis:
            basis = next((piece for piece in self._first_sentences(doc.text, limit=1) if piece.strip()), doc.doc_id)
        words = basis.split()
        return (" ".join(words[:12]).strip(" .,:;") or doc.doc_id).strip()

    def _infer_argument_stance(self, doc: DocRecord, title: str, claims: list[str]) -> str:
        doc_id = doc.doc_id.lower()
        if "-pro" in doc_id or doc_id.endswith("_pro"):
            return "pro"
        if "-con" in doc_id or doc_id.endswith("_con"):
            return "con"
        text = " ".join([title, *claims, doc.text[:320]]).lower()
        positive_hits = sum(1 for cue in {"should", "must", "benefit", "support", "improve", "protect", "effective"} if cue in text)
        negative_hits = sum(1 for cue in {"not", "never", "harm", "risk", "oppose", "against", "worse", "ineffective"} if cue in text)
        if positive_hits >= negative_hits + 2:
            return "pro"
        if negative_hits >= positive_hits + 2:
            return "con"
        return ""

    def _topic_cluster(self, title: str, claims: list[str], doc: DocRecord) -> str:
        cluster_terms = self._merge_unique(
            [term for term in top_terms(title, limit=4) if term not in {"study", "paper", "argument", "essay"}],
            [term for term in top_terms(" ".join(claims), limit=6) if term not in {"study", "paper", "argument", "essay"}],
            [term for term in top_terms(doc.text, limit=8) if term not in {"study", "paper", "argument", "essay"}],
            limit=4,
        )
        return "-".join(cluster_terms[:3])

    def _profile_dataset_hints(self, doc: DocRecord, title: str, summary: str, claims: list[str]) -> tuple[list[str], list[str], dict]:
        dataset = str(doc.metadata.get("source_dataset", "")).lower()
        metadata = dict(doc.metadata)
        if dataset:
            metadata["source_dataset"] = dataset
        keywords: list[str] = []
        relation_hints: list[str] = []
        cluster = self._topic_cluster(title, claims, doc)
        if cluster:
            metadata["topic_cluster"] = cluster
        if dataset == "arguana":
            metadata.setdefault("topic", "argument")
            metadata["doc_kind"] = "argument"
            stance = self._infer_argument_stance(doc, title, claims)
            if stance:
                metadata["stance"] = stance
                relation_hints.append(f"stance_{stance}")
            relation_hints.extend(["debate", "argument", "comparison"])
            keywords.extend(top_terms(" ".join([title, summary, *claims]), limit=6))
        elif dataset == "scifact":
            metadata.setdefault("topic", "scientific_claims")
            metadata["doc_kind"] = "evidence"
            relation_hints.extend(["claim", "evidence", "study"])
            keywords.extend(top_terms(" ".join([title, summary, *claims]), limit=6))
        elif dataset == "nfcorpus":
            metadata.setdefault("topic", "clinical_retrieval")
            metadata["doc_kind"] = "medical_passage"
            relation_hints.extend(["clinical", "condition", "treatment"])
            keywords.extend(top_terms(" ".join([title, summary, *claims]), limit=6))
        return keywords, relation_hints, metadata

    def _postprocess_profile(self, doc: DocRecord, payload: dict | None) -> DocBrief:
        payload = payload or {}
        fallback_claims = self._first_sentences(doc.text, limit=3)
        claims = self._merge_unique([str(item) for item in payload.get("claims", [])], fallback_claims, limit=4)
        title = self._derive_title(doc, claims)
        summary = str(payload.get("summary", "")).strip()[:320]
        if not summary:
            summary = " ".join(claims[:2])[:320]
        extra_keywords, extra_hints, metadata = self._profile_dataset_hints(doc, title, summary, claims)
        keywords = self._merge_unique(
            [str(item) for item in payload.get("keywords", [])],
            top_terms(title, limit=4),
            top_terms(summary, limit=6),
            extra_keywords,
            limit=10,
        )
        relation_hints = self._merge_unique(
            [str(item) for item in payload.get("relation_hints", [])],
            extra_hints,
            top_terms(" ".join(claims), limit=4),
            limit=8,
        )
        entities = self._merge_unique(
            [str(item) for item in payload.get("entities", [])],
            [title] if title and len(title.split()) <= 8 else [],
            limit=10,
        )
        brief = enrich_brief(
            DocBrief(
                doc_id=doc.doc_id,
                title=title,
                summary=summary,
                entities=entities,
                keywords=keywords,
                claims=claims,
                relation_hints=relation_hints,
                metadata=metadata,
            )
        )
        brief.metadata.update({key: value for key, value in metadata.items() if value})
        return brief

    def plan_query_strategy(self, payload: dict) -> dict:
        try:
            return self._invoke_json(
                self._query_strategy_instruction(),
                "\n".join(
                    part
                    for part in [
                        json.dumps(
                            {
                                "task": "Choose a retrieval strategy from local signals.",
                                "payload": payload,
                                "output_schema": {
                                    "mode": "string",
                                    "sparse_gate": "float",
                                    "allow_sparse_only": "boolean",
                                    "graph_gate": "float",
                                    "sparse_boost": "float",
                                    "novelty_bias": "float",
                                    "reason": "string",
                                    "uncertainty": "float",
                                },
                            },
                            ensure_ascii=False,
                        ),
                        "Few-shot examples:" if self.query_strategy_few_shot_text else "",
                        self.query_strategy_few_shot_text,
                    ]
                    if part
                ),
                thinking=self.live_reasoning["query_strategy"],
                stage="query_strategy",
            )
        except Exception as exc:
            self._handle_remote_failure("query_strategy", exc)
            return {}

    def _embed_local_bge_m3(self, texts: list[str]) -> np.ndarray:
        import torch

        batches: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(texts), 8):
                batch = texts[start : start + 8]
                encoded = self._local_embedding_tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                outputs = self._local_embedding_model(**encoded)
                hidden = outputs.last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
                batches.append(normalized.cpu().numpy().astype(np.float32))
        return np.vstack(batches)

    def embed_texts(self, texts: Iterable[str]) -> np.ndarray:
        items = list(texts)
        if self._local_embedding_model is not None:
            return self._embed_local_bge_m3(items)
        vectors = self._embeddings.embed_documents(items)
        return np.asarray(vectors, dtype=np.float32)

    def profile_doc(self, doc: DocRecord) -> DocBrief:
        try:
            payload = self._invoke_json(
                "You are a document profiler. Return JSON only.",
                "\n".join(
                    [
                        "Return a JSON object with keys: summary, entities, keywords, claims, relation_hints.",
                        f"title: {doc.title}",
                        f"text: {doc.text}",
                    ]
                ),
                thinking=False,
                stage="profile_doc",
            )
            return self._postprocess_profile(doc, payload)
        except Exception as exc:
            self._handle_remote_failure("profile_doc", exc)
            return self._postprocess_profile(doc, None)

    def profile_docs(self, docs: list[DocRecord]) -> list[DocBrief]:
        if not docs:
            return []
        results: dict[str, DocBrief] = {}
        for batch in self._chunk(docs, 4):
            try:
                payload = self._invoke_json(
                    "You are a document profiler. Return JSON only.",
                    json.dumps(
                        {
                            "task": "Profile each document and return a JSON array.",
                            "documents": [
                                {"doc_id": doc.doc_id, "title": doc.title, "text": doc.text}
                                for doc in batch
                            ],
                            "output_schema": [
                                {
                                    "doc_id": "string",
                                    "summary": "string",
                                    "entities": ["string"],
                                    "keywords": ["string"],
                                    "claims": ["string"],
                                    "relation_hints": ["string"],
                                }
                            ],
                        },
                        ensure_ascii=False,
                    ),
                    stage="profile_docs_batch",
                )
                items = payload if isinstance(payload, list) else payload.get("documents", [])
                for item in items:
                    doc_id = str(item.get("doc_id", ""))
                    source = next((doc for doc in batch if doc.doc_id == doc_id), None)
                    if source is None:
                        continue
                    results[doc_id] = self._postprocess_profile(source, item)
            except Exception as exc:
                self._handle_remote_failure("profile_docs_batch", exc)
            for doc in batch:
                if doc.doc_id in results:
                    continue
                if self.require_remote:
                    results[doc.doc_id] = self.profile_doc(doc)
                else:
                    results.setdefault(doc.doc_id, self.profile_doc(doc))
        return [results[doc.doc_id] for doc in docs]

    def propose_candidates(self, anchor: DocBrief, corpus: list[DocBrief]) -> list[CandidateProposal]:
        try:
            shortlist = self._candidate_shortlist(anchor, [brief for brief in corpus if brief.doc_id != anchor.doc_id], limit=8)
            candidate_rows = [
                {
                    "doc_id": brief.doc_id,
                    "title": brief.title,
                    "summary": brief.summary,
                    "keywords": brief.keywords,
                    "entities": brief.entities,
                }
                for brief in shortlist
            ]
            payload = self._invoke_json(
                "You are a corpus scout. Return JSON only.",
                json.dumps(
                    {
                        "task": "Select the most promising candidate docs for a logic edge from the anchor. Favor precise matches over recall.",
                        "anchor": {
                            "doc_id": anchor.doc_id,
                            "title": anchor.title,
                            "summary": anchor.summary,
                            "keywords": anchor.keywords,
                            "entities": anchor.entities,
                            "relation_hints": anchor.relation_hints,
                        },
                        "candidates": candidate_rows,
                        "output_schema": [
                            {"doc_id": "string", "reason": "string", "query": "string", "score_hint": "float"}
                        ],
                    },
                    ensure_ascii=False,
                ),
                thinking=self.live_reasoning["scout"],
                stage="corpus_scout",
            )
            items = payload if isinstance(payload, list) else payload.get("candidates", [])
            proposals: list[CandidateProposal] = []
            for item in items[:4]:
                doc_id = str(item.get("doc_id", ""))
                if not doc_id:
                    continue
                proposals.append(
                    CandidateProposal(
                        doc_id=doc_id,
                        reason=str(item.get("reason", ""))[:160],
                        query=str(item.get("query", ""))[:160],
                        score_hint=float(item.get("score_hint", 0.5)),
                    )
                )
            return proposals or super().propose_candidates(anchor, corpus)
        except Exception as exc:
            self._handle_remote_failure("corpus_scout", exc)
            return super().propose_candidates(anchor, corpus)

    def judge_relation(self, anchor: DocBrief, candidate: DocBrief) -> JudgeResult:
        return self.judge_relation_with_signals(
            anchor,
            candidate,
            JudgeSignals(
                dense_score=0.0,
                sparse_score=0.0,
                overlap_score=0.0,
                content_overlap_score=0.0,
                mention_score=0.0,
                role_listing_score=0.0,
                forward_reference_score=0.0,
                reverse_reference_score=0.0,
                direction_score=0.0,
                local_support=0.0,
                utility_score=0.0,
                best_relation="comparison",
                stage_pair="",
                risk_flags=[],
                relation_fit_scores={},
            ),
        )

    def judge_relation_with_signals(self, anchor: DocBrief, candidate: DocBrief, signals: JudgeSignals) -> JudgeResult:
        try:
            payload = self._invoke_json(
                self._judge_instruction(),
                "\n".join(
                    part
                    for part in [
                        json.dumps(
                            {
                                "task": "Judge whether the candidate should become a durable logic edge from the anchor.",
                                "anchor": to_jsonable(anchor),
                                "candidate": to_jsonable(candidate),
                                "signals": to_jsonable(signals),
                                "output_schema": {
                                    "accepted": "boolean",
                                    "canonical_relation": "string",
                                    "semantic_relation_label": "string",
                                    "confidence": "float",
                                    "utility_score": "float",
                                    "uncertainty": "float",
                                    "evidence_spans": ["string"],
                                    "rationale": "string",
                                    "support_score": "float",
                                    "contradiction_flags": ["string"],
                                    "decision_reason": "string",
                                },
                            },
                            ensure_ascii=False,
                        ),
                        "Few-shot examples:" if self.judge_few_shot_text else "",
                        self.judge_few_shot_text,
                    ]
                    if part
                ),
                thinking=self.live_reasoning["judge"],
                stage="judge_relation",
            )
            return self._verdict_from_payload(payload)
        except Exception as exc:
            self._handle_remote_failure("judge_relation", exc)
            return super().judge_relation(anchor, candidate)

    def judge_relations(self, anchor: DocBrief, candidates: list[DocBrief]) -> dict[str, JudgeResult]:
        default_signals = [
            (
                candidate,
                JudgeSignals(
                    dense_score=0.0,
                    sparse_score=0.0,
                    overlap_score=0.0,
                    content_overlap_score=0.0,
                    mention_score=0.0,
                    role_listing_score=0.0,
                    forward_reference_score=0.0,
                    reverse_reference_score=0.0,
                    direction_score=0.0,
                    local_support=0.0,
                    utility_score=0.0,
                    best_relation="comparison",
                    stage_pair="",
                    risk_flags=[],
                    relation_fit_scores={},
                ),
            )
            for candidate in candidates
        ]
        return self.judge_relations_with_signals(anchor, default_signals)

    def judge_relations_with_signals(self, anchor: DocBrief, candidates: list[tuple[DocBrief, JudgeSignals]]) -> dict[str, JudgeResult]:
        if not candidates:
            return {}
        if not candidates:
            return {}
        verdicts: dict[str, JudgeResult] = {}
        for batch in self._chunk(candidates, 6):
            try:
                payload = self._invoke_json(
                    self._judge_instruction(),
                    "\n".join(
                        part
                        for part in [
                            json.dumps(
                                {
                                    "task": "Judge each candidate against the anchor and return a JSON array.",
                                    "anchor": to_jsonable(anchor),
                                    "candidates": [
                                        {
                                            "candidate": to_jsonable(candidate),
                                            "signals": to_jsonable(signals),
                                        }
                                        for candidate, signals in batch
                                    ],
                                    "output_schema": [
                                        {
                                            "candidate_doc_id": "string",
                                            "accepted": "boolean",
                                            "canonical_relation": "string",
                                            "semantic_relation_label": "string",
                                            "confidence": "float",
                                            "utility_score": "float",
                                            "uncertainty": "float",
                                            "evidence_spans": ["string"],
                                            "rationale": "string",
                                            "support_score": "float",
                                            "contradiction_flags": ["string"],
                                            "decision_reason": "string",
                                        }
                                    ],
                                },
                                ensure_ascii=False,
                            ),
                            "Few-shot examples:" if self.judge_few_shot_text else "",
                            self.judge_few_shot_text,
                        ]
                        if part
                    ),
                    thinking=self.live_reasoning["judge"],
                    stage="judge_relations_batch",
                )
                items = payload if isinstance(payload, list) else payload.get("verdicts", [])
                for item in items:
                    candidate_doc_id = str(item.get("candidate_doc_id", ""))
                    if not candidate_doc_id:
                        continue
                    verdicts[candidate_doc_id] = self._verdict_from_payload(item)
            except Exception as exc:
                self._handle_remote_failure("judge_relations_batch", exc)
            for candidate, signals in batch:
                if candidate.doc_id in verdicts:
                    continue
                verdicts[candidate.doc_id] = self.judge_relation_with_signals(anchor, candidate, signals)
        return verdicts

    def curate_memory(self, anchor: DocBrief, accepted: list[LogicEdge], rejected: list[str]) -> dict:
        try:
            payload = self._invoke_json(
                "You are a memory curator. Return JSON only.",
                json.dumps(
                    {
                        "anchor": to_jsonable(anchor),
                        "accepted_edges": [to_jsonable(edge) for edge in accepted],
                        "rejected_docs": rejected,
                        "output_schema": {
                            "active_hypotheses": ["string"],
                            "successful_queries": ["string"],
                            "failed_queries": ["string"],
                            "aliases": {"entity": ["alias"]},
                            "relation_patterns": {"relation_type": ["doc ids or short patterns"]},
                        },
                    },
                    ensure_ascii=False,
                ),
                thinking=self.live_reasoning["curator"],
                stage="curate_memory",
            )
            return {
                "active_hypotheses": [str(item) for item in payload.get("active_hypotheses", [])][:4],
                "successful_queries": [str(item) for item in payload.get("successful_queries", [])][:4],
                "failed_queries": [str(item) for item in payload.get("failed_queries", [])][:4],
                "aliases": {
                    str(key): [str(alias) for alias in value][:4]
                    for key, value in payload.get("aliases", {}).items()
                },
                "relation_patterns": {
                    str(key): [str(value) for value in values][:6]
                    for key, values in payload.get("relation_patterns", {}).items()
                },
            }
        except Exception as exc:
            self._handle_remote_failure("curate_memory", exc)
            return super().curate_memory(anchor, accepted, rejected)


def build_provider(config: ProviderConfig, root_dir: Path | None = None) -> ProviderBase:
    if config.kind == "stub":
        return StubProvider(config, root_dir)
    if config.kind == "openai_compatible":
        return OpenAICompatibleProvider(config, root_dir)
    raise ValueError(f"Unsupported provider kind: {config.kind}")
