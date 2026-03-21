from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from hnsw_logic.config.schema import ProviderConfig
from hnsw_logic.core.constants import RELATION_TYPES
from hnsw_logic.core.facets import enrich_brief
from hnsw_logic.core.models import DocBrief, DocRecord, LogicEdge
from hnsw_logic.core.utils import append_jsonl, to_jsonable, top_terms, utc_now
from hnsw_logic.embedding.provider_base import ProviderBase
from hnsw_logic.embedding.provider_stub import StubProvider
from hnsw_logic.embedding.provider_types import CandidateProposal, JudgeResult, JudgeSignals


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
            timeout=60,
            max_retries=2,
        )
        self._embeddings = None
        self._local_embedding_model = None
        self._local_embedding_tokenizer = None
        self._brief_vector_cache: dict[str, np.ndarray] = {}
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
        self._trace_remote(stage, "remote_failure", str(exc))
        if self.require_remote:
            raise RuntimeError(f"Remote provider call failed during {stage}: {exc}") from exc

    def _is_content_filter_error(self, exc: Exception) -> bool:
        message = str(exc).lower()
        return "content_filter" in message or "moderation block" in message or "'code': '421'" in message

    def _is_response_parse_error(self, exc: Exception) -> bool:
        if isinstance(exc, json.JSONDecodeError):
            return True
        message = str(exc).lower()
        return (
            "jsondecodeerror" in message
            or "expecting ',' delimiter" in message
            or "unterminated string" in message
            or "extra data" in message
            or "empty response body" in message
        )

    def _is_output_limit_error(self, exc: Exception) -> bool:
        message = str(exc).lower()
        return "router_output_limitation" in message or "output token rate limit exceeded" in message

    def _is_connection_error(self, exc: Exception) -> bool:
        message = str(exc).lower()
        return (
            "connection error" in message
            or "apiconnectionerror" in message
            or "connecterror" in message
            or "unexpected eof while reading" in message
            or "ssl:" in message
            or "timed out" in message
            or "timeout" in message
            or "remoteprotocolerror" in message
        )

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

        thinking_enabled = thinking
        current_user_prompt = user_prompt
        last_exc: Exception | None = None
        max_attempts = 5
        for attempt in range(max_attempts):
            kwargs = {
                "extra_body": {
                    "thinking": {"type": "enabled" if thinking_enabled else "disabled"}
                }
            }
            try:
                response = self._chat.invoke([SystemMessage(content=system_prompt), HumanMessage(content=current_user_prompt)], **kwargs)
                content = response.content if isinstance(response.content, str) else "".join(part.get("text", "") for part in response.content)
                if not str(content).strip():
                    raise ValueError("empty response body")
                payload = self._parse_json(content)
                self._trace_remote(stage, "success", f"attempt={attempt + 1}")
                return payload
            except Exception as exc:
                last_exc = exc
                self._trace_remote(stage, "error", f"attempt={attempt + 1}: {exc}")
                if attempt == 0 and (self._is_response_parse_error(exc) or "empty response body" in str(exc).lower()):
                    current_user_prompt = (
                        f"{user_prompt}\n\n"
                        "Previous response was not valid JSON. Return exactly one JSON object or JSON array matching the schema. "
                        "Do not include markdown fences, prose, or explanations outside JSON."
                    )
                    continue
                if thinking_enabled and self._is_output_limit_error(exc):
                    thinking_enabled = False
                    current_user_prompt = (
                        f"{user_prompt}\n\n"
                        "Keep the response concise. Return only the required JSON fields. "
                        "Use short evidence spans and a brief decision_reason."
                    )
                    continue
                if self._is_connection_error(exc) and attempt < max_attempts - 1:
                    backoff_seconds = min(8.0, 1.5 * (2**attempt))
                    self._trace_remote(stage, "retry", f"sleep={backoff_seconds:.1f}s after attempt={attempt + 1}")
                    time.sleep(backoff_seconds)
                    continue
                raise
        if last_exc is not None:
            raise last_exc
        raise RuntimeError(f"Remote provider call failed during {stage}: unknown error")

    def _chunk(self, items: list, size: int) -> list[list]:
        return [items[i : i + size] for i in range(0, len(items), size)]

    def _brief_vector_key(self, brief: DocBrief) -> str:
        return f"{brief.doc_id}\n{brief.title}\n{brief.summary}"

    def _brief_vector_text(self, brief: DocBrief) -> str:
        return f"{brief.title}\n{brief.summary}"

    def _brief_vectors(self, briefs: list[DocBrief]) -> np.ndarray:
        if not briefs:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        vectors: list[np.ndarray | None] = [None] * len(briefs)
        missing_indices: list[int] = []
        missing_texts: list[str] = []
        for index, brief in enumerate(briefs):
            cached = self._brief_vector_cache.get(self._brief_vector_key(brief))
            if cached is None:
                missing_indices.append(index)
                missing_texts.append(self._brief_vector_text(brief))
            else:
                vectors[index] = cached
        if missing_texts:
            encoded = self._embed_local_bge_m3(missing_texts) if self._local_embedding_model is not None else np.asarray(self._embeddings.embed_documents(missing_texts), dtype=np.float32)
            for offset, vector in enumerate(encoded):
                index = missing_indices[offset]
                brief = briefs[index]
                cached = np.asarray(vector, dtype=np.float32)
                self._brief_vector_cache[self._brief_vector_key(brief)] = cached
                vectors[index] = cached
        return np.vstack([vector for vector in vectors if vector is not None]).astype(np.float32)

    def _candidate_shortlist(self, anchor: DocBrief, corpus: list[DocBrief], limit: int = 8) -> list[DocBrief]:
        if not corpus:
            return []
        anchor_terms = set(anchor.keywords + anchor.entities + anchor.relation_hints)
        vectors = self._brief_vectors([anchor] + corpus)
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

    def _prefer_local_scout(self, anchor: DocBrief) -> bool:
        return True

    def _judge_instruction(self) -> str:
        return (
            "You are a relation judge. Return JSON only. Prefer precision over recall. "
            "Use relation types: supporting_evidence, implementation_detail, same_concept, comparison, prerequisite. "
            "You may also use canonical_relation='none' when the pair is related but not useful as a durable retrieval edge. "
            "implementation_detail means the candidate defines a mechanism, formula, backend, or component directly used by the anchor. "
            "supporting_evidence means the candidate explains, constrains, or gates a claim in the anchor; do not use it for mere co-usage, API exposure, or two components that share the same registry or service surface. "
            "prerequisite means the candidate is an explicitly named role or earlier step required by the anchor. "
            "comparison is appropriate for debate or argument corpora when two documents address the same topic from contrasting or alternative positions. "
            "For scientific or clinical corpora, prefer supporting_evidence when the candidate adds aligned risk, treatment, outcome, or progression evidence; prefer same_concept only when both documents clearly describe the same finding family rather than merely related methodology. "
            "For argumentative corpora, require either contrasting stance or a strong alternative-position signal; same-side topical overlap is not enough. "
            "When a structured topic-family identifier or sibling document family is present, treat it as a strong same-topic signal but never as sufficient evidence on its own. "
            "Use the supplied signals to judge edge utility, and abstain when utility is low or risk flags dominate."
        )

    def _review_instruction(self) -> str:
        return (
            "You are an edge reviewer for offline retrieval indexing. Return JSON only. "
            "You do not start from scratch: you review an existing judge verdict together with local signals. "
            "Focus on durable retrieval utility, uncertainty, and generic failure modes such as weak direction, "
            "methodology-only overlap, generic service-surface overlap, and foundational-but-not-actionable support. "
            "Approve only when the edge is likely to help retrieval across many queries, not just because the pair is topically related. "
            "For scientific or clinical corpora, reward clinically specific bridge terms and aligned outcome or treatment language when they increase retrieval surface. "
            "For argumentative corpora, approve comparison edges only when the pair creates a reusable contrast bridge rather than repeating the same stance. "
            "Do not reject a comparison as a duplicate only because both sides share topic terms; strong stance contrast plus a reusable contrast bridge should survive duplicate checks. "
            "Prefer same-topic family comparison bridges over broader cross-family policy analogies when both remain plausible. "
            "If the pair is related but not durable or not useful, set canonical_relation='none'. "
            "You may keep the original relation, reject it, or replace it with a safer canonical relation."
        )

    def _planner_instruction(self) -> str:
        return (
            "You are an offline indexing planner. Return JSON only. "
            "You receive a proposed anchor plan. Improve only notes or graph_potential if there is clear utility. "
            "Do not invent anchors that do not exist in the payload."
        )

    def _counterevidence_instruction(self) -> str:
        return (
            "You are a counterevidence checker for durable retrieval edges. Return JSON only. "
            "Decide whether a tentative edge should be kept after looking for duplicate bridges, weak direction, "
            "topic-only overlap, methodology-only overlap, or low retrieval utility. "
            "For comparison edges in argument corpora, treat opposing stances on the same topic as a reusable contrast bridge rather than a duplicate unless the content is effectively the same claim."
        )

    def _memory_learning_instruction(self) -> str:
        return (
            "You summarize offline indexing learnings. Return JSON only. "
            "Condense accepted patterns and rejected patterns into short reusable bullets. "
            "Only produce updates suitable for AGENTS.md learned sections or skill references."
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

    def _profile_source_payload(self, doc: DocRecord) -> dict:
        excerpt_sentences = self._first_sentences(doc.text, limit=4)
        excerpt = " ".join(excerpt_sentences).strip()
        if not excerpt:
            excerpt = doc.text.strip()
        excerpt = excerpt[:1200]
        return {
            "doc_id": doc.doc_id,
            "title": doc.title,
            "text_excerpt": excerpt,
            "top_terms": top_terms(doc.text, limit=12),
            "source_dataset": str(doc.metadata.get("source_dataset", "")),
            "topic_hint": str(doc.metadata.get("topic", "")),
            "text_length": len(doc.text),
        }

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

    def _topic_family_key(self, doc: DocRecord) -> str:
        doc_id = doc.doc_id.lower().replace("_", "-").strip()
        if not doc_id:
            return ""
        parts = [part for part in doc_id.split("-") if part]
        if len(parts) < 4:
            return ""
        last = parts[-1]
        if not any(char.isdigit() for char in last):
            return ""
        family = "-".join(parts[:-1]).strip("-")
        if len(family) < 12:
            return ""
        return family

    def _profile_dataset_hints(self, doc: DocRecord, title: str, summary: str, claims: list[str]) -> tuple[list[str], list[str], dict]:
        dataset = str(doc.metadata.get("source_dataset", "")).lower()
        metadata = dict(doc.metadata)
        if dataset:
            metadata["source_dataset"] = dataset
        keywords: list[str] = []
        relation_hints: list[str] = []
        context_text = " ".join([title, summary, *claims, doc.text[:1600]]).lower()
        cluster = self._topic_cluster(title, claims, doc)
        if cluster:
            metadata["topic_cluster"] = cluster
        family = self._topic_family_key(doc)
        if family:
            metadata["topic_family"] = family
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
            if any(term in context_text for term in {"risk factor", "risks", "burden", "dalys", "mortality", "deaths", "exposure"}):
                relation_hints.extend(["population risk", "chronic disease burden"])
                keywords.extend(["risk factors", "chronic disease"])
            if any(term in context_text for term in {"nutrition", "diet", "dietary", "food", "fasting glucose", "body mass index", "bmi", "metabolic"}):
                relation_hints.extend(["nutrition", "metabolic risk"])
                keywords.extend(["nutrition", "metabolic risk"])
            if any(term in context_text for term in {"global", "regional", "national", "population", "public health"}):
                relation_hints.append("population health")
                keywords.append("population health")
        elif dataset == "nfcorpus":
            metadata.setdefault("topic", "clinical_retrieval")
            metadata["doc_kind"] = "medical_passage"
            relation_hints.extend(["clinical", "condition", "treatment"])
            keywords.extend(top_terms(" ".join([title, summary, *claims]), limit=6))
            if any(term in context_text for term in {"risk factor", "risks", "burden", "mortality", "outcome", "survival", "prognosis"}):
                relation_hints.extend(["clinical risk", "disease burden"])
                keywords.extend(["clinical risk", "disease burden"])
            if any(term in context_text for term in {"nutrition", "diet", "dietary", "obesity", "metabolic", "glucose"}):
                relation_hints.extend(["nutrition", "metabolic health"])
                keywords.extend(["nutrition", "metabolic health"])
        return keywords, relation_hints, metadata

    def _postprocess_profile(self, doc: DocRecord, payload: dict | None) -> DocBrief:
        payload = payload or {}
        default_claims = self._first_sentences(doc.text, limit=3)
        claims = self._merge_unique([str(item) for item in payload.get("claims", [])], default_claims, limit=4)
        title = self._derive_title(doc, claims)
        summary = str(payload.get("summary", "")).strip()[:320]
        if not summary:
            summary = " ".join(claims[:2])[:320]
        extra_keywords, extra_hints, metadata = self._profile_dataset_hints(doc, title, summary, claims)
        source_dataset = str(metadata.get("source_dataset", "")).lower()
        keyword_groups = [
            [str(item) for item in payload.get("keywords", [])],
            top_terms(title, limit=4),
            top_terms(summary, limit=6),
            extra_keywords,
        ]
        relation_groups = [
            [str(item) for item in payload.get("relation_hints", [])],
            extra_hints,
            top_terms(" ".join(claims), limit=4),
        ]
        keyword_limit = 10
        relation_limit = 8
        if source_dataset in {"scifact", "nfcorpus"}:
            keyword_groups = [
                extra_keywords,
                [str(item) for item in payload.get("keywords", [])],
                top_terms(title, limit=4),
                top_terms(summary, limit=6),
            ]
            relation_groups = [
                extra_hints,
                [str(item) for item in payload.get("relation_hints", [])],
                top_terms(" ".join(claims), limit=4),
            ]
            keyword_limit = 12
            relation_limit = 10
        keywords = self._merge_unique(
            *keyword_groups,
            limit=keyword_limit,
        )
        relation_hints = self._merge_unique(
            *relation_groups,
            limit=relation_limit,
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

    def plan_indexing_batch(self, payload: dict) -> dict:
        try:
            return self._invoke_json(
                self._planner_instruction(),
                json.dumps(
                    {
                        "task": "Review and lightly improve the offline indexing plan.",
                        "payload": payload,
                        "output_schema": {
                            "graph_potential": "float",
                            "notes": ["string"],
                        },
                    },
                    ensure_ascii=False,
                ),
                thinking=self.live_reasoning["scout"],
                stage="plan_indexing_batch",
            )
        except Exception as exc:
            self._handle_remote_failure("plan_indexing_batch", exc)
            return super().plan_indexing_batch(payload)

    def check_counterevidence(self, anchor: DocBrief, candidate: DocBrief, signals: JudgeSignals, verdict: JudgeResult) -> dict:
        try:
            payload = self._invoke_json(
                self._counterevidence_instruction(),
                json.dumps(
                    {
                        "task": "Check whether this tentative edge has strong counterevidence or duplicate risk.",
                        "anchor": to_jsonable(anchor),
                        "candidate": to_jsonable(candidate),
                        "signals": to_jsonable(signals),
                        "verdict": to_jsonable(verdict),
                        "output_schema": {
                            "keep": "boolean",
                            "risk_flags": ["string"],
                            "counterevidence": ["string"],
                            "decision_reason": "string",
                            "risk_penalty": "float",
                        },
                    },
                    ensure_ascii=False,
                ),
                thinking=self.live_reasoning["reviewer"],
                stage="check_counterevidence",
            )
            return self._normalize_counterevidence_result(anchor, candidate, signals, verdict, payload)
        except Exception as exc:
            if self._is_content_filter_error(exc):
                self._trace_remote("check_counterevidence", "blocked_local", candidate.doc_id)
                return super().check_counterevidence(anchor, candidate, signals, verdict)
            if self._is_response_parse_error(exc) or self._is_output_limit_error(exc):
                self._trace_remote("check_counterevidence", "local_decision", candidate.doc_id)
                return super().check_counterevidence(anchor, candidate, signals, verdict)
            self._handle_remote_failure("check_counterevidence", exc)
            return super().check_counterevidence(anchor, candidate, signals, verdict)

    def check_counterevidence_many(
        self,
        anchor: DocBrief,
        candidates: list[tuple[DocBrief, JudgeSignals, JudgeResult]],
    ) -> dict[str, dict]:
        if not candidates:
            return {}
        results: dict[str, dict] = {}
        for batch in self._chunk(candidates, 8):
            batch_by_id = {candidate.doc_id: (candidate, signals, verdict) for candidate, signals, verdict in batch}
            try:
                payload = self._invoke_json(
                    self._counterevidence_instruction(),
                    json.dumps(
                        {
                            "task": "Check each tentative edge for duplicate, direction, and low-utility risk.",
                            "anchor": to_jsonable(anchor),
                            "candidates": [
                                {
                                    "candidate_doc_id": candidate.doc_id,
                                    "candidate": to_jsonable(candidate),
                                    "signals": to_jsonable(signals),
                                    "verdict": to_jsonable(verdict),
                                }
                                for candidate, signals, verdict in batch
                            ],
                            "output_schema": [
                                {
                                    "candidate_doc_id": "string",
                                    "keep": "boolean",
                                    "risk_flags": ["string"],
                                    "counterevidence": ["string"],
                                    "decision_reason": "string",
                                    "risk_penalty": "float",
                                }
                            ],
                        },
                        ensure_ascii=False,
                    ),
                    thinking=self.live_reasoning["reviewer"],
                    stage="check_counterevidence_batch",
                )
                items = payload if isinstance(payload, list) else payload.get("checks", [])
                for item in items:
                    candidate_doc_id = str(item.get("candidate_doc_id", ""))
                    if not candidate_doc_id:
                        continue
                    candidate_entry = batch_by_id.get(candidate_doc_id)
                    result = {
                        "keep": bool(item.get("keep", True)),
                        "risk_flags": [str(flag) for flag in item.get("risk_flags", [])][:8],
                        "counterevidence": [str(flag) for flag in item.get("counterevidence", [])][:8],
                        "decision_reason": str(item.get("decision_reason", ""))[:220],
                        "risk_penalty": float(item.get("risk_penalty", 0.0)),
                    }
                    if candidate_entry is None:
                        results[candidate_doc_id] = result
                    else:
                        candidate_doc, candidate_signals, candidate_verdict = candidate_entry
                        results[candidate_doc_id] = self._normalize_counterevidence_result(
                            anchor,
                            candidate_doc,
                            candidate_signals,
                            candidate_verdict,
                            result,
                        )
            except Exception as exc:
                if (self._is_content_filter_error(exc) or self._is_response_parse_error(exc) or self._is_output_limit_error(exc)) and len(batch) > 1:
                    midpoint = max(1, len(batch) // 2)
                    for partial in (batch[:midpoint], batch[midpoint:]):
                        results.update(self.check_counterevidence_many(anchor, partial))
                    continue
                if self._is_content_filter_error(exc) or self._is_response_parse_error(exc) or self._is_output_limit_error(exc):
                    status = "blocked_local" if self._is_content_filter_error(exc) else "local_decision"
                    for candidate, signals, verdict in batch:
                        self._trace_remote("check_counterevidence_batch", status, candidate.doc_id)
                        results[candidate.doc_id] = super().check_counterevidence(anchor, candidate, signals, verdict)
                    continue
                self._handle_remote_failure("check_counterevidence_batch", exc)
                results.update(super().check_counterevidence_many(anchor, batch))
            for candidate, signals, verdict in batch:
                if candidate.doc_id in results:
                    continue
                results[candidate.doc_id] = super().check_counterevidence(anchor, candidate, signals, verdict)
        return results

    def summarize_memory_learnings(self, payload: dict) -> dict:
        try:
            return self._invoke_json(
                self._memory_learning_instruction(),
                json.dumps(
                    {
                        "task": "Summarize reusable learning patterns from offline indexing outcomes.",
                        "payload": payload,
                        "output_schema": {
                            "learned_patterns": ["string"],
                            "failure_patterns": ["string"],
                            "reference_updates": {"path": ["string"]},
                        },
                    },
                    ensure_ascii=False,
                ),
                thinking=self.live_reasoning["curator"],
                stage="summarize_memory_learnings",
            )
        except Exception as exc:
            self._handle_remote_failure("summarize_memory_learnings", exc)
            return super().summarize_memory_learnings(payload)

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
                (
                    "You are a document profiler. Return JSON only. "
                    "Profile the document in neutral analytical language. "
                    "When the excerpt contains provocative, harmful, or highly emotional phrasing, "
                    "paraphrase it instead of quoting directly unless an exact phrase is essential."
                ),
                json.dumps(
                    {
                        "task": (
                            "Return a JSON object with keys: summary, entities, keywords, claims, relation_hints. "
                            "Use the excerpt and top terms as the primary grounding context."
                        ),
                        "document": self._profile_source_payload(doc),
                        "output_schema": {
                            "summary": "string",
                            "entities": ["string"],
                            "keywords": ["string"],
                            "claims": ["string"],
                            "relation_hints": ["string"],
                        },
                    },
                    ensure_ascii=False,
                ),
                thinking=False,
                stage="profile_doc",
            )
            return self._postprocess_profile(doc, payload)
        except Exception as exc:
            if self._is_content_filter_error(exc) or self._is_response_parse_error(exc) or self._is_output_limit_error(exc):
                status = "blocked_local" if self._is_content_filter_error(exc) else "local_decision"
                self._trace_remote("profile_doc", status, doc.doc_id)
                return self._postprocess_profile(doc, None)
            self._handle_remote_failure("profile_doc", exc)
            return self._postprocess_profile(doc, None)

    def profile_docs(self, docs: list[DocRecord], on_brief: Callable[[DocBrief], None] | None = None) -> list[DocBrief]:
        if not docs:
            return []
        results: dict[str, DocBrief] = {}

        def remember(brief: DocBrief) -> None:
            existing = results.get(brief.doc_id)
            results[brief.doc_id] = brief
            if existing is None and on_brief is not None:
                on_brief(brief)

        for batch in self._chunk(docs, 4):
            batch_by_id = {doc.doc_id: doc for doc in batch}
            try:
                payload = self._invoke_json(
                    (
                        "You are a document profiler. Return JSON only. "
                        "Profile each document in neutral analytical language. "
                        "Paraphrase provocative or harmful wording when a neutral summary can preserve the stance, topic, and retrieval utility."
                    ),
                    json.dumps(
                        {
                            "task": "Profile each document and return a JSON array.",
                            "documents": [
                                self._profile_source_payload(doc)
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
                    source = batch_by_id.get(doc_id)
                    if source is None:
                        continue
                    remember(self._postprocess_profile(source, item))
            except Exception as exc:
                if self._is_content_filter_error(exc) or self._is_response_parse_error(exc) or self._is_output_limit_error(exc):
                    if len(batch) > 1:
                        midpoint = max(1, len(batch) // 2)
                        for partial in (batch[:midpoint], batch[midpoint:]):
                            for brief in self.profile_docs(partial, on_brief=on_brief):
                                remember(brief)
                        continue
                    blocked = batch[0]
                    status = "blocked_local" if self._is_content_filter_error(exc) else "retry_single"
                    self._trace_remote("profile_docs_batch", status, blocked.doc_id)
                    remember(self.profile_doc(blocked))
                    continue
                self._handle_remote_failure("profile_docs_batch", exc)
            for doc in batch:
                if doc.doc_id in results:
                    continue
                remember(self.profile_doc(doc))
        return [results[doc.doc_id] for doc in docs]

    def propose_candidates(self, anchor: DocBrief, corpus: list[DocBrief]) -> list[CandidateProposal]:
        if self._prefer_local_scout(anchor):
            shortlist = self._candidate_shortlist(anchor, [brief for brief in corpus if brief.doc_id != anchor.doc_id], limit=6)
            return [
                CandidateProposal(
                    doc_id=brief.doc_id,
                    reason="local semantic shortlist for offline discovery",
                    query=" ".join((brief.keywords + brief.relation_hints + [brief.title])[:4]),
                    score_hint=0.72,
                )
                for brief in shortlist
            ]
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
            if self._is_content_filter_error(exc):
                self._trace_remote("judge_relation", "blocked_local", candidate.doc_id)
                return super().judge_relation(anchor, candidate)
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
        verdicts: dict[str, JudgeResult] = {}
        for batch in self._chunk(candidates, 8):
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
                if (self._is_content_filter_error(exc) or self._is_response_parse_error(exc) or self._is_output_limit_error(exc)) and len(batch) > 1:
                    midpoint = max(1, len(batch) // 2)
                    for partial in (batch[:midpoint], batch[midpoint:]):
                        verdicts.update(self.judge_relations_with_signals(anchor, partial))
                    continue
                if self._is_content_filter_error(exc) or self._is_response_parse_error(exc) or self._is_output_limit_error(exc):
                    blocked_ids = ",".join(candidate.doc_id for candidate, _ in batch)
                    status = "blocked_local" if self._is_content_filter_error(exc) else "retry_single"
                    self._trace_remote("judge_relations_batch", status, blocked_ids)
                else:
                    self._handle_remote_failure("judge_relations_batch", exc)
            for candidate, signals in batch:
                if candidate.doc_id in verdicts:
                    continue
                verdicts[candidate.doc_id] = self.judge_relation_with_signals(anchor, candidate, signals)
        return verdicts

    def review_relation_with_signals(self, anchor: DocBrief, candidate: DocBrief, signals: JudgeSignals, verdict: JudgeResult) -> JudgeResult:
        try:
            payload = self._invoke_json(
                self._review_instruction(),
                "\n".join(
                    part
                    for part in [
                        json.dumps(
                            {
                                "task": "Review the judged edge and decide whether it should survive into the offline retrieval graph.",
                                "anchor": to_jsonable(anchor),
                                "candidate": to_jsonable(candidate),
                                "signals": to_jsonable(signals),
                                "judge_verdict": to_jsonable(verdict),
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
                        "Few-shot examples:" if self.review_few_shot_text else "",
                        self.review_few_shot_text,
                    ]
                    if part
                ),
                thinking=self.live_reasoning["reviewer"],
                stage="review_relation",
            )
            return self._verdict_from_payload(payload)
        except Exception as exc:
            if self._is_content_filter_error(exc):
                self._trace_remote("review_relation", "blocked_local", candidate.doc_id)
                return super().review_relation_with_signals(anchor, candidate, signals, verdict)
            self._handle_remote_failure("review_relation", exc)
            return super().review_relation_with_signals(anchor, candidate, signals, verdict)

    def review_relations_with_signals(self, anchor: DocBrief, candidates: list[tuple[DocBrief, JudgeSignals, JudgeResult]]) -> dict[str, JudgeResult]:
        if not candidates:
            return {}
        verdicts: dict[str, JudgeResult] = {}
        for batch in self._chunk(candidates, 4):
            try:
                payload = self._invoke_json(
                    self._review_instruction(),
                    "\n".join(
                        part
                        for part in [
                            json.dumps(
                                {
                                    "task": "Review each judged edge and return a JSON array.",
                                    "anchor": to_jsonable(anchor),
                                    "candidates": [
                                        {
                                            "candidate": to_jsonable(candidate),
                                            "signals": to_jsonable(signals),
                                            "judge_verdict": to_jsonable(verdict),
                                        }
                                        for candidate, signals, verdict in batch
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
                            "Few-shot examples:" if self.review_few_shot_text else "",
                            self.review_few_shot_text,
                        ]
                        if part
                    ),
                    thinking=self.live_reasoning["reviewer"],
                    stage="review_relations_batch",
                )
                items = payload if isinstance(payload, list) else payload.get("verdicts", [])
                for item in items:
                    candidate_doc_id = str(item.get("candidate_doc_id", ""))
                    if not candidate_doc_id:
                        continue
                    verdicts[candidate_doc_id] = self._verdict_from_payload(item)
            except Exception as exc:
                if (self._is_content_filter_error(exc) or self._is_response_parse_error(exc) or self._is_output_limit_error(exc)) and len(batch) > 1:
                    midpoint = max(1, len(batch) // 2)
                    for partial in (batch[:midpoint], batch[midpoint:]):
                        verdicts.update(self.review_relations_with_signals(anchor, partial))
                    continue
                if self._is_content_filter_error(exc) or self._is_response_parse_error(exc) or self._is_output_limit_error(exc):
                    blocked_ids = ",".join(candidate.doc_id for candidate, _, _ in batch)
                    status = "blocked_local" if self._is_content_filter_error(exc) else "retry_single"
                    self._trace_remote("review_relations_batch", status, blocked_ids)
                else:
                    self._handle_remote_failure("review_relations_batch", exc)
            for candidate, signals, verdict in batch:
                if candidate.doc_id in verdicts:
                    continue
                verdicts[candidate.doc_id] = self.review_relation_with_signals(anchor, candidate, signals, verdict)
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
            if self._is_content_filter_error(exc):
                self._trace_remote("curate_memory", "blocked_local", anchor.doc_id)
                return super().curate_memory(anchor, accepted, rejected)
            self._handle_remote_failure("curate_memory", exc)
            return super().curate_memory(anchor, accepted, rejected)


def build_provider(config: ProviderConfig, root_dir: Path | None = None) -> ProviderBase:
    if config.kind == "stub":
        return StubProvider(config, root_dir)
    if config.kind == "openai_compatible":
        return OpenAICompatibleProvider(config, root_dir)
    raise ValueError(f"Unsupported provider kind: {config.kind}")
