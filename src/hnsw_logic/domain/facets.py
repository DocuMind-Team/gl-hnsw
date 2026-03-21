from __future__ import annotations

from dataclasses import replace

from hnsw_logic.domain.models import DocBrief
from hnsw_logic.domain.tokens import tokenize

ROLE_TYPES = ("profiler", "scout", "judge", "curator")
MECHANISM_TERMS = {
    "alpha", "beta", "backend", "build", "candidate", "confidence", "curator", "discovery",
    "edge", "edges", "embedding", "expansion", "file", "files", "formula", "fusion", "gate",
    "graph", "hnsw", "index", "job", "judge", "logic", "memory", "neighbor", "neighbors",
    "payload", "persistent", "policy", "profiler", "query", "queue", "ranker", "registry",
    "revalidate", "revalidation", "routing", "scout", "search", "seed", "service", "shell",
    "sqlite", "storage", "subagent", "subagents", "summary", "target", "timestamp", "worker",
    "workers",
}
SURFACE_TERMS = {
    "api", "backend", "endpoint", "endpoints", "fastapi", "filesystem", "health", "inspection",
    "job", "jobs", "memory", "messages", "payload", "persistent", "queue", "registry", "search",
    "service", "shell", "sqlite", "state", "storage", "timestamp", "timestamps", "workspace",
}
PROCESS_TERMS = {
    "after", "before", "build", "candidate", "candidates", "discovery", "expand", "expansion",
    "judge", "merge", "merged", "polluting", "profile", "profiling", "propose", "proposes",
    "query", "rank", "ranking", "recall", "revalidate", "reviewed", "route", "scout", "search",
    "seed", "summarize", "worker",
}
OVERVIEW_TERMS = {"overview", "system", "project", "workflow", "architecture"}


def _all_terms(brief: DocBrief) -> set[str]:
    return {
        *tokenize(brief.title),
        *tokenize(brief.summary),
        *tokenize(" ".join(brief.claims)),
        *tokenize(" ".join(brief.keywords)),
        *tokenize(" ".join(brief.entities)),
        *tokenize(" ".join(brief.relation_hints)),
    }


def _infer_stage(brief: DocBrief, terms: set[str]) -> str:
    title_tokens = set(tokenize(brief.title))
    topic = str(brief.metadata.get("topic", "")).lower()
    if {"overlay", "graph"} & title_tokens:
        return "logic_graph"
    if {"jump", "policy"} & title_tokens:
        return "logic_policy"
    if {"fusion", "ranker"} & title_tokens:
        return "logic_fusion"
    if {"hybrid", "retrieval"} & title_tokens:
        return "retrieval_overview"
    if "overview" in title_tokens and "deepagents" in title_tokens:
        return "agent_overview"
    if {"subagents", "profiler", "scout", "judge", "curator"} & title_tokens:
        return "agent_roles"
    if {"memory", "backend"} & title_tokens:
        return "agent_memory"
    if {"jobs", "workers"} & title_tokens:
        return "ops_overview"
    if {"service", "fastapi"} & title_tokens:
        return "ops_service"
    if {"registry", "sqlite"} & title_tokens:
        return "ops_registry"
    if {"revalidation", "revalidate"} & title_tokens:
        return "ops_revalidation"
    if topic == "logic" and {"graph", "policy", "fusion", "overlay"} & terms:
        if {"policy", "jump"} & terms:
            return "logic_policy"
        if {"fusion", "ranker", "alpha", "beta"} & terms:
            return "logic_fusion"
        return "logic_graph"
    if topic in {"deepagents", "agents"} and {"memory", "backend"} & terms:
        return "agent_memory"
    if topic in {"deepagents", "agents"} and {"subagents", "profiler", "scout", "judge", "curator"} & terms:
        return "agent_roles"
    if topic == "ops" and {"registry", "sqlite"} & terms:
        return "ops_registry"
    if topic == "ops" and {"service", "fastapi"} & terms:
        return "ops_service"
    return ""


def _infer_doc_kind(brief: DocBrief, terms: set[str], stage: str) -> str:
    title_tokens = set(tokenize(brief.title))
    if {"overview", "retrieval"} & title_tokens or "overview" in stage:
        return "overview"
    if {"policy", "jump"} & title_tokens:
        return "policy"
    if {"fusion", "ranker"} & title_tokens:
        return "formula"
    if {"profiler", "scout", "judge", "curator"} & title_tokens:
        return "role"
    if {"memory", "backend"} & title_tokens:
        return "memory"
    if {"service", "fastapi"} & title_tokens:
        return "service"
    if {"registry", "sqlite"} & title_tokens:
        return "registry"
    if {"metrics", "report", "evaluation"} & title_tokens:
        return "report"
    if {"graph", "overlay"} & title_tokens:
        return "graph"
    if {"revalidation", "revalidate"} & title_tokens:
        return "maintenance"
    if OVERVIEW_TERMS & terms:
        return "overview"
    return "component"


def infer_brief_facets(brief: DocBrief) -> dict:
    terms = _all_terms(brief)
    title_tokens = set(tokenize(brief.title))
    stage = _infer_stage(brief, terms)
    doc_kind = _infer_doc_kind(brief, terms, stage)
    role_type = next((role for role in ROLE_TYPES if role in terms or role in title_tokens), "")
    mechanism_terms = sorted(term for term in terms & MECHANISM_TERMS if len(term) > 3)[:6]
    surface_terms = sorted(term for term in terms & SURFACE_TERMS if len(term) > 3)[:6]
    process_terms = sorted(term for term in terms & PROCESS_TERMS if len(term) > 3)[:6]
    aliases: list[str] = []
    if role_type:
        aliases.append(f"{role_type} role")
    if "memory" in terms and ("persistent" in terms or "storage" in terms):
        aliases.append("persistent memory")
    if "registry" in terms and "sqlite" in terms:
        aliases.append("sqlite registry")
    if "graph" in terms and "logic" in terms:
        aliases.append("logic graph")
    return {
        "stage": stage,
        "doc_kind": doc_kind,
        "role_type": role_type,
        "mechanism_terms": mechanism_terms,
        "surface_terms": surface_terms,
        "process_terms": process_terms,
        "aliases": aliases[:4],
    }


def enrich_brief(brief: DocBrief) -> DocBrief:
    facets = infer_brief_facets(brief)
    metadata = dict(brief.metadata)
    metadata.update({key: value for key, value in facets.items() if value})

    keywords: list[str] = []
    seen_keywords: set[str] = set()
    for term in [*brief.keywords, *facets["mechanism_terms"], *facets["surface_terms"]]:
        if term and term not in seen_keywords:
            keywords.append(term)
            seen_keywords.add(term)

    relation_hints: list[str] = []
    seen_hints: set[str] = set()
    stage_terms = str(facets["stage"]).replace("_", " ").split()
    doc_kind_terms = [str(facets["doc_kind"])] if facets["doc_kind"] else []
    role_terms = [str(facets["role_type"])] if facets["role_type"] else []
    for term in [*brief.relation_hints, *stage_terms, *doc_kind_terms, *role_terms, *facets["process_terms"]]:
        if term and term not in seen_hints:
            relation_hints.append(term)
            seen_hints.add(term)

    entities: list[str] = []
    seen_entities: set[str] = set()
    for term in [*brief.entities, *facets["aliases"]]:
        if term and term not in seen_entities:
            entities.append(term)
            seen_entities.add(term)

    return replace(
        brief,
        keywords=keywords[:10],
        relation_hints=relation_hints[:8],
        entities=entities[:10],
        metadata=metadata,
    )


def build_search_views(brief: DocBrief) -> dict[str, str]:
    facets = infer_brief_facets(brief)
    relation_text = " ".join([*brief.keywords, *brief.entities, *brief.relation_hints]).strip()
    structure_text = " ".join(
        [
            str(brief.metadata.get("topic", "")),
            facets["stage"].replace("_", " "),
            facets["doc_kind"],
            facets["role_type"],
            *facets["mechanism_terms"],
            *facets["surface_terms"],
            *facets["process_terms"],
            *facets["aliases"],
        ]
    ).strip()
    claims_text = " ".join(brief.claims).strip()
    summary_text = brief.summary.strip()
    title_text = brief.title.strip()
    full_text = "\n".join(part for part in [title_text, summary_text, claims_text, relation_text, structure_text] if part).strip()
    return {
        "title": title_text,
        "summary": summary_text,
        "claims": claims_text,
        "relation": relation_text,
        "structure": structure_text,
        "full": full_text,
    }
