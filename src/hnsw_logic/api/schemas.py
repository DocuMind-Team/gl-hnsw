from __future__ import annotations

from pydantic import BaseModel, Field


class JobRequest(BaseModel):
    payload: dict = Field(default_factory=dict)


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10


class SearchHitModel(BaseModel):
    doc_id: str
    title: str
    final_score: float
    geometric_score: float
    logical_score: float
    source_kind: str
    via_edge: str | None
    summary: str
    rank: int


class SearchResponseModel(BaseModel):
    query: str
    hits: list[SearchHitModel]
