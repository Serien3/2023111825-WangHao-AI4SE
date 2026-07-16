from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ErrorResponse(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    request_id: str


class HealthResponse(BaseModel):
    status: str
    repository: dict[str, Any]
    llm: dict[str, Any]
    ml: dict[str, Any]
    history: dict[str, Any]


class ReviewRequest(BaseModel):
    scope: Literal["file", "workspace"]
    path: str | None = None
    profile: Literal["fast", "strict"] = "strict"
    include_diff_in_history: bool = False
    expected_snapshot_hash: str


class PreflightResponse(BaseModel):
    snapshot_hash: str
    included_items: list[dict[str, Any]]
    excluded_items: list[dict[str, Any]]
    context_sources: list[dict[str, Any]]
    char_count: int
    batch_plan: list[dict[str, Any]]
    unreviewed_items: list[str]
    blocked: bool


class Finding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    line: int | None = Field(default=None, ge=1)
    severity: Literal["info", "warning", "blocker"]
    category: Literal[
        "correctness", "security", "compatibility", "performance",
        "maintainability", "testing", "style", "documentation",
    ]
    message: str = Field(min_length=1, max_length=1000)
    suggestion: str = Field(default="", max_length=1000)


class BatchReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["MERGE", "REJECT"]
    risk_level: Literal["low", "medium", "high"]
    summary: str = Field(min_length=1, max_length=2000)
    reasoning: str = Field(min_length=1, max_length=4000)
    blocking_defects: list[str] = Field(default_factory=list, max_length=20)
    findings: list[Finding] = Field(default_factory=list, max_length=20)
