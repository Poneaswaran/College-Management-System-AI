"""
timetable_agent/schemas.py

Pydantic request/response models for the AI timetable endpoints.
Consumed by FastAPI route handlers in api/timetable_chat.py and api/timetable_audit.py.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class UserContext(BaseModel):
    """Context about the calling user for memory scoping and RAG filtering."""
    role: str
    department_id: str | None = None
    tenant_id: str | None = None


# ─── Chat ──────────────────────────────────────────────────────────────────────

class ChatHistoryMessage(BaseModel):
    role: str   # "user" | "assistant"
    content: str


class TimetableChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000,
                         description="Admin's natural-language message")
    semester_id: int = Field(..., gt=0)
    timetable_state: dict[str, Any] = Field(
        ..., description="Current snapshot produced by Django TimetableStateView"
    )
    history: list[ChatHistoryMessage] = Field(
        default_factory=list,
        description="Previous turns so the LLM has context"
    )
    user_context: UserContext | None = None

    @field_validator("message")
    @classmethod
    def strip_message(cls, v: str) -> str:
        return v.strip()


class ConstraintItem(BaseModel):
    """A single machine-executable constraint extracted by the LLM."""
    type: str   # "move_entry" | "assign_room" | "swap_entries"
    # Remaining keys are type-specific; validated downstream by Django
    model_config = {"extra": "allow"}


class TimetableChatResponse(BaseModel):
    answer: str = Field(..., description="LLM's plain-English explanation for the admin")
    proposed_constraints: list[ConstraintItem] = Field(
        default_factory=list,
        description="Machine-executable constraint list (may be empty for informational replies)"
    )
    confidence: str = Field(
        default="medium",
        description="LLM's self-assessed confidence: high | medium | low"
    )
    requires_confirmation: bool = Field(
        default=True,
        description="Should the UI ask the admin to confirm before applying constraints?"
    )


# ─── Audit ─────────────────────────────────────────────────────────────────────

class ScheduleAuditRequest(BaseModel):
    audit_type: str = Field(default="soft_preferences")
    timetable_state: dict[str, Any]
    user_context: UserContext | None = None


class AuditFinding(BaseModel):
    severity: str     # "critical" | "warning" | "info"
    category: str     # "faculty_load" | "room_distribution" | "section_fairness" | ...
    description: str
    affected_entities: list[str] = Field(default_factory=list)
    suggestion: str = ""


class ScheduleAuditResponse(BaseModel):
    summary: str
    findings: list[AuditFinding]
    score: int = Field(description="Overall schedule quality 0–100")
