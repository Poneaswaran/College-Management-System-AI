"""
api/timetable_chat.py

FastAPI routes for the AI Timetable Copilot:
  POST /timetable/chat   — chat with the scheduler copilot
  POST /timetable/audit  — run a soft-preference audit
  POST /timetable/explain-conflicts — translate scheduler errors to English
"""

from fastapi import APIRouter, Depends, Request

from rag.errors import AppError
from rag.logging import get_logger
from rag.security import validate_internal_request
from timetable_agent.schemas import (
    ScheduleAuditRequest,
    ScheduleAuditResponse,
    TimetableChatRequest,
    TimetableChatResponse,
)
from pydantic import BaseModel
from typing import Any

router = APIRouter(prefix="/timetable", tags=["timetable_ai"])
logger = get_logger(__name__)


def get_timetable_agent(request: Request):
    agent = getattr(request.app.state, "timetable_agent", None)
    if agent is None:
        raise AppError(
            status_code=503,
            detail="Timetable agent is not initialised. Restart the AI service.",
            code="agent_not_ready",
        )
    return agent


@router.post("/chat", response_model=TimetableChatResponse)
async def timetable_chat(
    payload: TimetableChatRequest,
    _: None = Depends(validate_internal_request),
    agent=Depends(get_timetable_agent),
) -> TimetableChatResponse:
    """
    Main chat endpoint for the admin's scheduling assistant.

    The Django backend enriches the payload with the current timetable state
    snapshot before forwarding it here. The LLM response always contains:
      - A plain-English answer for the admin
      - An optional list of machine-executable constraints ready for Django to apply
    """
    logger.info(
        "timetable_chat message_len=%d semester_id=%d history_turns=%d",
        len(payload.message),
        payload.semester_id,
        len(payload.history),
    )

    result = await agent.chat(
        message=payload.message,
        history=[h.model_dump() for h in payload.history],
        timetable_state=payload.timetable_state,
    )

    logger.info(
        "timetable_chat constraints=%d confidence=%s",
        len(result.proposed_constraints),
        result.confidence,
    )
    return result


@router.post("/audit", response_model=ScheduleAuditResponse)
async def timetable_audit(
    payload: ScheduleAuditRequest,
    _: None = Depends(validate_internal_request),
    agent=Depends(get_timetable_agent),
) -> ScheduleAuditResponse:
    """
    Soft-preference schedule audit.

    Receives the full timetable state from Django and asks the LLM to review
    it for human-preference violations that the algorithmic scheduler cannot
    detect (e.g. faculty exhaustion, unfair displacement patterns).
    """
    logger.info("timetable_audit audit_type=%s", payload.audit_type)
    result = await agent.audit(timetable_state=payload.timetable_state)
    logger.info("timetable_audit findings=%d score=%d", len(result.findings), result.score)
    return result


class ConflictExplainRequest(TimetableChatRequest):
    """Re-uses TimetableChatRequest but the message field holds error strings."""
    error_messages: list[str] = []


@router.post("/explain-conflicts", response_model=TimetableChatResponse)
async def explain_conflicts(
    payload: ConflictExplainRequest,
    _: None = Depends(validate_internal_request),
    agent=Depends(get_timetable_agent),
) -> TimetableChatResponse:
    """
    Translate raw scheduler error strings into plain English with fix suggestions.

    Called automatically by Django's RescheduleService or GenerateSemesterTimetableView
    when the algorithmic engine surfaces violations.
    """
    errors = payload.error_messages or [payload.message]
    logger.info("explain_conflicts count=%d", len(errors))

    result = await agent.explain_conflicts(
        error_messages=errors,
        timetable_state=payload.timetable_state,
    )
    return result


# ── Explain Why Not ────────────────────────────────────────────────────────────

class ExplainWhyNotRequest(BaseModel):
    """
    Payload Django's ExplainWhyNotView sends after enriching the request
    with pre-computed diagnostic facts.
    """
    message: str
    semester_id: int
    timetable_state: dict[str, Any]
    diagnostic_context: dict[str, Any] = {}
    history: list[dict] = []


@router.post("/explain-why-not", response_model=TimetableChatResponse)
async def explain_why_not(
    payload: ExplainWhyNotRequest,
    _: None = Depends(validate_internal_request),
    agent=Depends(get_timetable_agent),
) -> TimetableChatResponse:
    """
    Answer a negative-space diagnostic question.

    Django pre-enriches the payload with structured facts about the
    specific room, section, or period being asked about, so the LLM
    can reason from facts rather than inference.

    Example questions handled:
      - "Why couldn't Section CSE-4A get Room 101 on Monday P3?"
      - "Why is Dr. Ramesh scheduled in two places on Wednesday P5?"
      - "Why does CSE-3B always end up without a room on Fridays?"
    """
    logger.info(
        "explain_why_not message_len=%d semester_id=%d diag_keys=%s",
        len(payload.message),
        payload.semester_id,
        list(payload.diagnostic_context.keys()),
    )

    result = await agent.explain_why_not(
        message=payload.message,
        history=payload.history,
        timetable_state=payload.timetable_state,
        diagnostic_context=payload.diagnostic_context,
    )

    logger.info("explain_why_not confidence=%s", result.confidence)
    return result
