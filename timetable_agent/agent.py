"""
timetable_agent/agent.py

TimetableAgent — the core intelligence layer.

Wraps Ollama LLM calls with:
  • JSON response parsing with fallback
  • Constraint extraction from LLM output
  • Conflict explanation pipeline
  • Audit pipeline

All methods are async-friendly (uses httpx.AsyncClient via OllamaLLMService).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from rag.errors import AppError
from timetable_agent.prompts import (
    build_audit_prompt,
    build_chat_prompt,
    build_conflict_explanation_prompt,
)
from timetable_agent.schemas import (
    AuditFinding,
    ConstraintItem,
    ScheduleAuditResponse,
    TimetableChatResponse,
)

logger = logging.getLogger(__name__)


def _extract_json_block(text: str) -> str:
    """
    Extract the first JSON object from LLM output.
    Handles both raw JSON and markdown ```json ... ``` blocks.
    """
    # Try ```json ... ``` fence
    fence_match = re.search(r"```(?:json)?\s*(\{.*?})\s*```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1)

    # Try a bare top-level object
    brace_match = re.search(r"(\{.*})", text, re.DOTALL)
    if brace_match:
        return brace_match.group(1)

    return text  # return as-is and let json.loads fail naturally


def _extract_constraint_list(text: str) -> list[dict]:
    """
    Pull a JSON array from the LLM answer field (for constraint lists
    embedded inside the answer string).
    """
    array_match = re.search(r"```(?:json)?\s*(\[.*?])\s*```", text, re.DOTALL)
    if array_match:
        try:
            return json.loads(array_match.group(1))
        except json.JSONDecodeError:
            return []
    return []


class TimetableAgent:
    """
    High-level timetable scheduling assistant.

    Parameters
    ----------
    llm_service : OllamaLLMService (or any object exposing
                  async generate_answer(question, context_chunks))
    """

    def __init__(self, llm_service) -> None:
        self.llm = llm_service

    # ─── Chat ──────────────────────────────────────────────────────────────────

    async def chat(
        self,
        *,
        message: str,
        history: list[dict],
        timetable_state: dict,
    ) -> TimetableChatResponse:
        """
        Process one admin message and return a structured response.

        If the LLM proposes constraints, they are extracted from the JSON block
        in the answer and returned separately so the Django view can apply them
        after admin confirmation.
        """
        prompt = build_chat_prompt(
            message=message,
            history=history,
            timetable_state=timetable_state,
        )

        raw = await self.llm._generate(prompt)
        logger.debug("timetable_agent.chat raw=%r", raw[:200])

        try:
            cleaned = _extract_json_block(raw)
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("LLM did not return valid JSON; wrapping raw text.")
            data = {"answer": raw, "proposed_constraints": [], "confidence": "low",
                    "requires_confirmation": True}

        # Extract any constraint list that may be embedded inside answer text
        embedded = _extract_constraint_list(data.get("answer", ""))
        raw_constraints = data.get("proposed_constraints", []) or embedded

        constraints = []
        for c in raw_constraints:
            try:
                constraints.append(ConstraintItem.model_validate(c))
            except Exception:
                logger.warning("Skipping invalid constraint: %r", c)

        return TimetableChatResponse(
            answer=data.get("answer", raw),
            proposed_constraints=constraints,
            confidence=data.get("confidence", "medium"),
            requires_confirmation=bool(data.get("requires_confirmation", True)),
        )

    # ─── Conflict Explanation ──────────────────────────────────────────────────

    async def explain_conflicts(
        self,
        *,
        error_messages: list[str],
        timetable_state: dict,
    ) -> TimetableChatResponse:
        """
        Translate raw scheduler error strings into plain English with fixes.
        """
        prompt = build_conflict_explanation_prompt(
            error_messages=error_messages,
            timetable_state=timetable_state,
        )

        raw = await self.llm._generate(prompt)

        try:
            cleaned = _extract_json_block(raw)
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            data = {"answer": raw, "proposed_constraints": [], "confidence": "low",
                    "requires_confirmation": True}

        return TimetableChatResponse(
            answer=data.get("answer", raw),
            proposed_constraints=[],
            confidence=data.get("confidence", "low"),
            requires_confirmation=True,
        )

    # ─── Audit ─────────────────────────────────────────────────────────────────

    async def audit(self, *, timetable_state: dict) -> ScheduleAuditResponse:
        """
        Run a soft-preference audit on the full timetable.
        """
        prompt = build_audit_prompt(timetable_state)
        raw = await self.llm._generate(prompt)

        try:
            cleaned = _extract_json_block(raw)
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            return ScheduleAuditResponse(
                summary="Audit could not be parsed. LLM response was malformed.",
                findings=[
                    AuditFinding(
                        severity="info",
                        category="parsing",
                        description=raw[:500],
                        suggestion="Re-run the audit or check Ollama logs.",
                    )
                ],
                score=0,
            )

        findings = []
        for f in data.get("findings", []):
            try:
                findings.append(AuditFinding.model_validate(f))
            except Exception:
                logger.warning("Skipping malformed finding: %r", f)

        return ScheduleAuditResponse(
            summary=data.get("summary", "No summary provided."),
            findings=findings,
            score=int(data.get("score", 0)),
        )


def create_timetable_agent(settings) -> TimetableAgent:
    """Factory — reuses the same Ollama client as the study-material pipeline."""
    from services.llm_service import create_llm_service
    llm = create_llm_service(settings)
    return TimetableAgent(llm_service=llm)
