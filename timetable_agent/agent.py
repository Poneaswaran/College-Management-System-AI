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
    build_explain_why_not_prompt,
)
from timetable_agent.schemas import (
    AuditFinding,
    ConstraintItem,
    ScheduleAuditResponse,
    TimetableChatResponse,
    UserContext,
)
from rag.vector_store import ChromaVectorStore, RetrievalResult
from services.embedder import Embedder


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

    def __init__(self, llm_service, vector_store: ChromaVectorStore | None = None, embedder: Embedder | None = None) -> None:
        self.llm = llm_service
        self.vector_store = vector_store
        self.embedder = embedder

    # ─── Chat ──────────────────────────────────────────────────────────────────

    async def chat(
        self,
        *,
        message: str,
        history: list[dict],
        timetable_state: dict,
        user_context: UserContext | None = None,
    ) -> TimetableChatResponse:
        """
        Process one admin message and return a structured response.

        If the LLM proposes constraints, they are extracted from the JSON block
        in the answer and returned separately so the Django view can apply them
        after admin confirmation.
        """
        # Retrieve relevant memories
        memories = await self._get_relevant_memories(message, user_context)
        
        prompt = build_chat_prompt(
            message=message + memories,
            history=history,
            timetable_state=timetable_state,
        )

        raw = await self.llm._generate(prompt)
        logger.debug("timetable_agent.chat raw=%r", raw[:200])
        
        # Check if we should store this as a memory
        await self._check_and_store_memory(message, user_context)

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

    # ─── Grid Chat ─────────────────────────────────────────────────────────────

    async def grid_chat(
        self,
        *,
        message: str,
        history: list[dict],
        department_name: str,
        collected_fields: dict,
        user_context: UserContext | None = None,
    ) -> dict:
        """
        Conversational agent for setting up timetable grids.
        """
        # Retrieve relevant memories
        memories = await self._get_relevant_memories(message, user_context)

        # TEMP DEBUG — remove after fix
        print(f"DEBUG LLM URL: {self.llm.settings.OLLAMA_URL}")
        print(f"DEBUG LLM MODEL: {self.llm.settings.LLM_MODEL}")

        system_prompt = f"""
You are a timetable configuration assistant for a college management system.
Your job is to extract a structured timetable grid TEMPLATE that will apply to a whole SEMESTER.

You must collect ALL of the following to build the semester structure:
1. Academic Year (e.g., 2025-26)
2. Class start time (day_start)
3. Class end time (day_end)
4. Lunch break start time and duration in minutes
5. Number of periods (prioritize this over duration)
6. Period duration (calculated or confirmed)

Rules:
- If any field is missing or ambiguous, ask ONE specific question at a time.
- Do NOT assume lunch break time — always confirm it explicitly.
- If the number of periods is known, PRIORITIZE it. Calculate the period duration automatically to fit exactly between day_start and day_end (after deducting lunch).
- If the user provides a duration that contradicts the number of periods or day length, POLITELY ignore their duration and use your calculated one, explaining the math to the user.
- Once all fields are collected, output a JSON block wrapped in <GRID_JSON>...</GRID_JSON>
  tags — do not output the JSON until ALL fields are confirmed.
- To help me track progress, ALWAYS include a JSON block wrapped in <UPDATE_FIELDS>...</UPDATE_FIELDS> 
  tags containing the current set of extracted fields. Supported keys: "day_start", "day_end", "lunch_start", "lunch_duration_mins", "period_duration_mins", "num_periods".
- Be concise, professional, and ask one question at a time.
- IMPORTANT: This is a TEMPLATE for the semester. Do NOT ask for specific calendar dates or daily attendance dates. 
- You only need the "Academic Year" (e.g. 2025-26).
- JSON data must be strictly valid. Do NOT include comments (// or /*) inside JSON blocks.
- CRITICAL: Never show JSON, code, or technical state to the user. All data MUST be wrapped in <UPDATE_FIELDS> or <GRID_JSON> tags.
- If you show raw curly braces {{ }} in your human reply, you have failed.
- ALWAYS include a friendly human-readable response alongside your data tags, even if it is just a simple greeting or a status update.

Current department: {department_name}
Collected so far: {json.dumps(collected_fields, indent=2)}

INSTITUTIONAL MEMORY (RULES TO FOLLOW):
{memories if memories else "No specific rules found yet."}
        """

        # Format history for Ollama LLM
        formatted_history = []
        for h in history:
            role = h.get("role", "user")
            # In some setups, parts[0]["text"] is used, but for ollama it's usually just role/content
            if "parts" in h and h["parts"]:
                content = h["parts"][0].get("text", "")
            else:
                content = h.get("content", "")
            formatted_history.append({"role": role, "content": content})

        # Construct a strict instructional prompt
        full_prompt = f"### INSTRUCTIONS:\n{system_prompt}\n\n### USER MESSAGE:\n{message}\n\n### ASSISTANT RESPONSE:"
        
        # We need the answer from the LLM
        raw = await self.llm._generate(full_prompt)
        logger.debug("timetable_agent.grid_chat raw=%r", raw[:200])
        
        # Check if we should store this as a memory
        await self._check_and_store_memory(message, user_context)
        
        # Check for GRID_JSON
        match = re.search(r'<GRID_JSON>(.*?)</GRID_JSON>', raw, re.DOTALL)
        resolved_grid = None
        state = "collecting"
        
        if match:
            try:
                resolved_grid = json.loads(match.group(1))
                state = "complete"
                # Strip out the JSON block for the human-readable reply
                raw = raw.replace(match.group(0), "").strip()
            except json.JSONDecodeError:
                raw += "\n\nError: Failed to parse generated grid JSON."
                
        # Check for UPDATE_FIELDS
        updated_fields = None
        update_match = re.search(r'<UPDATE_FIELDS>(.*?)</UPDATE_FIELDS>', raw, re.DOTALL)
        if update_match:
            try:
                updated_fields = json.loads(update_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # FINAL CLEANUP: Ensure NO tags or raw JSON leak into the human reply
        # 1. Scrub ALL XML-like tags (e.g. <CONTEXT_GATHERING>, <UPDATE_FIELDS>, etc.)
        raw = re.sub(r'<[^>]+>', '', raw)
        # 2. Scrub markdown code blocks
        raw = re.sub(r'```(?:json)?.*?```', '', raw, flags=re.DOTALL)
        # 3. Scrub any raw JSON objects that might have leaked
        raw = re.sub(r'\{[\s\S]*?\}', '', raw).strip()
        
        # Fallback for empty responses after scrubbing
        reply = raw.strip()
        if not reply:
            if state == "complete":
                reply = "Great! I've generated your timetable grid. You can see the preview on the right."
            else:
                reply = "Hello! I'm ready to help you set up your timetable. Could you provide the college start and end times?"

        return {
            "reply": reply,
            "state": state,
            "resolved_grid": resolved_grid,
            "updated_fields": updated_fields
        }

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

    # ─── Explain Why Not ───────────────────────────────────────────────────────

    async def explain_why_not(
        self,
        *,
        message: str,
        history: list[dict],
        timetable_state: dict,
        diagnostic_context: dict,
        user_context: UserContext | None = None,
    ) -> TimetableChatResponse:
        """
        Answer a negative-space diagnostic question.

        Uses the full timetable state + pre-computed diagnostic_context
        (room occupancy, section schedule) so the LLM can reason from
        structured facts rather than making assumptions.
        """
        # Retrieve relevant memories
        memories = await self._get_relevant_memories(message, user_context)
        
        prompt = build_explain_why_not_prompt(
            message=message + memories,
            history=history,
            timetable_state=timetable_state,
            diagnostic_context=diagnostic_context,
        )

        raw = await self.llm._generate(prompt)
        logger.debug("explain_why_not raw=%r", raw[:200])

        try:
            cleaned = _extract_json_block(raw)
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            data = {
                "answer": raw,
                "proposed_constraints": [],
                "confidence": "low",
                "requires_confirmation": False,
            }

        return TimetableChatResponse(
            answer=data.get("answer", raw),
            proposed_constraints=[],  # diagnostic answers never propose changes
            confidence=data.get("confidence", "medium"),
            requires_confirmation=False,
        )


    # ─── Memory Helpers ────────────────────────────────────────────────────────
    
    async def _get_relevant_memories(self, message: str, user_context: UserContext | None) -> str:
        if not self.vector_store or not self.embedder or not user_context:
            return ""
            
        try:
            query_embedding = self.embedder.embed_query(message)
            
            # Build scoped filter
            # Admins see global admin rules. 
            # HODs see global admin rules OR their own department rules.
            
            base_filters = [
                {"type": {"$eq": "memory"}},
            ]
            if user_context.tenant_id:
                base_filters.append({"tenant_id": {"$eq": str(user_context.tenant_id)}})

            if user_context.role == "ADMIN":
                # Admins only need global admin rules (or can define them)
                where_logic = {"role": {"$eq": "ADMIN"}}
            else:
                # HODs see global admin rules OR their own department rules
                where_logic = {
                    "$or": [
                        {"role": {"$eq": "ADMIN"}},
                        {
                            "$and": [
                                {"role": {"$eq": "HOD"}},
                                {"department_id": {"$eq": str(user_context.department_id)}}
                            ]
                        }
                    ]
                }
            
            where = {"$and": base_filters + [where_logic]}
                
            results = self.vector_store.query(
                query_embedding=query_embedding,
                top_k=3,
                where=where
            )
            
            if not results:
                return ""
                
            memories = "\n".join([f"- {r.text}" for r in results if r.score and r.score > 0.7])
            if memories:
                return f"\n\n[CONTEXT: Stored Memories/Rules]\n{memories}\n"
        except Exception as e:
            logger.error("Memory retrieval failed: %s", e)
            
        return ""

    async def _check_and_store_memory(self, message: str, user_context: UserContext | None):
        if not self.vector_store or not self.embedder or not user_context:
            return
            
        # Basic heuristic for facts worth remembering
        triggers = ["remember", "note:", "college timing", "starts at", "ends at", "lunch", "half day", "closed", "saturday", "sunday", "periods"]
        if any(t in message.lower() for t in triggers):
            try:
                embedding = self.embedder.embed_query(message)
                
                # Deduplication check: see if we already have a near-identical memory
                existing = self.vector_store.query(
                    query_embedding=embedding,
                    top_k=1,
                    where={
                        "$and": [
                            {"tenant_id": str(user_context.tenant_id) if user_context.tenant_id else "none"},
                            {"type": "memory"}
                        ]
                    }
                )
                if existing and existing[0].score and existing[0].score > 0.98:
                    logger.info("Similar memory already exists (score %.2f), skipping storage.", existing[0].score)
                    return

                metadata = {
                    "role": user_context.role,
                    "department_id": str(user_context.department_id) if user_context.department_id else "none",
                    "tenant_id": str(user_context.tenant_id) if user_context.tenant_id else "none",
                }
                self.vector_store.store_memory(text=message, embedding=embedding, metadata=metadata)
                logger.info("Stored new memory for %s", user_context.role)
            except Exception as e:
                logger.error("Memory storage failed: %s", e)


def create_timetable_agent(settings) -> TimetableAgent:
    """Factory - reuses the same Ollama client as the study-material pipeline."""
    from services.llm_service import create_llm_service
    from services.embedder import create_embedder
    from rag.vector_store import ChromaVectorStore
    
    llm = create_llm_service(settings)
    embedder = create_embedder(settings)
    vector_store = ChromaVectorStore(settings)
    
    return TimetableAgent(llm_service=llm, vector_store=vector_store, embedder=embedder)
