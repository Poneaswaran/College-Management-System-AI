"""
timetable_agent/prompts.py

All LLM prompt templates for the timetable copilot.

Four prompts:
  SYSTEM_PROMPT          — Injected at the start of every chat turn.
  constraint_translator  — Turns a natural-language constraint into JSON.
  conflict_explainer     — Turns a raw scheduler error into readable English.
  audit_checker          — Reviews timetable for soft-preference violations.
"""

from __future__ import annotations

# ─── System prompt injected into every chat ────────────────────────────────────
SYSTEM_PROMPT = """\
You are the Timetable Copilot for a College Management System. Your job is to help
the admin manage and optimise the weekly class schedule.

CAPABILITIES
  1. Natural-language → JSON constraints: Translate an admin's plain-English request
     (e.g. "Move all Final Year CS classes before lunch on Wednesdays") into a list
     of machine-executable JSON constraint objects that the Django backend can apply.

  2. Conflict explanation: When the scheduler returns an error or violation string,
     explain it in plain English and suggest at least one concrete fix.

  3. Schedule auditing: Analyse the full timetable for soft-preference violations
     (e.g. faculty with back-to-back overload, sections always displaced to last
     period, rooms spread across distant buildings on the same day).

CONSTRAINT JSON FORMAT
  Return constraints inside a JSON block ```json\n[...]\n``` in your answer.
  Each constraint object must have a "type" field and type-specific fields:

  { "type": "move_entry",   "entry_id": <int>, "target_period_id": <int> }
  { "type": "assign_room",  "entry_id": <int>, "room_id": <int> }
  { "type": "swap_entries", "entry1_id": <int>, "entry2_id": <int> }

STRICT RULES
  - NEVER invent room numbers, faculty names, section names, or IDs.
    Only use IDs/names that appear in the timetable_state provided.
  - If a request is impossible (e.g. no room available), say so clearly
    and explain why rather than fabricating a solution.
  - If you are uncertain, say "confidence: low" and ask for clarification.
  - Final-year sections (priority = 1) MUST always have a room. Never suggest
    moves that would displace them without providing an alternative.
  - Output constraints only when specifically asked to make a change.
    For informational questions, answer in plain text only.

RESPONSE FORMAT
  Always respond with a JSON object:
  {
    "answer": "<plain English explanation for the admin>",
    "proposed_constraints": [...],   // empty list if no changes needed
    "confidence": "high|medium|low",
    "requires_confirmation": true|false
  }
"""


def build_chat_prompt(
    message: str,
    history: list[dict],
    timetable_state: dict,
) -> str:
    """
    Build the full prompt string for a chat turn.

    Parameters
    ----------
    message          : current admin message
    history          : list of {"role": "user"|"assistant", "content": "..."}
    timetable_state  : dict from Django TimetableStateView
    """
    import json

    state_summary = {
        "semester": timetable_state.get("semester"),
        "meta": timetable_state.get("meta"),
        "sections": timetable_state.get("sections", [])[:20],  # cap for token budget
        "rooms": timetable_state.get("rooms", []),
        "overflow_summary": timetable_state.get("overflow_summary", [])[:10],
        "schedule_sample": timetable_state.get("schedule", [])[:30],
    }
    state_str = json.dumps(state_summary, indent=2, default=str)

    history_str = ""
    for turn in history[-6:]:  # last 3 exchanges
        role = turn.get("role", "user").capitalize()
        history_str += f"\n{role}: {turn.get('content', '')}"

    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"CURRENT TIMETABLE STATE:\n```json\n{state_str}\n```\n\n"
        f"CONVERSATION SO FAR:{history_str}\n\n"
        f"Admin: {message}\n\n"
        f"Assistant (respond with the JSON object described in RESPONSE FORMAT):"
    )


def build_conflict_explanation_prompt(
    error_messages: list[str],
    timetable_state: dict,
) -> str:
    """
    Prompt asking the LLM to explain scheduler errors in plain English.
    """
    import json

    errors_str = "\n".join(f"  • {e}" for e in error_messages)
    meta = json.dumps(timetable_state.get("meta", {}), indent=2)

    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"The timetable scheduler encountered the following errors:\n{errors_str}\n\n"
        f"Timetable meta (context):\n```json\n{meta}\n```\n\n"
        "Please explain each error in plain English and suggest a concrete fix for each. "
        "Return your answer as JSON in the format described in RESPONSE FORMAT.\n"
        "Assistant:"
    )


def build_audit_prompt(timetable_state: dict) -> str:
    """
    Prompt asking the LLM to audit the schedule for soft-preference violations.
    """
    import json

    schedule   = timetable_state.get("schedule", [])
    sections   = timetable_state.get("sections", [])
    overflow   = timetable_state.get("overflow_summary", [])
    state_str  = json.dumps(
        {"schedule": schedule, "sections": sections, "overflow_summary": overflow},
        indent=2, default=str
    )

    return (
        f"{SYSTEM_PROMPT}\n\n"
        "You are now performing a SCHEDULE AUDIT. Review the timetable below for "
        "SOFT-PREFERENCE violations — things that are valid but sub-optimal:\n\n"
        "  • Faculty with more than 4 consecutive periods on any single day\n"
        "  • Any section that overflows more than 3 times in the semester\n"
        "  • Labs scheduled immediately after exams or in the last period of the day\n"
        "  • Final-year sections NOT having priority in any period\n"
        "  • Rooms from different distant buildings assigned to the same section\n\n"
        f"TIMETABLE:\n```json\n{state_str}\n```\n\n"
        "Respond with:\n"
        "{\n"
        '  "summary": "<1-2 sentence overview>",\n'
        '  "findings": [\n'
        '    {\n'
        '      "severity": "critical|warning|info",\n'
        '      "category": "faculty_load|room_distribution|section_fairness|scheduling_pattern",\n'
        '      "description": "...",\n'
        '      "affected_entities": ["Section A", "Dr. Smith"],\n'
        '      "suggestion": "..."\n'
        '    }\n'
        '  ],\n'
        '  "score": <0-100 overall quality score>\n'
        "}\n"
        "Assistant:"
    )


def build_explain_why_not_prompt(
    message: str,
    history: list[dict],
    timetable_state: dict,
    diagnostic_context: dict,
) -> str:
    """
    Build a deep-dive diagnostic prompt for negative-space questions.

    Unlike the standard chat prompt, this injects:
      • The FULL schedule (not capped at 30 entries)
      • Pre-computed ``diagnostic_context`` with room occupancy and
        section-specific schedule data so the LLM doesn't have to search.

    Parameters
    ----------
    message            : Admin's "why couldn't ..." question
    history            : Previous turns for multi-turn context
    timetable_state    : Full snapshot from TimetableStateView
    diagnostic_context : Pre-computed facts from ExplainWhyNotView
                         {room_occupancy, section_schedule, ...}
    """
    import json

    # Include the full schedule — this query needs all the data
    full_schedule  = timetable_state.get("schedule", [])
    rooms          = timetable_state.get("rooms", [])
    sections       = timetable_state.get("sections", [])
    overflow       = timetable_state.get("overflow_summary", [])
    non_room_slots = timetable_state.get("non_room_slots", [])

    state_str = json.dumps(
        {
            "semester": timetable_state.get("semester"),
            "meta": timetable_state.get("meta"),
            "rooms": rooms,
            "sections": sections,
            "schedule": full_schedule,
            "non_room_slots": non_room_slots,
            "overflow_summary": overflow,
        },
        indent=2,
        default=str,
    )

    diag_str = json.dumps(diagnostic_context, indent=2, default=str) if diagnostic_context else "{}"

    history_str = ""
    for turn in history[-6:]:
        role = turn.get("role", "user").capitalize()
        history_str += f"\n{role}: {turn.get('content', '')}"

    return (
        f"{SYSTEM_PROMPT}\n\n"
        "You are answering a DIAGNOSTIC QUESTION about why a specific scheduling "
        "outcome occurred. Use ONLY the data provided below — do NOT guess or "
        "invent any room numbers, section names, faculty names, or IDs.\n\n"
        "YOUR TASK\n"
        "  Examine the timetable state and the pre-computed diagnostic context.\n"
        "  Identify the EXACT algorithmic bottleneck that caused the situation "
        "  described in the admin's question. Possible root causes include:\n"
        "    • Room already booked by another section at that period\n"
        "    • Room capacity too small for the section strength\n"
        "    • Room type mismatch (e.g. theory room requested for lab)\n"
        "    • Faculty double-booked at that slot\n"
        "    • Section already has a class (section conflict)\n"
        "    • Room under maintenance at that time\n"
        "    • Priority ordering caused higher-priority section to claim the room first\n\n"
        f"FULL TIMETABLE STATE:\n```json\n{state_str}\n```\n\n"
        f"PRE-COMPUTED DIAGNOSTIC FACTS:\n```json\n{diag_str}\n```\n\n"
        f"CONVERSATION SO FAR:{history_str}\n\n"
        f"Admin: {message}\n\n"
        "Assistant (respond as the JSON object from RESPONSE FORMAT — be specific, "
        "cite actual section/room/faculty names from the data, no vague answers):"
    )

