import json
from typing import Any, Dict, List, Optional

import ollama
from pydantic import ValidationError

from schemas.routing import RoutingResult
from schemas.planning import PlanResult
from helper_functions.funcs import _try_parse_plan, _repair_json_with_ollama
from config import PLANNER_MODEL, REPAIR_MODEL


"""
-------- LLM #2 System Prompt (Planner) --------
"""

PLANNER_SYSTEM_PROMPT = """
You are the Planner (LLM #2) for a senior-friendly, local-first desktop automation assistant on macOS.

Return JSON ONLY (no markdown, no code fences).

You NEVER execute actions. You ONLY produce a safe plan using universal primitives.
Execution will be performed later by deterministic executors after user confirmation.

CORE PHILOSOPHY:
- Open-ended request, closed-ended execution
- LLMs plan, never execute
- The user must know what will happen before it happens

ALLOWED UNIVERSAL PRIMITIVES (TOOLS):
1) shortcuts.run
Args:
{
  "name": "string (MUST be one of AVAILABLE_SHORTCUTS names exactly)",
  "input_json": "object (JSON-serializable)"
}

2) browser.run
Args:
{
  "steps": [
    { "op": "go_to", "url": "https://..." },
    { "op": "click", "text": "string" } |
    { "op": "click", "selector": "string" } |
    { "op": "type", "selector": "string", "value": "string" } |
    { "op": "wait_for", "selector": "string", "timeout_ms": 0 } |
    { "op": "pause_for_user", "reason": "string" }
  ]
}

3) files.apply
Args:
{
  "ops": [
    { "op": "find", "query": "string", "scope": "string" } |
    { "op": "move", "src": "string", "dst": "string" } |
    { "op": "copy", "src": "string", "dst": "string" } |
    { "op": "rename", "path": "string", "new_name": "string" } |
    { "op": "trash", "path": "string" }
  ]
}

4) system.open_app
Args: { "app_name": "string" }

5) pause_for_user
Args: { "reason": "string" }

SAFETY RULES:
- Use ONLY the allowed tools above.
- DO NOT output code (no AppleScript, no Python, no shell).
- DO NOT request or handle passwords/OTPs. Use pause_for_user for login/OTP.
- Default mail to draft unless the user explicitly says send now.
- Never permanently delete files; use trash only.
- If ROUTING_INFO.needs_clarification is true: DO NOT plan actions; return questions only.

CONFIRMATION RULES:
- If ROUTING_INFO.risk is medium or high, set needs_confirmation=true for ALL actions.
- Also set needs_confirmation=true if ROUTING_INFO.hints.require_confirmation is true.

OUTPUT SHAPE:
{
  "explanation": "Plain English summary for the user",
  "actions": [
    {
      "tool": "shortcuts.run|browser.run|files.apply|system.open_app|pause_for_user",
      "args": { ... },
      "needs_confirmation": true|false
    }
  ],
  "questions": []
}

PLANNING GUIDELINES:
- Keep actions minimal.
- Prefer shortcuts.run when ROUTING_INFO.hints.prefer includes "shortcuts".
- Ensure shortcut name matches AVAILABLE_SHORTCUTS.
- Ask questions instead of guessing.
""".strip()


"""
-------- Deterministic Policy Enforcement (Post-LLM) --------
"""

def enforce_policy(plan: PlanResult, routing: RoutingResult) -> PlanResult:
    """
    Deterministic, post-LLM guardrails.
    This ensures the model cannot accidentally bypass confirmation policy.
    """
    # If router says clarification needed: planner should not produce actions.
    if routing.needs_clarification:
        return PlanResult(
            explanation="I need a bit more information before I can plan this.",
            actions=[],
            questions=routing.questions or ["Could you clarify what you want to do?"],
        )

    must_confirm = (routing.risk in ("medium", "high")) or routing.hints.require_confirmation
    if must_confirm:
        for a in plan.actions:
            a.needs_confirmation = True

    return plan


"""
-------- Final LLM #2 call --------
"""

def plan_user_request(
    user_message: str,
    routing_json: Dict[str, Any],
    available_shortcuts: List[Dict[str, Any]],
    model: str = PLANNER_MODEL,
    timezone: str = "Asia/Kolkata",
    now_iso: Optional[str] = None,
    max_repair_attempts: int = 1,
    tool_policy: Optional[Dict[str, Any]] = None,
) -> PlanResult:
    """
    Input:
      - user_message (original user request)
      - routing_json (output of LLM #1)
      - available_shortcuts (list of shortcuts available to call)
      - now_iso/timezone context

    Output:
      - PlanResult containing explanation, actions, and questions

    Execution is NOT performed here.
    """

    routing = RoutingResult.model_validate(routing_json)

    # If router already says we need clarification, short-circuit early.
    if routing.needs_clarification:
        return PlanResult(
            explanation="I need a bit more information before I can plan this.",
            actions=[],
            questions=routing.questions,
        )

    input_block = {
        "user_message": user_message,
        "routing_info": routing.model_dump(),
        "now_iso": now_iso,
        "timezone": timezone,
        "available_shortcuts": available_shortcuts,
        "tool_policy": tool_policy or {},
    }

    prompt = f"""
{PLANNER_SYSTEM_PROMPT}

INPUT_JSON:
{json.dumps(input_block, ensure_ascii=False)}
""".strip()

    resp = ollama.generate(model=model, prompt=prompt, stream=False)
    raw = resp["response"]

    # Parse + validate
    try:
        plan = _try_parse_plan(raw)
    except (json.JSONDecodeError, ValidationError):
        # Repair attempts
        for _ in range(max_repair_attempts):
            repaired = _repair_json_with_ollama(raw, model=REPAIR_MODEL)
            try:
                plan = _try_parse_plan(repaired)
                break
            except (json.JSONDecodeError, ValidationError):
                raw = repaired
        else:
            # Safe fallback
            plan = PlanResult(
                explanation="I couldn't safely create a plan. Please rephrase your request.",
                actions=[],
                questions=["What would you like me to do? Please include any important details like time, person, or file name."],
            )

    # Deterministic post-LLM policy
    return enforce_policy(plan, routing)


if __name__ == "__main__":
    # Demo-only: run planner manually
    from llm.router import route_user_request

    available_shortcuts = [
        {
            "name": "Create Reminder",
            "description": "Creates a reminder",
            "input_schema": {"title": "string", "due_iso": "string|null"}
        },
        {
            "name": "Draft Email",
            "description": "Drafts an email in Mail",
            "input_schema": {"to": "string", "subject": "string", "body": "string"}
        }
    ]

    msg = input("User request: ").strip()

    routing = route_user_request(msg)
    print("ROUTING:\n", json.dumps(routing.model_dump(), indent=2))

    plan = plan_user_request(
        user_message=msg,
        routing_json=routing.model_dump(),
        available_shortcuts=available_shortcuts,
        now_iso=None,
    )
    print("PLAN:\n", json.dumps(plan.model_dump(), indent=2, ensure_ascii=False))
