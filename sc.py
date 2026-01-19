import json
from typing import List, Literal, Optional, Dict, Any

import ollama
from pydantic import BaseModel, Field, ValidationError



"""
-------- Schema Pydantic Definitions --------
"""


# Variable definitions for ouput schema
Domain = Literal["reminders", "mail", "files", "browser", "system", "unknown"]
Risk = Literal["low", "medium", "high"]


# Information for LLM #2
class RoutingHints(BaseModel):
    # Which tool should LLM #2 use for the task (what does it prefer)
    prefer: List[Literal["shortcuts", "browser", "files_api"]] = Field(default_factory=list)

    # What should writing mail tool default to
    mail_default: Optional[Literal["draft", "send"]] = None

    # Should the task require confirmation from the user to execute
    require_confirmation: bool = True


# Main router output schema --> This is the output of LLM #1
class RoutingResult(BaseModel):
    domains: List[Domain] = Field(min_length=1)
    risk: Risk
    needs_clarification: bool
    questions: List[str] = Field(default_factory=list)
    hints: RoutingHints = Field(default_factory=RoutingHints)



"""
-------- System Prompt --------
"""


# System Prompt for LLM #1
ROUTER_SYSTEM_PROMPT = """
You are a routing classifier for a senior-friendly desktop assistant on macOS.

Return JSON ONLY (no markdown, no code fences).

Your job:
- Choose the MINIMUM set of domains required to fulfill the request.
- If unsure between including a domain or not, DO NOT include it.
- Most requests involve only 1 domain, but others can include multiple.

DOMAINS:
- reminders: create/update/list/delete reminders, "remind me...", alarms, due times
- mail: draft/send email, subject/body/attachments
- files: find/move/rename/copy/delete local files/folders
- browser: navigate websites, fill web forms, online tasks, login flows
- system: open apps, basic OS settings
- unknown: unclear intent

Examples of requests involving multiple domains:
- "Remind me tomorrow at 9 PM to email my professor about the assignment". Domains = reminders, mail. Risk = low (reminder).
- "Find my latest resume and email it to Sarah". Domains = files, mail. Risk = low (files - high only if deleting files), medium only if auto-sending email (usually draft only)

CRITICAL RULES (avoid over-routing):
- DO NOT include "browser" unless the user explicitly mentions a website/app in the browser, online payment, login, form, or a URL.
- DO NOT include "files" unless the user explicitly mentions files/folders/downloads/documents/attachments or file operations.
- For "Remind me ..." requests, domains should typically be ONLY ["reminders"].

RISK LEVEL:
- low: creating/updating a reminder, drafting an email, opening an app/site, listing files
- medium: sending a message/email, moving files, scheduling events
- high: deleting files permanently, any financial/payment/account/security actions

NOTE: Never DELETE files permanantly, always move them to the trash.

CLARIFICATION:
- Set needs_clarification=true only if a required detail is missing and guessing would be unsafe and add questions accordingly, based on what needs to be clarified. It is always better to ask clarifying question over guessing and hallucinating crucial details.
Examples of when to set needs_clarification=true:
- "email John" with no email address and no known contact context
- "Pay my electricity bill" with no website, login details, or any other crucial piece
Otherwise set false.

OUTPUT SHAPE:
{
  "domains": ["..."],
  "risk": "low|medium|high",
  "needs_clarification": true|false,
  "questions": [],
  "hints": {
    "prefer": ["shortcuts|browser|files_api"],
    "mail_default": "draft|send|null",
    "require_confirmation": true|false
  }
}

FEW-SHOT EXAMPLES:

Input: {"user_message":"Remind me to do the dishes tomorrow at 7 PM"}
Output: {"domains":["reminders"],"risk":"low","needs_clarification":false,"questions":[],"hints":{"prefer":["shortcuts"],"mail_default":null,"require_confirmation":true}}

Input: {"user_message":"Move my latest resume from Downloads to Documents/Jobs"}
Output: {"domains":["files"],"risk":"medium","needs_clarification":false,"questions":[],"hints":{"prefer":["files_api"],"mail_default":null,"require_confirmation":true}}

Input: {"user_message":"Give me updates on the latest stock prices for Apple Inc."}
Output: {"domains":["browser"],"risk":"low","needs_clarification":false,"questions":[],"hints":{"prefer":["browser"],"mail_default":null,"require_confirmation":true}}
""".strip()



"""
-------- Helper Functions --------
"""


def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        # handle ```json ... ```
        parts = s.split("```")
        if len(parts) >= 2:
            return parts[1].strip()
    return s

def _try_parse_routing(raw: str) -> RoutingResult:
    raw = _strip_code_fences(raw)
    data = json.loads(raw)
    return RoutingResult.model_validate(data)

def _repair_json_with_ollama(raw: str, model: str = "qwen2.5-coder:3b") -> str:
    # Only used if model outputs malformed JSON.
    repair_prompt = f"""
Fix the following so it becomes valid JSON matching the required schema.
Rules:
- Output JSON only, no markdown.
- Do not change meaning; only fix formatting, quotes, commas, and field names if needed.

TEXT:
{raw}
""".strip()
    resp = ollama.generate(model=model, prompt=repair_prompt, stream=False)
    return resp["response"].strip()





"""
-------- Final LLM #1 call --------
"""

def route_user_request(user_message: str, model: str = "qwen2.5:7b-instruct", timezone: str = "Asia/Kolkata", now_iso: Optional[str] = None, max_repair_attempts: int = 1) -> RoutingResult:
    """
    Returns RoutingResult (domains, risk, clarification questions, hints).
    Uses Ollama local model by default.

    Pass now_iso if you want the router to use current time context (optional).
    """
    user_block = {
        "user_message": user_message,
        "timezone": timezone,
        "now_iso": now_iso,
    }

    prompt = f"""
    {ROUTER_SYSTEM_PROMPT}

    INPUT:
    {json.dumps(user_block, ensure_ascii=False)}
    """.strip()

    resp = ollama.generate(model=model, prompt=prompt, stream=False)
    raw = resp["response"]

    # Parse + validate
    try:
        return _try_parse_routing(raw)
    except (json.JSONDecodeError, ValidationError):
        # Optional single repair attempt
        for _ in range(max_repair_attempts):
            repaired = _repair_json_with_ollama(raw, model="qwen2.5-coder:3b")
            try:
                return _try_parse_routing(repaired)
            except (json.JSONDecodeError, ValidationError):
                raw = repaired

        # Final fallback: safe default
        return RoutingResult(
            domains=["unknown"],
            risk="medium",
            needs_clarification=True,
            questions=["I'm not sure what you want to do. Could you rephrase in one sentence?"],
            hints=RoutingHints(prefer=["shortcuts"], mail_default="draft", require_confirmation=True),
        )


if __name__ == "__main__":
    msg = input("User request: ").strip()
    result = route_user_request(user_message=msg)
    print(json.dumps(result.model_dump(), indent=2))