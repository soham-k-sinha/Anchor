import json
from typing import Type, TypeVar

import ollama
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)
from config import REPAIR_MODEL


"""
-------- Generic Helper Utilities --------
"""


def _strip_code_fences(s: str) -> str:
    """
    Removes ```json ...``` fences if the model returns them.
    (Even though prompts forbid it, models sometimes do it anyway.)
    """
    s = s.strip()
    if s.startswith("```"):
        # handle ```json ... ```
        parts = s.split("```")
        if len(parts) >= 2:
            return parts[1].strip()
    return s

def _parse_and_validate(raw: str, schema: Type[T]) -> T:
    """
    Generic: parse JSON string -> validate with provided Pydantic schema.
    """
    raw = _strip_code_fences(raw)
    data = json.loads(raw)
    return schema.model_validate(data)

def _repair_json_with_ollama(raw: str, model: str = REPAIR_MODEL) -> str:
    """
    Used ONLY if the LLM output is malformed JSON or fails schema validation.
    This is your LLM #3 "repair" role (optional).
    """
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
-------- Router-specific wrappers (LLM #1) --------
"""

from schemas.routing import RoutingResult

def _try_parse_routing(raw: str) -> RoutingResult:
    return _parse_and_validate(raw, RoutingResult)


"""
-------- Planner-specific wrappers (LLM #2) --------
"""

from schemas.planning import PlanResult

def _try_parse_plan(raw: str) -> PlanResult:
    return _parse_and_validate(raw, PlanResult)