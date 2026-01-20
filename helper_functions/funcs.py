import json
import ollama

from schemas.routing import RoutingResult

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


