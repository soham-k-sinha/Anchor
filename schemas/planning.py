from typing import Any, Dict, List, Literal
from pydantic import BaseModel, Field

# Allowed universal primitive tool names for LLM #2 output
ToolName = Literal[
    "shortcuts.run",
    "browser.run",
    "files.apply",
    "system.open_app",
    "pause_for_user"
]

class Action(BaseModel):
    tool: ToolName
    args: Dict[str, Any]
    needs_confirmation: bool = True

class PlanResult(BaseModel):
    explanation: str
    actions: List[Action] = Field(default_factory=list)
    questions: List[str] = Field(default_factory=list)


