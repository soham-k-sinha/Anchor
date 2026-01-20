from typing import List, Literal, Optional
from pydantic import BaseModel, Field


"""
-------- ROUTING: Schema Pydantic Definitions --------
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

