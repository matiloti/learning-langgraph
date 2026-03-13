"""
State definitions for the coding assistant graph.
Keeps all state schemas in one place for clarity.
"""

import operator
from typing import Literal
from typing_extensions import TypedDict, Annotated
from pydantic import BaseModel, Field
from langchain_core.messages import AnyMessage


# ---------------------------------------------------------------------------
# Task model – used by the planner to break work into atomic steps
# ---------------------------------------------------------------------------

class Task(BaseModel):
    """A single atomic task produced by the planner."""
    id: int = Field(..., description="Unique task id (1-based)")
    description: str = Field(..., description="Short, concrete description of what to do")
    type: Literal["code", "answer"] = Field(..., description="Whether this task requires coding or just answering")
    status: Literal["pending", "in_progress", "done", "skipped"] = Field(default="pending")


class PlanOutput(BaseModel):
    """Structured output from the planner node."""
    tasks: list[Task] = Field(..., description="Ordered list of tasks to complete the user request")
    reasoning: str = Field(..., description="Brief explanation of the plan")


class TriageDecision(BaseModel):
    """Quick router decision – does the message need planning or is it simple?"""
    needs_planning: bool = Field(..., description="True if the request has multiple steps or is complex")
    type: Literal["code", "answer", "plan"] = Field(..., description="Route: code, answer, or plan")


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    """Global state flowing through the graph."""
    # Core conversation
    messages: Annotated[list[AnyMessage], operator.add]

    # Task management
    tasks: list[dict]            # list of Task.model_dump() dicts
    current_task_id: int         # which task we're working on (0 = none)

    # Tool call tracking (for looping)
    tool_calls: list[AnyMessage]
    tool_call_attempts: int

    # Context engineering
    context_summary: str         # rolling summary of work done so far
    working_files: list[str]     # files the assistant has read/written

    # Routing
    route: str                   # current route decision
