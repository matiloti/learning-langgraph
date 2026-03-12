from pydantic import BaseModel, Field
import operator
from typing_extensions import TypedDict, Annotated
from langchain.messages import ToolMessage
from typing import Literal

class GlobalState(TypedDict):
    tool_calls: list[ToolMessage]
    messages: Annotated[list[ToolMessage], operator.add]
    state: str
    tool_call_attempts: int

class Decision(BaseModel):
    """Triage decision whether to seek clarifications or to code."""
    decision: Literal["code_node","talk_node"] = Field(..., description="The decision")
