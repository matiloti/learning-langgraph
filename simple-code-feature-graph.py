from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.messages import SystemMessage
import os
from pydantic import BaseModel, Field
from langchain.messages import AnyMessage, ToolMessage
from typing_extensions import TypedDict, Annotated
import operator
from typing import Literal
from langgraph.graph import StateGraph, START, END

# Define local model
base_model = init_chat_model(
    model="gpt-5", # this is ignored im using zai-org/glm-4.7-flash locally but if i put this name LangGraph throws error
    base_url="http://localhost:1234/v1", # Local model zai-org/glm-4.7-flash
)

class GlobalState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    clarifications_attempts: int
    state: str

class Decision(BaseModel):
    """Triage decision whether to seek clarifications or to code."""
    decision: Literal["clarify","code","answer"] = Field(..., description="The decision")

decision_model = base_model.with_structured_output(Decision)

def clarification_triage(state: dict):
    """LLM decides whether user input requires clarification or not"""

    return {
        "state": decision_model.invoke(
            state["messages"],
            [
                SystemMessage(content=f"Based on the conversation history, decide if you have enough information whether user only seeks information, or if it requires you to proceed to coding. If it requires to code, decide whether you need further clarification. Examples of vague requests are 'do tests' or 'fix everything' and require further clarification. If the user gives you a really generic and broad task, is considered for clarification. If the user provides a simple and well-known coding task, proceed to code directly. If the user only asks something (info), it's considered a query, so go to query. CRITICAL: max 3 clarifications. You don't need to reach 3 clarifications if you consider it's enough. Current clarifications: {state["clarifications_attempts"]}/3")
            ]
        ).decision
    }

def should_code_or_clarify_or_answer(state: GlobalState) -> Literal["code","clarify","answer"]:
    """Route decision to correct node"""
    return state["state"]


WORKSPACE = os.path.realpath("/Users/matias/Projects/learning-langgraph/test-folder")
def safe_path(filepath: str) -> str:
    """Resolve a path and ensure it's inside WORKSPACE. Raises ValueError if not."""
    full = os.path.realpath(os.path.join(WORKSPACE, filepath))
    if not full.startswith(WORKSPACE):
        raise ValueError(f"Access denied: {filepath} is outside workspace")
    return full

# Define tools
@tool
def read_file(filepath: str) -> str:
    """Read the contents of a file.

    Args:
        filepath: Path to the file to read
    """
    path = safe_path(filepath)
    with open(path, "r") as f:
        return f.read()

@tool
def write_file(filepath: str, content: str) -> str:
    """Write content to a file.

    Args:
        filepath: Path to the file to write
        content: The full content to write
    """
    path = safe_path(filepath)
    with open(path, "w") as f:
        f.write(content)
    return f"Wrote to {path}"

@tool
def list_files(directory: str) -> str:
    """List files in a directory.

    Args:
        directory: Path to the directory
    """
    path = safe_path(directory)
    return "\n".join(os.listdir(path))

# Augment the LLM with tools
read_write_tools=[read_file,write_file,list_files]
read_write_tools_by_name = {tool.name: tool for tool in read_write_tools}
read_write_model = base_model.bind_tools(read_write_tools)

read_only_tools=[read_file,write_file,list_files]
read_only_tools_by_name = {tool.name: tool for tool in read_only_tools}
read_only_model = base_model.bind_tools(read_only_tools)

def answer(state: GlobalState):
    """Seeks required info to answer the user directly and finishes"""
    return {
        "messages": [
            read_only_model.invoke(
                [
                    SystemMessage(
                        content="You are a helpful coding assistant tasked with helping the user answer queries about the code in /Users/matias/Projects/learning-langgraph/test-folder"
                    )
                ]
                + state["messages"]
            )
        ]
    }



def read_tool_node(state: dict):
    """Performs the tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = read_only_tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}

# Build workflow
agent_builder = StateGraph(GlobalState)
agent_builder.add_node("clarification_triage", clarification_triage)


