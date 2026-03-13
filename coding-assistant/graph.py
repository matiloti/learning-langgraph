"""
LangGraph graph definition for the coding assistant.

Graph topology:
                    ┌─────────────┐
                    │    START     │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   triage    │
                    └──────┬──────┘
                           │
               ┌───────────┼───────────┐
               │           │           │
        ┌──────▼──┐  ┌─────▼────┐ ┌───▼────┐
        │  plan   │  │   code   │ │ answer │
        └────┬────┘  └────┬─────┘ └───┬────┘
             │            │           │
             │      ┌─────▼─────┐ ┌──▼──────────┐
             │      │code_tools │ │ read_tools   │
             │      └─────┬─────┘ └──┬──────────┘
             │            │          │
             └────────┬───┘──────────┘
                      │
               ┌──────▼──────┐
               │  task_router │ ──→ next task or END
               └─────────────┘
"""

from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from state import AgentState
from nodes import (
    triage_node,
    planner_node,
    code_node,
    answer_node,
    read_tool_node,
    code_tool_node,
    summary_node,
)


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def route_after_triage(state: AgentState) -> Literal["planner_node", "code_node", "answer_node"]:
    """Route based on triage decision."""
    route = state.get("route", "answer")
    if route == "plan":
        return "planner_node"
    elif route == "code":
        return "code_node"
    return "answer_node"


def route_after_plan(state: AgentState) -> Literal["code_node", "answer_node"]:
    """After planning, start the first task."""
    route = state.get("route", "code")
    if route == "answer":
        return "answer_node"
    return "code_node"


def route_after_code(state: AgentState) -> Literal["code_tool_node", "task_router"]:
    """After code node: execute tools or move to task router."""
    tool_calls = state.get("tool_calls", [])
    if tool_calls and hasattr(tool_calls[-1], 'tool_calls') and tool_calls[-1].tool_calls:
        return "code_tool_node"
    return "task_router"


def route_after_answer(state: AgentState) -> Literal["read_tool_node", "task_router"]:
    """After answer node: execute tools or move to task router."""
    tool_calls = state.get("tool_calls", [])
    if tool_calls and hasattr(tool_calls[-1], 'tool_calls') and tool_calls[-1].tool_calls:
        return "read_tool_node"
    return "task_router"


def route_after_task(state: AgentState) -> Literal["code_node", "answer_node", "summary_node", END]:
    """After a task completes, route to the next task or finish."""
    route = state.get("route", "done")
    current_id = state.get("current_task_id", 0)

    if route == "done" or current_id == 0:
        # Check if we had tasks — if so, summarize before ending
        tasks = state.get("tasks", [])
        if tasks:
            return "summary_node"
        return END
    elif route == "code":
        return "code_node"
    elif route == "answer":
        return "answer_node"
    return END


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def build_graph():
    """Construct and compile the agent graph."""
    builder = StateGraph(AgentState)

    # Add nodes
    builder.add_node("triage_node", triage_node)
    builder.add_node("planner_node", planner_node)
    builder.add_node("code_node", code_node)
    builder.add_node("answer_node", answer_node)
    builder.add_node("code_tool_node", code_tool_node)
    builder.add_node("read_tool_node", read_tool_node)
    builder.add_node("task_router", lambda state: {})  # pass-through for routing
    builder.add_node("summary_node", summary_node)

    # Entry
    builder.add_edge(START, "triage_node")

    # Triage routes
    builder.add_conditional_edges("triage_node", route_after_triage,
                                   ["planner_node", "code_node", "answer_node"])

    # Plan → first task
    builder.add_conditional_edges("planner_node", route_after_plan,
                                   ["code_node", "answer_node"])

    # Code → tools or task router
    builder.add_conditional_edges("code_node", route_after_code,
                                   ["code_tool_node", "task_router"])

    # Answer → tools or task router
    builder.add_conditional_edges("answer_node", route_after_answer,
                                   ["read_tool_node", "task_router"])

    # Tools loop back to their parent nodes
    builder.add_edge("code_tool_node", "code_node")
    builder.add_edge("read_tool_node", "answer_node")

    # Task router → next task or end
    builder.add_conditional_edges("task_router", route_after_task,
                                   ["code_node", "answer_node", "summary_node", END])

    # Summary → end
    builder.add_edge("summary_node", END)

    # Compile with memory
    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


# Singleton graph instance
agent = build_graph()
