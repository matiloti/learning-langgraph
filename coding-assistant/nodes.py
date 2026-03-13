"""
Graph node implementations.

Each node is a pure function: state in → state updates out.
Context engineering is applied at every LLM call to keep small models effective.
"""

from langchain_core.messages import (
    SystemMessage, HumanMessage, AIMessage, ToolMessage
)
from state import AgentState
from tools import read_tools_by_name, read_write_tools_by_name
from context import assemble_context
from prompts import (
    triage_prompt, planner_prompt, code_prompt,
    answer_prompt, summary_prompt
)


# ---------------------------------------------------------------------------
# TRIAGE NODE — routes the user's message
# ---------------------------------------------------------------------------

def triage_node(state: AgentState) -> dict:
    """Classify the user's message and decide routing."""
    from models import triage_model

    messages = assemble_context(
        system_prompt=triage_prompt(),
        messages=state["messages"],
        max_context_tokens=2000,
    )

    try:
        decision = triage_model.invoke(messages)
        if decision.needs_planning:
            return {"route": "plan"}
        return {"route": decision.type}
    except Exception:
        # Fallback: if structured output fails, default to code
        return {"route": "code"}


# ---------------------------------------------------------------------------
# PLANNER NODE — breaks requests into tasks
# ---------------------------------------------------------------------------

def planner_node(state: AgentState) -> dict:
    """Break the user's request into atomic tasks."""
    from models import plan_model

    messages = assemble_context(
        system_prompt=planner_prompt(),
        messages=state["messages"],
        tasks=state.get("tasks", []),
        context_summary=state.get("context_summary", ""),
        max_context_tokens=3000,
    )

    try:
        plan = plan_model.invoke(messages)
        tasks = [t.model_dump() for t in plan.tasks]
        # Set first task to in_progress
        if tasks:
            tasks[0]["status"] = "in_progress"
        return {
            "tasks": tasks,
            "current_task_id": tasks[0]["id"] if tasks else 0,
            "route": tasks[0]["type"] if tasks else "answer",
        }
    except Exception as e:
        # Fallback: create a single task from the user's message
        user_msg = ""
        for m in reversed(state["messages"]):
            if isinstance(m, HumanMessage):
                user_msg = str(m.content)
                break
        fallback_task = {
            "id": 1,
            "description": user_msg[:200] or "Complete the user's request",
            "type": "code",
            "status": "in_progress",
        }
        return {
            "tasks": [fallback_task],
            "current_task_id": 1,
            "route": "code",
        }


# ---------------------------------------------------------------------------
# CODE NODE — executes coding tasks with tools
# ---------------------------------------------------------------------------

def code_node(state: AgentState) -> dict:
    """Execute a coding task. Can call read/write tools."""
    from models import code_model

    attempts = state.get("tool_call_attempts", 0)
    max_attempts = 12
    remaining = max_attempts - attempts

    # Build current task description for focused context
    current_task = _get_current_task(state)
    task_instruction = ""
    if current_task:
        task_instruction = f"\n\nCURRENT TASK: {current_task['description']}"

    messages = assemble_context(
        system_prompt=code_prompt(remaining) + task_instruction,
        messages=state["messages"],
        tasks=state.get("tasks", []),
        current_task_id=state.get("current_task_id", 0),
        context_summary=state.get("context_summary", ""),
        working_files=state.get("working_files", []),
        max_context_tokens=5000,
    )

    # Include pending tool call chain
    tool_chain = state.get("tool_calls", [])
    if tool_chain:
        messages = messages + tool_chain

    message = code_model.invoke(messages)

    # Model wants to call tools and has budget
    if message.tool_calls and attempts < max_attempts:
        return {
            "tool_calls": [message],
            "tool_call_attempts": attempts + 1,
        }

    # Model hit budget — force a text response
    if message.tool_calls:
        from models import base_model
        forced = messages + [
            AIMessage(content="I've used all my tool calls. Let me summarize what I've done so far.")
        ]
        message = base_model.invoke(forced)

    # Check if task is complete
    content = str(message.content)
    task_done = "TASK COMPLETE" in content.upper()

    updates = {
        "messages": [message],
        "tool_calls": [],
        "tool_call_attempts": 0,
    }

    if task_done:
        updates.update(_advance_task(state))

    return updates


# ---------------------------------------------------------------------------
# ANSWER NODE — answers questions (read-only tools)
# ---------------------------------------------------------------------------

def answer_node(state: AgentState) -> dict:
    """Answer a question, optionally reading files."""
    from models import answer_model

    attempts = state.get("tool_call_attempts", 0)
    max_attempts = 4
    remaining = max_attempts - attempts

    current_task = _get_current_task(state)
    task_instruction = ""
    if current_task:
        task_instruction = f"\n\nCURRENT TASK: {current_task['description']}"

    messages = assemble_context(
        system_prompt=answer_prompt(remaining) + task_instruction,
        messages=state["messages"],
        tasks=state.get("tasks", []),
        current_task_id=state.get("current_task_id", 0),
        context_summary=state.get("context_summary", ""),
        max_context_tokens=4000,
    )

    tool_chain = state.get("tool_calls", [])
    if tool_chain:
        messages = messages + tool_chain

    message = answer_model.invoke(messages)

    if message.tool_calls and attempts < max_attempts:
        return {
            "tool_calls": [message],
            "tool_call_attempts": attempts + 1,
        }

    if message.tool_calls:
        from models import base_model
        forced = messages + [
            AIMessage(content="I've used all my tool calls. Let me answer with what I know.")
        ]
        message = base_model.invoke(forced)

    content = str(message.content)
    task_done = "TASK COMPLETE" in content.upper()

    updates = {
        "messages": [message],
        "tool_calls": [],
        "tool_call_attempts": 0,
    }

    if task_done:
        updates.update(_advance_task(state))

    return updates


# ---------------------------------------------------------------------------
# TOOL EXECUTION NODES
# ---------------------------------------------------------------------------

def _execute_tools(state: AgentState, tools_by_name: dict) -> dict:
    """Shared tool execution logic with error handling."""
    tool_chain = state.get("tool_calls", [])
    if not tool_chain:
        return {"tool_calls": [], "tool_call_attempts": 0}

    last_ai_msg = tool_chain[-1]
    if not hasattr(last_ai_msg, 'tool_calls') or not last_ai_msg.tool_calls:
        return {"tool_calls": [], "tool_call_attempts": 0}

    results = []
    working_files = list(state.get("working_files", []))

    for tc in last_ai_msg.tool_calls:
        name = tc["name"]
        args = tc["args"]
        try:
            if name not in tools_by_name:
                raise KeyError(f"Unknown tool: {name}. Available: {', '.join(tools_by_name.keys())}")
            observation = tools_by_name[name].invoke(args)
            results.append(ToolMessage(content=str(observation), tool_call_id=tc["id"]))

            # Track files we've interacted with
            if "filepath" in args:
                fp = args["filepath"]
                if fp not in working_files:
                    working_files.append(fp)
        except Exception as e:
            error_msg = f"Error calling {name}: {e}"
            results.append(ToolMessage(content=error_msg, tool_call_id=tc["id"]))

    # Build the tool chain: AI message + tool results
    return {
        "tool_calls": [last_ai_msg] + results,
        "working_files": working_files,
    }


def read_tool_node(state: AgentState) -> dict:
    """Execute read-only tools."""
    return _execute_tools(state, read_tools_by_name)


def code_tool_node(state: AgentState) -> dict:
    """Execute read+write tools."""
    return _execute_tools(state, read_write_tools_by_name)


# ---------------------------------------------------------------------------
# SUMMARY NODE — compress context after task completion
# ---------------------------------------------------------------------------

def summary_node(state: AgentState) -> dict:
    """Generate a rolling summary of work done so far."""
    from models import base_model

    # Quick summary from recent messages
    recent = state["messages"][-6:]
    summary_msgs = [
        SystemMessage(content=summary_prompt()),
    ] + recent

    try:
        result = base_model.invoke(summary_msgs)
        old_summary = state.get("context_summary", "")
        new_summary = str(result.content)
        if old_summary:
            combined = f"{old_summary}\n\nLatest: {new_summary}"
        else:
            combined = new_summary
        # Keep summary compact
        if len(combined) > 1500:
            combined = combined[-1500:]
        return {"context_summary": combined}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _get_current_task(state: AgentState) -> dict | None:
    """Get the current task being worked on."""
    tasks = state.get("tasks", [])
    current_id = state.get("current_task_id", 0)
    for t in tasks:
        if t.get("id") == current_id:
            return t
    return None


def _advance_task(state: AgentState) -> dict:
    """Mark current task as done and advance to next pending task."""
    tasks = list(state.get("tasks", []))
    current_id = state.get("current_task_id", 0)

    # Mark current as done
    for t in tasks:
        if t.get("id") == current_id:
            t["status"] = "done"

    # Find next pending task
    next_task = None
    for t in tasks:
        if t.get("status") == "pending":
            t["status"] = "in_progress"
            next_task = t
            break

    if next_task:
        return {
            "tasks": tasks,
            "current_task_id": next_task["id"],
            "route": next_task["type"],
        }
    else:
        return {
            "tasks": tasks,
            "current_task_id": 0,
            "route": "done",
        }
