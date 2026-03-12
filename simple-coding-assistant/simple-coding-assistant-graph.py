from langchain.messages import SystemMessage, ToolMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
import warnings
from tools import (
    read_write_tools, read_write_tools_by_name,
    read_tools, read_tools_by_name,
)
from schemas import GlobalState, Decision
from models import (
    base_model, decision_model, read_only_model, read_write_model
)
from prompts import triage_prompt, answer_prompt, code_prompt
from typing import Literal

warnings.filterwarnings(
    "ignore",
    message="Pydantic serializer warnings"
)

def triage_node(state: GlobalState):
    """LLM decides whether user input requires coding or just talking"""
    decision = decision_model.invoke(
        [SystemMessage(content=triage_prompt())]
        + state["messages"]
    ).decision
    return {"state": decision}

def route(state: GlobalState) -> Literal["talk_node", "code_node", END]:
    return state["state"]

def talk_node(state: GlobalState):
    """Talks with the user. If it required info to answer, the coding assistant can decide whether to read files using a tool or not."""
    message = read_only_model.invoke(
                [SystemMessage(content=answer_prompt(state.get("tool_call_attempts", 0)))]
                + state["messages"]
                + state.get("tool_calls", [])
            )
    
    # LLM wants to call tools and we have budget left
    if message.tool_calls and state.get("tool_call_attempts", 0) < 3:
        return {"tool_calls": [message], "tool_call_attempts": state.get("tool_call_attempts", 0) + 1}

    # LLM wants to call tools but we've hit the limit — force a final answer
    if message.tool_calls:
        forced_msgs = state["messages"] + [AIMessage(content="I've reached the tool call limit, so I will respond with what I have.")]
        message = base_model.invoke(
                    [SystemMessage(content=answer_prompt(state.get("tool_call_attempts", 0)))]
                    + forced_msgs
                )

    # No tool calls (either naturally or after forced re-invoke) — return the answer
    return {"messages": [message], "tool_calls": [], "tool_call_attempts": 0}

def code_node(state: GlobalState):
    """Reads the necessary files to code and codes what the user asks for. Decides whether to call a tool or not."""
    print("--- CODE ---")
    message = read_write_model.invoke(
                [SystemMessage(content=code_prompt(state.get("tool_call_attempts", 0)))]
                + state["messages"]
                + state.get("tool_calls", [])
            )
    
    # LLM wants to call tools and we have budget left
    if message.tool_calls and state.get("tool_call_attempts", 0) < 10:
        return {"tool_calls": [message], "tool_call_attempts": state.get("tool_call_attempts", 0) + 1}
    
    # LLM wants to call tools but we've hit the limit — force a final answer
    if message.tool_calls:
        forced_msgs = state["messages"] + [AIMessage(content="I've reached the tool call limit, so I will respond with what I have.")]
        message = base_model.invoke(
                    [SystemMessage(content=answer_prompt(state.get("tool_call_attempts", 0)))]
                    + forced_msgs
                )
    
    # No tool calls (either naturally or after forced re-invoke) — return the answer
    return {"messages": [message], "tool_calls": [], "tool_call_attempts": 0}

def _execute_tool_calls(state: dict, tools_by_name: dict):
    """Shared logic for executing tool calls with error handling."""
    result = []
    summaries = []
    for tool_call in state["tool_calls"][-1].tool_calls:
        name = tool_call["name"]
        args = tool_call["args"]
        args_str = ", ".join(f"{k}={v}" for k, v in args.items())
        try:
            if name not in tools_by_name:
                raise KeyError(f"Unknown tool: {name}")
            tool = tools_by_name[name]
            observation = tool.invoke(args)
            result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
            summaries.append(f"{name}({args_str})")
        except Exception as e:
            error_msg = f"Error calling {name}({args_str}): {e}"
            result.append(ToolMessage(content=error_msg, tool_call_id=tool_call["id"]))
            summaries.append(f"{name}({args_str}) -> ERROR: {e}")
    # Preserve the AI message that requested the tool calls + the results
    # The LLM needs: AIMessage(tool_calls=[...]) followed by ToolMessage(tool_call_id=...) for each call
    ai_message = state["tool_calls"][-1]
    digest = "I've called: " + ", ".join(summaries)
    return {"messages": [AIMessage(content=digest)], "tool_calls": [ai_message] + result}

def read_tool_node(state: dict):
    """Performs the reading tool call"""
    return _execute_tool_calls(state, read_tools_by_name)

def read_write_tool_node(state: dict):
    """Performs the write/read tool call"""
    return _execute_tool_calls(state, read_write_tools_by_name)

def should_continue_node(state: GlobalState) -> Literal["read_tool_node", "read_write_tool_node", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    tool_calls = state["tool_calls"]
    if len(tool_calls) == 0:
        return END
    
    last_message = tool_calls[-1]

    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls and state["state"] == "talk_node" and not state['tool_call_attempts'] > 3:
        return "read_tool_node"
    elif last_message.tool_calls and state["state"] == "code_node" and not state['tool_call_attempts'] > 10:
        return "read_write_tool_node"

    # Otherwise, we stop (reply to the user)
    return END

# Build workflow
agent_builder = StateGraph(GlobalState)
agent_builder.add_node("triage_node", triage_node)
agent_builder.add_node("talk_node", talk_node)
agent_builder.add_node("code_node", code_node)
agent_builder.add_node("read_tool_node", read_tool_node)
agent_builder.add_node("read_write_tool_node", read_write_tool_node)

agent_builder.add_edge(START, "triage_node")
agent_builder.add_conditional_edges("triage_node", route, ["talk_node", "code_node"])
agent_builder.add_conditional_edges("talk_node", should_continue_node, ["read_tool_node", END])
agent_builder.add_conditional_edges("code_node", should_continue_node, ["read_write_tool_node", END])
agent_builder.add_edge("read_tool_node", "talk_node")
agent_builder.add_edge("read_write_tool_node", "code_node")

# Compile the agent with checkpointer for conversation persistence
checkpointer = InMemorySaver()
agent = agent_builder.compile(checkpointer=checkpointer)

# Show the agent graph
graph_png = agent.get_graph(xray=True).draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(graph_png)


# Invoke
config = {"configurable": {"thread_id": "1"}}
while True:
    user_input = input("> ")

    if user_input == "q":
        break

    for chunk in agent.stream(
        {"messages": [HumanMessage(content=user_input)]},
        stream_mode="updates",
        config=config
    ):
        for node_name, state in chunk.items():
            if state and "messages" in state:
                for msg in state["messages"]:
                    print(f"+ {msg.content}")