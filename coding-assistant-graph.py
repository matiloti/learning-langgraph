from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.messages import SystemMessage, AnyMessage, ToolMessage, HumanMessage
import os
from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Annotated
import operator
from typing import Literal
from langgraph.graph import StateGraph, START, END
import warnings

warnings.filterwarnings(
    "ignore",
    message="Pydantic serializer warnings"
)

# Define local model
base_model = init_chat_model(
    model="gpt-5", # this is ignored im using zai-org/glm-4.7-flash locally but if i put this name LangGraph throws error
    base_url="http://localhost:1234/v1", # Local model zai-org/glm-4.7-flash
)

class GlobalState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    clarifications_attempts: int
    state: str
    tool_call_attempts: int

class Decision(BaseModel):
    """Triage decision whether to seek clarifications or to code."""
    decision: Literal["clarify_node","code_node","answer_node","random_fact_node"] = Field(..., description="The decision")

decision_model = base_model.with_structured_output(Decision)

def clarification_triage_node(state: dict):
    """LLM decides whether user input requires clarification or not"""

    return {
        "state": str(decision_model.invoke(
            [
                SystemMessage(content=f"""
                              Based on the conversation history, decide if you have enough information whether user only seeks information, or if it requires you to proceed to coding. 

                              - If it requires to code, decide whether you need further clarification. Examples of vague requests are, but not limited to, 'continue' or 'fix it' without further information. 
                              - If the user gives you a really generic and broad task, is considered for clarification. 
                              - If the user provides a simple and well-known coding task, or the user has been cristal clear, or the user has clarified its intent, proceed to code. 
                              - If the user only asks something (info), it's considered a query, so answer directly. 
                              - If user spits nonsense and current input its completely non-actionable or not related to coding, spit a random fact. Nonsense can range from "tell me a joke", "how much R's in strawberry" to "asdfghj" kind of inputs.
                              - If the latest input is not code-related, spit a random fact

                              CRITICAL: max 3 clarifications. You don't need to reach 3 clarifications if you consider it's enough. Current clarifications: {state.get("clarifications_attempts", 0)}/3
                            """
                )
            ] + state["messages"]
        ).decision)
    }

def should_code_or_clarify_or_answer(state: GlobalState) -> Literal["code_node","clarify_node","answer_node"]:
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

read_only_tools=[read_file,list_files]
read_only_tools_by_name = {tool.name: tool for tool in read_only_tools}
read_only_model = base_model.bind_tools(read_only_tools)

def answer_node(state: GlobalState):
    """Seeks required info to answer the user directly and finishes. Decides whether to call a tool or not."""
    print("--- ANSWER ---")
    message = read_only_model.invoke(
                [
                    SystemMessage(
                        content=f"""
                            # ROLE
                            You are a helpful coding assistant tasked with helping the user answer queries about the code. 

                            # RULES
                            - You don't write code, you only read files to answer the user queries.
                            - CRITICAL: Your working dir is '/Users/matias/Projects/learning-langgraph/test-folder'. Never try to use another working dir.
                            - Do not perform more than 3 tool calls. Current tool calls: {state.get("tool_call_attempts", 0)}/3
                            - You CAN'T run or execute files

                            # TOOLS
                            Use the 'read_file' and 'list_files' tools to help yourself in this task:

                            - read_file: reads the whole contents of a file. Args: 
                                - filepath: Path to the file to read
                            - list_files: lists all the files in a directory. Args: 
                                - directory: Path to the directory
                         """
                    )
                ]
                + state["messages"]
            )
    print(message.content)
    return {"messages": [message], "tool_call_attempts": state.get("tool_call_attempts", 0) + 1}

def code_node(state: GlobalState):
    """Reads the necessary files to code and codes what the user asks for. Decides whether to call a tool or not."""
    print("--- CODE ---")
    message = read_write_model.invoke(
                [
                    SystemMessage(
                        content=f"""
                            # ROLE
                            You are a helpful coding assistant tasked with helping the user code.
                            
                            # RULES
                            - CRITICAL: Your working dir is '/Users/matias/Projects/learning-langgraph/test-folder'. Never try to use another working dir.
                            - Do not perform more than 3 tool calls. Current tool calls: {state.get("tool_call_attempts", 0)}/3
                            - You CAN'T run or execute files

                            # TOOLS
                            Use the 'write_file', 'read_file' and 'list_files' tools to help yourself in this task:

                            - write_file: write content to a file. This tool overwrites the whole file, so be careful. Args: 
                                filepath: Path to the file to write
                                content: The full content to write
                            - read_file: reads the whole contents of a file. Args: 
                                filepath: Path to the file to read
                            - list_files: lists all the files in a directory. Args: 
                                directory: Path to the directory
                         """
                    )
                ]
                + state["messages"]
            )
    print(message.content)
    return {"messages": [message], "tool_call_attempts": state.get("tool_call_attempts", 0) + 1}

def clarify_node(state: GlobalState):
    """Clarifies user intent."""
    print("--- CLARIFY ---")
    message = read_write_model.invoke(
                [
                    SystemMessage(
                        content="""
                            # Role
                            You are a helpful coding assistant tasked with clarifying user intent.
                         """
                    )
                ]
                + state["messages"]
            )
    print(message.content)
    return {"messages": [message], "clarifications_attempts": state.get("clarifications_attempts", 0) + 1}

def random_fact_node(state: GlobalState):
    """Random node that spits random facts"""
    print("--- RANDOM FACT ---")
    message = read_write_model.invoke(
                [
                    SystemMessage(
                        content="""
                            # Role
                            If you have arrived here, its because the user spitted nonsense and we dont know how to act.
                            Therefore, we applied the smartest strategy: just spit a random world fact.
                            It can be about whatever: animals, countries, food, people, cinema, technology, pets, travels, sports, politics...
                            Choose your style complete freedom go my boy you got it
                         """
                    )
                ]
                + state["messages"]
            )
    print(message.content)
    return {"messages": [message], "clarifications_attempts": state["clarifications_attempts"] + 1}

def read_tool_node(state: dict):
    """Performs the reading tool call"""
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = read_only_tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}

def read_write_tool_node(state: dict):
    """Performs the write/read tool call"""
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = read_write_tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}

def should_continue_node(state: GlobalState) -> Literal["read_tool_node", "read_write_tool_node", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls and state['state'] == 'answer_node' and not state['tool_call_attempts'] > 3:
        return "read_tool_node"
    elif last_message.tool_calls and state['state'] == 'code_node' and not state['tool_call_attempts'] > 3:
        return "read_write_tool_node"

    # Otherwise, we stop (reply to the user)
    return END

# Build workflow
agent_builder = StateGraph(GlobalState)
agent_builder.add_node("clarification_triage_node", clarification_triage_node)
agent_builder.add_node("code_node", code_node)
agent_builder.add_node("answer_node", answer_node)
agent_builder.add_node("clarify_node", clarify_node)
agent_builder.add_node("random_fact_node", random_fact_node)
agent_builder.add_node("read_tool_node", read_tool_node)
agent_builder.add_node("read_write_tool_node", read_write_tool_node)

agent_builder.add_edge(START, "clarification_triage_node")
agent_builder.add_conditional_edges(
    "clarification_triage_node", 
    should_code_or_clarify_or_answer,
    ["code_node","clarify_node","answer_node","random_fact_node"]
)
agent_builder.add_conditional_edges(
    "code_node", 
    should_continue_node,
    ["read_write_tool_node",END]
)
agent_builder.add_conditional_edges(
    "answer_node", 
    should_continue_node,
    ["read_tool_node",END]
)
agent_builder.add_edge("read_write_tool_node", "code_node")
agent_builder.add_edge("read_tool_node", "answer_node")
agent_builder.add_edge("clarify_node", END)
agent_builder.add_edge("random_fact_node", END)

# Compile the agent
agent = agent_builder.compile()

# Show the agent graph
graph_png = agent.get_graph(xray=True).draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(graph_png)


# Invoke
state = {
    "messages": [],
    "clarifications_attempts": 0,
    "tool_call_attempts": 0,
    "state": ""
}
while True:
    user_input = input("Enter something (type 'q' to quit): ")

    print(user_input)
    if user_input == "q":
        break

    state["messages"].append(HumanMessage(content=user_input))
    state["clarifications_attempts"] = 0
    state["tool_call_attempts"] = 0
    state["state"] = ""
    state = agent.invoke(state)
