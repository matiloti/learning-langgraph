#!/usr/bin/env python3
"""
Code Assist — LangGraph Coding Assistant
Optimized for small local models (~10B parameters).

Usage:
    python main.py [--workspace PATH] [--model MODEL] [--base-url URL]
"""

import os
import sys
import argparse
import warnings

warnings.filterwarnings("ignore", message="Pydantic serializer warnings")
warnings.filterwarnings("ignore", category=DeprecationWarning)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Code Assist — LangGraph Coding Assistant for Local Models"
    )
    parser.add_argument(
        "--workspace", "-w",
        default=os.getcwd(),
        help="Working directory for the assistant (default: current directory)"
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="Model name (default: qwen2.5-coder:14b)"
    )
    parser.add_argument(
        "--base-url", "-u",
        default=None,
        help="Model API endpoint (default: http://localhost:11434/v1 for Ollama)"
    )
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=None,
        help="Model temperature (default: 0.1)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Set environment variables before importing modules that use them
    workspace = os.path.realpath(args.workspace)
    os.environ["CODING_ASSISTANT_WORKSPACE"] = workspace

    if args.model:
        os.environ["CODING_ASSISTANT_MODEL"] = args.model
    if args.base_url:
        os.environ["CODING_ASSISTANT_BASE_URL"] = args.base_url
    if args.temperature is not None:
        os.environ["CODING_ASSISTANT_TEMPERATURE"] = str(args.temperature)

    # Now import everything (after env vars are set)
    from graph import agent
    from ui import (
        console, print_banner, print_config, print_welcome,
        print_assistant_message, print_tool_call, print_tool_result,
        print_tasks, print_plan_created, print_task_started,
        print_task_completed, print_all_done, print_error,
        print_warning, print_info, get_input, get_spinner,
    )
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

    model_name = os.environ.get("CODING_ASSISTANT_MODEL", "qwen2.5-coder:14b")
    base_url = os.environ.get("CODING_ASSISTANT_BASE_URL", "http://localhost:11434/v1")

    # Startup
    print_banner()
    print_config(model=model_name, base_url=base_url, workspace=workspace)
    print_welcome()

    # Conversation config (thread persistence)
    config = {"configurable": {"thread_id": "main"}}
    thread_counter = 1

    while True:
        user_input = get_input().strip()

        if not user_input:
            continue

        # Handle commands
        if user_input.lower() in ("quit", "exit", "q"):
            console.print("\n[dim]Goodbye! 👋[/]\n")
            break

        if user_input.lower() == "help":
            print_welcome()
            continue

        if user_input.lower() == "tasks":
            # Get current state from checkpointer
            try:
                state = agent.get_state(config)
                tasks = state.values.get("tasks", [])
                current_id = state.values.get("current_task_id", 0)
                print_tasks(tasks, current_id)
            except Exception:
                print_info("No active tasks.")
            continue

        if user_input.lower() == "clear":
            thread_counter += 1
            config = {"configurable": {"thread_id": f"thread-{thread_counter}"}}
            print_info("Conversation cleared. Starting fresh.")
            continue

        # Track state for UI updates
        prev_tasks = []
        prev_task_id = 0
        shown_task_starts = set()

        try:
            with get_spinner("Thinking..."):
                # We'll collect all events first then display
                pass

            # Stream the graph execution
            for event in agent.stream(
                {"messages": [HumanMessage(content=user_input)]},
                stream_mode="updates",
                config=config,
            ):
                for node_name, updates in event.items():
                    if not updates:
                        continue

                    # Show task plan creation
                    if node_name == "planner_node" and "tasks" in updates:
                        print_plan_created(updates["tasks"])
                        prev_tasks = updates["tasks"]
                        prev_task_id = updates.get("current_task_id", 0)

                    # Show task transitions
                    if "tasks" in updates:
                        current_tasks = updates["tasks"]
                        current_id = updates.get("current_task_id", prev_task_id)

                        # Check for newly completed tasks
                        for t in current_tasks:
                            tid = t.get("id")
                            if t.get("status") == "done":
                                # Find if it was previously not done
                                old = next((pt for pt in prev_tasks if pt.get("id") == tid), None)
                                if old and old.get("status") != "done":
                                    print_task_completed(t)

                            if t.get("status") == "in_progress" and tid not in shown_task_starts:
                                shown_task_starts.add(tid)
                                if node_name != "planner_node":
                                    print_task_started(t)

                        prev_tasks = current_tasks
                        prev_task_id = current_id

                    # Show tool calls
                    if "tool_calls" in updates and updates["tool_calls"]:
                        for msg in updates["tool_calls"]:
                            if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
                                for tc in msg.tool_calls:
                                    print_tool_call(tc["name"], tc["args"])
                            elif isinstance(msg, ToolMessage):
                                print_tool_result(msg.content)

                    # Show assistant messages
                    if "messages" in updates:
                        for msg in updates["messages"]:
                            if isinstance(msg, AIMessage):
                                content = str(msg.content)
                                if content.strip() and not msg.tool_calls:
                                    print_assistant_message(content)

            # Check if all tasks completed
            try:
                state = agent.get_state(config)
                tasks = state.values.get("tasks", [])
                if tasks and all(t.get("status") in ("done", "skipped") for t in tasks):
                    print_all_done()
            except Exception:
                pass

        except KeyboardInterrupt:
            print_warning("Interrupted. Type 'quit' to exit.")
        except Exception as e:
            print_error(str(e))
            if "Connection" in str(e) or "refused" in str(e).lower():
                print_info(f"Make sure your model is running at {base_url}")
                print_info("For Ollama: ollama serve && ollama pull qwen2.5-coder:14b")


if __name__ == "__main__":
    main()
