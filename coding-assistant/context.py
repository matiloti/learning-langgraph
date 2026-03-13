"""
Context engineering utilities for small local models.

Key strategies (informed by ACE, JetBrains Research, and Anthropic's guide):
1. Rolling summaries — compress old conversation into a summary
2. Selective context — only include relevant file contents
3. Token budgeting — keep total context under model limits
4. Structured scaffolding — format context so small models parse it reliably
5. Observation masking — truncate tool outputs while preserving reasoning
   (JetBrains Research showed this matches LLM summarization quality)
6. Context isolation — each task gets focused context, not the full history
"""

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage


# ---------------------------------------------------------------------------
# Token estimation (rough, avoids needing tiktoken for local models)
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English/code."""
    return len(text) // 4


def estimate_messages_tokens(messages: list) -> int:
    """Estimate tokens across a list of messages."""
    total = 0
    for msg in messages:
        content = msg.content if hasattr(msg, 'content') else str(msg)
        total += estimate_tokens(str(content)) + 10  # overhead per message
    return total


# ---------------------------------------------------------------------------
# Observation masking (JetBrains Research technique)
# Agent turns skew heavily toward tool output. Mask/truncate observations
# while preserving action and reasoning history in full.
# ---------------------------------------------------------------------------

def mask_tool_observations(messages: list, max_observation_chars: int = 500) -> list:
    """Truncate tool output messages while keeping AI reasoning intact.

    Research shows agent context is ~80% tool output. Masking observations
    to a summary preserves problem-solving ability while saving massive
    amounts of context window space.
    """
    masked = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            content = str(msg.content)
            if len(content) > max_observation_chars:
                # Keep first and last portions, add a count of what was trimmed
                lines = content.split("\n")
                if len(lines) > 10:
                    kept = "\n".join(lines[:5]) + f"\n[... {len(lines) - 10} lines masked ...]\n" + "\n".join(lines[-5:])
                else:
                    kept = content[:max_observation_chars] + f"\n[... truncated, {len(content)} chars total ...]"
                masked.append(ToolMessage(content=kept, tool_call_id=msg.tool_call_id))
            else:
                masked.append(msg)
        else:
            masked.append(msg)
    return masked


# ---------------------------------------------------------------------------
# Conversation compression
# ---------------------------------------------------------------------------

def compress_conversation(messages: list, max_tokens: int = 3000) -> list:
    """Keep recent messages, compress older ones into a summary.

    Strategy for small models:
    - Always keep the last 6 messages (3 turns) intact
    - Summarize everything before that into a compact context block
    - This prevents small models from getting confused by long histories
    """
    if estimate_messages_tokens(messages) <= max_tokens:
        return messages

    # Keep last 6 messages intact
    keep_count = min(6, len(messages))
    recent = messages[-keep_count:]
    older = messages[:-keep_count]

    if not older:
        return recent

    # Build a compact summary of older messages
    summary_parts = []
    for msg in older:
        content = msg.content if hasattr(msg, 'content') else str(msg)
        if isinstance(msg, HumanMessage):
            # Truncate long user messages
            text = str(content)[:200]
            summary_parts.append(f"User: {text}")
        elif isinstance(msg, AIMessage):
            text = str(content)[:150]
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                tools_used = [tc['name'] for tc in msg.tool_calls]
                summary_parts.append(f"Assistant used tools: {', '.join(tools_used)}")
            elif text.strip():
                summary_parts.append(f"Assistant: {text}")
        elif isinstance(msg, ToolMessage):
            text = str(content)[:100]
            summary_parts.append(f"Tool result: {text}")

    summary_text = "\n".join(summary_parts)
    summary_msg = SystemMessage(content=f"[CONVERSATION HISTORY SUMMARY]\n{summary_text}\n[END SUMMARY]")

    return [summary_msg] + recent


# ---------------------------------------------------------------------------
# File context builder
# ---------------------------------------------------------------------------

def build_file_context(working_files: list[str], max_per_file: int = 1500) -> str:
    """Build a compact context string from working files.

    For small models, we:
    - Only include files the assistant has actively worked with
    - Truncate large files with a note
    - Format with clear delimiters so the model knows where file content starts/ends
    """
    if not working_files:
        return ""

    import os
    parts = ["[FILES IN CONTEXT]"]
    for filepath in working_files[-5:]:  # cap at 5 most recent files
        if not os.path.isfile(filepath):
            continue
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except Exception:
            continue

        if len(content) > max_per_file * 4:  # rough char limit
            content = content[:max_per_file * 4] + f"\n... (truncated, {len(content)} chars total)"

        parts.append(f"--- {filepath} ---\n{content}\n--- end ---")

    parts.append("[END FILES]")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Task context builder
# ---------------------------------------------------------------------------

def build_task_context(tasks: list[dict], current_task_id: int) -> str:
    """Format task list for the model with clear status indicators.

    Uses simple, unambiguous formatting that small models parse reliably.
    """
    if not tasks:
        return ""

    lines = ["[TASK LIST]"]
    for t in tasks:
        status_icon = {
            "pending": "[ ]",
            "in_progress": "[>]",
            "done": "[x]",
            "skipped": "[-]",
        }.get(t.get("status", "pending"), "[ ]")

        marker = " <-- CURRENT" if t.get("id") == current_task_id else ""
        lines.append(f"{status_icon} Task {t['id']}: {t['description']} ({t['type']}){marker}")

    lines.append("[END TASKS]")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Assemble full context for a node
# ---------------------------------------------------------------------------

def assemble_context(
    system_prompt: str,
    messages: list,
    tasks: list[dict] = None,
    current_task_id: int = 0,
    context_summary: str = "",
    working_files: list[str] = None,
    max_context_tokens: int = 6000,
) -> list:
    """Assemble the full message list for an LLM call.

    This is the main context engineering function. It:
    1. Builds a rich system prompt with task context
    2. Compresses conversation history
    3. Stays within token budget
    """
    # Build enriched system prompt
    parts = [system_prompt]

    if context_summary:
        parts.append(f"\n[WORK DONE SO FAR]\n{context_summary}\n[END WORK SUMMARY]")

    if tasks:
        parts.append(f"\n{build_task_context(tasks, current_task_id)}")

    if working_files:
        file_ctx = build_file_context(working_files)
        if file_ctx:
            parts.append(f"\n{file_ctx}")

    full_system = "\n".join(parts)

    # Budget: system prompt gets priority, then recent messages
    system_tokens = estimate_tokens(full_system)
    remaining_budget = max(max_context_tokens - system_tokens, 1500)

    # Apply observation masking before compression (saves ~60% of context)
    masked = mask_tool_observations(messages)
    compressed = compress_conversation(masked, max_tokens=remaining_budget)

    return [SystemMessage(content=full_system)] + compressed
