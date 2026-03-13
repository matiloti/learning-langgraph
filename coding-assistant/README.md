# Code Assist — LangGraph Coding Assistant

A task-planning coding assistant built with [LangGraph](https://langchain-ai.github.io/langgraph/), specifically optimized to run with **small local models** (~10B parameters) via [Ollama](https://ollama.ai), [LM Studio](https://lmstudio.ai/), or any OpenAI-compatible endpoint.

```
  ██████╗ ██████╗ ██████╗ ███████╗    █████╗ ███████╗███████╗██╗███████╗████████╗
 ██╔════╝██╔═══██╗██╔══██╗██╔════╝   ██╔══██╗██╔════╝██╔════╝██║██╔════╝╚══██╔══╝
 ██║     ██║   ██║██║  ██║█████╗     ███████║███████╗███████╗██║███████╗   ██║
 ██║     ██║   ██║██║  ██║██╔══╝     ██╔══██║╚════██║╚════██║██║╚════██║   ██║
 ╚██████╗╚██████╔╝██████╔╝███████╗   ██║  ██║███████║███████║██║███████║   ██║
  ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝   ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝╚══════╝   ╚═╝
```

## What Makes This Different?

Most coding assistants assume access to powerful cloud models (GPT-4, Claude, etc.). This one is **designed from the ground up** to work reliably with small local models. It uses several techniques to compensate for smaller model limitations:

| Technique | What It Does |
|-----------|-------------|
| **Task Decomposition** | Complex requests are broken into small, atomic tasks that small models can handle one at a time |
| **Context Compression** | Rolling conversation summaries keep context within small model token limits |
| **Structured Output Scaffolding** | Pydantic models + explicit examples guide small models toward valid responses |
| **Token Budgeting** | Automatic context window management ensures prompts stay within limits |
| **Tool Call Guardrails** | Budget limits and fallback mechanisms prevent infinite tool loops |
| **Explicit Prompting** | Prompts use simple language, clear delimiters, and few-shot examples — optimized for 10B-class models |

## Architecture

The assistant is built as a LangGraph **StateGraph** with the following topology:

```
                     ┌─────────────┐
                     │    START     │
                     └──────┬──────┘
                            │
                     ┌──────▼──────┐
                     │   Triage    │  ← Classifies: code / answer / plan
                     └──────┬──────┘
                            │
                ┌───────────┼───────────┐
                │           │           │
         ┌──────▼──┐  ┌─────▼────┐ ┌───▼────┐
         │ Planner │  │   Code   │ │ Answer │
         └────┬────┘  └────┬─────┘ └───┬────┘
              │            │           │
              │      ┌─────▼─────┐ ┌──▼──────────┐
              │      │Code Tools │ │ Read Tools   │
              │      └─────┬─────┘ └──┬──────────┘
              │            │          │       ↑ loops
              └────────┬───┴──────────┘       │
                       │                      │
                ┌──────▼──────┐               │
                │ Task Router │───────────────┘
                └──────┬──────┘
                       │
                ┌──────▼──────┐
                │   Summary   │  ← Compresses context
                └──────┬──────┘
                       │
                     ┌─▼──┐
                     │ END│
                     └────┘
```

### Node Descriptions

| Node | Purpose |
|------|---------|
| **Triage** | Routes messages — simple questions go to Answer, simple code tasks go to Code, complex requests go to Planner |
| **Planner** | Breaks complex requests into 1-8 atomic tasks with structured output |
| **Code** | Executes coding tasks using read/write file tools |
| **Answer** | Answers questions using read-only file tools |
| **Code Tools** | Executes file write operations (create, edit, delete) |
| **Read Tools** | Executes file read operations (list, read, search, find) |
| **Task Router** | Advances to the next task or finishes when all tasks are done |
| **Summary** | Generates a rolling context summary to keep conversation compact |

### State Schema

```python
class AgentState(TypedDict):
    messages: list[AnyMessage]        # Conversation history
    tasks: list[dict]                 # Task list from planner
    current_task_id: int              # Active task
    tool_calls: list[AnyMessage]      # Pending tool call chain
    tool_call_attempts: int           # Budget tracking
    context_summary: str              # Rolling work summary
    working_files: list[str]          # Files touched this session
    route: str                        # Current routing decision
```

## Quick Start

### Prerequisites

- **Python 3.10+**
- **Ollama** (recommended) or any OpenAI-compatible local model server

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd learning-langgraph/coding-assistant

# Run the setup script (creates venv, installs deps, adds alias)
./setup.sh

# Reload your shell
source ~/.zshrc
```

### Pull a Model

```bash
# Recommended: Qwen 2.5 Coder 14B (best balance of size and coding ability)
ollama pull qwen2.5-coder:14b

# Alternatives:
ollama pull qwen2.5-coder:7b        # Lighter, still good
ollama pull deepseek-coder-v2:16b    # Strong at code generation
ollama pull codellama:13b            # Meta's code model
ollama pull qwen2.5:14b              # Good general + code
```

### Run

```bash
# Make sure Ollama is serving
ollama serve

# Navigate to any project and run
cd /path/to/your/project
codeassist

# Or specify workspace explicitly
codeassist --workspace /path/to/project

# Use a different model
codeassist --model codellama:13b

# Use LM Studio instead of Ollama
codeassist --base-url http://localhost:1234/v1
```

## Usage

### Interactive Commands

| Command | Description |
|---------|-------------|
| `quit` / `exit` | Exit the assistant |
| `tasks` | Show current task list with status |
| `clear` | Clear conversation history |
| `help` | Show available commands |

### Example Sessions

**Simple question:**
```
You > What files are in this project?
Assistant > [Lists files and describes the project structure]
```

**Single coding task:**
```
You > Create a Python function that calculates fibonacci numbers in utils.py
Assistant > [Creates the file using tools, shows tool calls and results]
```

**Complex multi-step request:**
```
You > Build a REST API with Flask that has user registration, login, and a protected endpoint

📋 Plan created with 5 task(s)
┌───┬───────┬──────────────────────────────────────────────────────────────┬────────┐
│ # │Status │ Task                                                         │ Type   │
├───┼───────┼──────────────────────────────────────────────────────────────┼────────┤
│ 1 │  ◉    │ Create app.py with Flask app setup and config                │ code   │
│ 2 │  ○    │ Add User model and database setup                            │ code   │
│ 3 │  ○    │ Implement registration endpoint POST /register               │ code   │
│ 4 │  ○    │ Implement login endpoint POST /login with JWT                │ code   │
│ 5 │  ○    │ Add protected endpoint GET /profile with auth middleware     │ code   │
└───┴───────┴──────────────────────────────────────────────────────────────┴────────┘

▶ Starting task 1: Create app.py with Flask app setup and config
  ⚡ list_files(directory=".")
  ⚡ write_file(filepath="app.py", content=<file content>)
✓ Completed task 1
...
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CODING_ASSISTANT_MODEL` | `qwen2.5-coder:14b` | Model name |
| `CODING_ASSISTANT_BASE_URL` | `http://localhost:11434/v1` | API endpoint |
| `CODING_ASSISTANT_API_KEY` | `not-needed` | API key (not needed for Ollama) |
| `CODING_ASSISTANT_TEMPERATURE` | `0.1` | Model temperature |
| `CODING_ASSISTANT_WORKSPACE` | Current directory | Working directory |

### CLI Arguments

```bash
codeassist --model MODEL --base-url URL --workspace PATH --temperature FLOAT
```

## Recommended Models

Tested and ranked for coding assistant tasks:

| Model | Size | Strengths | Tool Calling | Recommended |
|-------|------|-----------|-------------|-------------|
| `qwen2.5-coder:14b` | 14B | Best overall for coding, good tool use | Native | Yes |
| `qwen2.5-coder:7b` | 7B | Good coding, fast, lower VRAM | Native | Budget pick |
| `deepseek-coder-v2:16b` | 16B | Strong generation, good reasoning | Good | Yes |
| `codellama:13b` | 13B | Solid coding, well-known | Basic | Okay |
| `qwen2.5:14b` | 14B | Good general purpose + code | Native | General use |

**VRAM Requirements:**
- 7B models: ~6GB VRAM
- 13-14B models: ~10-12GB VRAM
- 16B models: ~12-14GB VRAM

## Project Structure

```
coding-assistant/
├── main.py            # Entry point — CLI argument parsing, main loop
├── graph.py           # LangGraph StateGraph definition and compilation
├── nodes.py           # All graph node implementations
├── state.py           # Pydantic models and TypedDict state schemas
├── models.py          # LLM configuration and specialized model variants
├── tools.py           # File operation tools (read, write, edit, search)
├── prompts.py         # All prompts — optimized for small models
├── context.py         # Context engineering (compression, budgeting, assembly)
├── ui.py              # Rich terminal UI (colors, panels, tables, spinners)
├── requirements.txt   # Python dependencies
├── setup.sh           # One-command setup script
└── README.md          # This file
```

## Context Engineering Deep Dive

The assistant uses several techniques to work within the constraints of small models:

### 1. Conversation Compression

Old messages are summarized into compact context blocks, keeping only the last 6 messages (3 turns) intact. This prevents context window overflow while preserving essential information.

```python
# Before compression: 50 messages, ~8000 tokens
# After compression: 1 summary + 6 recent messages, ~2000 tokens
```

### 2. Token Budgeting

Every LLM call goes through `assemble_context()` which:
1. Builds the system prompt with task context
2. Estimates token usage
3. Allocates remaining budget to conversation history
4. Compresses if needed

### 3. Structured Output Scaffolding

Instead of asking small models to figure out complex routing or planning in free text, we use Pydantic structured output:

```python
class TriageDecision(BaseModel):
    needs_planning: bool    # Simple boolean — easy for small models
    type: Literal["code", "answer", "plan"]  # Constrained choices
```

### 4. Explicit Prompting

Every prompt follows these rules for small models:
- **EXPLICIT instructions** with numbered steps
- **FEW-SHOT examples** showing expected behavior
- **CLEAR delimiters** (`[SECTION]...[END SECTION]`) for structured data
- **SIMPLE language** avoiding complex phrasing
- **GUARDRAILS** telling the model what NOT to do

### 5. Task Decomposition

Complex requests are broken into tasks of 1-3 tool calls each. This means the model only needs to focus on one small thing at a time, dramatically improving reliability with smaller models.

### 6. Observation Masking

Inspired by [JetBrains Research](https://blog.jetbrains.com/research/2025/12/efficient-context-management/): agent conversation turns are ~80% tool output. We mask/truncate tool observations while preserving AI reasoning and action history in full. This matches LLM-based summarization quality while being instant and free.

```python
# Before masking: read_file returns 500 lines → 2000 tokens
# After masking: first 5 + last 5 lines + "[... 490 lines masked ...]" → 100 tokens
```

### 7. Context Isolation

Each task in the plan gets focused context — only the task description, relevant files, and recent history. Inspired by the [Deep Agents](https://github.com/langchain-ai/deepagents) subagent isolation pattern and the [ACE framework](https://arxiv.org/abs/2510.04618) which showed that DeepSeek-V3.1 (a smaller model) matched GPT-4.1 performance through better context engineering alone.

## Design Decisions and Tradeoffs

| Decision | Rationale |
|----------|-----------|
| **Separate triage/plan/code/answer nodes** | Small models work better with focused, single-purpose prompts |
| **Tool call budgets** | Prevents small models from getting stuck in infinite tool loops |
| **Rolling summaries** | More reliable than asking small models to handle long contexts |
| **File content in system prompt** | Small models reference files better when they're in the system context |
| **Structured output for routing** | Eliminates parsing errors that plague small models with free-text routing |
| **Explicit "TASK COMPLETE" markers** | Clear signal small models can reliably produce to indicate completion |

## Troubleshooting

### "Connection refused" error
```bash
# Make sure Ollama is running
ollama serve

# Check it's accessible
curl http://localhost:11434/v1/models
```

### Model not producing tool calls
Some models have weak tool calling support. Try:
1. Use `qwen2.5-coder:14b` — it has the best tool calling among small models
2. Lower temperature: `codeassist --temperature 0.05`
3. If using LM Studio, ensure "Tool Use" is enabled in model settings

### Out of memory
- Use a smaller model: `codeassist --model qwen2.5-coder:7b`
- Reduce Ollama's context: `OLLAMA_NUM_CTX=4096 ollama serve`

### Model hallucinating file contents
This is a known issue with small models. The assistant's design mitigates this by:
- Always reading files before editing (enforced in prompts)
- Using tool call budgets to prevent runaway behavior
- Including explicit "NEVER guess file contents" in prompts

## Inspiration and References

This project draws from several sources:

### Projects
- **[LangGraph](https://langchain-ai.github.io/langgraph/)** — StateGraph patterns, conditional edges, checkpointing
- **[Open SWE](https://github.com/langchain-ai/open-swe)** — LangGraph-based Manager → Planner → Programmer pipeline architecture
- **[Deep Agents](https://github.com/langchain-ai/deepagents)** — TodoListMiddleware, SummarizationMiddleware, subagent isolation pattern
- **[LangGraph CodeAct](https://github.com/langchain-ai/langgraph-codeact)** — CodeAct pattern (LLM writes executable code instead of JSON tool calls)
- **[Aider](https://github.com/Aider-AI/aider)** — Gold standard terminal UI, repository map, auto-commits, multi-mode chat
- **[Goose](https://github.com/block/goose)** — Block's 30k+ star agent, MCP-first extensibility, multi-model configuration
- **[Open Interpreter](https://github.com/OpenInterpreter/open-interpreter)** — Natural language CLI, streaming output patterns
- **[ollama-code](https://github.com/tcsenpai/ollama-code)** — Privacy-first CLI coding agent built specifically for Ollama
- **[Rich](https://github.com/Textualize/rich)** — Terminal formatting library used for the UI

### Research Papers
- **[ACE Framework](https://arxiv.org/abs/2510.04618)** — Showed DeepSeek-V3.1 matched GPT-4.1 through context engineering alone (59.4% vs 60.3% on AppWorld)
- **[Confucius Code Agent](https://arxiv.org/pdf/2512.10398)** — 59% on SWE-Bench-Pro: "principled scaffolding can substantially amplify the same underlying LLM"
- **[OpenDev](https://arxiv.org/html/2603.05344v1)** — Four-level hierarchy for compound AI coding systems, scaffolding/harness phase separation
- **[JetBrains Observation Masking](https://blog.jetbrains.com/research/2025/12/efficient-context-management/)** — Truncating tool output saves ~60% context while matching LLM summarization quality
- **[Anthropic Context Engineering Guide](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)** — Write/Select/Compress/Isolate context framework

## License

MIT — use it however you want.
