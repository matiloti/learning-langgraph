"""
Model configuration — designed for local models via Ollama, LM Studio, or any
OpenAI-compatible endpoint.

Context engineering note: we create specialized model wrappers that
constrain the model's behavior through tool binding and structured output,
reducing the cognitive load on small models.
"""

import os
from langchain_openai import ChatOpenAI
from state import TriageDecision, PlanOutput
from tools import read_tools, read_write_tools

# ---------------------------------------------------------------------------
# Configuration — override with environment variables
# ---------------------------------------------------------------------------

MODEL_NAME = os.environ.get("CODING_ASSISTANT_MODEL", "qwen2.5-coder:14b")
BASE_URL = os.environ.get("CODING_ASSISTANT_BASE_URL", "http://localhost:11434/v1")
API_KEY = os.environ.get("CODING_ASSISTANT_API_KEY", "not-needed")
TEMPERATURE = float(os.environ.get("CODING_ASSISTANT_TEMPERATURE", "0.1"))

# ---------------------------------------------------------------------------
# Base model — low temperature for deterministic coding behavior
# ---------------------------------------------------------------------------

base_model = ChatOpenAI(
    model=MODEL_NAME,
    base_url=BASE_URL,
    api_key=API_KEY,
    temperature=TEMPERATURE,
    max_tokens=4096,
)

# ---------------------------------------------------------------------------
# Specialized model variants
# ---------------------------------------------------------------------------

# For routing decisions — structured output keeps small models on track
triage_model = base_model.with_structured_output(TriageDecision)

# For task planning — structured output ensures valid task lists
plan_model = base_model.with_structured_output(PlanOutput)

# For coding — has read+write tools available
code_model = base_model.bind_tools(read_write_tools)

# For answering questions — read-only tools
answer_model = base_model.bind_tools(read_tools)
