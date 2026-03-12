from schemas import (
    Decision
)
from langchain.chat_models import init_chat_model
from tools import (
    read_write_tools, read_write_tools_by_name,
    read_tools, read_tools_by_name,
)

# Define local model
base_model = init_chat_model(
    model="gpt-5", # this is ignored im using zai-org/glm-4.7-flash locally but if i put this name LangGraph throws error
    base_url="http://localhost:1234/v1", # Local model zai-org/glm-4.7-flash
)

decision_model = base_model.with_structured_output(Decision)

# Augment the LLM with tools
read_write_model = base_model.bind_tools(read_write_tools)
read_only_model = base_model.bind_tools(read_tools)