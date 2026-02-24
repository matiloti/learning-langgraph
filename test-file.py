from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from typing import Literal

model = init_chat_model(
    "gpt-5",
    base_url="http://localhost:1234/v1"
)

class Decision(BaseModel):
    """Triage decision whether to seek clarifications or to code."""
    decision: Literal["clarify","code"] = Field(..., description="The decision")

model_with_structure = model.with_structured_output(Decision)
response = model_with_structure.invoke("Decide whether to seek clarification or to code based on this clue: i asking questions and i know nothing about code")
print(response.decision)