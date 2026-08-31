from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., description="User's message")
    session_id: str = Field(default="default", description="Conversation/session identifier")


class Source(BaseModel):
    chunk_id: str
    text: str
    score: float


class ChatResponse(BaseModel):
    """Structured, validated JSON contract returned to the client."""
    answer: str
    used_tools: List[str] = Field(default_factory=list)
    sources: List[Source] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"] = "medium"


class IngestResponse(BaseModel):
    filename: str
    chunks_created: int
    collection: str

class ClassifyResponse(BaseModel):
    predicted_class: str
    confidence: float
    top_k: List[dict] = Field(default_factory=list)
    error: Optional[str] = None


# --- JSON schema handed to llama.cpp for constrained decoding of the ---
# --- model's own turn (tool_call OR final_answer, never garbage text) ---
AGENT_TURN_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["tool_call", "final_answer"]},
        "tool_name": {"type": "string"},
        "tool_args": {"type": "object"},
        "final_answer": {"type": "string"},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    "required": ["action"],
}
