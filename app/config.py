import os

# --- LLM (vLLM, OpenAI-compatible server) ---
# host.docker.internal lets the app container reach vLLM running on your host machine.
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://host.docker.internal:8000/v1")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "mlx-community/Qwen2.5-3B-Instruct-4bit")  # must match how you launched vllm_metal.server
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.4"))
LLM_TOP_P = float(os.getenv("LLM_TOP_P", "0.9"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))
# Agent turns (tool_call / final_answer JSON) are short by design — capping
# this tightly limits the damage if the model loops instead of stopping cleanly.
AGENT_MAX_TOKENS = int(os.getenv("AGENT_MAX_TOKENS", "300"))
# Discourages repetition loops in smaller/quantized models that don't stop
# cleanly on their own. Standard OpenAI-API fields, widely supported.
LLM_FREQUENCY_PENALTY = float(os.getenv("LLM_FREQUENCY_PENALTY", "0.4"))
LLM_PRESENCE_PENALTY = float(os.getenv("LLM_PRESENCE_PENALTY", "0.4"))

# --- RAG ---
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
# "auto" picks mps (Apple Silicon GPU) when available, else cpu. Only relevant
# when running the app natively on the host — Docker Desktop on Mac has no
# Metal passthrough, so this always falls back to cpu inside a container.
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "auto")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "/data/chroma")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "documents")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))       # characters
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))  # characters
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "4"))

# --- Agent loop ---
MAX_TOOL_ITERATIONS = int(os.getenv("MAX_TOOL_ITERATIONS", "4"))

SYSTEM_PROMPT = """You are a precise, helpful AI assistant with access to tools and a document knowledge base.

Rules:
- If the user's question likely needs information from the knowledge base, call the `retrieve_documents` tool before answering.
- Use tools only when they genuinely help; do not call a tool if you already have enough information.
- After you have enough information, respond with a final answer to the user. Never leave a tool call unanswered.
- Be concise and factual. If the retrieved context does not contain the answer, say so honestly instead of guessing.
- Output exactly ONE JSON object and then STOP. Do not repeat yourself, do not output more than one JSON object.
"""