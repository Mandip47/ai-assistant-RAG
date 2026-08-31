import os

# --- LLM providers (primary + optional fallback for reliability) ---
def _build_llm_providers() -> list[dict]:
    providers = [{
        "base_url": os.getenv("PRIMARY_LLM_BASE_URL", "http://host.docker.internal:8000/v1"),
        "model": os.getenv("PRIMARY_LLM_MODEL_NAME", "mlx-community/Qwen2.5-3B-Instruct-4bit"),
    }]
    fallback_url = os.getenv("FALLBACK_LLM_BASE_URL", "")
    fallback_model = os.getenv("FALLBACK_LLM_MODEL_NAME", "")
    if fallback_url and fallback_model:
        providers.append({"base_url": fallback_url, "model": fallback_model})
    return providers

LLM_PROVIDERS = _build_llm_providers()

LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.4"))
LLM_TOP_P = float(os.getenv("LLM_TOP_P", "0.9"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))
# Agent turns (tool_call / final_answer JSON) are short by design.
AGENT_MAX_TOKENS = int(os.getenv("AGENT_MAX_TOKENS", "300"))
LLM_FREQUENCY_PENALTY = float(os.getenv("LLM_FREQUENCY_PENALTY", "0.4"))
LLM_PRESENCE_PENALTY = float(os.getenv("LLM_PRESENCE_PENALTY", "0.4"))

# --- Reliability ---
RETRY_MAX_ATTEMPTS = int(os.getenv("RETRY_MAX_ATTEMPTS", "3"))
RETRY_WAIT_MIN_SECONDS = float(os.getenv("RETRY_WAIT_MIN_SECONDS", "1"))
RETRY_WAIT_MAX_SECONDS = float(os.getenv("RETRY_WAIT_MAX_SECONDS", "8"))
RATE_LIMIT = os.getenv("RATE_LIMIT", "20/minute")

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
# How many past user+assistant turn pairs to keep per session and feed back
# into the prompt as conversation history.
SESSION_MAX_TURNS = int(os.getenv("SESSION_MAX_TURNS", "6"))
# Mirrors in-memory session history to disk so it survives an app container
# restart. Lives on the same volume as CHROMA_PERSIST_DIR by default. Set to
# "" to disable persistence and keep history purely in-memory.
SESSION_STORE_PATH = os.getenv("SESSION_STORE_PATH", "/data/sessions.json")

SYSTEM_PROMPT = """You are a precise, helpful AI assistant with access to tools and a document knowledge base.

Rules:
- If the user's question likely needs information from the knowledge base, call the `retrieve_documents` tool before answering.
- Use tools only when they genuinely help; do not call a tool if you already have enough information.
- After you have enough information, respond with a final answer to the user. Never leave a tool call unanswered.
- Be concise and factual. If the retrieved context does not contain the answer, say so honestly instead of guessing.
- Output exactly ONE JSON object and then STOP. Do not repeat yourself, do not output more than one JSON object.
"""

# --- Image classification (ONNX) ---
ONNX_MODEL_PATH = os.getenv("ONNX_MODEL_PATH", "/models/resnet50_cifar10.onnx")
ONNX_INTRA_OP_THREADS = int(os.getenv("ONNX_INTRA_OP_THREADS", "4"))
# Your model expects 224x224 (standard ResNet50/ImageNet input size) — CIFAR-10
# images are upsampled to this size, a common pattern when fine-tuning a
# pretrained ResNet50 on CIFAR-10 rather than training from scratch at 32x32.
CIFAR10_INPUT_SIZE = int(os.getenv("CIFAR10_INPUT_SIZE", "224"))
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
# ImageNet normalization stats — the right default if your model started from
# ImageNet-pretrained ResNet50 weights. Switch to CIFAR-10-native stats
# ([0.4914,0.4822,0.4465] / [0.2470,0.2435,0.2616]) if you trained from
# scratch instead, or if predictions look off.
CIFAR10_MEAN = [0.485, 0.456, 0.406]
CIFAR10_STD = [0.229, 0.224, 0.225]