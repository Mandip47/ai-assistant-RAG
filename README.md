# Local AI Assistant (RAG + Tool Calling)

A self-contained AI assistant running entirely on local infrastructure: **Qwen 2.5 3B via vLLM**, **Chroma** as an embedded vector store (no Redis, no external DB server), and a **FastAPI** backend with JSON-schema-constrained structured output and tool calling.

See `architecture.png` for the architecture diagram.

## Stack

| Requirement        | Implementation                                                                                   |
| ------------------ | ------------------------------------------------------------------------------------------------ |
| LLM Integration    | vLLM's OpenAI-compatible `/v1/chat/completions` endpoint serving Gemma 3B                        |
| Prompt Engineering | System prompt in `app/config.py`; `temperature`/`top_p` tunable via env vars                     |
| Structured Output  | Guided JSON decoding (vLLM's `guided_json` field) — every model turn is valid JSON               |
| Tool Calling       | JSON-driven tool-call loop (`app/agent.py`, `app/tools.py`) — `retrieve_documents`, `calculator` |
| RAG Pipeline       | Chunking (`app/rag.py`) → `sentence-transformers` embeddings → Chroma (persisted on disk)        |
| Local Deployment   | vLLM serving Gemma 3B locally                                                                    |
| Containerization   | `Dockerfile` + `docker-compose.yml` for the app, no Redis                                        |

## Prerequisites

- Docker + Docker Compose
- `vllm_metal` already installed and running on your **host** machine (this is the Apple Silicon/MLX build of vLLM), serving your model on port 8000:

  ```bash
  python -m vllm_metal.server --model mlx-community/Qwen2.5-3B-Instruct-4bit --port 8000
  ```

  Set `LLM_MODEL_NAME` in `docker-compose.yml` to the exact model id you passed with `--model` — it must match verbatim, or requests are rejected.

  > Note: `vllm_metal` is a lighter community server than mainline vLLM and may not honor the `guided_json` constrained-decoding field the same way full vLLM does. The code degrades gracefully either way — `parse_agent_turn()` in `app/llm_client.py` falls back to extracting JSON from free text if the model doesn't return a strictly schema-conformant response — but answers may be a little less consistently structured than with full vLLM or Ollama.

## Setup

```bash
docker compose up --build
```

The app container reaches your host's vLLM server at `http://host.docker.internal:8000/v1` — no vLLM container needed. Since vLLM already owns port 8000 on your host, the assistant API is exposed on **`localhost:8080`** instead (mapped to container port 8000).

> On Linux, `host.docker.internal` requires the `extra_hosts` entry already in `docker-compose.yml` (Docker 20.10+). If it still can't connect, set `LLM_BASE_URL` to your host's LAN IP instead, e.g. `http://172.17.0.1:8000/v1`.

## Usage

**Ingest a document into the knowledge base:**

```bash
curl -X POST http://localhost:8080/ingest \
  -F "file=@./data/some_notes.txt"
```

**Chat with the assistant (RAG + tool calling happens automatically):**

```bash
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What does the document say about X?"}'
```

Example response (structured, validated JSON):

```json
{
  "answer": "According to the ingested document, ...",
  "used_tools": ["retrieve_documents"],
  "sources": [
    { "chunk_id": "some_notes.txt-a1b2c3-0", "text": "...", "score": 0.83 }
  ],
  "confidence": "medium"
}
```

**Health check:**

```bash
curl http://localhost:8080/health
```

## How it works

1. `/chat` hands the user message to `run_agent()` in `app/agent.py`.
2. The model is called with `guided_json` constrained to a strict JSON schema (`action`: `tool_call` or `final_answer`), so we never have to regex-parse free text.
3. If the model requests `retrieve_documents`, we embed the query with `sentence-transformers`, query Chroma for the top-k nearest chunks (cosine similarity), and feed the results back to the model as a tool result message.
4. This loops (bounded by `MAX_TOOL_ITERATIONS`) until the model returns `final_answer`, which is validated against the `ChatResponse` Pydantic schema before being returned to the client.
5. `/ingest` chunks uploaded `.txt`/`.md` files (sliding window with sentence-boundary snapping), embeds each chunk, and upserts into the local Chroma collection persisted at `/data/chroma`.

## Configuration

All tunable via environment variables in `docker-compose.yml` (see `app/config.py` for defaults): `LLM_TEMPERATURE`, `LLM_TOP_P`, `LLM_MAX_TOKENS`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `RETRIEVAL_TOP_K`, `MAX_TOOL_ITERATIONS`.
