import asyncio
import logging

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler

from app.agent import run_agent
from app.schemas import ChatRequest, ChatResponse, IngestResponse, ClassifyResponse
from app import rag, vision, config, sessions
from app.resilience import limiter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Local AI Assistant", version="2.0.0")

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def graceful_degradation_handler(request: Request, exc: Exception):
    """
    Catch-all so unexpected failures return a structured JSON error instead
    of a bare 500 with a stack trace leaking to the client.
    """
    logger.error("Unhandled error on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=503,
        content={
            "error": "The service hit an unexpected problem and could not complete this request.",
            "detail": str(exc),
        },
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
@limiter.limit(config.RATE_LIMIT)
async def chat(request: Request, req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(400, "message cannot be empty")
    return await run_agent(req.message, session_id=req.session_id)


@app.delete("/chat/{session_id}")
@limiter.limit(config.RATE_LIMIT)
async def clear_chat(request: Request, session_id: str):
    sessions.clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}


@app.post("/ingest", response_model=IngestResponse)
@limiter.limit(config.RATE_LIMIT)
async def ingest(request: Request, file: UploadFile = File(...)):
    if not file.filename.endswith((".txt", ".md")):
        raise HTTPException(400, "Only .txt and .md files are supported in this scope.")
    raw = await file.read()
    text = raw.decode("utf-8", errors="ignore")
    # Embedding is CPU-bound and synchronous — offload so it doesn't block
    # the event loop under concurrent load.
    chunk_count = await asyncio.to_thread(rag.ingest_text, text, file.filename)
    return IngestResponse(
        filename=file.filename,
        chunks_created=chunk_count,
        collection=config.CHROMA_COLLECTION_NAME,
    )


@app.post("/classify", response_model=ClassifyResponse)
@limiter.limit(config.RATE_LIMIT)
async def classify(request: Request, file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Please upload an image file.")
    image_bytes = await file.read()
    # ONNX inference is CPU-bound and synchronous — offload to a thread.
    result = await asyncio.to_thread(vision.classify, image_bytes)
    return ClassifyResponse(**result)