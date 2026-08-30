from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.agent import run_agent
from app.schemas import ChatRequest, ChatResponse, IngestResponse
from app import rag, config

app = FastAPI(title="Local AI Assistant", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(400, "message cannot be empty")
    return await run_agent(req.message)


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    if not file.filename.endswith((".txt", ".md")):
        raise HTTPException(400, "Only .txt and .md files are supported in this scope.")
    raw = await file.read()
    text = raw.decode("utf-8", errors="ignore")
    chunk_count = rag.ingest_text(text, source_name=file.filename)
    return IngestResponse(
        filename=file.filename,
        chunks_created=chunk_count,
        collection=config.CHROMA_COLLECTION_NAME,
    )
