"""
RAG pipeline. No Redis, no external vector DB server:
Chroma runs embedded (persisted to local disk inside the container/volume).
"""
import uuid
import chromadb
from sentence_transformers import SentenceTransformer

from app import config

_embedder: SentenceTransformer | None = None
_chroma_client: chromadb.ClientAPI | None = None
_collection = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    return _embedder


def _get_collection():
    global _chroma_client, _collection
    if _collection is None:
        _chroma_client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
        _collection = _chroma_client.get_or_create_collection(
            name=config.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> list[str]:
    """Simple sliding-window character chunker with sentence-ish boundaries."""
    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        # try to break on a sentence/paragraph boundary near the end
        if end < n:
            boundary = text.rfind(". ", start, end)
            if boundary != -1 and boundary > start + chunk_size * 0.5:
                end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap if end - overlap > start else end
    return chunks


def ingest_text(text: str, source_name: str) -> int:
    """Chunk, embed, and upsert text into the vector store. Returns chunk count."""
    chunks = chunk_text(text)
    if not chunks:
        return 0

    embedder = _get_embedder()
    embeddings = embedder.encode(chunks, normalize_embeddings=True).tolist()
    ids = [f"{source_name}-{uuid.uuid4().hex[:8]}-{i}" for i in range(len(chunks))]
    metadatas = [{"source": source_name, "chunk_index": i} for i in range(len(chunks))]

    collection = _get_collection()
    collection.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
    return len(chunks)


def retrieve(query: str, top_k: int = None) -> list[dict]:
    """Embed the query and return the top-k most similar chunks with scores."""
    top_k = top_k or config.RETRIEVAL_TOP_K
    collection = _get_collection()
    if collection.count() == 0:
        return []

    embedder = _get_embedder()
    query_embedding = embedder.encode([query], normalize_embeddings=True).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, collection.count()),
    )

    out = []
    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    dists = results.get("distances", [[]])[0]
    for _id, doc, dist in zip(ids, docs, dists):
        similarity = 1 - dist  # cosine distance -> similarity
        out.append({"chunk_id": _id, "text": doc, "score": round(float(similarity), 4)})
    return out
