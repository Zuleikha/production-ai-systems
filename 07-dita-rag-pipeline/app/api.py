"""
api.py
FastAPI application for the DITA RAG pipeline.
Endpoints: index, search, ask
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path

from parser import parse_all_dita_files
from indexer import index_chunks
from retriever import search, generate_answer

DATA_DIR = os.getenv("DATA_DIR", "./data")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_store")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Auto-index on startup if data directory exists
    if Path(DATA_DIR).exists():
        print("Auto-indexing DITA files on startup...")
        chunks = parse_all_dita_files(DATA_DIR)
        if chunks:
            index_chunks(chunks, persist_dir=CHROMA_DIR)
    yield


app = FastAPI(
    title="DITA RAG Pipeline",
    description="Semantic search and Q&A over DITA XML documentation",
    version="1.0.0",
    lifespan=lifespan
)


class QueryRequest(BaseModel):
    query: str
    n_results: int = 4


class SearchResult(BaseModel):
    chunk_id: str
    text: str
    metadata: dict
    score: float


class AskResponse(BaseModel):
    query: str
    answer: str
    sources: list


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/index")
def reindex():
    """Re-parse and re-index all DITA files."""
    if not Path(DATA_DIR).exists():
        raise HTTPException(status_code=404, detail=f"Data directory not found: {DATA_DIR}")
    chunks = parse_all_dita_files(DATA_DIR)
    if not chunks:
        raise HTTPException(status_code=404, detail="No DITA files found in data directory")
    count = index_chunks(chunks, persist_dir=CHROMA_DIR)
    return {"indexed": count, "status": "ok"}


@app.post("/search", response_model=list[SearchResult])
def semantic_search(request: QueryRequest):
    """Return top matching chunks for a query."""
    try:
        results = search(request.query, n_results=request.n_results, persist_dir=CHROMA_DIR)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask", response_model=AskResponse)
def ask(request: QueryRequest):
    """Retrieve relevant chunks and generate an answer."""
    try:
        chunks = search(request.query, n_results=request.n_results, persist_dir=CHROMA_DIR)
        answer = generate_answer(request.query, chunks)
        sources = [
            {
                "topic": c["metadata"]["topic_title"],
                "section": c["metadata"]["section_title"],
                "score": c["score"],
                "tags": c["metadata"]["tags"],
            }
            for c in chunks
        ]
        return AskResponse(query=request.query, answer=answer, sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
