"""
Phase 1 — FastAPI entrypoint.
Two endpoints: POST /ingest (upload doc) + POST /query (ask question).
"""

import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile

load_dotenv()

from app.generation import generate
from app.ingestion import ingest_file
from app.models import IngestResponse, QueryRequest, QueryResponse, SourceChunk
from app.retrieval import retrieve

app = FastAPI(
    title="Doc Q&A — Phase 1",
    description="RAG from scratch: no frameworks, pure Python + OpenAI",
    version="1.0.0",
)

UPLOAD_DIR = Path("data/docs")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/")
def root():
    return {"status": "ok", "phase": 1, "endpoints": ["/ingest", "/query"]}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    """
    Upload a .txt or .pdf file.
    Pipeline: save → chunk → embed → store.
    """
    allowed = {".txt", ".pdf"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(400, f"Only {allowed} supported")

    dest = UPLOAD_DIR / file.filename
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    chunks_count = ingest_file(str(dest))
    return IngestResponse(
        filename=file.filename,
        chunks_created=chunks_count,
        message=f"Ingested {chunks_count} chunks from {file.filename}",
    )


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """
    Ask a question.
    Pipeline: embed question → cosine search → prompt → GPT → answer.
    """
    top_k = req.top_k or int(os.getenv("TOP_K", 3))
    chunks = retrieve(req.question, top_k=top_k)

    if not chunks:
        raise HTTPException(404, "No documents ingested yet. POST to /ingest first.")

    result = generate(req.question, chunks)

    return QueryResponse(
        answer=result["answer"],
        sources=[SourceChunk(**c) for c in chunks],
        prompt_tokens=result["prompt_tokens"],
        completion_tokens=result["completion_tokens"],
    )
