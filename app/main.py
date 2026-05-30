"""
Phase 1 + 2 — FastAPI entrypoint.
Endpoints: /ingest, /query, /agent, /query/structured, /evaluate
"""

import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile

load_dotenv()

from app.agent import run_agent
from app.evaluation import evaluate
from app.generation import generate
from app.ingestion import ingest_file
from app.models import (
    AgentRequest,
    AgentResponse,
    AgentStep,
    EvalRequest,
    EvalResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SourceChunk,
    StructuredAnswer,
    StructuredQueryRequest,
)
from app.retrieval import retrieve
from app.structured import generate_structured

app = FastAPI(
    title="Doc Q&A — Phase 1 + 2",
    description="RAG from scratch + Agents, Function Calling, Structured Outputs, Evaluation",
    version="2.0.0",
)

UPLOAD_DIR = Path("data/docs")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/")
def root():
    return {
        "status": "ok",
        "phase": 2,
        "endpoints": ["/ingest", "/query", "/agent", "/query/structured", "/evaluate"],
    }


# --- Phase 1 Endpoints ---

@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    """Upload a .txt or .pdf file. Pipeline: save → chunk → embed → store."""
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
    """Single-shot RAG: embed question → cosine search → prompt → GPT → answer."""
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


# --- Phase 2 Endpoints ---

@app.post("/agent", response_model=AgentResponse)
def agent_query(req: AgentRequest):
    """
    Agent with function calling.
    GPT decides which tools to call, loops until it has an answer.
    """
    result = run_agent(req.question)

    return AgentResponse(
        answer=result["answer"],
        steps=[AgentStep(**s) for s in result["steps"]],
        iterations=result["iterations"],
        prompt_tokens=result["prompt_tokens"],
        completion_tokens=result["completion_tokens"],
    )


@app.post("/query/structured", response_model=StructuredAnswer)
def structured_query(req: StructuredQueryRequest):
    """
    Query with structured output.
    Forces GPT to return a strict JSON schema with confidence, sources, follow-ups.
    """
    top_k = req.top_k or int(os.getenv("TOP_K", 3))
    chunks = retrieve(req.question, top_k=top_k)

    if not chunks:
        raise HTTPException(404, "No documents ingested yet. POST to /ingest first.")

    result = generate_structured(req.question, chunks)
    return StructuredAnswer(**result)


@app.post("/evaluate", response_model=EvalResponse)
def evaluate_answer(req: EvalRequest):
    """
    Evaluate an answer using LLM-as-judge.
    Scores: faithfulness, relevance, completeness.
    """
    result = evaluate(req.question, req.answer, req.context_chunks)
    return EvalResponse(**result)
