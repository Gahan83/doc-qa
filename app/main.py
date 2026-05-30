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


import logging
import sys
from fastapi.responses import JSONResponse
from fastapi.requests import Request
from fastapi.concurrency import run_in_threadpool
import asyncio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger("doc-qa")

app = FastAPI(
    title="Doc Q&A — Phase 1 + 2",
    description="RAG from scratch + Agents, Function Calling, Structured Outputs, Evaluation",
    version="2.0.0",
)

# Middleware must be defined after app is created
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Request: {request.method} {request.url}")
    try:
        response = await call_next(request)
        logger.info(f"Response: {request.method} {request.url} {response.status_code}")
        return response
    except Exception as e:
        logger.exception(f"Error handling request: {request.method} {request.url}")
        raise

UPLOAD_DIR = Path("data/docs")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)



@app.get("/")
async def root():
    return {
        "status": "ok",
        "phase": 2,
        "endpoints": ["/ingest", "/query", "/agent", "/query/structured", "/evaluate", "/healthz"],
    }

@app.get("/healthz")
async def healthz():
    return {"status": "healthy"}


# --- Phase 1 Endpoints ---


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    """Upload a .txt or .pdf file. Pipeline: save → chunk → embed → store."""
    allowed = {".txt", ".pdf"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(400, f"Only {allowed} supported")
    dest = UPLOAD_DIR / file.filename
    try:
        # Async file write
        async with asyncio.to_thread(open, dest, "wb") as f:
            await asyncio.to_thread(shutil.copyfileobj, file.file, f)
        # Ingestion is CPU-bound, run in threadpool
        chunks_count = await run_in_threadpool(ingest_file, str(dest))
        return IngestResponse(
            filename=file.filename,
            chunks_created=chunks_count,
            message=f"Ingested {chunks_count} chunks from {file.filename}",
        )
    except Exception as e:
        logging.exception("Ingestion failed")
        raise HTTPException(500, f"Ingestion failed: {e}")



@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """Single-shot RAG: embed question → cosine search → prompt → GPT → answer."""
    try:
        top_k = req.top_k or int(os.getenv("TOP_K", 3))
        chunks = await run_in_threadpool(retrieve, req.question, top_k)
        if not chunks:
            raise HTTPException(404, "No documents ingested yet. POST to /ingest first.")
        result = await run_in_threadpool(generate, req.question, chunks)
        return QueryResponse(
            answer=result["answer"],
            sources=[SourceChunk(**c) for c in chunks],
            prompt_tokens=result["prompt_tokens"],
            completion_tokens=result["completion_tokens"],
        )
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Query failed")
        raise HTTPException(500, f"Query failed: {e}")


# --- Phase 2 Endpoints ---


@app.post("/agent", response_model=AgentResponse)
async def agent_query(req: AgentRequest):
    """
    Agent with function calling.
    GPT decides which tools to call, loops until it has an answer.
    """
    try:
        result = await run_in_threadpool(run_agent, req.question)
        return AgentResponse(
            answer=result["answer"],
            steps=[AgentStep(**s) for s in result["steps"]],
            iterations=result["iterations"],
            prompt_tokens=result["prompt_tokens"],
            completion_tokens=result["completion_tokens"],
        )
    except Exception as e:
        logging.exception("Agent failed")
        raise HTTPException(500, f"Agent failed: {e}")



@app.post("/query/structured", response_model=StructuredAnswer)
async def structured_query(req: StructuredQueryRequest):
    """
    Query with structured output.
    Forces GPT to return a strict JSON schema with confidence, sources, follow-ups.
    """
    try:
        top_k = req.top_k or int(os.getenv("TOP_K", 3))
        chunks = await run_in_threadpool(retrieve, req.question, top_k)
        if not chunks:
            raise HTTPException(404, "No documents ingested yet. POST to /ingest first.")
        result = await run_in_threadpool(generate_structured, req.question, chunks)
        return StructuredAnswer(**result)
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Structured query failed")
        raise HTTPException(500, f"Structured query failed: {e}")



@app.post("/evaluate", response_model=EvalResponse)
async def evaluate_answer(req: EvalRequest):
    """
    Evaluate an answer using LLM-as-judge.
    Scores: faithfulness, relevance, completeness.
    """
    try:
        result = await run_in_threadpool(evaluate, req.question, req.answer, req.context_chunks)
        return EvalResponse(**result)
    except Exception as e:
        logging.exception("Evaluation failed")
        raise HTTPException(500, f"Evaluation failed: {e}")
