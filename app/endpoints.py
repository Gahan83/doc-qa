"""
HTTP endpoints (APIRouter).

Mounted by app/main.py. Keeps main.py limited to app creation,
middleware, and logging configuration.
"""

import logging
import os
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response

from app.agent import run_agent
from app.agent_multi import run_multi_agent
from app.agent_scratch import run_agent_scratch
from app.evaluation import batch_evaluate, evaluate
from app.observability import notify_query
from app.generation import generate
from app.ingestion import AUDIO_EXTS, IMAGE_EXTS, VIDEO_EXTS, ingest_file
from app.models import (
    AgentCompareResponse,
    AgentRequest,
    AgentResponse,
    AgentStep,
    BatchEvalRequest,
    BatchEvalResponse,
    EvalRequest,
    MultiAgentResponse,
    EvalResponse,
    ExplainChunkRequest,
    ExplainChunkResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SourceChunk,
    StructuredAnswer,
    StructuredQueryRequest,
    TranscribeResponse,
    TranscriptSegment,
    VoiceSpeakRequest,
)
from app.retrieval import retrieve
from app.structured import generate_structured
from app.tokens import (
    DEFAULT_MODEL,
    _price_for,
    build_usage,
    count_tokens,
    estimate_cost,
    log_usage,
)

logger = logging.getLogger("doc-qa")

router = APIRouter()

ALLOWED_EXTS = {".txt", ".pdf"} | AUDIO_EXTS | VIDEO_EXTS | IMAGE_EXTS

UPLOAD_DIRS = {
    "doc":   Path("data/docs"),
    "audio": Path("data/audio"),
    "video": Path("data/video"),
    "image": Path("data/images"),
}
for d in UPLOAD_DIRS.values():
    d.mkdir(parents=True, exist_ok=True)


def _upload_dir(suffix: str) -> Path:
    if suffix in AUDIO_EXTS:
        return UPLOAD_DIRS["audio"]
    if suffix in VIDEO_EXTS:
        return UPLOAD_DIRS["video"]
    if suffix in IMAGE_EXTS:
        return UPLOAD_DIRS["image"]
    return UPLOAD_DIRS["doc"]


def _make_source_chunks(chunks: list[dict]) -> list[SourceChunk]:
    """Convert retrieval dicts → SourceChunk, forwarding only known fields."""
    fields = set(SourceChunk.model_fields)
    return [SourceChunk(**{k: v for k, v in c.items() if k in fields}) for c in chunks]


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@router.get("/")
async def root():
    return {
        "status": "ok",
        "version": "9.0.0",
        "endpoints": [
            "/ingest", "/ingest/visual",
            "/query", "/query/visual", "/query/structured",
            "/agent", "/agent/scratch", "/agent/compare", "/agent/multi",
            "/evaluate", "/evaluate/batch",
            "/voice/transcribe", "/voice/speak",
            "/explain-chunk",
            "/healthz",
        ],
    }


@router.get("/healthz")
async def healthz():
    return {"status": "healthy"}


# ---------------------------------------------------------------------------
# LLM internals: explain a chunk (tiktoken token count + cost, no API call)
# ---------------------------------------------------------------------------

@router.post("/explain-chunk", response_model=ExplainChunkResponse)
async def explain_chunk(req: ExplainChunkRequest):
    """
    Inspect a doc chunk before ingesting/querying.
    Local tiktoken count + estimated input cost. No GPT call, no spend.
    """
    model = req.model or DEFAULT_MODEL
    tokens = count_tokens(req.text, model)
    price_in, price_out = _price_for(model)
    return ExplainChunkResponse(
        model=model,
        char_count=len(req.text),
        token_count=tokens,
        estimated_input_cost_usd=estimate_cost(tokens, 0, model),
        price_input_per_1m=price_in,
        price_output_per_1m=price_out,
    )


# ---------------------------------------------------------------------------
# Ingest any file type
# ---------------------------------------------------------------------------

@router.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    """
    Upload PDF, TXT, audio (.mp3 .wav .m4a .ogg .flac .aac .opus),
    video (.mp4 .mov .avi .mkv .webm), or image (.png .jpg .jpeg .gif .webp .bmp).
    Audio → Whisper transcript chunks.
    Video → Whisper transcript + GPT-4o frame description chunks (Phase 6+7).
    Image → GPT-4o vision description chunk (multimodal).
    """
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTS:
        raise HTTPException(400, f"Unsupported file type '{suffix}'. Allowed: {sorted(ALLOWED_EXTS)}")

    dest = _upload_dir(suffix) / file.filename
    try:
        def _save():
            with open(dest, "wb") as f_out:
                shutil.copyfileobj(file.file, f_out)

        await run_in_threadpool(_save)
        n = await run_in_threadpool(ingest_file, str(dest))
        return IngestResponse(
            filename=file.filename,
            chunks_created=n,
            message=f"Ingested {n} chunks from '{file.filename}'",
        )
    except Exception as e:
        logger.exception("Ingestion failed")
        raise HTTPException(500, f"Ingestion failed: {e}")


# ---------------------------------------------------------------------------
# CLIP visual ingest
# ---------------------------------------------------------------------------

@router.post("/ingest/visual", response_model=IngestResponse)
async def ingest_visual(file: UploadFile = File(...)):
    """
    Ingest video frames using CLIP embeddings.
    Frames land in a separate 'visual_chunks' Chroma collection.
    Query via POST /query/visual.
    Requires: pip install sentence-transformers Pillow
    """
    suffix = Path(file.filename).suffix.lower()
    if suffix not in VIDEO_EXTS:
        raise HTTPException(400, f"Visual ingest requires video. Got '{suffix}'")

    dest = UPLOAD_DIRS["video"] / file.filename
    try:
        def _save():
            with open(dest, "wb") as f_out:
                shutil.copyfileobj(file.file, f_out)

        await run_in_threadpool(_save)

        from app.clip_store import ingest_frames_clip
        n = await run_in_threadpool(ingest_frames_clip, str(dest))
        return IngestResponse(
            filename=file.filename,
            chunks_created=n,
            message=f"CLIP-embedded {n} frames from '{file.filename}'",
        )
    except Exception as e:
        logger.exception("Visual ingest failed")
        raise HTTPException(500, f"Visual ingest failed: {e}")


# ---------------------------------------------------------------------------
# Text/audio/video RAG query (with timestamp citations)
# ---------------------------------------------------------------------------

@router.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest, background: BackgroundTasks):
    """RAG over all ingested content. Audio/video answers cite timestamps.

    After responding, a background task logs the query cost (storage/query_costs.csv)
    and, if SLACK_WEBHOOK_URL is set, posts the answer + eval score to Slack.
    """
    try:
        top_k  = req.top_k or int(os.getenv("TOP_K", 3))
        chunks = await run_in_threadpool(retrieve, req.question, top_k)
        if not chunks:
            raise HTTPException(404, "No documents ingested. POST to /ingest first.")
        result = await run_in_threadpool(generate, req.question, chunks)
        usage = build_usage(result["prompt_tokens"], result["completion_tokens"])
        log_usage("/query", usage)
        background.add_task(notify_query, "/query", req.question, result["answer"], usage, chunks)
        return QueryResponse(
            answer=result["answer"],
            sources=_make_source_chunks(chunks),
            prompt_tokens=result["prompt_tokens"],
            completion_tokens=result["completion_tokens"],
            usage=usage,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Query failed")
        raise HTTPException(500, f"Query failed: {e}")


# ---------------------------------------------------------------------------
# CLIP visual query
# ---------------------------------------------------------------------------

@router.post("/query/visual", response_model=QueryResponse)
async def query_visual(req: QueryRequest):
    """
    Search video frames using CLIP embeddings (image-text shared space).
    Ingest video first via POST /ingest/visual.
    """
    try:
        from app.clip_store import query_visual as _qv
        top_k  = req.top_k or int(os.getenv("TOP_K", 3))
        chunks = await run_in_threadpool(_qv, req.question, top_k)
        if not chunks:
            raise HTTPException(404, "No visual content ingested. POST to /ingest/visual first.")
        result = await run_in_threadpool(generate, req.question, chunks)
        usage = build_usage(result["prompt_tokens"], result["completion_tokens"])
        log_usage("/query/visual", usage)
        return QueryResponse(
            answer=result["answer"],
            sources=_make_source_chunks(chunks),
            prompt_tokens=result["prompt_tokens"],
            completion_tokens=result["completion_tokens"],
            usage=usage,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Visual query failed")
        raise HTTPException(500, f"Visual query failed: {e}")


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

def _to_agent_response(result: dict, route: str) -> AgentResponse:
    """Build an AgentResponse + usage block from either agent's result dict."""
    usage = build_usage(result["prompt_tokens"], result["completion_tokens"])
    log_usage(route, usage)
    return AgentResponse(
        answer=result["answer"],
        plan=result.get("plan"),
        steps=[AgentStep(**s) for s in result["steps"]],
        iterations=result["iterations"],
        tool_calls=result.get("tool_calls"),
        scratchpad=result.get("scratchpad"),
        prompt_tokens=result["prompt_tokens"],
        completion_tokens=result["completion_tokens"],
        usage=usage,
        trace_id=result.get("trace_id"),
        session_id=result.get("session_id"),
        stop_reason=result.get("stop_reason"),
        elapsed_ms=result.get("elapsed_ms"),
    )


@router.post("/agent", response_model=AgentResponse)
async def agent_query(req: AgentRequest):
    """Tool-calling agent loop (LangChain). GPT decides which tools to call."""
    try:
        result = await run_in_threadpool(run_agent, req.question, req.session_id)
        return _to_agent_response(result, "/agent")
    except Exception as e:
        logger.exception("Agent failed")
        raise HTTPException(500, f"Agent failed: {e}")


@router.post("/agent/scratch", response_model=AgentResponse)
async def agent_scratch_query(req: AgentRequest):
    """Same ReAct loop, ZERO frameworks — raw Azure OpenAI SDK (app/agent_scratch.py)."""
    try:
        result = await run_in_threadpool(run_agent_scratch, req.question, req.session_id)
        return _to_agent_response(result, "/agent/scratch")
    except Exception as e:
        logger.exception("Scratch agent failed")
        raise HTTPException(500, f"Scratch agent failed: {e}")


@router.post("/agent/multi", response_model=MultiAgentResponse)
async def agent_multi(req: QueryRequest):
    """Orchestrator dispatches to retriever_agent (search) + synthesizer_agent (write).
    Trace shows every inter-agent handoff."""
    try:
        top_k = req.top_k or int(os.getenv("TOP_K", 3))
        result = await run_in_threadpool(run_multi_agent, req.question, top_k)
        usage = build_usage(result["prompt_tokens"], result["completion_tokens"])
        log_usage("/agent/multi", usage)
        return MultiAgentResponse(
            answer=result["answer"],
            refined_query=result["refined_query"],
            handoffs=result["handoffs"],
            sources=_make_source_chunks(result["sources"]),
            prompt_tokens=result["prompt_tokens"],
            completion_tokens=result["completion_tokens"],
            usage=usage,
            trace_id=result["trace_id"],
            elapsed_ms=result["elapsed_ms"],
        )
    except Exception as e:
        logger.exception("Multi-agent failed")
        raise HTTPException(500, f"Multi-agent failed: {e}")


@router.post("/agent/compare", response_model=AgentCompareResponse)
async def agent_compare(req: AgentRequest):
    """Run LangChain agent and the from-scratch agent on the same question; diff them."""
    try:
        lc_result = await run_in_threadpool(run_agent, req.question, req.session_id)
        sc_result = await run_in_threadpool(run_agent_scratch, req.question, req.session_id)
        lc = _to_agent_response(lc_result, "/agent/compare:langchain")
        sc = _to_agent_response(sc_result, "/agent/compare:scratch")
        comparison = {
            "iterations":   {"langchain": lc.iterations, "scratch": sc.iterations},
            "tool_calls":   {"langchain": lc.tool_calls, "scratch": sc.tool_calls},
            "total_tokens": {"langchain": lc.usage.total_tokens, "scratch": sc.usage.total_tokens},
            "cost_usd":     {"langchain": lc.usage.estimated_cost_usd, "scratch": sc.usage.estimated_cost_usd},
            "elapsed_ms":   {"langchain": lc.elapsed_ms, "scratch": sc.elapsed_ms},
            "answers_equal": lc.answer.strip() == sc.answer.strip(),
        }
        return AgentCompareResponse(question=req.question, langchain=lc, scratch=sc, comparison=comparison)
    except Exception as e:
        logger.exception("Agent compare failed")
        raise HTTPException(500, f"Agent compare failed: {e}")


# ---------------------------------------------------------------------------
# Structured output
# ---------------------------------------------------------------------------

@router.post("/query/structured", response_model=StructuredAnswer)
async def structured_query(req: StructuredQueryRequest):
    """Returns a strict JSON schema with confidence, sources, follow-ups."""
    try:
        top_k  = req.top_k or int(os.getenv("TOP_K", 3))
        chunks = await run_in_threadpool(retrieve, req.question, top_k)
        if not chunks:
            raise HTTPException(404, "No documents ingested.")
        result = await run_in_threadpool(generate_structured, req.question, chunks)
        usage = build_usage(result["prompt_tokens"], result["completion_tokens"])
        log_usage("/query/structured", usage)
        return StructuredAnswer(**result, usage=usage)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Structured query failed")
        raise HTTPException(500, f"Structured query failed: {e}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@router.post("/evaluate", response_model=EvalResponse)
async def evaluate_answer(req: EvalRequest):
    """LLM-as-judge scoring: faithfulness, relevance, completeness."""
    try:
        result = await run_in_threadpool(evaluate, req.question, req.answer, req.context_chunks)
        et = result["eval_tokens"]
        usage = build_usage(et["prompt_tokens"], et["completion_tokens"])
        log_usage("/evaluate", usage)
        return EvalResponse(**result, usage=usage)
    except Exception as e:
        logger.exception("Evaluation failed")
        raise HTTPException(500, f"Evaluation failed: {e}")


@router.post("/evaluate/batch", response_model=BatchEvalResponse)
async def evaluate_batch(req: BatchEvalRequest):
    """Score many Q/A/context items in one run; export results to CSV."""
    try:
        items = [i.model_dump() for i in req.items]
        result = await run_in_threadpool(batch_evaluate, items, req.export_csv, req.csv_path)
        return BatchEvalResponse(**result)
    except Exception as e:
        logger.exception("Batch evaluation failed")
        raise HTTPException(500, f"Batch evaluation failed: {e}")


# ---------------------------------------------------------------------------
# Voice I/O
# ---------------------------------------------------------------------------

@router.post("/voice/transcribe", response_model=TranscribeResponse)
async def voice_transcribe(file: UploadFile = File(...)):
    """
    Transcribe audio without storing it.
    Returns full text + per-segment timestamps.
    Client use: record mic → POST here → pipe text to /agent → POST /voice/speak.
    """
    try:
        from app.voice import transcribe_raw

        suffix = Path(file.filename).suffix.lower()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        raw = await run_in_threadpool(transcribe_raw, tmp_path)
        Path(tmp_path).unlink(missing_ok=True)

        return TranscribeResponse(
            text=raw["text"],
            language=raw["language"],
            duration=raw["duration"],
            segments=[TranscriptSegment(**s) for s in raw["segments"]],
        )
    except Exception as e:
        logger.exception("Voice transcription failed")
        raise HTTPException(500, f"Transcription failed: {e}")


@router.post("/voice/speak")
async def voice_speak(req: VoiceSpeakRequest):
    """
    Convert text to speech. Returns MP3 audio.
    Requires a TTS deployment in your Azure OpenAI resource (TTS_MODEL in .env).
    Voices: alloy, echo, fable, onyx, nova, shimmer.
    """
    try:
        from app.voice import synthesize
        audio_bytes = await run_in_threadpool(synthesize, req.text, req.voice)
        return Response(content=audio_bytes, media_type="audio/mpeg")
    except Exception as e:
        logger.exception("TTS failed")
        raise HTTPException(500, f"TTS failed: {e}")
