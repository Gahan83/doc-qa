"""
Multi-agent system: one orchestrator, two specialized subagents.

  orchestrator      — owns the workflow; dispatches tasks, never searches or
                      writes prose itself. Records every inter-agent handoff.
  retriever_agent   — ONLY does search: refines the question into a search query
                      (LLM), then runs vector retrieval. Returns chunks. No prose.
  synthesizer_agent — ONLY writes answers: takes question + chunks, produces the
                      final grounded answer. Never searches.

The deliverable is the handoff trace: every arrow between agents is logged with
a direction, an action, a one-line summary, and latency, so you can see the
control flow:

    orchestrator -> retriever_agent   (dispatch search)
    retriever_agent -> orchestrator   (returned N chunks)
    orchestrator -> synthesizer_agent (dispatch synthesis)
    synthesizer_agent -> orchestrator (returned answer)

Exposed via POST /agent/multi.
"""

import logging
import time
import uuid

from langchain_core.prompts import ChatPromptTemplate

from app.generation import generate
from app.llm import get_chat_llm
from app.retrieval import retrieve

logger = logging.getLogger("doc-qa.multi")


_REFINE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You turn a user question into a concise keyword search query for a "
               "vector database. Return ONLY the query text, no quotes, no explanation."),
    ("user", "{question}"),
])


# ---------------------------------------------------------------------------
# Subagent: retriever — ONLY search
# ---------------------------------------------------------------------------

def retriever_agent(question: str, top_k: int) -> dict:
    """Refine the question into a search query, then retrieve. No answer writing."""
    llm = get_chat_llm(temperature=0.0, max_tokens=64)
    resp = (_REFINE_PROMPT | llm).invoke({"question": question})
    refined = (resp.content or question).strip() or question
    usage = resp.usage_metadata or {}

    chunks = retrieve(refined, top_k=top_k)
    return {
        "refined_query": refined,
        "chunks": chunks,
        "prompt_tokens": usage.get("input_tokens", 0),
        "completion_tokens": usage.get("output_tokens", 0),
    }


# ---------------------------------------------------------------------------
# Subagent: synthesizer — ONLY writes answers (reuses generation.generate)
# ---------------------------------------------------------------------------

def synthesizer_agent(question: str, chunks: list[dict]) -> dict:
    """Write the grounded answer from supplied chunks. No searching."""
    return generate(question, chunks)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _handoff(handoffs: list[dict], frm: str, to: str, action: str, summary: str,
             started: float, trace_id: str) -> None:
    latency_ms = int((time.monotonic() - started) * 1000)
    handoffs.append({
        "step": len(handoffs) + 1,
        "from_agent": frm,
        "to_agent": to,
        "action": action,
        "summary": summary,
        "latency_ms": latency_ms,
    })
    logger.info("multi.handoff trace_id=%s %s -> %s action=%s summary=%s",
                trace_id, frm, to, action, summary[:160])


def run_multi_agent(question: str, top_k: int = 3) -> dict:
    """Orchestrate retriever_agent then synthesizer_agent; trace every handoff."""
    trace_id = uuid.uuid4().hex[:10]
    overall_start = time.monotonic()
    handoffs: list[dict] = []
    total_prompt = 0
    total_completion = 0

    logger.info("multi.start trace_id=%s top_k=%d", trace_id, top_k)

    # --- Orchestrator -> retriever_agent ---
    t = time.monotonic()
    _handoff(handoffs, "orchestrator", "retriever_agent", "dispatch_search",
             f"Find context for: {question}", t, trace_id)

    t = time.monotonic()
    r = retriever_agent(question, top_k)
    total_prompt += r["prompt_tokens"]
    total_completion += r["completion_tokens"]
    _handoff(handoffs, "retriever_agent", "orchestrator", "return_chunks",
             f"refined_query='{r['refined_query']}', returned {len(r['chunks'])} chunk(s)",
             t, trace_id)

    # Orchestrator guard: nothing retrieved -> stop before synthesis.
    if not r["chunks"]:
        elapsed = int((time.monotonic() - overall_start) * 1000)
        _handoff(handoffs, "orchestrator", "orchestrator", "halt",
                 "No chunks retrieved; skipping synthesis", time.monotonic(), trace_id)
        return {
            "answer": "I don't have enough information — no relevant documents were found.",
            "refined_query": r["refined_query"],
            "handoffs": handoffs,
            "sources": [],
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "trace_id": trace_id,
            "elapsed_ms": elapsed,
        }

    # --- Orchestrator -> synthesizer_agent ---
    t = time.monotonic()
    _handoff(handoffs, "orchestrator", "synthesizer_agent", "dispatch_synthesis",
             f"Write answer from {len(r['chunks'])} chunk(s)", t, trace_id)

    t = time.monotonic()
    s = synthesizer_agent(question, r["chunks"])
    total_prompt += s["prompt_tokens"]
    total_completion += s["completion_tokens"]
    answer = s["answer"]
    _handoff(handoffs, "synthesizer_agent", "orchestrator", "return_answer",
             f"answer written ({len(answer)} chars)", t, trace_id)

    elapsed = int((time.monotonic() - overall_start) * 1000)
    logger.info("multi.done trace_id=%s handoffs=%d ms=%d", trace_id, len(handoffs), elapsed)
    return {
        "answer": answer,
        "refined_query": r["refined_query"],
        "handoffs": handoffs,
        "sources": r["chunks"],
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "trace_id": trace_id,
        "elapsed_ms": elapsed,
    }
