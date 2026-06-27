"""
ReAct agent from scratch — ZERO frameworks.

Same tools, same ReAct loop, same return shape as app/agent.py, but:
  * No LangChain. Raw `openai.AzureOpenAI` client.
  * Tool schemas are hand-written JSON (OpenAI function-calling format).
  * Tools are plain Python functions returning the standard envelope; we reuse
    the underlying business logic (retrieve / load_store / memory_store) directly,
    NOT the LangChain @tool wrappers in app/tools.py.

Purpose: see exactly what LangChain does for you. Everything here — message
plumbing, tool dispatch, the Thought/Action/Observation loop, token accounting,
retries, guardrails — is explicit.

Exposed via POST /agent/scratch. Compare with POST /agent (LangChain) via
POST /agent/compare.
"""

import json
import logging
import os
import time
import uuid

from openai import AzureOpenAI

from app.ingestion import load_store
from app.memory_store import append_turn, new_session_id, recall_facts, recent_turns, save_fact
from app.retrieval import retrieve

logger = logging.getLogger("doc-qa.agent_scratch")

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# --- Safety rails (same budgets as the LangChain agent) ---------------------
MAX_ITERATIONS    = 6
MAX_TOOL_CALLS    = 12
MAX_WALL_SECONDS  = 45
TOOL_MAX_RETRIES  = 2
TOOL_BACKOFF_SECS = (0.5, 1.5)


# ---------------------------------------------------------------------------
# Raw client (built once per run; reads same env as app/llm.py)
# ---------------------------------------------------------------------------

def _client() -> AzureOpenAI:
    return AzureOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_API_VERSION", "2024-12-01-preview"),
    )


# ---------------------------------------------------------------------------
# Tools — plain functions + standard envelope (mirrors app/tools.py logic)
# ---------------------------------------------------------------------------

def _ok(payload):
    return {"status": "ok", "error_type": None, "retryable": False, "payload": payload}


def _err(error_type: str, message: str, retryable: bool = False):
    return {"status": "error", "error_type": error_type, "retryable": retryable,
            "payload": {"message": message}}


def t_plan_step(goal: str, sub_steps: list, tool_sequence: list, stop_condition: str):
    if not goal.strip() or not sub_steps:
        return _err("invalid_plan", "Plan must have a goal and at least one sub-step")
    return _ok({"goal": goal.strip(), "sub_steps": sub_steps,
                "tool_sequence": tool_sequence, "stop_condition": stop_condition.strip()})


def t_search_documents(query: str, top_k: int = 3):
    if not query or not query.strip():
        return _err("bad_input", "query must not be empty")
    if not (1 <= top_k <= 10):
        return _err("bad_input", "top_k must be between 1 and 10")
    try:
        results = retrieve(query, top_k=top_k)
    except Exception as e:
        return _err("retrieval_failed", str(e), retryable=True)
    return _ok([{"source": r["source"], "text": r["text"], "score": round(r["score"], 4)}
                for r in results])


def t_list_documents():
    try:
        store = load_store()
    except Exception as e:
        return _err("store_failed", str(e), retryable=True)
    if not store:
        return _ok([])
    stats: dict = {}
    for item in store:
        src = item["source"]
        stats.setdefault(src, {"filename": src, "chunk_count": 0})
        stats[src]["chunk_count"] += 1
    return _ok(list(stats.values()))


def t_get_document_summary(filename: str):
    if not filename or not filename.strip():
        return _err("bad_input", "filename must not be empty")
    try:
        store = load_store()
    except Exception as e:
        return _err("store_failed", str(e), retryable=True)
    chunks = [item["text"] for item in store if item["source"] == filename]
    if not chunks:
        return _err("not_found", f"No document named '{filename}'")
    return _ok({"filename": filename, "total_chunks": len(chunks),
                "preview": " ".join(chunks[:3])[:1000]})


def t_remember(fact: str, tags: list | None = None):
    if not fact or not fact.strip():
        return _err("bad_input", "fact must not be empty")
    return _ok(save_fact(fact, tags))


def t_recall(query: str, limit: int = 5):
    if not (1 <= limit <= 20):
        return _err("bad_input", "limit must be between 1 and 20")
    return _ok(recall_facts(query or "", limit=limit))


TOOLS_BY_NAME = {
    "plan_step": t_plan_step,
    "search_documents": t_search_documents,
    "list_documents": t_list_documents,
    "get_document_summary": t_get_document_summary,
    "remember": t_remember,
    "recall": t_recall,
}

# Hand-written OpenAI function-calling schemas (what @tool generates for you).
TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "plan_step",
        "description": "Record a plan BEFORE executing tools. Call FIRST for any multi-step question.",
        "parameters": {"type": "object", "properties": {
            "goal": {"type": "string", "description": "One-sentence statement of the user's goal."},
            "sub_steps": {"type": "array", "items": {"type": "string"}, "description": "Ordered sub-tasks."},
            "tool_sequence": {"type": "array", "items": {"type": "string"}, "description": "Tool names in call order."},
            "stop_condition": {"type": "string", "description": "Rule for when to stop and answer."},
        }, "required": ["goal", "sub_steps", "tool_sequence", "stop_condition"]}}},
    {"type": "function", "function": {
        "name": "search_documents",
        "description": "Semantic search over ingested documents. Use for document-content questions.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "The search query."},
            "top_k": {"type": "integer", "description": "Number of results (1-10). Default 3."},
        }, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "list_documents",
        "description": "List all ingested documents and their chunk counts.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "get_document_summary",
        "description": "Get a preview/summary of a specific document.",
        "parameters": {"type": "object", "properties": {
            "filename": {"type": "string", "description": "Filename to summarize."},
        }, "required": ["filename"]}}},
    {"type": "function", "function": {
        "name": "remember",
        "description": "Persist a compact fact the user is likely to refer back to.",
        "parameters": {"type": "object", "properties": {
            "fact": {"type": "string", "description": "One short sentence to remember."},
            "tags": {"type": "array", "items": {"type": "string"}, "description": "Optional tags."},
        }, "required": ["fact"]}}},
    {"type": "function", "function": {
        "name": "recall",
        "description": "Recall previously stored facts relevant to a query.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Keywords or topic."},
            "limit": {"type": "integer", "description": "Max facts to return."},
        }, "required": ["query"]}}},
]


SYSTEM_PROMPT = """You are a document assistant agent with tools. You reason in the
ReAct style: Thought -> Action -> Observation, looping until you can answer.

REASONING RULE:
- BEFORE every tool call, write a brief Thought (prefix "Thought:") in your message
  content stating WHY you call this tool and WHAT you expect to learn.
- The tool call is the Action; the tool's returned envelope is the Observation.
- When you have enough Observations, reply with ONLY the final answer — clean prose,
  no "Thought:" prefix, no tool call.

Workflow:
1. For ANY multi-step question, call plan_step FIRST (goal, sub_steps, tool_sequence, stop_condition).
2. Execute the plan, narrating a Thought before each tool call.
3. Tool envelopes are {status, error_type, retryable, payload}. On error, pick a fallback.
4. Use recall before assuming context is missing; remember durable facts.
5. Cite sources. Never invent sources or chunk text.
"""


# ---------------------------------------------------------------------------
# Tool execution with retry (same policy as app/agent.py)
# ---------------------------------------------------------------------------

def _invoke_tool_with_retry(name: str, args: dict):
    fn = TOOLS_BY_NAME.get(name)
    if fn is None:
        return _err("unknown_tool", f"Unknown tool: {name}"), 0, "unknown_tool"

    last = None
    for attempt in range(TOOL_MAX_RETRIES + 1):
        try:
            last = fn(**args)
        except Exception as e:
            last = _err("tool_exception", f"{type(e).__name__}: {e}", retryable=True)
        if last.get("status") == "ok":
            return last, attempt, None
        if not last.get("retryable"):
            return last, attempt, last.get("error_type")
        if attempt < TOOL_MAX_RETRIES:
            time.sleep(TOOL_BACKOFF_SECS[min(attempt, len(TOOL_BACKOFF_SECS) - 1)])
    return last, TOOL_MAX_RETRIES, last.get("error_type")


def _summarize_observation(envelope: dict) -> str:
    status = envelope.get("status", "unknown")
    if status == "ok":
        payload = envelope.get("payload")
        if isinstance(payload, list):
            return f"ok: {len(payload)} result(s)"
        return f"ok: {json.dumps(payload, default=str)[:200]}"
    return f"error[{envelope.get('error_type')}]: {envelope.get('payload', {}).get('message', '')[:160]}"


def _clean_answer(text: str | None) -> str:
    """Drop any leading 'Thought:' reasoning lines the model leaks into the final
    answer (prompt asks for clean prose, but enforce it deterministically)."""
    if not text:
        return text or ""
    lines = text.splitlines()
    while lines and lines[0].strip().lower().startswith("thought:"):
        lines.pop(0)
    return "\n".join(lines).strip() or text.strip()


def _short_term_context(session_id: str):
    turns = recent_turns(session_id)
    if not turns:
        return None
    return "Recent conversation:\n" + "\n".join(f"{t['role']}: {t['content']}" for t in turns)


# ---------------------------------------------------------------------------
# Public entry — hand-rolled ReAct loop
# ---------------------------------------------------------------------------

def run_agent_scratch(question: str, session_id: str | None = None) -> dict:
    """Raw-SDK ReAct loop. Returns the SAME dict shape as app/agent.run_agent."""
    trace_id = uuid.uuid4().hex[:10]
    session_id = session_id or new_session_id()
    started = time.monotonic()
    client = _client()

    system_blocks = [SYSTEM_PROMPT]
    if (ctx := _short_term_context(session_id)):
        system_blocks.append(ctx)

    messages = [
        {"role": "system", "content": "\n\n".join(system_blocks)},
        {"role": "user", "content": question},
    ]

    steps: list[dict] = []
    scratchpad: list[str] = []
    plan: dict | None = None
    total_prompt_tokens = 0
    total_completion_tokens = 0
    tool_calls_used = 0
    stop_reason = "completed"

    logger.info("scratch.start trace_id=%s session=%s", trace_id, session_id)

    for iteration in range(1, MAX_ITERATIONS + 1):
        if time.monotonic() - started > MAX_WALL_SECONDS:
            stop_reason = "wall_clock_exceeded"
            break

        t_llm = time.monotonic()
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=0.2,
            max_completion_tokens=1024,
        )
        llm_ms = int((time.monotonic() - t_llm) * 1000)

        usage = resp.usage
        if usage:
            total_prompt_tokens += usage.prompt_tokens
            total_completion_tokens += usage.completion_tokens

        msg = resp.choices[0].message
        thought = (msg.content or "").strip() or "(no explicit reasoning provided)"
        scratchpad.append(f"Thought {iteration}: {thought}")
        logger.info("scratch.thought trace_id=%s iter=%d thought=%s", trace_id, iteration, thought[:300])

        # --- No tool calls => final answer -----------------------------------
        if not msg.tool_calls:
            answer = _clean_answer(msg.content)
            steps.append({"iteration": iteration, "thought": thought,
                          "action": "final_answer", "latency_ms": llm_ms, "status": "ok"})
            scratchpad.append(f"Answer: {answer}")
            append_turn(session_id, "user", question)
            append_turn(session_id, "assistant", answer)
            elapsed = int((time.monotonic() - started) * 1000)
            logger.info("scratch.done trace_id=%s iters=%d tool_calls=%d ms=%d",
                        trace_id, iteration, tool_calls_used, elapsed)
            return _result(answer, plan, steps, iteration, tool_calls_used, scratchpad,
                           total_prompt_tokens, total_completion_tokens, trace_id, session_id,
                           stop_reason, elapsed)

        # Append the assistant turn (with its tool_calls) before tool results.
        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ],
        })

        for tc in msg.tool_calls:
            if tool_calls_used >= MAX_TOOL_CALLS:
                stop_reason = "tool_call_budget_exceeded"
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": json.dumps(_err("budget_exceeded",
                                                            "Tool call budget exhausted"))})
                continue

            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            scratchpad.append(f"Action {iteration}: {name}({json.dumps(args, default=str)[:160]})")
            logger.info("scratch.action trace_id=%s iter=%d tool=%s args=%s",
                        trace_id, iteration, name, json.dumps(args, default=str)[:200])

            t_tool = time.monotonic()
            envelope, retries, err_type = _invoke_tool_with_retry(name, args)
            tool_ms = int((time.monotonic() - t_tool) * 1000)
            tool_calls_used += 1

            if name == "plan_step" and envelope.get("status") == "ok":
                plan = envelope.get("payload")

            payload_json = json.dumps(envelope, default=str)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": payload_json})

            observation = _summarize_observation(envelope)
            scratchpad.append(f"Observation {iteration}: {observation}")
            steps.append({
                "iteration": iteration, "thought": thought, "tool": name, "arguments": args,
                "observation": observation, "status": envelope.get("status"),
                "error_type": err_type, "retries": retries, "latency_ms": tool_ms,
                "result_preview": payload_json[:240],
            })
            logger.info("scratch.observation trace_id=%s iter=%d tool=%s status=%s retries=%d ms=%d obs=%s",
                        trace_id, iteration, name, envelope.get("status"), retries, tool_ms, observation[:200])

        if stop_reason == "tool_call_budget_exceeded":
            break
    else:
        stop_reason = "max_iterations"

    # --- Safe-stop ----------------------------------------------------------
    failed = [s for s in steps if s.get("status") == "error"]
    safe_answer = (
        f"I couldn't fully answer within the safety budget (stop_reason={stop_reason}). "
        f"Partial progress: {len(steps)} step(s), {len(failed)} failure(s). "
        "Next best action: rephrase, or narrow to a single document."
    )
    elapsed = int((time.monotonic() - started) * 1000)
    logger.warning("scratch.safe_stop trace_id=%s reason=%s iters=%d tool_calls=%d ms=%d",
                   trace_id, stop_reason, len(steps), tool_calls_used, elapsed)
    return _result(safe_answer, plan, steps, len(steps), tool_calls_used, scratchpad,
                   total_prompt_tokens, total_completion_tokens, trace_id, session_id,
                   stop_reason, elapsed)


def _result(answer, plan, steps, iterations, tool_calls, scratchpad,
            prompt_tokens, completion_tokens, trace_id, session_id, stop_reason, elapsed) -> dict:
    return {
        "answer": answer, "plan": plan, "steps": steps, "iterations": iterations,
        "tool_calls": tool_calls, "scratchpad": scratchpad,
        "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
        "trace_id": trace_id, "session_id": session_id,
        "stop_reason": stop_reason, "elapsed_ms": elapsed,
    }
