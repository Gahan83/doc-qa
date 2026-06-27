"""
Agentic loop.

Differences from a plain bind_tools loop:

1. Planning — the model is instructed to call `plan_step` first for any
   non-trivial question. The plan is captured in the trace.
2. Memory — short-term turns (from app/memory_store) are injected as
   system context so follow-ups don't need full restated context.
3. Retry/recovery — tool envelopes carry a `retryable` flag. The loop
   retries transient failures (exponential backoff), and on terminal
   failure it tells the model what failed so it can pick a fallback.
4. Guardrails — hard caps on iterations, tool calls, and wall-clock.
   On hit, we return a safe-stop response: partial answer + what failed
   + next best action.
5. Observability — every step records tool, args, status, latency_ms,
   error_type, token usage, and a trace_id for the whole run.
"""

import json
import logging
import time
import uuid

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from app.llm import get_chat_llm
from app.memory_store import (
    append_turn,
    new_session_id,
    recent_turns,
)
from app.tools import TOOLS, TOOLS_BY_NAME

logger = logging.getLogger("doc-qa.agent")

# --- Safety rails -----------------------------------------------------------
MAX_ITERATIONS    = 6      # full LLM round-trips
MAX_TOOL_CALLS    = 12     # individual tool invocations across the run
MAX_WALL_SECONDS  = 45     # total wall-clock budget
TOOL_MAX_RETRIES  = 2      # per tool call, on retryable errors
TOOL_BACKOFF_SECS = (0.5, 1.5)   # exponential-ish backoff sequence


AGENT_SYSTEM_PROMPT = """You are a document assistant agent with tools. You reason
in the ReAct style: Thought -> Action -> Observation, looping until you can answer.

REASONING RULE (important):
- BEFORE every tool call, write a brief Thought in your message content: state WHY
  you are calling this tool and WHAT you expect to learn. One or two sentences.
- The tool call itself is the Action. The tool's returned envelope is the Observation.
- Prefix each reasoning sentence that precedes a tool call with "Thought:".
- When you have enough Observations to satisfy the stop_condition, reply with ONLY
  the final answer — clean prose, NO "Thought:" prefix and no tool call.
- Never leave the reasoning empty when you act — always verbalize the Thought.

Workflow:
1. For ANY question that needs more than a single lookup, call `plan_step` FIRST.
   - Provide: goal, sub_steps, tool_sequence, stop_condition.
   - Skip planning ONLY for trivial single-shot questions.
2. Execute your plan by calling tools in order, narrating a Thought before each.
3. Each tool returns an envelope: {status, error_type, retryable, payload}.
   - On status="error", read error_type. If retryable, the runtime already
     retried — pick a fallback strategy or ask a clarifying question.
4. Use `recall` to check stored facts before assuming context is missing.
5. Use `remember` to save durable facts the user is likely to ask again.
6. When stop_condition is met, produce a concise final answer that cites sources.

Rules:
- Never invent sources or chunk text.
- If retrieval returns nothing relevant, say so plainly.
- Prefer ONE clarifying question over a wrong answer when intent is ambiguous.
"""


def _summarize_observation(envelope: dict) -> str:
    """Compact Observation string from a tool envelope for the ReAct trace."""
    status = envelope.get("status", "unknown")
    if status == "ok":
        payload = envelope.get("payload")
        if isinstance(payload, list):
            return f"ok: {len(payload)} result(s)"
        text = json.dumps(payload, default=str)
        return f"ok: {text[:200]}"
    return f"error[{envelope.get('error_type')}]: {envelope.get('payload', {}).get('message', '')[:160]}"


# ---------------------------------------------------------------------------
# Tool execution with retry
# ---------------------------------------------------------------------------

def _invoke_tool_with_retry(tool_name: str, args: dict) -> tuple[dict, int, str | None]:
    """Returns (envelope_dict, retries_used, transport_error_or_None)."""
    tool_fn = TOOLS_BY_NAME.get(tool_name)
    if tool_fn is None:
        return (
            {"status": "error", "error_type": "unknown_tool",
             "retryable": False, "payload": {"message": f"Unknown tool: {tool_name}"}},
            0,
            "unknown_tool",
        )

    last_envelope: dict | None = None
    for attempt in range(TOOL_MAX_RETRIES + 1):
        try:
            result = tool_fn.invoke(args)
        except Exception as e:  # transport-level failure
            result = {
                "status": "error",
                "error_type": "tool_exception",
                "retryable": True,
                "payload": {"message": f"{type(e).__name__}: {e}"},
            }

        last_envelope = result if isinstance(result, dict) else {"status": "ok", "payload": result}
        if last_envelope.get("status") == "ok":
            return last_envelope, attempt, None
        if not last_envelope.get("retryable"):
            return last_envelope, attempt, last_envelope.get("error_type")

        if attempt < TOOL_MAX_RETRIES:
            time.sleep(TOOL_BACKOFF_SECS[min(attempt, len(TOOL_BACKOFF_SECS) - 1)])

    return last_envelope, TOOL_MAX_RETRIES, last_envelope.get("error_type")


# ---------------------------------------------------------------------------
# Memory injection
# ---------------------------------------------------------------------------

def _short_term_context(session_id: str) -> str | None:
    turns = recent_turns(session_id)
    if not turns:
        return None
    lines = [f"{t['role']}: {t['content']}" for t in turns]
    return "Recent conversation:\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def run_agent(question: str, session_id: str | None = None) -> dict:
    """Run the agentic loop. Returns answer + trace + usage + status."""
    trace_id = uuid.uuid4().hex[:10]
    session_id = session_id or new_session_id()
    started = time.monotonic()

    llm = get_chat_llm(temperature=0.2, max_tokens=1024)
    llm_with_tools = llm.bind_tools(TOOLS)

    system_blocks = [AGENT_SYSTEM_PROMPT]
    if (ctx := _short_term_context(session_id)):
        system_blocks.append(ctx)

    messages = [
        SystemMessage(content="\n\n".join(system_blocks)),
        HumanMessage(content=question),
    ]

    steps: list[dict] = []
    scratchpad: list[str] = []   # ordered Thought/Action/Observation trace
    plan: dict | None = None
    total_prompt_tokens = 0
    total_completion_tokens = 0
    tool_calls_used = 0
    stop_reason = "completed"

    logger.info("agent.start trace_id=%s session=%s", trace_id, session_id)

    for iteration in range(1, MAX_ITERATIONS + 1):
        if time.monotonic() - started > MAX_WALL_SECONDS:
            stop_reason = "wall_clock_exceeded"
            break

        t_llm = time.monotonic()
        response = llm_with_tools.invoke(messages)
        llm_ms = int((time.monotonic() - t_llm) * 1000)
        messages.append(response)

        usage = response.usage_metadata or {}
        total_prompt_tokens += usage.get("input_tokens", 0)
        total_completion_tokens += usage.get("output_tokens", 0)

        # --- ReAct: Thought (model's reasoning emitted in message content) ---
        thought = (response.content or "").strip() or "(no explicit reasoning provided)"
        scratchpad.append(f"Thought {iteration}: {thought}")
        logger.info("agent.thought trace_id=%s iter=%d thought=%s",
                    trace_id, iteration, thought[:300])

        if not response.tool_calls:
            steps.append({
                "iteration": iteration,
                "thought": thought,
                "action": "final_answer",
                "latency_ms": llm_ms,
                "status": "ok",
            })
            scratchpad.append(f"Answer: {thought}")
            answer = response.content
            append_turn(session_id, "user", question)
            append_turn(session_id, "assistant", answer)
            elapsed = int((time.monotonic() - started) * 1000)
            logger.info(
                "agent.done trace_id=%s iters=%d tool_calls=%d ms=%d",
                trace_id, iteration, tool_calls_used, elapsed,
            )
            return {
                "answer": answer,
                "plan": plan,
                "steps": steps,
                "iterations": iteration,
                "tool_calls": tool_calls_used,
                "scratchpad": scratchpad,
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "trace_id": trace_id,
                "session_id": session_id,
                "stop_reason": stop_reason,
                "elapsed_ms": elapsed,
            }

        # Execute every tool call requested this turn
        for tool_call in response.tool_calls:
            if tool_calls_used >= MAX_TOOL_CALLS:
                stop_reason = "tool_call_budget_exceeded"
                messages.append(ToolMessage(
                    content=json.dumps({"status": "error", "error_type": "budget_exceeded",
                                        "retryable": False,
                                        "payload": {"message": "Tool call budget exhausted"}}),
                    tool_call_id=tool_call["id"],
                ))
                continue

            tool_name = tool_call["name"]
            args = tool_call["args"]

            # --- ReAct: Action ---
            scratchpad.append(f"Action {iteration}: {tool_name}({json.dumps(args, default=str)[:160]})")
            logger.info("agent.action trace_id=%s iter=%d tool=%s args=%s",
                        trace_id, iteration, tool_name, json.dumps(args, default=str)[:200])

            t_tool = time.monotonic()
            envelope, retries, err_type = _invoke_tool_with_retry(tool_name, args)
            tool_ms = int((time.monotonic() - t_tool) * 1000)
            tool_calls_used += 1

            if tool_name == "plan_step" and envelope.get("status") == "ok":
                plan = envelope.get("payload")

            payload_json = json.dumps(envelope, default=str)
            messages.append(ToolMessage(content=payload_json, tool_call_id=tool_call["id"]))

            # --- ReAct: Observation ---
            observation = _summarize_observation(envelope)
            scratchpad.append(f"Observation {iteration}: {observation}")

            step_record = {
                "iteration": iteration,
                "thought": thought,
                "tool": tool_name,
                "arguments": args,
                "observation": observation,
                "status": envelope.get("status"),
                "error_type": err_type,
                "retries": retries,
                "latency_ms": tool_ms,
                "result_preview": payload_json[:240],
            }
            steps.append(step_record)
            logger.info(
                "agent.observation trace_id=%s iter=%d tool=%s status=%s retries=%d ms=%d obs=%s",
                trace_id, iteration, tool_name, envelope.get("status"), retries, tool_ms, observation[:200],
            )

        if stop_reason == "tool_call_budget_exceeded":
            break
    else:
        stop_reason = "max_iterations"

    # ---- Safe-stop response -------------------------------------------------
    failed = [s for s in steps if s.get("status") == "error"]
    safe_answer = (
        "I couldn't fully answer within the safety budget "
        f"(stop_reason={stop_reason}). "
        f"Partial progress: {len(steps)} step(s), {len(failed)} failure(s). "
        "Next best action: rephrase the question, or narrow it to a single document."
    )
    elapsed = int((time.monotonic() - started) * 1000)
    logger.warning(
        "agent.safe_stop trace_id=%s reason=%s iters=%d tool_calls=%d ms=%d",
        trace_id, stop_reason, len(steps), tool_calls_used, elapsed,
    )
    return {
        "answer": safe_answer,
        "plan": plan,
        "steps": steps,
        "iterations": len(steps),
        "tool_calls": tool_calls_used,
        "scratchpad": scratchpad,
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "trace_id": trace_id,
        "session_id": session_id,
        "stop_reason": stop_reason,
        "elapsed_ms": elapsed,
    }
