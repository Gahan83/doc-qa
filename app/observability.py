"""
Per-query observability: cost tracking (CSV) + Slack notifications.

- append_query_cost(): one row per query in storage/query_costs.csv
  (timestamp, route, model, tokens, cost, question). The durable audit of spend.
- post_query_to_slack(): posts answer + cost + eval score to a Slack incoming
  webhook (SLACK_WEBHOOK_URL). Fail-safe — never raises into the request path.
- notify_query(): orchestrator meant to run as a FastAPI BackgroundTask after the
  response is sent. Logs cost, optionally runs LLM-as-judge eval, posts to Slack.

Env:
  SLACK_WEBHOOK_URL    — if set, Slack posting is enabled.
  SLACK_INCLUDE_EVAL   — "false" to skip the eval step in Slack (default true).
"""

import csv
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

import httpx

logger = logging.getLogger("doc-qa.observability")

COST_CSV = Path("storage/query_costs.csv")
_CSV_LOCK = threading.Lock()
_CSV_HEADER = [
    "timestamp", "route", "model", "prompt_tokens", "completion_tokens",
    "total_tokens", "estimated_cost_usd", "question",
]


# ---------------------------------------------------------------------------
# Cost tracking
# ---------------------------------------------------------------------------

def append_query_cost(route: str, question: str, usage: dict) -> None:
    """Append one cost row to storage/query_costs.csv (thread-safe, fail-safe)."""
    try:
        COST_CSV.parent.mkdir(parents=True, exist_ok=True)
        with _CSV_LOCK:
            new_file = not COST_CSV.exists()
            with COST_CSV.open("a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if new_file:
                    w.writerow(_CSV_HEADER)
                w.writerow([
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    route, usage["model"], usage["prompt_tokens"],
                    usage["completion_tokens"], usage["total_tokens"],
                    usage["estimated_cost_usd"], question.replace("\n", " ")[:500],
                ])
    except Exception:
        logger.warning("Failed to append query cost CSV", exc_info=True)


# ---------------------------------------------------------------------------
# Slack
# ---------------------------------------------------------------------------

def _slack_enabled() -> bool:
    return bool(os.getenv("SLACK_WEBHOOK_URL"))


def post_query_to_slack(question: str, answer: str, usage: dict,
                        eval_result: dict | None = None) -> bool:
    """Post a query summary to Slack. Returns True on 2xx. Never raises."""
    url = os.getenv("SLACK_WEBHOOK_URL")
    if not url:
        return False

    lines = [
        "*Doc Q&A — query*",
        f"*Q:* {question[:400]}",
        f"*A:* {answer[:1500]}",
        (f"*Cost:* ${usage['estimated_cost_usd']:.6f} "
         f"({usage['total_tokens']} tok | {usage['model']})"),
    ]
    if eval_result:
        lines.append(
            f"*Eval:* overall {eval_result['overall_score']}/5 "
            f"(faith {eval_result['faithfulness']['score']}, "
            f"rel {eval_result['relevance']['score']}, "
            f"comp {eval_result['completeness']['score']})"
        )

    try:
        resp = httpx.post(url, json={"text": "\n".join(lines)}, timeout=10.0)
        ok = resp.is_success
        if not ok:
            logger.warning("Slack post returned %s: %s", resp.status_code, resp.text[:200])
        return ok
    except Exception:
        logger.warning("Slack post failed", exc_info=True)
        return False


# ---------------------------------------------------------------------------
# Background orchestrator
# ---------------------------------------------------------------------------

def notify_query(route: str, question: str, answer: str, usage: dict,
                 context_chunks: list[dict] | None = None) -> None:
    """Run AFTER the response is sent (FastAPI BackgroundTask).

    Always records cost. If Slack is configured, optionally runs LLM-as-judge
    eval and posts answer + cost + score. Eval is skipped when Slack is off
    (no point paying for it) or SLACK_INCLUDE_EVAL=false.
    """
    append_query_cost(route, question, usage)

    if not _slack_enabled():
        return

    eval_result = None
    include_eval = os.getenv("SLACK_INCLUDE_EVAL", "true").lower() != "false"
    if include_eval and context_chunks:
        try:
            from app.evaluation import evaluate
            eval_result = evaluate(question, answer, context_chunks)
        except Exception:
            logger.warning("Eval for Slack failed; posting without score", exc_info=True)

    post_query_to_slack(question, answer, usage, eval_result)
