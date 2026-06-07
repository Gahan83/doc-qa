"""
Agent tools.

Each tool returns a *standardized envelope*:
    {
        "status": "ok" | "error",
        "error_type": str | None,
        "retryable": bool,
        "payload": <tool-specific result>,
    }

The agent loop in app/agent.py reads `status` + `retryable` to decide
whether to retry, fall back, or surface an error to the user.
"""

from typing import Any

from langchain_core.tools import tool

from app.ingestion import load_store
from app.memory_store import recall_facts, save_fact
from app.retrieval import retrieve


def _ok(payload: Any) -> dict:
    return {"status": "ok", "error_type": None, "retryable": False, "payload": payload}


def _err(error_type: str, message: str, retryable: bool = False) -> dict:
    return {
        "status": "error",
        "error_type": error_type,
        "retryable": retryable,
        "payload": {"message": message},
    }


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

@tool
def plan_step(goal: str, sub_steps: list[str], tool_sequence: list[str], stop_condition: str) -> dict:
    """Record a plan BEFORE executing tools. Call this FIRST for any multi-step question.

    Args:
        goal: One-sentence statement of the user's goal.
        sub_steps: Ordered list of sub-tasks needed to reach the goal.
        tool_sequence: Names of tools, in the order you intend to call them.
        stop_condition: A short rule that tells you when to stop and answer.
    """
    if not goal.strip() or not sub_steps:
        return _err("invalid_plan", "Plan must have a goal and at least one sub-step")
    return _ok({
        "goal": goal.strip(),
        "sub_steps": sub_steps,
        "tool_sequence": tool_sequence,
        "stop_condition": stop_condition.strip(),
    })


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

@tool
def search_documents(query: str, top_k: int = 3) -> dict:
    """Semantic search over ingested documents.

    Use when the user asks about document content.

    Args:
        query: The search query.
        top_k: Number of top results (1-10). Default 3.
    """
    if not query or not query.strip():
        return _err("bad_input", "query must not be empty")
    if not (1 <= top_k <= 10):
        return _err("bad_input", "top_k must be between 1 and 10")

    try:
        results = retrieve(query, top_k=top_k)
    except Exception as e:
        return _err("retrieval_failed", str(e), retryable=True)

    return _ok([
        {"source": r["source"], "text": r["text"], "score": round(r["score"], 4)}
        for r in results
    ])


@tool
def list_documents() -> dict:
    """List all ingested documents and their chunk counts."""
    try:
        store = load_store()
    except Exception as e:
        return _err("store_failed", str(e), retryable=True)

    if not store:
        return _ok([])

    doc_stats: dict[str, dict] = {}
    for item in store:
        source = item["source"]
        doc_stats.setdefault(source, {"filename": source, "chunk_count": 0})
        doc_stats[source]["chunk_count"] += 1
    return _ok(list(doc_stats.values()))


@tool
def get_document_summary(filename: str) -> dict:
    """Get a preview/summary of a specific document.

    Args:
        filename: Filename of the document to summarize.
    """
    if not filename or not filename.strip():
        return _err("bad_input", "filename must not be empty")

    try:
        store = load_store()
    except Exception as e:
        return _err("store_failed", str(e), retryable=True)

    chunks = [item["text"] for item in store if item["source"] == filename]
    if not chunks:
        return _err("not_found", f"No document named '{filename}'")

    preview = " ".join(chunks[:3])[:1000]
    return _ok({"filename": filename, "total_chunks": len(chunks), "preview": preview})


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

@tool
def remember(fact: str, tags: list[str] | None = None) -> dict:
    """Persist a compact fact the user is likely to refer back to.

    Args:
        fact: One short sentence to remember.
        tags: Optional tags for retrieval.
    """
    if not fact or not fact.strip():
        return _err("bad_input", "fact must not be empty")
    return _ok(save_fact(fact, tags))


@tool
def recall(query: str, limit: int = 5) -> dict:
    """Recall previously stored facts relevant to a query.

    Args:
        query: Keywords or topic to search remembered facts.
        limit: Max number of facts to return.
    """
    if not (1 <= limit <= 20):
        return _err("bad_input", "limit must be between 1 and 20")
    return _ok(recall_facts(query or "", limit=limit))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TOOLS = [
    plan_step,
    search_documents,
    list_documents,
    get_document_summary,
    remember,
    recall,
]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}
