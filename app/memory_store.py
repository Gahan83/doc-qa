"""
Lightweight agent memory.

Short-term: per-session conversation buffer (in-process dict).
Long-term:  durable JSON store of facts the agent chooses to remember.

Kept intentionally simple — no DB, no vector index.
Swap to Chroma or Redis later without changing call sites.
"""

import json
import threading
import uuid
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque

# ---------------------------------------------------------------------------
# Short-term memory: last N (role, content) pairs per session
# ---------------------------------------------------------------------------

_SHORT_TERM_MAX = 10
_short_term: dict[str, Deque[dict]] = defaultdict(lambda: deque(maxlen=_SHORT_TERM_MAX))
_lock = threading.Lock()


def new_session_id() -> str:
    return uuid.uuid4().hex[:12]


def append_turn(session_id: str, role: str, content: str) -> None:
    with _lock:
        _short_term[session_id].append({"role": role, "content": content})


def recent_turns(session_id: str) -> list[dict]:
    with _lock:
        return list(_short_term.get(session_id, []))


# ---------------------------------------------------------------------------
# Long-term memory: JSON-backed list of compact facts
# ---------------------------------------------------------------------------

_MEM_PATH = Path("storage/agent_memory.json")
_MEM_PATH.parent.mkdir(parents=True, exist_ok=True)


def _load() -> list[dict]:
    if not _MEM_PATH.exists():
        return []
    try:
        return json.loads(_MEM_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def _save(items: list[dict]) -> None:
    _MEM_PATH.write_text(json.dumps(items, indent=2), encoding="utf-8")


def save_fact(fact: str, tags: list[str] | None = None) -> dict:
    items = _load()
    entry = {"id": uuid.uuid4().hex[:8], "fact": fact.strip(), "tags": tags or []}
    items.append(entry)
    _save(items)
    return entry


def recall_facts(query: str, limit: int = 5) -> list[dict]:
    """Naive keyword recall — good enough for a single-user dev agent."""
    q = query.lower().strip()
    if not q:
        return _load()[-limit:]
    tokens = [t for t in q.split() if len(t) > 2]
    items = _load()
    scored = []
    for it in items:
        hay = (it["fact"] + " " + " ".join(it.get("tags", []))).lower()
        score = sum(hay.count(t) for t in tokens)
        if score > 0:
            scored.append((score, it))
    scored.sort(key=lambda p: p[0], reverse=True)
    return [it for _, it in scored[:limit]]
