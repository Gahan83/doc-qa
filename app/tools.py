"""
Phase 2 — Tools: define functions the agent can call.
Each tool has a schema (for OpenAI function calling) and an implementation.
"""

import json

from app.ingestion import load_store
from app.retrieval import retrieve


# --- Tool Implementations ---

def search_documents(query: str, top_k: int = 3) -> list[dict]:
    """Search ingested documents by semantic similarity."""
    results = retrieve(query, top_k=top_k)
    return [{"source": r["source"], "text": r["text"], "score": round(r["score"], 4)} for r in results]


def list_documents() -> list[dict]:
    """List all ingested documents with chunk counts."""
    store = load_store()
    if not store:
        return []

    doc_stats = {}
    for item in store:
        source = item["source"]
        if source not in doc_stats:
            doc_stats[source] = {"filename": source, "chunk_count": 0}
        doc_stats[source]["chunk_count"] += 1

    return list(doc_stats.values())


def get_document_summary(filename: str) -> dict:
    """Get the first few chunks of a document as a summary preview."""
    store = load_store()
    chunks = [item["text"] for item in store if item["source"] == filename]

    if not chunks:
        return {"error": f"No document found with name '{filename}'"}

    preview = " ".join(chunks[:3])[:1000]
    return {
        "filename": filename,
        "total_chunks": len(chunks),
        "preview": preview,
    }


# --- Tool Registry ---
TOOL_IMPLEMENTATIONS = {
    "search_documents": search_documents,
    "list_documents": list_documents,
    "get_document_summary": get_document_summary,
}


# --- OpenAI Function Schemas ---
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Search ingested documents by semantic similarity to find relevant chunks. Use when the user asks a question about document content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant document chunks.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of top results to return. Default 3.",
                        "default": 3,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_documents",
            "description": "List all ingested documents with their chunk counts. Use when the user asks what documents are available or how many documents exist.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_document_summary",
            "description": "Get a preview/summary of a specific document by filename. Use when the user wants an overview of a particular document.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "The filename of the document to summarize.",
                    },
                },
                "required": ["filename"],
            },
        },
    },
]


def execute_tool(tool_name: str, arguments: dict) -> str:
    """Execute a tool by name with given arguments. Returns JSON string."""
    if tool_name not in TOOL_IMPLEMENTATIONS:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    func = TOOL_IMPLEMENTATIONS[tool_name]
    result = func(**arguments)
    return json.dumps(result, ensure_ascii=False)
