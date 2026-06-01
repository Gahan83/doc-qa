"""
Phase 2 — Tools defined with LangChain's @tool decorator.
Schemas are auto-generated from type hints and docstrings.
"""

from langchain_core.tools import tool

from app.ingestion import load_store
from app.retrieval import retrieve


@tool
def search_documents(query: str, top_k: int = 3) -> list[dict]:
    """Search ingested documents by semantic similarity to find relevant chunks.

    Use when the user asks a question about document content.

    Args:
        query: The search query to find relevant document chunks.
        top_k: Number of top results to return. Default 3.
    """
    results = retrieve(query, top_k=top_k)
    return [
        {"source": r["source"], "text": r["text"], "score": round(r["score"], 4)}
        for r in results
    ]


@tool
def list_documents() -> list[dict]:
    """List all ingested documents with their chunk counts.

    Use when the user asks what documents are available or how many documents exist.
    """
    store = load_store()
    if not store:
        return []

    doc_stats: dict[str, dict] = {}
    for item in store:
        source = item["source"]
        if source not in doc_stats:
            doc_stats[source] = {"filename": source, "chunk_count": 0}
        doc_stats[source]["chunk_count"] += 1
    return list(doc_stats.values())


@tool
def get_document_summary(filename: str) -> dict:
    """Get a preview/summary of a specific document by filename.

    Use when the user wants an overview of a particular document.

    Args:
        filename: The filename of the document to summarize.
    """
    store = load_store()
    chunks = [item["text"] for item in store if item["source"] == filename]

    if not chunks:
        return {"error": f"No document found with name '{filename}'"}

    preview = " ".join(chunks[:3])[:1000]
    return {"filename": filename, "total_chunks": len(chunks), "preview": preview}


TOOLS = [search_documents, list_documents, get_document_summary]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}
