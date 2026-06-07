"""
Phase 1 — Retrieval using LangChain's Chroma vector store.
"""

from cachetools import LRUCache, cached

from app.llm import get_vectorstore

_retrieve_cache = LRUCache(maxsize=128)


@cached(_retrieve_cache)
def retrieve(question: str, top_k: int = 3) -> list[dict]:
    """
    Use Chroma similarity_search_with_score for top-K retrieval.
    Returns list of {source, text, score} dicts.
    """
    vectorstore = get_vectorstore()
    results = vectorstore.similarity_search_with_score(question, k=top_k)

    # Chroma returns cosine distance (lower = better); convert to similarity
    return [
        {
            "source":          doc.metadata.get("source", "unknown"),
            "text":            doc.page_content,
            "score":           round(1 - distance, 4),
            # Phase 4: timestamps — present for audio/video chunks, absent for PDFs
            "start":           doc.metadata.get("start"),
            "end":             doc.metadata.get("end"),
            "timestamp_label": doc.metadata.get("timestamp_label"),
        }
        for doc, distance in results
    ]
