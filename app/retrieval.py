"""
Phase 1 — Retrieval: embed query → ChromaDB vector search → top-K chunks.
ChromaDB handles cosine similarity and ANN indexing internally.
"""

from cachetools import LRUCache, cached

from app.ingestion import collection, embed_texts


_retrieve_cache = LRUCache(maxsize=128)

@cached(_retrieve_cache)
def retrieve(question: str, top_k: int = 3) -> list[dict]:
    """
    1. Embed the question.
    2. Query ChromaDB for nearest neighbors.
    3. Return top_k results with scores.
    """
    query_vector = embed_texts([question])[0]

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    # ChromaDB returns cosine distance (0 = identical, 2 = opposite)
    # Convert to similarity score: similarity = 1 - distance
    scored = []
    for i in range(len(results["ids"][0])):
        scored.append({
            "source": results["metadatas"][0][i]["source"],
            "text": results["documents"][0][i],
            "score": round(1 - results["distances"][0][i], 4),
        })

    return scored
