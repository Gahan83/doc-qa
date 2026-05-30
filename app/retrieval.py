"""
Phase 1 — Retrieval: embed query → cosine similarity → top-K chunks.
No vector DB. Pure numpy so you see the math.
"""


import numpy as np
from cachetools import LRUCache, cached

from app.ingestion import embed_texts, load_store


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    cos(θ) = (A · B) / (|A| * |B|)
    Returns 1.0 = identical direction, 0.0 = orthogonal, -1.0 = opposite.
    Embeddings are normalized by OpenAI so dot product ≈ cosine sim,
    but we compute it properly here as a learning exercise.
    """
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    dot = np.dot(va, vb)
    norm = np.linalg.norm(va) * np.linalg.norm(vb)
    if norm == 0:
        return 0.0
    return float(dot / norm)


_retrieve_cache = LRUCache(maxsize=128)

@cached(_retrieve_cache)
def retrieve(question: str, top_k: int = 3) -> list[dict]:
    """
    1. Embed the question.
    2. Score every stored chunk against it.
    3. Return top_k by score.
    """
    store = load_store()
    if not store:
        return []

    query_vector = embed_texts([question])[0]

    scored = [
        {
            "source": item["source"],
            "text": item["text"],
            "score": cosine_similarity(query_vector, item["embedding"]),
        }
        for item in store
    ]

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]
