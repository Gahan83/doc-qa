"""
Phase 1 — Document ingestion: read → chunk → embed → store.
Uses ChromaDB for vector storage (replaces manual JSON store).
"""

import os
import re
import uuid
from pathlib import Path

import chromadb
import PyPDF2
from openai import AzureOpenAI

client = AzureOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_API_VERSION", "2024-12-01-preview"),
)

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

# --- ChromaDB Setup ---
CHROMA_PATH = Path("storage/chromadb")
chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
collection = chroma_client.get_or_create_collection(
    name="doc_chunks",
    metadata={"hnsw:space": "cosine"},  # Use cosine similarity
)


def read_file(path: str) -> str:
    """Extract raw text from .txt or .pdf."""
    p = Path(path)
    if p.suffix == ".pdf":
        text = []
        with open(p, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text.append(page.extract_text() or "")
        return "\n".join(text)
    return p.read_text(encoding="utf-8")


def chunk_text(text: str) -> list[str]:
    """
    Split text into overlapping chunks by word count.
    Overlap helps preserve context at chunk boundaries.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + CHUNK_SIZE
        chunk = " ".join(words[start:end])
        # Clean excess whitespace
        chunk = re.sub(r"\s+", " ", chunk).strip()
        if chunk:
            chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Call OpenAI Embeddings API.
    Returns list of vectors (one per text).
    Batch in one call — cheaper than N calls.
    """
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in response.data]


def load_store() -> list[dict]:
    """Load all documents from ChromaDB (for backward compatibility with tools.py)."""
    results = collection.get(include=["documents", "metadatas", "embeddings"])
    store = []
    for i in range(len(results["ids"])):
        store.append({
            "source": results["metadatas"][i]["source"],
            "text": results["documents"][i],
            "embedding": results["embeddings"][i] if results["embeddings"] else [],
        })
    return store


def save_store(store: list[dict]) -> None:
    """No-op: ChromaDB persists automatically. Kept for backward compatibility."""
    pass


def ingest_file(filepath: str) -> int:
    """
    Full ingestion pipeline:
    read → chunk → embed → store in ChromaDB.
    Returns number of chunks created.
    """
    filename = Path(filepath).name
    text = read_file(filepath)
    chunks = chunk_text(text)

    if not chunks:
        return 0

    vectors = embed_texts(chunks)

    # Generate unique IDs for each chunk
    ids = [str(uuid.uuid4()) for _ in chunks]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=vectors,
        metadatas=[{"source": filename} for _ in chunks],
    )

    return len(chunks)
