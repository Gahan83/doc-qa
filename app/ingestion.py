"""
Phase 1 — Document ingestion: read → chunk → embed → store.
No frameworks. Pure logic so you see every step.
"""

import json
import os
import re
from pathlib import Path

import PyPDF2
from openai import AzureOpenAI

client = AzureOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_API_VERSION", "2024-12-01-preview"),
)

STORAGE_PATH = Path("storage/embeddings.json")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")


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
    if STORAGE_PATH.exists():
        return json.loads(STORAGE_PATH.read_text())
    return []


def save_store(store: list[dict]) -> None:
    STORAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STORAGE_PATH.write_text(json.dumps(store, indent=2))


def ingest_file(filepath: str) -> int:
    """
    Full ingestion pipeline:
    read → chunk → embed → append to JSON store.
    Returns number of chunks created.
    """
    filename = Path(filepath).name
    text = read_file(filepath)
    chunks = chunk_text(text)

    if not chunks:
        return 0

    vectors = embed_texts(chunks)

    store = load_store()
    for chunk_text_val, vector in zip(chunks, vectors):
        store.append({
            "source": filename,
            "text": chunk_text_val,
            "embedding": vector,
        })

    save_store(store)
    return len(chunks)
