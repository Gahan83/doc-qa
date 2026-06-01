"""
Phase 1 — Document ingestion using LangChain.
Loaders → splitter → embeddings → Chroma vector store.
"""

import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.llm import get_vectorstore

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))

# LangChain splits by characters; multiply our word-based config by ~5 chars/word
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE * 5,
    chunk_overlap=CHUNK_OVERLAP * 5,
)


def ingest_file(filepath: str) -> int:
    """
    Pipeline: load file → split into chunks → embed → store in Chroma.
    Returns number of chunks created.
    """
    p = Path(filepath)
    filename = p.name

    if p.suffix.lower() == ".pdf":
        loader = PyPDFLoader(filepath)
    else:
        loader = TextLoader(filepath, encoding="utf-8")

    docs = loader.load()
    chunks = splitter.split_documents(docs)

    if not chunks:
        return 0

    for chunk in chunks:
        chunk.metadata["source"] = filename

    vectorstore = get_vectorstore()
    vectorstore.add_documents(chunks)

    return len(chunks)


def load_store() -> list[dict]:
    """Return all chunks from Chroma (used by tools.py for list/summary)."""
    vectorstore = get_vectorstore()
    data = vectorstore.get(include=["documents", "metadatas"])
    return [
        {"source": data["metadatas"][i].get("source", "unknown"), "text": data["documents"][i]}
        for i in range(len(data["ids"]))
    ]
