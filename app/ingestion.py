"""
Document ingestion — Phase 1 (PDF/text) + Phase 3 (audio) + Phase 6 (video).
Routes by file extension to the right loader, then embeds into Chroma.
"""

import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.llm import get_vectorstore

CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE",   500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))

AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".opus"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE * 5,
    chunk_overlap=CHUNK_OVERLAP * 5,
)


def _store_media_chunks(chunks: list[dict]) -> int:
    """Convert media chunk dicts to LangChain Documents and add to Chroma."""
    if not chunks:
        return 0

    docs = [
        Document(
            page_content=c["text"],
            metadata={
                "source":          c["source"],
                "start":           float(c.get("start") or 0.0),
                "end":             float(c.get("end")   or 0.0),
                "timestamp_label": c.get("timestamp_label") or "",
            },
        )
        for c in chunks
    ]
    get_vectorstore().add_documents(docs)
    return len(docs)


def ingest_file(filepath: str) -> int:
    """
    Load → chunk → embed → store in Chroma.
    Returns number of chunks created.
    """
    p      = Path(filepath)
    suffix = p.suffix.lower()
    source = p.name

    # --- Audio (Phase 3) ---
    if suffix in AUDIO_EXTS:
        from app.audio_loader import load_audio
        return _store_media_chunks(load_audio(filepath))

    # --- Video (Phase 6 + 7) ---
    if suffix in VIDEO_EXTS:
        from app.video_loader import load_video
        describe = os.getenv("VISUAL_MODE", "describe") in ("describe", "both")
        return _store_media_chunks(load_video(filepath, describe_frames=describe))

    # --- PDF / text ---
    if suffix == ".pdf":
        loader = PyPDFLoader(filepath)
    else:
        loader = TextLoader(filepath, encoding="utf-8")

    docs   = loader.load()
    chunks = splitter.split_documents(docs)
    if not chunks:
        return 0

    for chunk in chunks:
        chunk.metadata["source"] = source

    get_vectorstore().add_documents(chunks)
    return len(chunks)


def load_store() -> list[dict]:
    """Return all chunks from Chroma (used by tools.py for list/summary)."""
    vectorstore = get_vectorstore()
    data = vectorstore.get(include=["documents", "metadatas"])
    return [
        {"source": data["metadatas"][i].get("source", "unknown"), "text": data["documents"][i]}
        for i in range(len(data["ids"]))
    ]
