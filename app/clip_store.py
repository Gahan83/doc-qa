"""
Phase 8 — CLIP visual search.

Images and text share the same embedding space (CLIP-ViT-B-32).
No GPT description step: frames are embedded directly as images,
and a text question is embedded the same way → cosine search finds matching frames.

Separate Chroma collection ("visual_chunks") so CLIP embeddings (512-dim)
don't mix with OpenAI text embeddings (1536-dim).

Usage:
    ingest_frames_clip(video_path)   # called from /ingest/visual
    query_visual(question, top_k)    # called from /query/visual
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings

CHROMA_DIR       = "storage/chromadb"
CLIP_COLLECTION  = "visual_chunks"
CLIP_MODEL_NAME  = "clip-ViT-B-32"


# ---------------------------------------------------------------------------
# Lazy-loaded singleton so the 340 MB model loads once per process
# ---------------------------------------------------------------------------
_clip_model: Optional[object] = None

def _get_clip():
    global _clip_model
    if _clip_model is None:
        from sentence_transformers import SentenceTransformer
        _clip_model = SentenceTransformer(CLIP_MODEL_NAME)
    return _clip_model


class _CLIPEmbedFn(EmbeddingFunction):
    """Chroma-compatible embedding function backed by sentence-transformers CLIP."""

    def __call__(self, input: Documents) -> Embeddings:  # type: ignore[override]
        model = _get_clip()
        return model.encode(list(input), convert_to_numpy=True).tolist()


def _collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(
        name=CLIP_COLLECTION,
        embedding_function=_CLIPEmbedFn(),
        metadata={"hnsw:space": "cosine"},
    )


def _extract_frames(video_path: str, frame_dir: str, interval: int) -> list[tuple[float, str]]:
    ffmpeg = os.getenv("FFMPEG_PATH", "ffmpeg")
    subprocess.run(
        [
            ffmpeg, "-i", video_path,
            "-vf", f"fps=1/{interval}",
            "-q:v", "5",
            str(Path(frame_dir) / "f%05d.jpg"),
            "-y",
        ],
        check=True, capture_output=True,
    )
    paths = sorted(Path(frame_dir).glob("f*.jpg"))
    return [(i * interval, str(p)) for i, p in enumerate(paths)]


def ingest_frames_clip(video_path: str) -> int:
    """
    Extract frames from video, embed each with CLIP, store in visual_chunks.
    Returns number of frames stored.
    """
    from PIL import Image

    source         = Path(video_path).name
    frame_interval = int(os.getenv("FRAME_INTERVAL", 5))
    model          = _get_clip()
    col            = _collection()

    with tempfile.TemporaryDirectory() as tmp:
        frame_dir = str(Path(tmp) / "frames")
        Path(frame_dir).mkdir()
        frames = _extract_frames(video_path, frame_dir, frame_interval)

        ids, embeddings, docs, metas = [], [], [], []
        for i, (ts, jpeg) in enumerate(frames):
            img = Image.open(jpeg)
            emb = model.encode(img, convert_to_numpy=True).tolist()
            m, s = divmod(int(ts), 60)
            label = f"{m:02d}:{s:02d}"

            ids.append(f"{source}_f{i:05d}")
            embeddings.append(emb)
            docs.append(f"[Frame from {source} at {label}]")
            metas.append({
                "source":          source,
                "start":           float(ts),
                "end":             float(ts + frame_interval),
                "timestamp_label": label,
            })

        if ids:
            col.add(ids=ids, embeddings=embeddings, documents=docs, metadatas=metas)

    return len(ids)


def query_visual(question: str, top_k: int = 3) -> list[dict]:
    """
    Embed `question` with CLIP, search visual_chunks.
    Returns [{source, text, start, end, timestamp_label, score}]
    """
    model    = _get_clip()
    q_emb    = model.encode(question, convert_to_numpy=True).tolist()
    col      = _collection()

    results  = col.query(
        query_embeddings=[q_emb],
        n_results=min(top_k, col.count() or 1),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for i, doc in enumerate(results["documents"][0]):
        meta = results["metadatas"][0][i]
        dist = results["distances"][0][i]
        chunks.append({
            "source":          meta.get("source", "unknown"),
            "text":            doc,
            "start":           meta.get("start", 0.0),
            "end":             meta.get("end", 0.0),
            "timestamp_label": meta.get("timestamp_label"),
            "score":           round(1 - dist, 4),
        })
    return chunks
