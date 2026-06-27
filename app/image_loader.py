"""
Image loader — multimodal ingest.

Upload an image -> GPT-4o (vision) writes a detailed description -> stored as a
Chroma chunk with the SAME shape as audio/video chunks
({source, text, start, end, timestamp_label}), so it flows through retrieval,
/query, and the agents unchanged. Images have no timeline, so start/end are 0
and timestamp_label is empty.

Reuses the vision approach from app/video_loader._describe_frame, but asks for a
fuller description (objects, scene, AND any visible text / OCR) so the image is
searchable by its content.
"""

import base64
import os
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}

_PROMPT = (
    "Describe this image in detail for a search index. Cover: main subjects/objects, "
    "the scene or setting, notable colors or layout, and TRANSCRIBE any visible text, "
    "labels, numbers, or diagram content verbatim. Write a single dense paragraph."
)


def load_image(filepath: str) -> list[dict]:
    """Describe an image with GPT-4o vision; return one media-style chunk."""
    from openai import AzureOpenAI

    source = Path(filepath).name
    with open(filepath, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    suffix = Path(filepath).suffix.lower().lstrip(".")
    mime = "jpeg" if suffix in ("jpg", "jpeg") else suffix

    client = AzureOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version=os.environ["AZURE_API_VERSION"],
    )

    resp = client.chat.completions.create(
        model=os.environ.get("CHAT_MODEL", "gpt-4o"),
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/{mime};base64,{b64}", "detail": "high"}},
                {"type": "text", "text": _PROMPT},
            ],
        }],
        max_completion_tokens=500,
    )
    desc = (resp.choices[0].message.content or "").strip()
    if not desc:
        return []

    return [{
        "source": source,
        "text": f"[Image: {source}] {desc}",
        "start": 0.0,
        "end": 0.0,
        "timestamp_label": "",
    }]
