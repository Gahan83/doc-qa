"""
Phase 9 — Voice I/O.

Speech-in  : audio file → Whisper (raw transcription, no ingestion)
Speech-out : text → Azure OpenAI TTS → MP3 bytes

The agent brain (app/agent.py) stays untouched; voice is just I/O glue:
  record audio  →  POST /voice/transcribe  →  POST /agent  →  POST /voice/speak
"""

import os
from pathlib import Path


def transcribe_raw(filepath: str) -> dict:
    """
    Transcribe an audio file without storing anything.
    Returns {text, language, duration, segments: [{start, end, text}]}
    """
    from openai import AzureOpenAI

    client = AzureOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version=os.environ["AZURE_API_VERSION"],
    )
    model = os.environ["WHISPER_MODEL"]

    with open(filepath, "rb") as f:
        result = client.audio.transcriptions.create(
            model=model,
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )

    def _get(seg, field):
        return seg[field] if isinstance(seg, dict) else getattr(seg, field)

    return {
        "text":     result.text or "",
        "language": result.language or "",
        "duration": result.duration or 0.0,
        "segments": [
            {
                "start": _get(s, "start"),
                "end":   _get(s, "end"),
                "text":  _get(s, "text").strip(),
            }
            for s in (result.segments or [])
        ],
    }


def synthesize(text: str, voice: str = "alloy") -> bytes:
    """
    Convert text to MP3 audio via Azure OpenAI TTS.
    Supported voices: alloy, echo, fable, onyx, nova, shimmer.
    Returns raw MP3 bytes.

    TTS may live on a different Azure resource than the main one.
    Set AZURE_TTS_ENDPOINT + AZURE_TTS_KEY in .env if so (falls back to main resource).
    """
    from openai import AzureOpenAI

    client = AzureOpenAI(
        api_key=os.environ.get("AZURE_TTS_KEY", os.environ["OPENAI_API_KEY"]),
        azure_endpoint=os.environ.get("AZURE_TTS_ENDPOINT", os.environ["AZURE_OPENAI_ENDPOINT"]),
        api_version=os.environ["AZURE_API_VERSION"],
    )
    tts_model = os.getenv("TTS_MODEL", "tts")

    response = client.audio.speech.create(
        model=tts_model,
        voice=voice,
        input=text,
        response_format="mp3",
    )
    return response.content
