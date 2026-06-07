"""
Audio loader — Phase 2 + Phase 5.

Phase 2: single file → Whisper → timestamp chunks
Phase 5: files > 20 MB → ffmpeg-sliced into 10-min segments → stitch timestamps

Public API:
    load_audio(filepath) -> list[dict]
        Each dict: {source, text, start, end, timestamp_label}
        Same shape the rest of the app gets from PDFs, plus timestamps.
"""

import os
import subprocess
import tempfile
from pathlib import Path

WHISPER_MAX_BYTES = 20 * 1024 * 1024  # 20 MB safety margin (Whisper hard limit is 25 MB)
SLICE_SECONDS = 600                    # 10 minutes per slice
WORDS_PER_CHUNK = 150                  # ~60 seconds of speech


def _ffmpeg() -> str:
    return os.getenv("FFMPEG_PATH", "ffmpeg")


def _fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _get_seg(seg, field: str):
    """Segment may be dict or object depending on SDK version."""
    return seg[field] if isinstance(seg, dict) else getattr(seg, field)


def _to_wav(src: str, dst: str) -> None:
    """Convert any audio to 16 kHz mono WAV — optimal for Whisper."""
    subprocess.run(
        [_ffmpeg(), "-i", src, "-ar", "16000", "-ac", "1", dst, "-y"],
        check=True, capture_output=True,
    )


def _duration(filepath: str) -> float:
    """Get media duration in seconds via ffmpeg."""
    r = subprocess.run(
        [_ffmpeg(), "-i", filepath, "-f", "null", "-"],
        capture_output=True, text=True,
    )
    for line in r.stderr.splitlines():
        if "Duration:" in line:
            t = line.split("Duration:")[1].split(",")[0].strip()
            h, m, s = t.split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)
    return 0.0


def _transcribe_file(client, model: str, wav_path: str) -> list[dict]:
    """Whisper API call → raw segment list [{start, end, text}]."""
    with open(wav_path, "rb") as f:
        result = client.audio.transcriptions.create(
            model=model,
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )
    return [
        {
            "start": _get_seg(s, "start"),
            "end":   _get_seg(s, "end"),
            "text":  _get_seg(s, "text").strip(),
        }
        for s in (result.segments or [])
    ]


def _transcribe_long(client, model: str, wav_path: str) -> list[dict]:
    """
    Phase 5: slice into SLICE_SECONDS-length WAV files, transcribe each,
    offset timestamps so the stitched result has absolute times.
    """
    total = _duration(wav_path)
    all_segs: list[dict] = []
    offset = 0.0
    slice_n = 0

    with tempfile.TemporaryDirectory() as tmp:
        while offset < total:
            slice_path = str(Path(tmp) / f"s{slice_n:04d}.wav")
            subprocess.run(
                [
                    _ffmpeg(), "-i", wav_path,
                    "-ss", str(offset),
                    "-t",  str(SLICE_SECONDS),
                    "-ar", "16000", "-ac", "1",
                    slice_path, "-y",
                ],
                check=True, capture_output=True,
            )
            for seg in _transcribe_file(client, model, slice_path):
                all_segs.append({
                    "start": seg["start"] + offset,
                    "end":   seg["end"]   + offset,
                    "text":  seg["text"],
                })
            offset += SLICE_SECONDS
            slice_n += 1

    return all_segs


def _segments_to_chunks(segments: list[dict], source: str) -> list[dict]:
    """
    Group raw segments into ~WORDS_PER_CHUNK-word chunks.
    Preserves start/end of the group so timestamps stay accurate.
    """
    chunks: list[dict] = []
    buf_texts: list[str] = []
    buf_start: float | None = None
    buf_end:   float | None = None
    word_count = 0

    def _flush():
        nonlocal buf_texts, buf_start, buf_end, word_count
        if not buf_texts:
            return
        ts = buf_start or 0.0
        chunks.append({
            "source":          source,
            "text":            " ".join(buf_texts),
            "start":           ts,
            "end":             buf_end or 0.0,
            "timestamp_label": _fmt(ts),
        })
        buf_texts, buf_start, buf_end, word_count = [], None, None, 0

    for seg in segments:
        if buf_start is None:
            buf_start = seg["start"]
        buf_texts.append(seg["text"])
        buf_end = seg["end"]
        word_count += len(seg["text"].split())
        if word_count >= WORDS_PER_CHUNK:
            _flush()

    _flush()
    return chunks


def load_audio(filepath: str) -> list[dict]:
    """
    Main entry point. Works for any ffmpeg-readable audio format.
    Returns chunks: [{source, text, start, end, timestamp_label}]
    """
    from openai import AzureOpenAI

    client = AzureOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version=os.environ["AZURE_API_VERSION"],
    )
    model  = os.environ["WHISPER_MODEL"]
    source = Path(filepath).name

    with tempfile.TemporaryDirectory() as tmp:
        wav = str(Path(tmp) / "audio.wav")
        _to_wav(filepath, wav)

        if os.path.getsize(wav) > WHISPER_MAX_BYTES:
            segments = _transcribe_long(client, model, wav)
        else:
            segments = _transcribe_file(client, model, wav)

    return _segments_to_chunks(segments, source)
