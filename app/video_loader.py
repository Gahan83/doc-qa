"""
Video loader — Phase 6 + Phase 7.

Phase 6: rip audio track → hand to audio_loader → Whisper transcript chunks
Phase 7: extract frames every N seconds → GPT-4o vision → description chunks

Both sets of chunks land in Chroma with the same shape:
  {source, text, start, end, timestamp_label}
"""

import base64
import os
import subprocess
import tempfile
from pathlib import Path

from app.audio_loader import load_audio


def _ffmpeg() -> str:
    return os.getenv("FFMPEG_PATH", "ffmpeg")


def _fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _extract_audio(video_path: str, out_wav: str) -> None:
    """Rip audio track from video to 16 kHz mono WAV."""
    subprocess.run(
        [
            _ffmpeg(), "-i", video_path,
            "-vn",                     # no video
            "-ar", "16000", "-ac", "1",
            out_wav, "-y",
        ],
        check=True, capture_output=True,
    )


def _extract_frames(video_path: str, frame_dir: str, interval: int) -> list[tuple[float, str]]:
    """
    Dump 1 frame every `interval` seconds as JPEG.
    Returns [(timestamp_seconds, jpeg_path), ...]
    """
    subprocess.run(
        [
            _ffmpeg(), "-i", video_path,
            "-vf", f"fps=1/{interval}",
            "-q:v", "5",
            str(Path(frame_dir) / "f%05d.jpg"),
            "-y",
        ],
        check=True, capture_output=True,
    )
    paths = sorted(Path(frame_dir).glob("f*.jpg"))
    return [(i * interval, str(p)) for i, p in enumerate(paths)]


def _describe_frame(client, jpeg_path: str, label: str) -> str:
    """Ask GPT-4o to describe a single video frame. Returns one sentence."""
    with open(jpeg_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    resp = client.chat.completions.create(
        model=os.environ.get("CHAT_MODEL", "gpt-4o"),
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url":    f"data:image/jpeg;base64,{b64}",
                            "detail": "low",
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            f"Describe what is on screen at {label} in one concise sentence. "
                            "Focus on visible text, diagrams, slides, or key visual content."
                        ),
                    },
                ],
            }
        ],
        max_tokens=120,
    )
    return resp.choices[0].message.content.strip()


def load_video(filepath: str, describe_frames: bool = True) -> list[dict]:
    """
    Phase 6 + 7: transcribe audio and (optionally) describe frames.
    Returns all chunks with {source, text, start, end, timestamp_label}.
    """
    from openai import AzureOpenAI

    source         = Path(filepath).name
    frame_interval = int(os.getenv("FRAME_INTERVAL", 5))

    with tempfile.TemporaryDirectory() as tmp:
        # Phase 6 — audio
        wav = str(Path(tmp) / "track.wav")
        _extract_audio(filepath, wav)

        audio_chunks = load_audio(wav)
        for c in audio_chunks:
            c["source"] = source  # fix: load_audio names source after the tmp wav

        if not describe_frames:
            return audio_chunks

        # Phase 7 — frames
        client = AzureOpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=os.environ["AZURE_API_VERSION"],
        )

        frame_dir = str(Path(tmp) / "frames")
        Path(frame_dir).mkdir()
        frames = _extract_frames(filepath, frame_dir, frame_interval)

        visual_chunks: list[dict] = []
        for ts, jpeg in frames:
            label = _fmt(ts)
            try:
                desc = _describe_frame(client, jpeg, label)
                visual_chunks.append({
                    "source":          source,
                    "text":            f"[Visual at {label}] {desc}",
                    "start":           float(ts),
                    "end":             float(ts + frame_interval),
                    "timestamp_label": label,
                })
            except Exception:
                pass  # single frame failure doesn't abort the whole video

    return audio_chunks + visual_chunks
