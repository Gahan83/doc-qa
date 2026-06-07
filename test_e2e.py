"""
End-to-end smoke test for all deployed models.
Run after deploying whisper + tts.

Steps:
  1. TTS: synthesize a sentence → save as test_speech.mp3
  2. Whisper: transcribe test_speech.mp3 → print segments
  3. Ingest: push test_speech.mp3 through the full pipeline
  4. Query: ask about the content → should cite timestamp
  5. Voice speak: synthesize the answer → save as answer.mp3
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

AUDIO_DIR = Path("data/audio")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

SPEECH_FILE = str(AUDIO_DIR / "test_speech.mp3")
ANSWER_FILE = str(AUDIO_DIR / "answer.mp3")

TEST_TEXT = (
    "Welcome to the doc Q&A system. "
    "This system can answer questions about documents, audio files, and videos. "
    "It supports timestamps so you can jump directly to the relevant moment."
)
TEST_QUESTION = "What does this system support?"


def step(n, label):
    print(f"\n{'='*60}")
    print(f"  Step {n}: {label}")
    print('='*60)


def test_tts():
    step(1, "TTS — synthesize test sentence")
    from app.voice import synthesize
    mp3 = synthesize(TEST_TEXT, voice="alloy")
    with open(SPEECH_FILE, "wb") as f:
        f.write(mp3)
    size_kb = len(mp3) // 1024
    print(f"  Saved {size_kb} KB → {SPEECH_FILE}")


def test_whisper():
    step(2, "Whisper — transcribe test_speech.mp3")
    import sys
    sys.argv = ["", SPEECH_FILE]
    import importlib
    tw = importlib.import_module("test_whisper")
    tw.transcribe(SPEECH_FILE)


def test_ingest():
    step(3, "Ingest — push audio through pipeline into Chroma")
    from app.ingestion import ingest_file
    n = ingest_file(SPEECH_FILE)
    print(f"  Created {n} chunks in Chroma")


def test_query():
    step(4, "Query — RAG over ingested audio (expect timestamp citation)")
    from app.retrieval import retrieve
    from app.generation import generate
    chunks = retrieve(TEST_QUESTION, top_k=3)
    print(f"  Retrieved {len(chunks)} chunks")
    for c in chunks:
        ts = c.get("timestamp_label", "")
        print(f"    [{ts or 'no-ts'}] score={c['score']}  {c['text'][:60]}...")
    result = generate(TEST_QUESTION, chunks)
    print(f"\n  Answer:\n  {result['answer']}")
    return result["answer"]


def test_speak(answer: str):
    step(5, "Voice speak — TTS the answer → answer.mp3")
    from app.voice import synthesize
    mp3 = synthesize(answer, voice="nova")
    with open(ANSWER_FILE, "wb") as f:
        f.write(mp3)
    print(f"  Saved → {ANSWER_FILE}")


if __name__ == "__main__":
    test_tts()
    test_whisper()
    test_ingest()
    answer = test_query()
    test_speak(answer)
    print("\n\nAll steps passed. Open data/audio/answer.mp3 to hear the answer.")
