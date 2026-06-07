# Doc Q&A — RAG over Documents, Audio & Video

A Retrieval-Augmented Generation (RAG) application built with **FastAPI**, **LangChain**, **Azure OpenAI**, and **ChromaDB**. Ingests PDFs, text, audio, and video. Answers cite timestamps. Talks and listens.

## Architecture

```
                        ┌─────────────────────────────────────────┐
                        │              INGEST PIPELINE            │
                        │                                         │
  PDF / TXT ────────────► LangChain loaders → text splitter       │
  MP3 / WAV / M4A ──────► ffmpeg → Whisper → timestamp chunks     │
  MP4 / MOV / MKV ──────► ffmpeg audio → Whisper                  │
                        │ ffmpeg frames → GPT-4o vision           │
                        │ ffmpeg frames → CLIP embeddings         │
                        └──────────────┬──────────────────────────┘
                                       │ embed (Azure OpenAI)
                                       ▼
                                  ChromaDB
                                       │
                        ┌──────────────▼──────────────────────────┐
                        │              QUERY PIPELINE              │
  Question ─────────────► similarity search → top-K chunks        │
                        │ format context (with timestamps)         │
                        │ GPT → answer citing "file.mp3 @ 03:42"  │
                        └──────────────────────────────────────────┘

  Voice loop:
  Mic → /voice/transcribe (Whisper) → /agent → /voice/speak (TTS) → Speaker
```

## Project Structure

```
doc-qa/
├── app/
│   ├── main.py           # FastAPI app setup, middleware, router mounting
│   ├── endpoints.py      # APIRouter with all HTTP endpoints
│   ├── llm.py            # Azure LLM, embeddings, Chroma setup
│   ├── ingestion.py      # Routes PDF/TXT/audio/video to correct loader
│   ├── retrieval.py      # Chroma similarity search (returns timestamps)
│   ├── generation.py     # Prompt chain — cites "file @ MM:SS"
│   ├── models.py         # Pydantic schemas for all endpoints
│   ├── audio_loader.py   # Whisper transcription → timestamp chunks (Phase 2+5)
│   ├── video_loader.py   # Audio rip + GPT-4o frame descriptions (Phase 6+7)
│   ├── clip_store.py     # CLIP visual embeddings, separate collection (Phase 8)
│   ├── voice.py          # TTS synthesis + raw transcription (Phase 9)
│   ├── agent.py          # LangChain bind_tools() agent loop
│   ├── tools.py          # @tool definitions for the agent
│   ├── structured.py     # with_structured_output() responses
│   └── evaluation.py     # LLM-as-judge scoring
├── data/
│   ├── docs/             # Uploaded PDFs and text files
│   ├── audio/            # Uploaded audio files
│   └── video/            # Uploaded video files
├── storage/chromadb/     # Persistent vector store (text + visual collections)
├── test_whisper.py       # Phase 1 — one-shot Whisper proof script
├── test_e2e.py           # End-to-end TTS → Whisper → ingest → query → TTS
├── Dockerfile
├── .env
├── requirements.txt
└── README.md
```

## Setup

### Prerequisites

- Python 3.10+
- ffmpeg (installed at path set in `FFMPEG_PATH`)
- Azure OpenAI resource(s) with these deployments:

| Deployment name          | Model                   | Used for                    |
|--------------------------|-------------------------|-----------------------------|
| `text-embedding-3-small` | text-embedding-3-small  | Document + audio embeddings |
| `gpt-5.4-mini`           | gpt-5.4-mini            | Chat, agent, evaluation     |
| `whisper`                | whisper                 | Audio/video transcription   |
| `tts`                    | tts-1                   | Text-to-speech              |

> TTS may need a separate Azure region (Sweden Central has it; West Europe may not).

### Installation

```bash
git clone <repo-url>
cd doc-qa
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
Swagger UI: **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**
```

## API Endpoints

### Ingest

| Method | Endpoint         | Description                                                       |
|--------|------------------|-------------------------------------------------------------------|
| POST   | `/ingest`        | Upload PDF, TXT, MP3, WAV, M4A, OGG, FLAC, MP4, MOV, MKV, WEBM    |
| POST   | `/ingest/visual` | Embed video frames via CLIP into a separate Chroma collection     |

Audio/video ingest runs Whisper automatically. Video also runs GPT-4o frame descriptions if `VISUAL_MODE=describe`.

### Query

| Method | Endpoint              | Description                                              |
|--------|-----------------------|----------------------------------------------------------|
| POST   | `/query`              | RAG with timestamp citations for audio and video         |
| POST   | `/query/visual`       | CLIP frame search over ingested video (Phase 8)          |
| POST   | `/query/structured`   | Returns JSON with confidence and follow-up questions     |
| POST   | `/agent`              | Tool-calling agent, loops until it has an answer         |
| POST   | `/evaluate`           | LLM-as-judge: faithfulness, relevance, completeness      |

### Voice

| Method | Endpoint              | Description                                                    |
|--------|-----------------------|----------------------------------------------------------------|
| POST   | `/voice/transcribe`   | Upload audio to get timestamped transcript, no storage         |
| POST   | `/voice/speak`        | {text, voice} in request body, returns MP3 audio bytes         |



## Key Concepts

- **RAG pipeline** — LangChain + ChromaDB, timestamps flow through retrieval → prompt → answer
- **Whisper** — converts any audio/video to searchable text with segment-level timestamps
- **Long-file handling** — ffmpeg slices files >20 MB; timestamps stitched back as absolute
- **GPT-4o vision** — describes video frames; descriptions embedded and searchable
- **CLIP embeddings** — images and text in same vector space; no description step needed
- **Agent loop** — `bind_tools()` lets GPT decide which tools to call and loop until done
- **LLM-as-judge** — structured evaluation of faithfulness, relevance, completeness
- **Voice loop** — mic audio → Whisper → agent → TTS → speaker; same brain, new senses
