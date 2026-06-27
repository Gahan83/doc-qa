# Doc Q&A — RAG over Documents, Audio & Video

A Retrieval-Augmented Generation (RAG) application built with **FastAPI**, **LangChain**, **Azure OpenAI**, and **ChromaDB**. Ingests PDFs, text, audio, and video. Answers cite timestamps. Talks and listens.

## Architecture

### Ingest pipeline (shared by both flows)

```
  PDF / TXT ────────────► LangChain loaders → text splitter      ┐
  MP3 / WAV / M4A ──────► ffmpeg → Whisper → timestamp chunks    │
  MP4 / MOV / MKV ──────► ffmpeg audio  → Whisper                ├─► embed (Azure OpenAI) ─► ChromaDB
                          ffmpeg frames → GPT-4o vision          │
                          ffmpeg frames → CLIP embeddings        ┘
```

### Normal flow — `POST /query` (deterministic RAG)

```
  Question
     │
     ▼
  retrieve(top_k)  ───►  Chroma similarity search
     │
     ▼
  format context (with timestamps)
     │
     ▼
  GPT (single call) ───► answer citing "file.mp3 @ 03:42"
     │
     ▼
  QueryResponse { answer, sources, tokens }
```

Properties: one LLM call, no tools, no looping. Fast, cheap, predictable.
Use for direct factual Q&A over indexed content.

### Agentic flow — `POST /agent` (decision loop)

```
  Question  +  optional session_id
       │
       ▼
  load short-term memory (recent turns)            ◄── app/memory_store.py
       │
       ▼
  ┌─────────────────────────  agent loop  ────────────────────────────┐
  │                                                                   │
  │   LLM (bind_tools)  ──► tool_calls?                               │
  │       │                    │                                      │
  │       │ no                 │ yes                                  │
  │       ▼                    ▼                                      │
  │   final answer       for each tool_call:                          │
  │                          ├─ plan_step          (first, if needed) │
  │                          ├─ search_documents  / list / summary    │
  │                          ├─ remember / recall  (long-term facts)  │
  │                          │                                        │
  │                          ▼                                        │
  │                     invoke_with_retry                             │
  │                       │  envelope { status, retryable, ... }      │
  │                       │   ├─ ok       ─► append ToolMessage       │
  │                       │   ├─ retry    ─► backoff, try again       │
  │                       │   └─ fail     ─► tell model, fallback     │
  │                       ▼                                           │
  │                  step trace (latency, retries, tokens)            │
  │                                                                   │
  │   safety rails: MAX_ITERATIONS / MAX_TOOL_CALLS / MAX_WALL_SECONDS│
  │   on cap hit ─► safe-stop response (partial + stop_reason)        │
  └───────────────────────────────────────────────────────────────────┘
       │
       ▼
  AgentResponse { answer, plan, steps[], trace_id, session_id,
                  stop_reason, elapsed_ms, tool_calls, tokens }
       │
       ▼
  append (user, assistant) turns to short-term memory
```

Properties: planning, tool selection, retries, memory, guardrails, full trace.
Use for ambiguous or multi-step requests where the right tool sequence isn't obvious.

### Voice loop

```
  Mic ─► /voice/transcribe (Whisper) ─► /query OR /agent ─► /voice/speak (TTS) ─► Speaker
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
│   ├── agent.py          # Agentic loop: plan → tools → retry → trace → safe-stop
│   ├── tools.py          # @tool definitions with standardized result envelope
│   ├── memory_store.py   # Short-term session buffer + long-term JSON facts
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

## Manual Testing (No Test Files)

Use these steps to validate the app end-to-end without creating any Python test scripts.

### A) Swagger flow (fastest)

1. Start the API:
```bash
uvicorn app.main:app --reload --port 8000
```
2. Open Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

3. Test normal RAG:
    - `POST /ingest` with one file (PDF/TXT/MP3/MP4)
    - `POST /query` with:
```json
{
   "question": "What is this file about?",
   "top_k": 3
}
```

4. Test agentic flow:
    - `POST /agent` with:
```json
{
   "question": "List documents, then summarize the most relevant one."
}
```
    - Confirm response includes `plan`, `steps`, `trace_id`, `stop_reason`.
    
5. Test session memory:
    - Reuse returned `session_id` in a second call:
```json
{
   "question": "Now give me just 3 bullet points.",
   "session_id": "<paste-session-id>"
}
```

### b) Voice path check

1. `POST /voice/transcribe` with an audio file.
2. Send returned text to `/query` or `/agent`.
3. `POST /voice/speak` with:
```json
{
   "text": "Testing speech output",
   "voice": "alloy"
}
```
4. Expect MP3 audio bytes in the response.

### Expected pass criteria

1. `/query` returns `answer` + `sources`.
2. `/agent` returns `answer` + `plan` + `steps` + `trace_id`.
3. Reusing `session_id` changes follow-up behavior (short-term memory works).
4. `/voice/transcribe` returns segment timestamps.
5. `/voice/speak` returns `audio/mpeg` bytes.

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
| POST   | `/agent`              | Agentic loop: plans, calls tools, retries, traces, safe-stops |
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
- **Agentic loop** — planning, memory, retry, guardrails, observability (see below)
- **LLM-as-judge** — structured evaluation of faithfulness, relevance, completeness
- **Voice loop** — mic audio → Whisper → agent → TTS → speaker; same brain, new senses

## Agentic Loop

The `/agent` endpoint is not a single prompt — it is a constrained decision loop with five capabilities layered on top of `bind_tools()`:

1. **Planning** — `plan_step` is the first tool the model is told to call for any non-trivial question. It records `goal`, `sub_steps`, `tool_sequence`, and `stop_condition`. The plan is returned in the response so you can inspect what the agent intended to do.
2. **Memory** — short-term per-session conversation turns are auto-injected into the system prompt. Long-term facts are saved via `remember` and retrieved via `recall`, persisted in `storage/agent_memory.json`.
3. **Retry/recovery** — every tool returns a standardized envelope `{status, error_type, retryable, payload}`. The loop retries `retryable=true` errors with backoff, then surfaces the failure to the model so it can pick a fallback or ask a clarifying question.
4. **Guardrails** — hard caps: `MAX_ITERATIONS=6`, `MAX_TOOL_CALLS=12`, `MAX_WALL_SECONDS=45`, `TOOL_MAX_RETRIES=2`. If a cap is hit, the agent returns a safe-stop response with `stop_reason` and a next best action.
5. **Observability** — every run gets a `trace_id`. Each step logs `tool`, `status`, `error_type`, `retries`, `latency_ms`, and token usage. Aggregate `elapsed_ms`, `tool_calls`, and `stop_reason` are returned to the client.

### Agent tools

| Tool                  | Purpose                                                    |
|-----------------------|------------------------------------------------------------|
| `plan_step`           | Commit to a plan before acting                             |
| `search_documents`    | Semantic search over ingested content                      |
| `list_documents`      | Enumerate ingested sources                                 |
| `get_document_summary`| Preview a specific document                                |
| `remember`            | Persist a compact fact for future runs                     |
| `recall`              | Retrieve previously stored facts by keyword                |

## MCP Server

Exposes the document corpus to MCP clients (e.g. Claude Desktop) via stdio — no HTTP layer. Defined in `app/mcp_server.py`.

| Tool | Purpose |
|------|---------|
| `search_documents(query, top_k=3)` | Semantic search over ingested chunks → JSON `{source, text, score, timestamp}` |
| `list_documents()`                  | List ingested files → JSON `{filename, chunk_count}` |

Reuses the same retrieval/store logic as the REST API (`app/retrieval.retrieve`, `app/ingestion.load_store`), so MCP results match `POST /query`. Transport is **stdio**: the client launches `python -m app.mcp_server` and talks over stdin/stdout.

### MCP prerequisites
- `pip install -r requirements.txt` (includes `mcp`).
- `.env` with Azure OpenAI creds (embeddings power search).
- Documents already ingested (reuses the existing Chroma store).

### Connect Claude Desktop
1. Open (create if missing) the config file:
   - **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`
   - **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
2. Merge in the `doc-qa` block (adjust `command` to your `python.exe` and `cwd` to the project root):
```json
{
  "mcpServers": {
    "doc-qa": {
      "command": "C:\\Users\\Gahan.K\\AppData\\Local\\Programs\\Python\\Python312\\python.exe",
      "args": ["-m", "app.mcp_server"],
      "cwd": "c:\\Gahan\\Practice Projects for Work\\doc-qa"
    }
  }
}
```
3. **Fully quit and reopen** Claude Desktop (not just close the window).
4. New chat → tools icon lists `search_documents` and `list_documents` under `doc-qa`.

### Try it in Claude Desktop
- "What documents do you have access to?" → calls `list_documents`.
- "Search my documents: how tall is the Eiffel Tower?" → calls `search_documents`.

### Verify without Claude Desktop (MCP Inspector)
Launch **from the project root** so `app` is importable and the relative `storage/chromadb` path resolves:
```bash
cd "c:\Gahan\Practice Projects for Work\doc-qa"
npx @modelcontextprotocol/inspector python -m app.mcp_server
```
In the Inspector: Transport Type **STDIO** (pre-filled) → **Connect** → Tools tab → run `list_documents` / `search_documents`. There is **no URL** — stdio, not HTTP.

> **Note:** MCP tool functions in `app/mcp_server.py` intentionally have no return-type annotation — FastMCP's structured-output schema builder is incompatible with the pinned `pydantic==2.9.2`. Results return as JSON text content, which Claude reads fine.


