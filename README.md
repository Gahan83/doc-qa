# Doc Q&A — RAG from Scratch

A Retrieval-Augmented Generation (RAG) application built from scratch with **FastAPI** and **Azure OpenAI**. No LangChain, no vector databases — pure Python so you can see every step of the pipeline.

## Architecture

```
Upload (.txt/.pdf) → Chunk text → Embed (Azure OpenAI) → Store (JSON)
Ask question → Embed query → Cosine similarity (numpy) → Top-K chunks → GPT → Answer
```

## Project Structure

```
doc-qa/
├── app/
│   ├── main.py          # FastAPI entrypoint with /ingest and /query endpoints
│   ├── ingestion.py     # Read → chunk → embed → store pipeline
│   ├── retrieval.py     # Embed query → cosine similarity → top-K results
│   ├── generation.py    # Build RAG prompt → call GPT → return answer
│   ├── models.py        # Pydantic request/response schemas
│   ├── tools.py         # NEW: function calling tool definitions
│   ├── agent.py         # NEW: ReAct agent loop
│   ├── structured.py    # NEW: structured JSON outputs
│   └── evaluation.py    # NEW: LLM-as-judge scoring
├── storage/             # Embeddings JSON store (created at runtime)
├── data/docs/           # Uploaded documents (created at runtime)
├── .env                 # Azure OpenAI credentials and config
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

### Prerequisites

- Python 3.10+
- Azure OpenAI resource with:
  - An embedding model deployment (`text-embedding-3-small`)
  - A chat model deployment (`gpt-4o-mini`)

### Installation

```bash
# Clone the repo
git clone <repo-url>
cd doc-qa

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=<your-azure-openai-key>
AZURE_OPENAI_ENDPOINT=<your-azure-endpoint>
AZURE_API_VERSION=2024-12-01-preview
EMBED_MODEL=text-embedding-3-small
CHAT_MODEL=emi-gpt-4o-mini
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K=3
```

## Running the Application

```bash
uvicorn app.main:app --reload
```

Server starts at **http://127.0.0.1:8000**

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET    | `/`      | Health check |
| POST   | `/ingest`| Upload a .txt or .pdf file |
| POST   | `/query` | Ask a question about ingested documents |

## Phase 2: Agents, Function Calling, Structured Outputs, Evaluation

### New Architecture

```
User Question
   ↓
Agent (LLM decides what to do)
   ↓
[Tools: search_documents, list_documents, get_document_summary]
   ↓
LLM can loop, call tools, and combine results
   ↓
Structured output or answer
   ↓
Optional: LLM-as-judge evaluation
```

### New API Endpoints

| Method | Endpoint             | Description |
|--------|----------------------|-------------|
| POST   | `/agent`             | Agent with function calling and multi-step reasoning |
| POST   | `/query/structured`  | Query with strict JSON output (answer, confidence, sources, follow-ups) |
| POST   | `/evaluate`          | LLM-as-judge: score answer for faithfulness, relevance, completeness |

### Example: /evaluate

Request body:
```json
{
  "question": "What is Azure Cognitive Services?",
  "answer": "Azure Cognitive Services is a collection of AI services and APIs that help developers build intelligent applications without needing direct AI or data science skills.",
  "context_chunks": [
    {
      "source": "AI-102 Exam - Free Actual Q&As, Page 1 _ ExamTopics_wd-compressed (1).pdf",
      "text": "Azure Cognitive Services are cloud-based artificial intelligence (AI) services that help developers build cognitive intelligence into applications without having direct AI or data science skills or knowledge."
    },
    {
      "source": "AI-102 Exam - Free Actual Q&As, Page 1 _ ExamTopics_wd-compressed (1).pdf",
      "text": "Cognitive Services provides machine learning capabilities to solve general problems such as analyzing text for sentiment or analyzing images to recognize objects."
    }
  ]
}
```

Response:
```json
{
  "faithfulness": { "score": 5, "reason": "Answer directly paraphrases the context" },
  "relevance": { "score": 5, "reason": "Directly answers what Azure Cognitive Services is" },
  "completeness": { "score": 3, "reason": "Misses the machine learning capabilities and image/text analysis details" },
  "overall_score": 4.3,
  "summary": "Accurate but incomplete — covers the definition but omits specific capabilities mentioned in context.",
  "eval_tokens": { "prompt_tokens": 312, "completion_tokens": 89 }
}
```

## Key Concepts Demonstrated

- **RAG pipeline** end-to-end without frameworks
- **Chunking strategies** with overlap for context preservation
- **Embedding** and **cosine similarity** math from scratch
- **Prompt engineering** — system prompt, context injection, grounding
- **Token tracking** for cost awareness

## Key Concepts Demonstrated (Phase 2)

- **Function calling**: LLM can invoke Python tools for search, listing, and summarization
- **Agent loop**: LLM can reason, call tools, observe, and iterate until it has an answer
- **Structured outputs**: Force LLM to return JSON with answer, confidence, sources, follow-ups
- **LLM-as-judge**: Automated answer evaluation for faithfulness, relevance, completeness
