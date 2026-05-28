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
│   └── models.py        # Pydantic request/response schemas
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

## Usage

### Interactive Docs

Open **http://127.0.0.1:8000/docs** for the Swagger UI.

```

## How It Works

1. **Ingestion** — Files are read (PDF text extraction via PyPDF2), split into overlapping chunks (~500 words), embedded using Azure OpenAI's embedding model, and stored in `storage/embeddings.json`.

2. **Retrieval** — The user's question is embedded with the same model. Cosine similarity is computed against all stored chunk embeddings using numpy, and the top-K most relevant chunks are returned.

3. **Generation** — The top-K chunks are injected into a structured prompt with a system message constraining GPT to answer only from the provided context. The response includes token usage for cost tracking.

## Key Concepts Demonstrated

- **RAG pipeline** end-to-end without frameworks
- **Chunking strategies** with overlap for context preservation
- **Embedding** and **cosine similarity** math from scratch
- **Prompt engineering** — system prompt, context injection, grounding
- **Token tracking** for cost awareness
