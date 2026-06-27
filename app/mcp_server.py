"""
MCP server for doc-qa — exposes document search over the Model Context Protocol.

Lets an MCP client (e.g. Claude Desktop) query this project's ingested document
corpus directly, with no HTTP layer. Transport is stdio: the client launches this
process and talks to it over stdin/stdout.

Tools:
  * search_documents(query, top_k) — semantic search over ingested chunks.
  * list_documents()               — list ingested files + chunk counts.

Reuses the same retrieval/store logic as the REST API and agents
(app/retrieval.retrieve, app/ingestion.load_store), so MCP results match /query.

Run standalone:   python -m app.mcp_server
Wire into Claude Desktop: see the "MCP Server" section in README.md.
"""

import json
import logging

from dotenv import load_dotenv

# Vector store + embeddings read Azure creds from .env — load before importing
# anything that builds the Chroma client.
load_dotenv()

from mcp.server.fastmcp import FastMCP

from app.ingestion import load_store
from app.retrieval import retrieve

# stdio servers MUST NOT write logs to stdout (it carries the MCP protocol).
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("doc-qa.mcp")

mcp = FastMCP("doc-qa")

# NOTE: tool functions intentionally have NO return-type annotation. With this
# project's pinned pydantic==2.9.2, FastMCP's structured-output schema builder
# (create_model(..., result=<type>)) raises PydanticUserError. Omitting the
# annotation skips that path; results are returned as plain JSON text content,
# which Claude Desktop reads fine. Do not add `-> ...` back without bumping pydantic.


@mcp.tool()
def search_documents(query: str, top_k: int = 3):
    """Semantic search over the ingested document corpus.

    Use this to answer questions about the user's documents. Returns the most
    relevant chunks with their source filename and a similarity score (0-1),
    as a JSON array.

    Args:
        query: Natural-language search query.
        top_k: Number of chunks to return (1-10). Default 3.
    """
    top_k = max(1, min(top_k, 10))
    results = retrieve(query, top_k=top_k)
    payload = [
        {
            "source": r["source"],
            "text": r["text"],
            "score": r["score"],
            "timestamp": r.get("timestamp_label"),
        }
        for r in results
    ]
    return json.dumps(payload, indent=2)


@mcp.tool()
def list_documents():
    """List all ingested documents and how many chunks each contributed.

    Use this to see what source material is available before searching.
    Returns a JSON array of {filename, chunk_count}.
    """
    store = load_store()
    stats: dict[str, dict] = {}
    for item in store:
        src = item["source"]
        stats.setdefault(src, {"filename": src, "chunk_count": 0})
        stats[src]["chunk_count"] += 1
    return json.dumps(list(stats.values()), indent=2)


def main() -> None:
    logger.info("Starting doc-qa MCP server (stdio)")
    mcp.run()  # stdio transport by default


if __name__ == "__main__":
    main()
