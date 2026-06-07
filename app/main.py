"""
FastAPI entrypoint — Phases 1-9.

App creation, logging, middleware. All routes live in app/endpoints.py.
"""

import logging
import sys

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.requests import Request

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("doc-qa")

from app.endpoints import router

app = FastAPI(
    title="Doc Q&A — Phases 1-9",
    description="RAG over docs, audio, and video with timestamp citations and voice I/O",
    version="9.0.0",
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"→ {request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(f"← {response.status_code} {request.url.path}")
    return response


app.include_router(router)
