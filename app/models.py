from pydantic import BaseModel
from typing import Optional


class IngestResponse(BaseModel):
    filename: str
    chunks_created: int
    message: str


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3


class SourceChunk(BaseModel):
    text: str
    source: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    prompt_tokens: int
    completion_tokens: int
