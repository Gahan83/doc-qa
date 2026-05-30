from pydantic import BaseModel
from typing import Optional


# --- Phase 1 Models ---

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


# --- Phase 2 Models ---

class AgentRequest(BaseModel):
    question: str


class AgentStep(BaseModel):
    iteration: int
    tool: Optional[str] = None
    arguments: Optional[dict] = None
    result_preview: Optional[str] = None
    action: Optional[str] = None


class AgentResponse(BaseModel):
    answer: str
    steps: list[AgentStep]
    iterations: int
    prompt_tokens: int
    completion_tokens: int


class StructuredQueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3


class StructuredAnswer(BaseModel):
    answer: str
    confidence: float
    sources_cited: list[str]
    follow_up_questions: list[str]
    prompt_tokens: int
    completion_tokens: int


class EvalRequest(BaseModel):
    question: str
    answer: str
    context_chunks: list[dict]


class EvalResponse(BaseModel):
    faithfulness: dict
    relevance: dict
    completeness: dict
    overall_score: float
    summary: str
    eval_tokens: dict
