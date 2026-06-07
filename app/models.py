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
    # Phase 4: timestamp fields — None for non-audio sources
    start: Optional[float] = None
    end: Optional[float] = None
    timestamp_label: Optional[str] = None


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


# --- Phase 3/4: Audio/video ingest ---

class AudioIngestResponse(BaseModel):
    filename: str
    chunks_created: int
    duration_seconds: Optional[float] = None
    message: str


# --- Phase 9: Voice I/O ---

class VoiceSpeakRequest(BaseModel):
    text: str
    voice: str = "alloy"  # alloy | echo | fable | onyx | nova | shimmer


class TranscriptSegment(BaseModel):
    start: float
    end: float
    text: str


class TranscribeResponse(BaseModel):
    text: str
    language: str
    duration: float
    segments: list[TranscriptSegment]
