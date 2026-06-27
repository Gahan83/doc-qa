from pydantic import BaseModel
from typing import Optional


# --- Token usage / cost (shared) ---

class Usage(BaseModel):
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float


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
    usage: Optional[Usage] = None


# --- Phase 2 Models ---

class AgentRequest(BaseModel):
    question: str
    session_id: Optional[str] = None


class AgentPlan(BaseModel):
    goal: str
    sub_steps: list[str]
    tool_sequence: list[str]
    stop_condition: str


class AgentStep(BaseModel):
    iteration: int
    thought: Optional[str] = None       # ReAct: model's reasoning before acting (CoT scratchpad)
    tool: Optional[str] = None          # ReAct: Action
    arguments: Optional[dict] = None
    observation: Optional[str] = None   # ReAct: Observation (tool result summary)
    result_preview: Optional[str] = None
    action: Optional[str] = None
    status: Optional[str] = None
    error_type: Optional[str] = None
    retries: Optional[int] = None
    latency_ms: Optional[int] = None


class AgentResponse(BaseModel):
    answer: str
    plan: Optional[AgentPlan] = None
    steps: list[AgentStep]
    iterations: int
    tool_calls: Optional[int] = None
    scratchpad: Optional[list[str]] = None   # ReAct: ordered Thought/Action/Observation trace
    prompt_tokens: int
    completion_tokens: int
    usage: Optional[Usage] = None
    trace_id: Optional[str] = None
    session_id: Optional[str] = None
    stop_reason: Optional[str] = None
    elapsed_ms: Optional[int] = None


class AgentCompareResponse(BaseModel):
    question: str
    langchain: AgentResponse
    scratch: AgentResponse
    comparison: dict   # quick side-by-side diff (tokens, iterations, tool_calls, answers)


class Handoff(BaseModel):
    step: int
    from_agent: str
    to_agent: str
    action: str
    summary: str
    latency_ms: int


class MultiAgentResponse(BaseModel):
    answer: str
    refined_query: str
    handoffs: list[Handoff]
    sources: list[SourceChunk]
    prompt_tokens: int
    completion_tokens: int
    usage: Optional[Usage] = None
    trace_id: str
    elapsed_ms: int


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
    usage: Optional[Usage] = None


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
    usage: Optional[Usage] = None


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


# --- LLM internals: explain a chunk ---

class ExplainChunkRequest(BaseModel):
    text: str
    model: Optional[str] = None  # defaults to CHAT_MODEL


class ExplainChunkResponse(BaseModel):
    model: str
    char_count: int
    token_count: int
    estimated_input_cost_usd: float   # cost if sent as prompt/context
    price_input_per_1m: float
    price_output_per_1m: float
