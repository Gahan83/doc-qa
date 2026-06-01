"""
Phase 2 — Structured Outputs using LangChain's with_structured_output().
Pydantic schema is enforced automatically.
"""

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.generation import format_context
from app.llm import get_chat_llm


class StructuredAnswerSchema(BaseModel):
    answer: str = Field(description="Concise answer based only on the context")
    confidence: float = Field(
        description="0.0 to 1.0 — how well the context supports the answer", ge=0.0, le=1.0
    )
    sources_cited: list[str] = Field(description="Filenames cited in the answer")
    follow_up_questions: list[str] = Field(description="2-3 suggested follow-up questions")


STRUCTURED_SYSTEM_PROMPT = """You are a precise document assistant.
Answer ONLY from the provided context chunks.

Rules:
- confidence 0.9+ = answer is clearly stated in context
- confidence 0.5-0.8 = answer is implied but not explicit
- confidence <0.5 = context barely supports the answer
- Always suggest 2-3 follow-up questions the user might ask"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", STRUCTURED_SYSTEM_PROMPT),
    ("user", "CONTEXT:\n{context}\n\nQUESTION: {question}\n\nAnswer based only on the context above."),
])


def generate_structured(question: str, chunks: list[dict]) -> dict:
    """Generate a Pydantic-validated structured response."""
    llm = get_chat_llm(temperature=0.2, max_tokens=512)
    structured_llm = llm.with_structured_output(StructuredAnswerSchema, include_raw=True)
    chain = PROMPT | structured_llm

    result = chain.invoke({
        "context": format_context(chunks),
        "question": question,
    })

    parsed: StructuredAnswerSchema = result["parsed"]
    raw_response = result["raw"]
    usage = raw_response.usage_metadata or {}

    return {
        **parsed.model_dump(),
        "prompt_tokens": usage.get("input_tokens", 0),
        "completion_tokens": usage.get("output_tokens", 0),
    }
