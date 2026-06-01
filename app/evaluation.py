"""
Phase 2 — LLM-as-judge evaluation using LangChain structured output.
"""

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.llm import get_chat_llm


class Score(BaseModel):
    score: int = Field(ge=1, le=5, description="Score from 1 to 5")
    reason: str = Field(description="Brief justification")


class EvalSchema(BaseModel):
    faithfulness: Score = Field(description="Is the answer grounded in the context?")
    relevance: Score = Field(description="Does it address the question?")
    completeness: Score = Field(description="Does it cover all relevant info?")
    overall_score: float = Field(description="Average of three scores rounded to 1 decimal")
    summary: str = Field(description="One sentence overall assessment")


EVAL_SYSTEM_PROMPT = """You are an impartial evaluator. Score the given answer on three criteria.

For each criterion, provide a score from 1-5 and a brief justification.

Criteria:
1. Faithfulness — Is the answer grounded in the provided context? (5 = fully grounded, 1 = hallucinated)
2. Relevance — Does the answer address the question asked? (5 = perfectly relevant, 1 = off-topic)
3. Completeness — Does the answer cover all relevant info from the context? (5 = complete, 1 = missing key info)"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", EVAL_SYSTEM_PROMPT),
    ("user", "CONTEXT PROVIDED TO THE MODEL:\n{context}\n\nQUESTION: {question}\n\nANSWER GIVEN: {answer}\n\nEvaluate the answer based on the criteria."),
])


def evaluate(question: str, answer: str, context_chunks: list[dict]) -> dict:
    """LLM-as-judge evaluation via structured output."""
    context_block = "\n\n---\n\n".join(
        f"[Source: {c.get('source', 'unknown')}]\n{c.get('text', '')}" for c in context_chunks
    )

    llm = get_chat_llm(temperature=0.0, max_tokens=512)
    structured_llm = llm.with_structured_output(EvalSchema, include_raw=True)
    chain = PROMPT | structured_llm

    result = chain.invoke({
        "context": context_block,
        "question": question,
        "answer": answer,
    })

    parsed: EvalSchema = result["parsed"]
    usage = result["raw"].usage_metadata or {}

    output = parsed.model_dump()
    output["eval_tokens"] = {
        "prompt_tokens": usage.get("input_tokens", 0),
        "completion_tokens": usage.get("output_tokens", 0),
    }
    return output
