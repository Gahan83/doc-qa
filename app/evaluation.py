"""
Phase 2 — Evaluation: LLM-as-judge to score answer quality.
Evaluates faithfulness, relevance, and completeness.
"""

import json
import os

from openai import AzureOpenAI

client = AzureOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_API_VERSION", "2024-12-01-preview"),
)

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

EVAL_SYSTEM_PROMPT = """You are an impartial evaluator. Score the given answer on three criteria.

For each criterion, provide a score from 1-5 and a brief justification.

Criteria:
1. **Faithfulness** — Is the answer grounded in the provided context? (5 = fully grounded, 1 = hallucinated)
2. **Relevance** — Does the answer address the question asked? (5 = perfectly relevant, 1 = off-topic)
3. **Completeness** — Does the answer cover all relevant info from the context? (5 = complete, 1 = missing key info)

You MUST respond in this exact JSON format:
{
    "faithfulness": {"score": <1-5>, "reason": "<brief justification>"},
    "relevance": {"score": <1-5>, "reason": "<brief justification>"},
    "completeness": {"score": <1-5>, "reason": "<brief justification>"},
    "overall_score": <average of three scores rounded to 1 decimal>,
    "summary": "<one sentence overall assessment>"
}"""


def evaluate(question: str, answer: str, context_chunks: list[dict]) -> dict:
    """
    Use LLM-as-judge to evaluate an answer.

    Args:
        question: The original question
        answer: The generated answer to evaluate
        context_chunks: The chunks that were used as context

    Returns:
        Evaluation scores and justifications.
    """
    context_block = "\n\n---\n\n".join(
        f"[Source: {c.get('source', 'unknown')}]\n{c.get('text', '')}" for c in context_chunks
    )

    eval_prompt = f"""CONTEXT PROVIDED TO THE MODEL:
{context_block}

QUESTION: {question}

ANSWER GIVEN: {answer}

Evaluate the answer based on the criteria."""

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": EVAL_SYSTEM_PROMPT},
            {"role": "user", "content": eval_prompt},
        ],
        temperature=0.0,
        max_completion_tokens=512,
        response_format={"type": "json_object"},
    )

    eval_text = response.choices[0].message.content

    try:
        evaluation = json.loads(eval_text)
    except json.JSONDecodeError:
        evaluation = {
            "faithfulness": {"score": 0, "reason": "Parse error"},
            "relevance": {"score": 0, "reason": "Parse error"},
            "completeness": {"score": 0, "reason": "Parse error"},
            "overall_score": 0.0,
            "summary": "Failed to parse evaluation response",
        }

    evaluation["eval_tokens"] = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
    }

    return evaluation
