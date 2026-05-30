"""
Phase 2 — Structured Outputs: force GPT to return validated JSON.
Uses response_format to guarantee schema compliance.
"""

import json
import os

from openai import AzureOpenAI

from app.generation import build_prompt

client = AzureOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_API_VERSION", "2024-12-01-preview"),
)

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

STRUCTURED_SYSTEM_PROMPT = """You are a precise document assistant.
Answer ONLY from the provided context chunks.

You MUST respond in this exact JSON format:
{
    "answer": "<your concise answer>",
    "confidence": <0.0 to 1.0 based on how well context supports the answer>,
    "sources_cited": ["<filename1>", "<filename2>"],
    "follow_up_questions": ["<suggested question 1>", "<suggested question 2>"]
}

Rules:
- confidence 0.9+ = answer is clearly stated in context
- confidence 0.5-0.8 = answer is implied but not explicit
- confidence <0.5 = context barely supports the answer
- Always suggest 2-3 follow-up questions the user might ask"""


def generate_structured(question: str, chunks: list[dict]) -> dict:
    """
    Generate a structured JSON answer with confidence and metadata.
    Uses response_format=json_object to enforce valid JSON output.
    """
    user_prompt = build_prompt(question, chunks)

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": STRUCTURED_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_completion_tokens=512,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {
            "answer": raw,
            "confidence": 0.0,
            "sources_cited": [],
            "follow_up_questions": [],
        }

    parsed["prompt_tokens"] = response.usage.prompt_tokens
    parsed["completion_tokens"] = response.usage.completion_tokens

    return parsed
