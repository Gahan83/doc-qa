"""
Phase 1 — Generation: build prompt → call GPT → return answer + token counts.
Prompt engineering concepts live here: system role, context injection, instruction.
"""

import os

from openai import AzureOpenAI

client = AzureOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_API_VERSION", "2024-12-01-preview"),
)

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# --- PROMPT ENGINEERING ZONE ---
# System prompt sets the model's persona + constraints.
# This is where prompt engineering begins — tweak this and watch behavior change.
SYSTEM_PROMPT = """You are a precise document assistant.
Answer ONLY from the provided context chunks.
If the answer is not in the context, say "I don't have enough information."
Be concise. Cite the source filename when relevant."""


def build_prompt(question: str, chunks: list[dict]) -> str:
    """
    Few-shot RAG prompt pattern:
    CONTEXT block → QUESTION → instruction to answer only from context.
    """
    context_block = "\n\n---\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in chunks
    )
    return f"""CONTEXT:
{context_block}

QUESTION: {question}

Answer based only on the context above."""


def generate(question: str, chunks: list[dict]) -> dict:
    """
    Call GPT with system prompt + user prompt.
    Returns answer string + token usage for cost tracking.
    """
    user_prompt = build_prompt(question, chunks)

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,  # Low temp = factual, deterministic answers
        max_tokens=512,
    )

    return {
        "answer": response.choices[0].message.content,
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
    }
