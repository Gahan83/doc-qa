"""
Phase 1 — Generation using LangChain ChatPromptTemplate + AzureChatOpenAI.
"""

from langchain_core.prompts import ChatPromptTemplate

from app.llm import get_chat_llm

SYSTEM_PROMPT = """You are a precise document assistant.
Answer ONLY from the provided context chunks.
If the answer is not in the context, say "I don't have enough information."
Be concise. Cite the source filename when relevant."""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("user", "CONTEXT:\n{context}\n\nQUESTION: {question}\n\nAnswer based only on the context above."),
])


def format_context(chunks: list[dict]) -> str:
    return "\n\n---\n\n".join(f"[Source: {c['source']}]\n{c['text']}" for c in chunks)


def generate(question: str, chunks: list[dict]) -> dict:
    """Call GPT via LangChain chain. Returns answer + token usage."""
    llm = get_chat_llm(temperature=0.2, max_tokens=512)
    chain = PROMPT | llm

    response = chain.invoke({
        "context": format_context(chunks),
        "question": question,
    })

    usage = response.usage_metadata or {}
    return {
        "answer": response.content,
        "prompt_tokens": usage.get("input_tokens", 0),
        "completion_tokens": usage.get("output_tokens", 0),
    }
