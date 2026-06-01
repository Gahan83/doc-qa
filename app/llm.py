"""
Centralized LangChain LLM and embedding clients.
"""

import os

from langchain_chroma import Chroma
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHROMA_DIR = "storage/chromadb"
COLLECTION_NAME = "doc_chunks"


def get_chat_llm(temperature: float = 0.2, max_tokens: int = 512) -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_deployment=CHAT_MODEL,
        api_key=os.getenv("OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_API_VERSION", "2024-12-01-preview"),
        temperature=temperature,
        max_tokens=max_tokens,
    )


def get_embeddings() -> AzureOpenAIEmbeddings:
    return AzureOpenAIEmbeddings(
        azure_deployment=EMBED_MODEL,
        api_key=os.getenv("OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_API_VERSION", "2024-12-01-preview"),
    )


def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DIR,
        collection_metadata={"hnsw:space": "cosine"},
    )
