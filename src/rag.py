"""
conquest.ai — RAG Pipeline with Conversation Memory

Implements a stateful chat engine using LlamaIndex's CondensePlusContextChatEngine.
Each session maintains a rolling conversation history; new queries are automatically
condensed with that history before retrieval — this is the "reverse prompt injection"
mechanism that keeps the conversation coherent.

How reverse prompt injection works here:
    Turn 1: User asks "What is gradient descent?"
    Turn 2: User asks "How does the learning rate affect it?"
    → The engine condenses turns 1+2 into: "How does the learning rate affect
      gradient descent convergence?" before querying ChromaDB.
    → Claude receives full conversation history + fresh retrieved context,
      producing a response that builds on what was already discussed.
"""

import logging
import os
from dataclasses import dataclass
from typing import AsyncIterator

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.llms.anthropic import Anthropic

from src.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# Number of top chunks to retrieve per query
TOP_K = 5
# Claude model to use
CLAUDE_MODEL = os.getenv("CONQUEST_MODEL", "claude-sonnet-4-6")
# Max tokens in Claude's response
MAX_TOKENS = 4096
# Token budget for conversation memory (keeps last ~3000 tokens of history)
MEMORY_TOKEN_LIMIT = 3000


@dataclass
class ChatResult:
    """Holds the response text and source documents from a chat turn."""
    response: str
    sources: list[dict]  # List of {"filename": str, "doc_type": str, "score": float}


def build_chat_engine(index: VectorStoreIndex) -> CondensePlusContextChatEngine:
    """
    Build a CondensePlusContextChatEngine — a stateful, memory-aware RAG engine.

    This replaces the stateless RetrieverQueryEngine. Key differences:
    - Maintains a ChatMemoryBuffer across turns within a session
    - Before each retrieval, condenses the conversation history + new query into
      a standalone question (reverse prompt injection)
    - Claude receives both the condensed context AND full chat history, enabling
      coherent multi-turn conversations about DS/ML/AI topics

    Args:
        index: The VectorStoreIndex backed by ChromaDB.

    Returns:
        A configured CondensePlusContextChatEngine ready for async chat.

    Raises:
        EnvironmentError: If ANTHROPIC_API_KEY is not set.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Copy .env.example to .env and add your Anthropic API key."
        )

    llm = Anthropic(
        model=CLAUDE_MODEL,
        api_key=api_key,
        max_tokens=MAX_TOKENS,
        system_prompt=SYSTEM_PROMPT,
    )
    Settings.llm = llm

    # ChatMemoryBuffer stores the rolling conversation history.
    # token_limit controls how far back the engine remembers — older turns
    # are dropped first when the limit is reached.
    memory = ChatMemoryBuffer.from_defaults(token_limit=MEMORY_TOKEN_LIMIT)

    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=TOP_K,
    )

    # CondensePlusContextChatEngine:
    # 1. Condenses conversation history + new message → standalone query
    # 2. Retrieves top-k chunks using the condensed query
    # 3. Synthesizes a response using retrieved context + full chat history
    chat_engine = CondensePlusContextChatEngine.from_defaults(
        retriever=retriever,
        memory=memory,
        llm=llm,
        system_prompt=SYSTEM_PROMPT,
        verbose=False,
    )

    return chat_engine


def extract_sources(source_nodes: list) -> list[dict]:
    """
    Extract unique source document metadata from retrieved nodes.

    Deduplicates by filename so each source document appears only once,
    even if multiple chunks came from the same file.

    Args:
        source_nodes: List of NodeWithScore objects from the chat response.

    Returns:
        List of source dicts with 'filename', 'doc_type', and 'score' keys.
    """
    seen = set()
    sources = []
    for node_with_score in source_nodes:
        metadata = node_with_score.node.metadata
        filename = metadata.get("filename", "Unknown source")
        if filename not in seen:
            seen.add(filename)
            sources.append({
                "filename": filename,
                "doc_type": metadata.get("doc_type", "document"),
                "score": round(node_with_score.score or 0.0, 3),
            })
    return sources


async def chat_stream(
    chat_engine: CondensePlusContextChatEngine,
    user_message: str,
) -> tuple[AsyncIterator[str], list[dict]]:
    """
    Send a message to the chat engine and return a streaming response.

    The chat engine automatically:
    - Condenses conversation history + this message into a retrieval query
    - Fetches top-5 relevant chunks from ChromaDB
    - Streams Claude's response with full conversation context

    Args:
        chat_engine: The configured CondensePlusContextChatEngine (holds memory).
        user_message: The user's latest message.

    Returns:
        Tuple of (async token iterator, list of source citation dicts).
    """
    try:
        streaming_response = await chat_engine.astream_chat(user_message)

        sources = extract_sources(
            getattr(streaming_response, "source_nodes", [])
        )

        async def token_stream() -> AsyncIterator[str]:
            if hasattr(streaming_response, "async_response_gen"):
                async for token in streaming_response.async_response_gen():
                    yield token
            else:
                yield str(streaming_response)

        return token_stream(), sources

    except Exception as e:
        logger.error(f"Chat stream failed: {e}")
        raise
