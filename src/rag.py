"""
conquest.ai — RAG Pipeline

Wraps the LlamaIndex query engine with Claude Sonnet 4.6 as the LLM.
Provides the core query function used by the Chainlit app.

Architecture:
    User query
        → ChromaDB similarity search (top-5 chunks, sentence-transformers embeddings)
        → Context + query assembled with prompt template
        → Claude Sonnet 4.6 generates a streamed response
        → Source metadata returned for citation display
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Optional

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.llms.anthropic import Anthropic

from src.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# Number of top chunks to retrieve per query
TOP_K = 5
# Claude model to use
CLAUDE_MODEL = os.getenv("CONQUEST_MODEL", "claude-sonnet-4-6")
# Max tokens in Claude's response
MAX_TOKENS = 2048


@dataclass
class QueryResult:
    """Holds the response text and source documents from a RAG query."""
    response: str
    sources: list[dict]  # List of {"filename": str, "doc_type": str}


def build_query_engine(index: VectorStoreIndex) -> RetrieverQueryEngine:
    """
    Configure and return a LlamaIndex RetrieverQueryEngine.

    Uses:
    - VectorIndexRetriever for similarity search over ChromaDB
    - Claude Sonnet 4.6 as the response synthesizer LLM
    - 'compact' response mode: merges retrieved chunks before sending to LLM

    Args:
        index: The VectorStoreIndex backed by ChromaDB.

    Returns:
        Configured RetrieverQueryEngine.
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

    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=TOP_K,
    )

    response_synthesizer = get_response_synthesizer(
        response_mode="compact",
        streaming=True,
    )

    return RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
    )


def extract_sources(response_nodes: list) -> list[dict]:
    """
    Extract unique source document metadata from retrieved nodes.

    Deduplicates by filename so each source document appears only once
    in the citation list, even if multiple chunks came from the same file.

    Args:
        response_nodes: List of NodeWithScore objects from the query response.

    Returns:
        List of source dicts with 'filename' and 'doc_type' keys.
    """
    seen = set()
    sources = []
    for node_with_score in response_nodes:
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


async def query_rag(
    query_engine: RetrieverQueryEngine,
    user_query: str,
) -> QueryResult:
    """
    Run a RAG query and return the complete response with sources.

    This is the non-streaming version, returning the full response at once.
    For streaming, use query_rag_stream().

    Args:
        query_engine: The configured RetrieverQueryEngine.
        user_query: The user's question as a string.

    Returns:
        QueryResult with response text and source citations.
    """
    try:
        response = await query_engine.aquery(user_query)
        sources = extract_sources(response.source_nodes)
        return QueryResult(
            response=str(response),
            sources=sources,
        )
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        raise


async def query_rag_stream(
    query_engine: RetrieverQueryEngine,
    user_query: str,
) -> tuple[AsyncIterator[str], list[dict]]:
    """
    Run a RAG query with streaming response.

    Retrieves context from ChromaDB first, then streams Claude's response
    token-by-token. Sources are returned alongside the stream iterator.

    This function first performs the retrieval step synchronously to get
    source metadata, then initiates the streaming LLM call.

    Args:
        query_engine: The configured RetrieverQueryEngine.
        user_query: The user's question.

    Returns:
        Tuple of (async token iterator, list of source dicts).
    """
    try:
        # Use the streaming query (LlamaIndex handles this via streaming=True in synthesizer)
        streaming_response = await query_engine.aquery(user_query)
        sources = extract_sources(streaming_response.source_nodes)

        # LlamaIndex streaming response exposes async_response_gen for token-by-token streaming
        async def token_stream() -> AsyncIterator[str]:
            if hasattr(streaming_response, "async_response_gen"):
                async for token in streaming_response.async_response_gen():
                    yield token
            else:
                # Fallback: yield the full response at once if streaming unavailable
                yield str(streaming_response)

        return token_stream(), sources

    except Exception as e:
        logger.error(f"Streaming RAG query failed: {e}")
        raise
