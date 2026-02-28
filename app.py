"""
conquest.ai — Chainlit Chat Application

Entry point for the conquest.ai chatbot. Run with:
    chainlit run app.py

Architecture:
    - @cl.on_chat_start  : Loads the ChromaDB index and builds the query engine
    - @cl.on_message     : Handles user messages, runs RAG, streams Claude's response,
                           and displays source citations
    - @cl.on_settings_update: Handles any future UI settings

The app streams Claude's response token-by-token for a responsive feel,
and shows citations collapsed below each answer so users can trace back
to the original DREAM library document.
"""

import logging
import os
import sys
from pathlib import Path

# Ensure the project root is on sys.path so `src` is importable
# regardless of the working directory Chainlit uses at startup.
sys.path.insert(0, str(Path(__file__).parent))

import chainlit as cl
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Paths (use env vars or defaults)
CHROMA_DB_PATH = Path(os.getenv("CHROMA_DB_PATH", "./data/chroma_db"))

# Session key for the query engine
QUERY_ENGINE_KEY = "query_engine"


# ─── Chat Lifecycle Handlers ─────────────────────────────────────────────────

@cl.on_chat_start
async def on_chat_start():
    """
    Called once when a user opens a new chat session.

    1. Loads the ChromaDB vector index (built by ingest.py)
    2. Configures the LlamaIndex RetrieverQueryEngine with Claude Sonnet 4.6
    3. Stores the query engine in the Chainlit session for reuse across messages
    4. Displays an error message if the index hasn't been built yet
    """
    from src.indexer import load_existing_index
    from src.rag import build_query_engine

    await cl.Message(
        content="Initializing conquest.ai... Loading knowledge base.",
        author="conquest.ai",
    ).send()

    # Load the pre-built ChromaDB index
    index = load_existing_index(CHROMA_DB_PATH)

    if index is None:
        await cl.Message(
            content=(
                "**Knowledge base not found.**\n\n"
                "The DREAM library index has not been built yet. Please run:\n\n"
                "```bash\n"
                "python ingest.py\n"
                "```\n\n"
                "This will clone the DREAM library and build the vector index. "
                "It only needs to be done once."
            ),
            author="conquest.ai",
        ).send()
        return

    # Validate API key before building the query engine
    if not os.getenv("ANTHROPIC_API_KEY"):
        await cl.Message(
            content=(
                "**Anthropic API key not found.**\n\n"
                "Please create a `.env` file in the project root:\n\n"
                "```bash\n"
                "cp .env.example .env\n"
                "```\n\n"
                "Then open `.env` and set your key:\n\n"
                "```\n"
                "ANTHROPIC_API_KEY=your_key_here\n"
                "```\n\n"
                "Get your key at [console.anthropic.com](https://console.anthropic.com). "
                "Restart the app after saving."
            ),
            author="conquest.ai",
        ).send()
        return

    # Build query engine and store in session
    try:
        query_engine = build_query_engine(index)
    except Exception as e:
        await cl.Message(
            content=f"**Failed to initialize the query engine.**\n\n```\n{e}\n```",
            author="conquest.ai",
        ).send()
        return

    cl.user_session.set(QUERY_ENGINE_KEY, query_engine)

    await cl.Message(
        content=(
            "Knowledge base loaded successfully! "
            "I have access to the DREAM library with books and papers on "
            "Data Science, Machine Learning, and AI.\n\n"
            "Ask me anything — I can explain concepts, derive equations, and draw diagrams."
        ),
        author="conquest.ai",
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """
    Called for every user message in the chat.

    1. Retrieves the query engine from the session
    2. Shows a "Searching knowledge base..." step indicator
    3. Runs the RAG query against ChromaDB + Claude
    4. Streams Claude's response token-by-token
    5. Appends collapsible source citations below the response
    """
    query_engine = cl.user_session.get(QUERY_ENGINE_KEY)

    if query_engine is None:
        await cl.Message(
            content=(
                "The knowledge base is not loaded. "
                "Please refresh the page or run `python ingest.py` first."
            ),
            author="conquest.ai",
        ).send()
        return

    user_query = message.content.strip()
    if not user_query:
        return

    # Show a step indicator while searching the vector DB
    async with cl.Step(name="Searching DREAM library...", type="retrieval") as step:
        step.input = user_query

    # Stream the response
    response_msg = cl.Message(content="", author="conquest.ai")
    await response_msg.send()

    try:
        from src.rag import query_rag_stream

        token_stream, sources = await query_rag_stream(query_engine, user_query)

        # Stream tokens into the response message
        async for token in token_stream:
            await response_msg.stream_token(token)

        await response_msg.update()

        # Append source citations as collapsible elements
        if sources:
            citation_lines = _format_citations(sources)
            citation_text = "**Sources from DREAM library:**\n" + "\n".join(citation_lines)
            await cl.Message(
                content=citation_text,
                author="conquest.ai",
                parent_id=response_msg.id,
            ).send()

    except EnvironmentError as e:
        # API key missing or misconfigured
        await cl.Message(
            content=f"**Configuration error:** {e}",
            author="conquest.ai",
        ).send()
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        await cl.Message(
            content=(
                "I encountered an error while processing your question. "
                "Please try again or rephrase your question.\n\n"
                f"*Error details: {type(e).__name__}: {e}*"
            ),
            author="conquest.ai",
        ).send()


# ─── Helper Functions ─────────────────────────────────────────────────────────

def _format_citations(sources: list[dict]) -> list[str]:
    """
    Format source metadata into readable citation lines.

    Each source is shown with a document-type icon and the filename.
    Relevance score is included to help users gauge how closely the
    document matched their query.

    Args:
        sources: List of source dicts from extract_sources().

    Returns:
        List of formatted citation strings (markdown).
    """
    icons = {
        "textbook": "📘",
        "paper": "📄",
        "reference": "📋",
        "document": "📄",
    }
    lines = []
    for i, src in enumerate(sources, 1):
        icon = icons.get(src["doc_type"], "📄")
        score_pct = int(src.get("score", 0) * 100)
        lines.append(
            f"{i}. {icon} `{src['filename']}` — relevance: {score_pct}%"
        )
    return lines
