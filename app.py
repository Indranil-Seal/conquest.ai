"""
conquest.ai — Chainlit Chat Application

Entry point for the conquest.ai chatbot. Run with:
    chainlit run app.py

Architecture:
    - @cl.on_chat_start : Loads ChromaDB index, builds the CondensePlusContextChatEngine
                          (stateful — maintains conversation memory across turns)
    - @cl.on_message    : Sends each user message through the chat engine, which:
                            1. Condenses history + new query (reverse prompt injection)
                            2. Retrieves top-5 DREAM library chunks
                            3. Streams Claude's response with full context
                            4. Displays source citations

The chat engine is stored per Chainlit session, so each browser tab gets
its own independent conversation history.
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

# Session key for the chat engine
CHAT_ENGINE_KEY = "chat_engine"


# ─── Chat Lifecycle Handlers ─────────────────────────────────────────────────

@cl.on_chat_start
async def on_chat_start():
    """
    Called once when a user opens a new chat session.

    1. Loads the ChromaDB vector index (built by ingest.py)
    2. Validates the Anthropic API key
    3. Builds a CondensePlusContextChatEngine with fresh ChatMemoryBuffer
    4. Stores the engine in the Chainlit session (isolated per browser tab)
    """
    from src.indexer import load_existing_index
    from src.rag import build_chat_engine

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

    # Validate API key before building the chat engine
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

    # Build chat engine (with fresh memory for this session)
    try:
        chat_engine = build_chat_engine(index)
    except Exception as e:
        await cl.Message(
            content=f"**Failed to initialize the chat engine.**\n\n```\n{e}\n```",
            author="conquest.ai",
        ).send()
        return

    cl.user_session.set(CHAT_ENGINE_KEY, chat_engine)

    await cl.Message(
        content=(
            "Knowledge base loaded. conquest.ai is ready.\n\n"
            "I remember our full conversation — feel free to ask follow-up questions "
            "and I'll build on what we've already discussed.\n\n"
            "Ask me anything about Data Science, Machine Learning, or AI."
        ),
        author="conquest.ai",
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """
    Called for every user message in the chat.

    1. Retrieves the stateful chat engine from the session
    2. Shows a step indicator while the engine condenses history and retrieves context
    3. Streams Claude's response token-by-token
    4. Appends source citations from the DREAM library
    """
    chat_engine = cl.user_session.get(CHAT_ENGINE_KEY)

    if chat_engine is None:
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

    # Show a step indicator while retrieving context
    async with cl.Step(name="Searching DREAM library...", type="retrieval") as step:
        step.input = user_query

    # Stream the response
    response_msg = cl.Message(content="", author="conquest.ai")
    await response_msg.send()

    try:
        from src.rag import chat_stream

        token_stream, sources = await chat_stream(chat_engine, user_query)

        # Stream tokens into the response message
        async for token in token_stream:
            await response_msg.stream_token(token)

        await response_msg.update()

        # Append source citations
        if sources:
            citation_lines = _format_citations(sources)
            citation_text = "**Sources from DREAM library:**\n" + "\n".join(citation_lines)
            await cl.Message(
                content=citation_text,
                author="conquest.ai",
                parent_id=response_msg.id,
            ).send()

    except EnvironmentError as e:
        await cl.Message(
            content=f"**Configuration error:** {e}",
            author="conquest.ai",
        ).send()
    except Exception as e:
        logger.error(f"Error processing message: {e}", exc_info=True)
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

    Args:
        sources: List of source dicts from extract_sources().

    Returns:
        List of formatted markdown citation strings.
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
