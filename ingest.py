"""
conquest.ai — Document Ingestion Script

Run this script ONCE to clone the DREAM library and build the ChromaDB vector index.
Subsequent runs of the chatbot (app.py) load the pre-built index directly,
giving fast startup times.

Usage:
    python ingest.py                # Standard first-time setup
    python ingest.py --force        # Re-index even if index already exists
    python ingest.py --dream-path ./my_dream  # Use a custom DREAM path

Steps:
    1. Clone the DREAM repository (if not already present)
    2. Load and parse all documents (PDF, EPUB, DOCX) from DREAM/library/
    3. Chunk documents with SentenceSplitter (512 tokens, 50 overlap)
    4. Embed chunks with sentence-transformers/all-MiniLM-L6-v2 (local)
    5. Store embeddings in ChromaDB at data/chroma_db/
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Default paths (can be overridden via env vars or CLI args)
DREAM_REPO_URL = "https://github.com/Indranil-Seal/DREAM.git"
DEFAULT_DREAM_PATH = Path(os.getenv("DREAM_PATH", "./DREAM"))
DEFAULT_CHROMA_PATH = Path(os.getenv("CHROMA_DB_PATH", "./data/chroma_db"))


def clone_dream(dream_path: Path) -> None:
    """
    Clone the DREAM library repository if it doesn't already exist.

    If the directory exists and is a valid git repo, the clone is skipped.
    If the directory exists but isn't a git repo, raises an error.

    Args:
        dream_path: Target local path for the DREAM repository.
    """
    if dream_path.exists():
        git_dir = dream_path / ".git"
        if git_dir.exists():
            logger.info(f"DREAM repository already exists at {dream_path}. Skipping clone.")
            return
        else:
            raise RuntimeError(
                f"Directory {dream_path} exists but is not a git repository. "
                "Remove it or specify a different path with --dream-path."
            )

    logger.info(f"Cloning DREAM library from {DREAM_REPO_URL} ...")
    logger.info("This may take a few minutes (the library contains ~400MB of PDFs).")

    result = subprocess.run(
        ["git", "clone", "--depth=1", DREAM_REPO_URL, str(dream_path)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        logger.error(f"Git clone failed:\n{result.stderr}")
        raise RuntimeError(
            f"Failed to clone DREAM repository. "
            "Check your internet connection and git installation.\n"
            f"Error: {result.stderr}"
        )

    logger.info(f"DREAM library cloned successfully to {dream_path}")


def ensure_chroma_dir(chroma_path: Path) -> None:
    """Create the ChromaDB storage directory if it doesn't exist."""
    chroma_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"ChromaDB storage directory: {chroma_path.resolve()}")


def run_ingestion(
    dream_path: Path,
    chroma_path: Path,
    force_reindex: bool = False,
) -> None:
    """
    Main ingestion pipeline: load documents, chunk, embed, and index.

    Imports are deferred to here so that missing packages give a clear
    error message pointing to requirements.txt.

    Args:
        dream_path: Path to the DREAM repository root.
        chroma_path: Path to the ChromaDB persistent storage directory.
        force_reindex: Re-index even if the collection already has data.
    """
    try:
        from src.indexer import build_index
    except ImportError as e:
        logger.error(
            f"Import error: {e}\n"
            "Please install dependencies first:\n"
            "  pip install -r requirements.txt"
        )
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("conquest.ai — Document Ingestion Pipeline")
    logger.info("=" * 60)
    logger.info(f"DREAM path:    {dream_path.resolve()}")
    logger.info(f"ChromaDB path: {chroma_path.resolve()}")
    logger.info(f"Force re-index: {force_reindex}")
    logger.info("")

    index = build_index(
        dream_path=dream_path,
        chroma_db_path=chroma_path,
        force_reindex=force_reindex,
    )

    logger.info("")
    logger.info("=" * 60)
    logger.info("Ingestion complete!")
    logger.info("You can now start the chatbot with:")
    logger.info("  chainlit run app.py")
    logger.info("=" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="conquest.ai document ingestion — builds the ChromaDB vector index from the DREAM library.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ingest.py                         # First-time setup
  python ingest.py --force                 # Force full re-index
  python ingest.py --dream-path /my/dream  # Custom DREAM path
        """,
    )
    parser.add_argument(
        "--dream-path",
        type=Path,
        default=DEFAULT_DREAM_PATH,
        help=f"Path to the DREAM library repository (default: {DEFAULT_DREAM_PATH})",
    )
    parser.add_argument(
        "--chroma-path",
        type=Path,
        default=DEFAULT_CHROMA_PATH,
        help=f"Path to ChromaDB storage directory (default: {DEFAULT_CHROMA_PATH})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-indexing even if the ChromaDB collection already has data",
    )
    parser.add_argument(
        "--no-clone",
        action="store_true",
        help="Skip cloning DREAM (use if you already have the library at --dream-path)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Step 1: Clone DREAM if needed
    if not args.no_clone:
        clone_dream(args.dream_path)

    # Step 2: Ensure ChromaDB directory exists
    ensure_chroma_dir(args.chroma_path)

    # Step 3: Run ingestion
    run_ingestion(
        dream_path=args.dream_path,
        chroma_path=args.chroma_path,
        force_reindex=args.force,
    )
