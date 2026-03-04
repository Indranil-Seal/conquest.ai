"""
conquest.ai — Document Ingestion Script

Clones (or updates) the DREAM library and keeps the ChromaDB index in sync.
Safe to run repeatedly — only changed documents are re-indexed.

Usage:
    python ingest.py                # First-time setup OR incremental update
    python ingest.py --force        # Wipe and fully rebuild the index
    python ingest.py --dream-path ./my_dream  # Use a custom DREAM path

How it works on subsequent runs:
    1. git fetch origin --depth=1 to check for upstream changes
    2. git diff detects which documents were added, modified, or deleted
    3. Deleted/modified documents are removed from ChromaDB
    4. New/modified documents are parsed, chunked, embedded, and inserted
    5. Unchanged documents are untouched — no wasted re-embedding
"""

import argparse
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Ensure the project root is on sys.path so `src` is importable.
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DREAM_REPO_URL = "https://github.com/Indranil-Seal/DREAM.git"
DEFAULT_DREAM_PATH = Path(os.getenv("DREAM_PATH", "./DREAM"))
DEFAULT_CHROMA_PATH = Path(os.getenv("CHROMA_DB_PATH", "./data/chroma_db"))

# Only track these file types — mirrors SUPPORTED_EXTENSIONS in indexer.py
TRACKED_EXTENSIONS = {".pdf", ".epub", ".docx"}


@dataclass
class SyncResult:
    """
    Result of syncing the DREAM library with the remote.

    Attributes:
        was_cloned: True if DREAM was freshly cloned (first run).
        changed:    True if git pull brought in new commits.
        added:      Absolute paths of files that are new or modified (to index).
        removed:    Filenames (basename) of deleted or modified files (to remove from ChromaDB).
    """
    was_cloned: bool = False
    changed: bool = False
    added: list[Path] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)


# ─── Git helpers ──────────────────────────────────────────────────────────────

def _git(dream_path: Path, args: list[str]) -> str:
    """
    Run a git command inside dream_path and return stdout.
    Raises RuntimeError on non-zero exit.
    """
    result = subprocess.run(
        ["git", "-C", str(dream_path)] + args,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed:\n{result.stderr.strip()}")
    return result.stdout.strip()


def _parse_diff(dream_path: Path, diff_output: str) -> tuple[list[Path], list[str]]:
    """
    Parse `git diff --name-status` output into lists of files to add and remove.

    Status codes handled:
        A  — Added:    add to index
        D  — Deleted:  remove from index
        M  — Modified: remove old chunks, re-index with new content
        R<n>— Renamed: remove old name, index under new name

    Only files with tracked extensions (.pdf, .epub, .docx) are included.
    Files in non-library directories (e.g., README, scripts) are ignored.

    Args:
        dream_path: DREAM repository root (to resolve absolute paths).
        diff_output: Raw stdout from `git diff --name-status`.

    Returns:
        Tuple of (files_to_add: list[Path], filenames_to_remove: list[str]).
    """
    to_add: list[Path] = []
    to_remove: list[str] = []

    for line in diff_output.splitlines():
        line = line.strip()
        if not line:
            continue

        parts = line.split("\t")
        status = parts[0]

        if status.startswith("R"):
            # Renamed: parts = ["R100", "old/path", "new/path"]
            old_rel, new_rel = parts[1], parts[2]
            old_path = dream_path / old_rel
            new_path = dream_path / new_rel
            if old_path.suffix.lower() in TRACKED_EXTENSIONS:
                to_remove.append(old_path.name)
            if new_path.suffix.lower() in TRACKED_EXTENSIONS:
                to_add.append(new_path)

        elif status == "A":
            path = dream_path / parts[1]
            if path.suffix.lower() in TRACKED_EXTENSIONS:
                to_add.append(path)

        elif status == "D":
            path = dream_path / parts[1]
            if path.suffix.lower() in TRACKED_EXTENSIONS:
                to_remove.append(path.name)

        elif status == "M":
            path = dream_path / parts[1]
            if path.suffix.lower() in TRACKED_EXTENSIONS:
                # Remove old chunks, then re-index with fresh content
                to_remove.append(path.name)
                to_add.append(path)

    return to_add, to_remove


# ─── DREAM sync ───────────────────────────────────────────────────────────────

def sync_dream(dream_path: Path) -> SyncResult:
    """
    Ensure the local DREAM library is in sync with the remote repository.

    First run:
        Clones the repository with --depth=1 (shallow, saves bandwidth).
        Returns SyncResult(was_cloned=True) — caller should do a full index build.

    Subsequent runs:
        1. git fetch origin --depth=1  — check upstream for new commits
        2. Compare local HEAD to FETCH_HEAD
        3. If equal: library is up to date, no action needed
        4. If different: parse git diff to find added/modified/deleted documents,
           then git reset --hard FETCH_HEAD to apply the update
        Returns SyncResult with the delta for incremental ChromaDB update.

    Args:
        dream_path: Target local path for the DREAM repository.

    Returns:
        SyncResult describing what changed.
    """
    # ── First time: clone ──────────────────────────────────────────────────
    if not dream_path.exists():
        logger.info(f"Cloning DREAM library from {DREAM_REPO_URL} ...")
        logger.info("This may take a few minutes (~400 MB of PDFs).")
        result = subprocess.run(
            ["git", "clone", "--depth=1", DREAM_REPO_URL, str(dream_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"git clone failed. Check your internet connection.\n{result.stderr.strip()}"
            )
        logger.info(f"DREAM cloned to {dream_path}")
        return SyncResult(was_cloned=True, changed=True)

    # ── Validate it's a git repo ───────────────────────────────────────────
    if not (dream_path / ".git").exists():
        raise RuntimeError(
            f"{dream_path} exists but is not a git repository. "
            "Remove it or specify a different path with --dream-path."
        )

    # ── Fetch latest from remote ───────────────────────────────────────────
    logger.info("Checking DREAM library for updates...")
    try:
        _git(dream_path, ["fetch", "origin", "--depth=1"])
    except RuntimeError as e:
        logger.warning(f"Could not reach remote: {e}. Proceeding with local copy.")
        return SyncResult(was_cloned=False, changed=False)

    local_head = _git(dream_path, ["rev-parse", "HEAD"])
    remote_head = _git(dream_path, ["rev-parse", "FETCH_HEAD"])

    if local_head == remote_head:
        logger.info("DREAM library is already up to date.")
        return SyncResult(was_cloned=False, changed=False)

    # ── Detect what changed ────────────────────────────────────────────────
    logger.info("Updates found — detecting changed documents...")
    diff_output = _git(dream_path, ["diff", "--name-status", local_head, remote_head])
    to_add, to_remove = _parse_diff(dream_path, diff_output)

    # ── Apply the update ───────────────────────────────────────────────────
    _git(dream_path, ["reset", "--hard", "FETCH_HEAD"])
    logger.info(
        f"DREAM library updated: "
        f"{len(to_add)} document(s) added/modified, "
        f"{len(to_remove)} document(s) removed."
    )

    return SyncResult(was_cloned=False, changed=True, added=to_add, removed=to_remove)


# ─── Ingestion pipeline ───────────────────────────────────────────────────────

def ensure_chroma_dir(chroma_path: Path) -> None:
    """Create the ChromaDB storage directory if it doesn't exist."""
    chroma_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"ChromaDB path: {chroma_path.resolve()}")


def run_ingestion(
    dream_path: Path,
    chroma_path: Path,
    sync_result: SyncResult,
    force_reindex: bool = False,
) -> None:
    """
    Update ChromaDB based on the result of syncing DREAM.

    Decision logic:
        --force           → wipe ChromaDB and full rebuild from all DREAM docs
        was_cloned=True   → full build (first run, ChromaDB is empty)
        ChromaDB empty    → full build (e.g., user deleted data/)
        changed=False     → nothing to do (library and index are in sync)
        changed=True      → incremental: remove deleted docs, index new docs

    Args:
        dream_path:    Path to the DREAM repository root.
        chroma_path:   Path to the ChromaDB persistent storage directory.
        sync_result:   Result from sync_dream().
        force_reindex: If True, ignore sync result and rebuild everything.
    """
    try:
        import chromadb as _chromadb
        from src.indexer import (
            CHROMA_COLLECTION,
            build_index,
            index_specific_documents,
            remove_documents_from_index,
        )
    except ImportError as e:
        logger.error(f"Import error: {e}\nRun: pip install -r requirements.txt")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("conquest.ai — Document Ingestion Pipeline")
    logger.info("=" * 60)

    # ── Force: full rebuild ────────────────────────────────────────────────
    if force_reindex:
        logger.info("--force specified: rebuilding entire index from scratch.")
        build_index(dream_path, chroma_path, force_reindex=True)
        _finish()
        return

    # ── First-time clone: full build ───────────────────────────────────────
    if sync_result.was_cloned:
        logger.info("First-time setup: building full index from DREAM library.")
        build_index(dream_path, chroma_path, force_reindex=False)
        _finish()
        return

    # ── Check if ChromaDB is empty (e.g., user deleted data/) ─────────────
    chroma_client = _chromadb.PersistentClient(path=str(chroma_path))
    collection = chroma_client.get_or_create_collection(CHROMA_COLLECTION)
    if collection.count() == 0:
        logger.info("ChromaDB is empty — building full index.")
        build_index(dream_path, chroma_path, force_reindex=False)
        _finish()
        return

    # ── No upstream changes: nothing to do ────────────────────────────────
    if not sync_result.changed:
        logger.info(
            f"DREAM library unchanged. ChromaDB has {collection.count()} chunks. "
            "Index is up to date — nothing to do."
        )
        _finish()
        return

    # ── Incremental update ─────────────────────────────────────────────────
    logger.info("Applying incremental index update...")

    if sync_result.removed:
        logger.info(f"Removing {len(sync_result.removed)} deleted/modified document(s)...")
        removed = remove_documents_from_index(chroma_path, sync_result.removed)
        logger.info(f"  {removed} chunks removed from ChromaDB.")

    if sync_result.added:
        logger.info(f"Indexing {len(sync_result.added)} new/modified document(s)...")
        indexed = index_specific_documents(dream_path, chroma_path, sync_result.added)
        logger.info(f"  {indexed} document(s) indexed.")

    logger.info(f"ChromaDB now has {collection.count()} total chunks.")
    _finish()


def _finish() -> None:
    logger.info("")
    logger.info("=" * 60)
    logger.info("Done! Start the chatbot with: chainlit run app.py")
    logger.info("=" * 60)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "conquest.ai ingestion — syncs the DREAM library and updates ChromaDB. "
            "Safe to run repeatedly; only changed documents are re-indexed."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ingest.py                         # First-time setup or incremental update
  python ingest.py --force                 # Wipe index and rebuild from scratch
  python ingest.py --dream-path /my/dream  # Use a custom DREAM path
  python ingest.py --no-sync               # Skip git pull, use local DREAM as-is
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
        help="Wipe ChromaDB and rebuild the full index from scratch",
    )
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help="Skip git clone/pull — use whatever is already at --dream-path",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Step 1: Sync DREAM library with remote
    if args.no_sync:
        if not args.dream_path.exists():
            logger.error(f"--no-sync specified but {args.dream_path} does not exist.")
            sys.exit(1)
        sync_result = SyncResult(was_cloned=False, changed=False)
        logger.info(f"Skipping git sync. Using DREAM at {args.dream_path}")
    else:
        sync_result = sync_dream(args.dream_path)

    # Step 2: Ensure ChromaDB directory exists
    ensure_chroma_dir(args.chroma_path)

    # Step 3: Update index based on what changed
    run_ingestion(
        dream_path=args.dream_path,
        chroma_path=args.chroma_path,
        sync_result=sync_result,
        force_reindex=args.force,
    )
