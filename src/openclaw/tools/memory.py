"""Memory tools — persistent long-term recall for the agent.

Mirrors openclaw's memory-tool.ts:
- memory_search: Mandatory recall step — semantically search memory files
- memory_get: Safe snippet read from specific memory files with line ranges
- memory_save: Save important information to long-term memory
- memory_list: List recent memories chronologically

The memory system indexes MEMORY.md + memory/*.md and optionally session transcripts.
Uses ChromaDB for vector search (or BM25 fallback), with hybrid scoring.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Where workspace memory files are stored
_MEMORY_BASE_PATH = Path.home() / ".langclaw" / "workspace"


def _resolve_memory_path(rel_path: str) -> Path | None:
    """Resolve a relative memory path to an absolute path."""
    # Strip leading slashes or dots
    clean = rel_path.lstrip("./").strip()
    if not clean:
        return None

    # Allow access to MEMORY.md and memory/ directory only
    candidates = [
        _MEMORY_BASE_PATH / clean,
        _MEMORY_BASE_PATH / "memory" / clean,
        Path.home() / ".langclaw" / "memory" / clean,
    ]

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            # Security: ensure it stays within ~/.langclaw
            try:
                candidate.resolve().relative_to(Path.home() / ".langclaw")
                return candidate
            except ValueError:
                pass

    return None


@tool
def memory_search_tool(
    query: str,
    max_results: int = 6,
    min_score: float = 0.0,
) -> str:
    """Mandatory recall step: semantically search MEMORY.md + memory/*.md before answering questions
    about prior work, decisions, dates, people, preferences, or todos.

    Returns top snippets with path + line numbers so you can use memory_get for full context.

    Args:
        query: What to search for (natural language).
        max_results: Maximum results to return (default 6).
        min_score: Minimum relevance score threshold 0.0–1.0 (default 0).

    Returns:
        JSON with matching memory snippets including path and line information.
    """
    from openclaw.memory import MemoryStore

    store = MemoryStore()

    try:
        results = store.hybrid_search(query, top_k=max_results, min_score=min_score)
    except AttributeError:
        results = store.search(query, top_k=max_results)

    if not results:
        return json.dumps({
            "results": [],
            "query": query,
            "text": f"No memories found matching: {query}",
        })

    result_list = []
    for entry in results:
        result_list.append({
            "memory_id": entry.memory_id,
            "content": entry.content,
            "snippet": entry.content[:500],
            "score": getattr(entry, "score", 1.0),
            "metadata": entry.metadata,
        })

    return json.dumps({
        "results": result_list,
        "query": query,
        "count": len(result_list),
    }, indent=2)


@tool
def memory_get_tool(
    path: str,
    from_line: Optional[int] = None,
    lines: Optional[int] = None,
) -> str:
    """Safe snippet read from MEMORY.md or memory/*.md with optional line range.

    Use after memory_search to pull only the needed lines and keep context small.
    Relative to ~/.langclaw/workspace/.

    Args:
        path: Relative path to the memory file (e.g., "MEMORY.md", "memory/user.md").
        from_line: Starting line number (1-based, optional).
        lines: Number of lines to read (optional, default = all remaining).

    Returns:
        JSON with file path, content, and line info.
    """
    resolved = _resolve_memory_path(path)
    if not resolved:
        return json.dumps({
            "path": path,
            "text": "",
            "error": f"Memory file not found: {path}",
        })

    try:
        all_lines = resolved.read_text(encoding="utf-8").splitlines()
        total = len(all_lines)

        if from_line is not None:
            start = max(0, from_line - 1)  # Convert 1-based to 0-based
        else:
            start = 0

        if lines is not None:
            end = min(start + lines, total)
        else:
            end = total

        selected = all_lines[start:end]
        text = "\n".join(selected)

        return json.dumps({
            "path": path,
            "text": text,
            "from_line": start + 1,
            "to_line": end,
            "total_lines": total,
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "path": path,
            "text": "",
            "error": str(e),
        })


@tool
def memory_save_tool(content: str, tags: str = "") -> str:
    """Save important information to long-term memory for future recall.

    Use this to remember user preferences, project details, decisions,
    or any information that should persist across sessions.

    Args:
        content: The information to remember. Be descriptive and specific.
        tags: Optional comma-separated tags for categorization.

    Returns:
        Confirmation with memory ID.
    """
    from openclaw.memory import MemoryStore

    metadata: dict = {}
    if tags:
        metadata["tags"] = [t.strip() for t in tags.split(",") if t.strip()]

    store = MemoryStore()
    entry = store.save(content, metadata)

    return json.dumps({
        "status": "ok",
        "memory_id": entry.memory_id,
        "content_preview": content[:100] + ("..." if len(content) > 100 else ""),
        "text": f"Memory saved (id={entry.memory_id})",
    })


@tool
def memory_list_tool(count: int = 10) -> str:
    """List recent memories chronologically.

    Args:
        count: Number of recent memories to show (default 10, max 50).

    Returns:
        JSON list of recent memories.
    """
    from openclaw.memory import MemoryStore

    count = max(1, min(50, count))
    store = MemoryStore()
    entries = store.list_recent(count)

    if not entries:
        return json.dumps({
            "count": 0,
            "entries": [],
            "text": "No memories saved yet. Use memory_save to store information.",
        })

    result_entries = []
    for entry in entries:
        from datetime import datetime, timezone
        dt = datetime.fromtimestamp(entry.timestamp, tz=timezone.utc)
        result_entries.append({
            "memory_id": entry.memory_id,
            "content": entry.content,
            "preview": entry.content[:120] + ("..." if len(entry.content) > 120 else ""),
            "timestamp": dt.strftime("%Y-%m-%d %H:%M UTC"),
            "tags": entry.metadata.get("tags", []),
        })

    return json.dumps({
        "count": len(result_entries),
        "entries": result_entries,
    }, indent=2)
