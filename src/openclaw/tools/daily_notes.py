"""Daily memory notes — date-based markdown journal for the agent.

Provides tools for reading, writing, and listing daily memory notes
stored as markdown files in ~/.langclaw/workspace/memory/YYYY-MM-DD.md.

Examples:
- "Read today's notes"
- "Add a note about the deployment we did"
- "Show me the last 7 days of notes"
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.tools import tool

DAILY_NOTES_DIR = Path.home() / ".langclaw" / "workspace" / "memory"


def _resolve_date(date: str) -> str:
    """Resolve a date string to YYYY-MM-DD format, defaulting to today."""
    if not date or not date.strip():
        return datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    # Validate the format
    try:
        datetime.strptime(date.strip(), "%Y-%m-%d")
    except ValueError:
        raise ValueError(
            f"Invalid date format: {date!r}. Expected YYYY-MM-DD."
        )
    return date.strip()


def _note_path(date: str) -> Path:
    """Return the file path for a given date's daily note."""
    return DAILY_NOTES_DIR / f"{date}.md"


@tool
def daily_note_read_tool(date: str = "") -> str:
    """Read daily memory notes for a given date (YYYY-MM-DD). Defaults to today.

    Args:
        date: Date in YYYY-MM-DD format. Leave empty for today.

    Returns:
        Contents of the daily note, or a message if none exists.
    """
    try:
        resolved = _resolve_date(date)
    except ValueError as e:
        return str(e)

    path = _note_path(resolved)

    if not path.exists():
        return f"No daily note found for {resolved}."

    content = path.read_text(encoding="utf-8")
    return f"**Daily Note — {resolved}**\n\n{content}"


@tool
def daily_note_write_tool(content: str, date: str = "") -> str:
    """Write/append to daily memory notes for a given date. Defaults to today.

    Each entry is appended with a timestamp header so multiple entries
    can be added to the same day.

    Args:
        content: The note content to append.
        date: Date in YYYY-MM-DD format. Leave empty for today.

    Returns:
        Confirmation that the note was written.
    """
    try:
        resolved = _resolve_date(date)
    except ValueError as e:
        return str(e)

    DAILY_NOTES_DIR.mkdir(parents=True, exist_ok=True)

    path = _note_path(resolved)
    is_new = not path.exists()

    now = datetime.now(tz=timezone.utc)
    timestamp = now.strftime("%H:%M:%S UTC")

    entry = f"\n## {timestamp}\n\n{content}\n"

    if is_new:
        # Create file with a top-level date heading
        header = f"# Daily Notes — {resolved}\n"
        path.write_text(header + entry, encoding="utf-8")
    else:
        with open(path, "a", encoding="utf-8") as f:
            f.write(entry)

    size = path.stat().st_size
    return (
        f"Note {'created' if is_new else 'appended'} for {resolved} "
        f"at {timestamp} ({size} bytes total)."
    )


@tool
def daily_note_list_tool(count: int = 7) -> str:
    """List recent daily note files.

    Args:
        count: Number of recent daily notes to list (default 7).

    Returns:
        List of recent daily note files with dates and sizes.
    """
    if not DAILY_NOTES_DIR.exists():
        return "No daily notes directory found. No notes have been written yet."

    # Collect all YYYY-MM-DD.md files
    notes: list[tuple[str, Path]] = []
    for entry in DAILY_NOTES_DIR.iterdir():
        if entry.suffix == ".md" and entry.stem != "":
            # Validate that the stem is a valid date
            try:
                datetime.strptime(entry.stem, "%Y-%m-%d")
                notes.append((entry.stem, entry))
            except ValueError:
                continue

    if not notes:
        return "No daily notes found."

    # Sort by date descending (most recent first)
    notes.sort(key=lambda x: x[0], reverse=True)
    notes = notes[:count]

    lines = [f"**Recent Daily Notes** ({len(notes)} shown)\n"]
    for date_str, path in notes:
        size = path.stat().st_size
        # Count non-empty lines as a rough measure of content
        line_count = sum(
            1 for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        lines.append(f"- **{date_str}** — {size:,} bytes, {line_count} lines")

    return "\n".join(lines)
