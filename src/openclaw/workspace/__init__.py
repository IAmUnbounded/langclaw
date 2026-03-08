"""Workspace manager — manages agent workspace and default files.

Mirrors OpenClaw's workspace structure:
~/.openclaw/workspace/
    AGENTS.md     — operating instructions
    SOUL.md       — persona, boundaries, tone
    TOOLS.md      — user tool notes
    IDENTITY.md   — agent name/vibe/emoji
    USER.md       — user profile
    skills/       — workspace skills
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

DEFAULT_WORKSPACE = Path.home() / ".openclaw" / "workspace"

# Path to default files bundled with the package
DEFAULTS_DIR = Path(__file__).parent / "defaults"


def ensure_workspace(workspace: Path | None = None) -> Path:
    """Create workspace directory with default files if it doesn't exist."""
    ws = workspace or DEFAULT_WORKSPACE
    ws.mkdir(parents=True, exist_ok=True)

    # Create skills directory
    (ws / "skills").mkdir(exist_ok=True)

    # Copy default files if they don't exist
    for default_file in DEFAULTS_DIR.glob("*.md"):
        target = ws / default_file.name
        if not target.exists():
            target.write_text(
                default_file.read_text(encoding="utf-8"),
                encoding="utf-8",
            )

    return ws


def get_workspace_files(workspace: Path | None = None) -> dict[str, str]:
    """Get all workspace files and their contents."""
    ws = workspace or DEFAULT_WORKSPACE
    files: dict[str, str] = {}
    if ws.exists():
        for f in ws.glob("*.md"):
            try:
                files[f.name] = f.read_text(encoding="utf-8")
            except Exception:
                files[f.name] = ""
    return files


def update_workspace_file(name: str, content: str, workspace: Path | None = None) -> None:
    """Update a workspace file."""
    ws = workspace or DEFAULT_WORKSPACE
    ws.mkdir(parents=True, exist_ok=True)
    (ws / name).write_text(content, encoding="utf-8")
