"""File operation tools — read, write, edit, and list files."""

from __future__ import annotations

import os
from pathlib import Path

from langchain_core.tools import tool


@tool
def read_file_tool(path: str, start_line: int = 0, end_line: int = 0) -> str:
    """Read the contents of a file.

    Args:
        path: Absolute or relative path to the file.
        start_line: Optional 1-indexed start line (0 = from beginning).
        end_line: Optional 1-indexed end line (0 = to end).

    Returns:
        File contents, optionally with line numbers.
    """
    try:
        file_path = Path(path).expanduser().resolve()
        if not file_path.exists():
            return f"[error] File not found: {path}"
        if not file_path.is_file():
            return f"[error] Not a file: {path}"

        content = file_path.read_text(encoding="utf-8")
        lines = content.split("\n")

        if start_line > 0 or end_line > 0:
            start = max(0, start_line - 1) if start_line > 0 else 0
            end = end_line if end_line > 0 else len(lines)
            selected = lines[start:end]
            numbered = [f"{i + start + 1:4d} | {line}" for i, line in enumerate(selected)]
            return "\n".join(numbered)

        # For full file, add line numbers if file is reasonably sized
        if len(lines) <= 500:
            numbered = [f"{i + 1:4d} | {line}" for i, line in enumerate(lines)]
            return "\n".join(numbered)

        # Truncate very large files
        if len(content) > 50000:
            return content[:50000] + f"\n\n... (file truncated, {len(lines)} total lines)"
        return content

    except UnicodeDecodeError:
        return f"[error] Cannot read binary file: {path}"
    except Exception as e:
        return f"[error] {e}"


@tool
def write_file_tool(path: str, content: str, create_dirs: bool = True) -> str:
    """Create or overwrite a file with the given content.

    Args:
        path: Absolute or relative path to the file.
        content: The content to write.
        create_dirs: Whether to create parent directories if needed (default True).

    Returns:
        Success or error message.
    """
    try:
        file_path = Path(path).expanduser().resolve()
        if create_dirs:
            file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return f"✅ Written {len(content)} chars to {file_path}"
    except Exception as e:
        return f"[error] Failed to write file: {e}"


@tool
def edit_file_tool(path: str, search: str, replace: str) -> str:
    """Edit a file by replacing a specific text pattern.

    Finds the exact `search` string in the file and replaces it with `replace`.

    Args:
        path: Path to the file to edit.
        search: The exact text to find (must be unique in the file).
        replace: The replacement text.

    Returns:
        Success or error message.
    """
    try:
        file_path = Path(path).expanduser().resolve()
        if not file_path.exists():
            return f"[error] File not found: {path}"

        content = file_path.read_text(encoding="utf-8")
        count = content.count(search)

        if count == 0:
            return f"[error] Search text not found in {path}"
        if count > 1:
            return f"[error] Search text found {count} times — must be unique. Be more specific."

        new_content = content.replace(search, replace, 1)
        file_path.write_text(new_content, encoding="utf-8")
        return f"✅ Replaced 1 occurrence in {file_path}"

    except Exception as e:
        return f"[error] Failed to edit file: {e}"


@tool
def list_directory_tool(path: str = ".", max_items: int = 100) -> str:
    """List contents of a directory.

    Args:
        path: Directory path (default: current directory).
        max_items: Maximum items to list (default 100).

    Returns:
        Formatted directory listing with file types and sizes.
    """
    try:
        dir_path = Path(path).expanduser().resolve()
        if not dir_path.exists():
            return f"[error] Directory not found: {path}"
        if not dir_path.is_dir():
            return f"[error] Not a directory: {path}"

        entries = sorted(dir_path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        lines = [f"📁 {dir_path}\n"]
        count = 0

        for entry in entries:
            if count >= max_items:
                remaining = len(list(dir_path.iterdir())) - max_items
                lines.append(f"  ... and {remaining} more items")
                break

            if entry.is_dir():
                child_count = sum(1 for _ in entry.iterdir()) if entry.exists() else 0
                lines.append(f"  📂 {entry.name}/ ({child_count} items)")
            else:
                size = entry.stat().st_size
                if size < 1024:
                    size_str = f"{size} B"
                elif size < 1024 * 1024:
                    size_str = f"{size / 1024:.1f} KB"
                else:
                    size_str = f"{size / (1024 * 1024):.1f} MB"
                lines.append(f"  📄 {entry.name} ({size_str})")
            count += 1

        return "\n".join(lines)

    except PermissionError:
        return f"[error] Permission denied: {path}"
    except Exception as e:
        return f"[error] {e}"
