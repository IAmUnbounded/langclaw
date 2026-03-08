"""Code search tools — grep and file finding for codebase exploration.

Provides the agent with fast code search capabilities using
system commands (ripgrep/grep/findstr) and Python glob.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from langchain_core.tools import tool


@tool
def grep_search_tool(
    query: str,
    path: str = ".",
    file_pattern: str = "",
    case_insensitive: bool = True,
    max_results: int = 50,
) -> str:
    """Search for text patterns across files using grep/ripgrep.

    Searches file contents for a pattern and returns matching lines
    with file names and line numbers.

    Args:
        query: Search pattern (literal text or regex).
        path: Directory or file to search in (default: current dir).
        file_pattern: Glob pattern to filter files (e.g. "*.py", "*.js").
        case_insensitive: Whether to ignore case (default True).
        max_results: Maximum matches to return (default 50).

    Returns:
        Formatted search results with file:line:content.
    """
    search_path = Path(path).expanduser().resolve()
    if not search_path.exists():
        return f"[error] Path not found: {path}"

    try:
        # Try ripgrep first (fastest), then grep, then findstr (Windows)
        if _has_command("rg"):
            args = ["rg", "--no-heading", "--line-number", f"--max-count={max_results}"]
            if case_insensitive:
                args.append("-i")
            if file_pattern:
                args.extend(["-g", file_pattern])
            args.extend([query, str(search_path)])
        elif sys.platform != "win32" and _has_command("grep"):
            args = ["grep", "-rn", f"--max-count={max_results}"]
            if case_insensitive:
                args.append("-i")
            if file_pattern:
                args.extend(["--include", file_pattern])
            args.extend([query, str(search_path)])
        elif sys.platform == "win32":
            # Windows fallback using findstr
            args = ["findstr", "/S", "/N"]
            if case_insensitive:
                args.append("/I")
            args.extend([query, str(search_path / (file_pattern or "*"))])
        else:
            return _python_grep(query, search_path, file_pattern, case_insensitive, max_results)

        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=15,
        )

        output = result.stdout.strip()
        if not output:
            return f"No matches found for: {query}"

        # Truncate if too long
        lines = output.split("\n")
        if len(lines) > max_results:
            lines = lines[:max_results]
            output = "\n".join(lines) + f"\n\n... ({len(lines)} matches shown)"
        else:
            output = "\n".join(lines)

        return f"🔍 Search results for '{query}':\n\n{output}"

    except subprocess.TimeoutExpired:
        return "[error] Search timed out"
    except Exception as e:
        # Fallback to Python implementation
        return _python_grep(query, search_path, file_pattern, case_insensitive, max_results)


@tool
def find_files_tool(
    pattern: str,
    path: str = ".",
    file_type: str = "any",
    max_results: int = 50,
) -> str:
    """Find files and directories by name pattern.

    Args:
        pattern: Glob pattern to match (e.g. "*.py", "test_*", "**/*.json").
        path: Directory to search in (default: current dir).
        file_type: Filter by type: "file", "dir", or "any" (default).
        max_results: Maximum results to return (default 50).

    Returns:
        List of matching file paths with sizes.
    """
    search_path = Path(path).expanduser().resolve()
    if not search_path.exists():
        return f"[error] Path not found: {path}"

    try:
        matches = []
        for item in search_path.rglob(pattern):
            # Skip hidden directories and common ignore patterns
            parts = item.relative_to(search_path).parts
            if any(p.startswith(".") or p in ("node_modules", "__pycache__", ".git", "venv") for p in parts):
                continue

            if file_type == "file" and not item.is_file():
                continue
            if file_type == "dir" and not item.is_dir():
                continue

            if item.is_file():
                size = item.stat().st_size
                if size < 1024:
                    size_str = f"{size} B"
                elif size < 1024 * 1024:
                    size_str = f"{size / 1024:.1f} KB"
                else:
                    size_str = f"{size / (1024 * 1024):.1f} MB"
                matches.append(f"  📄 {item.relative_to(search_path)} ({size_str})")
            else:
                matches.append(f"  📂 {item.relative_to(search_path)}/")

            if len(matches) >= max_results:
                break

        if not matches:
            return f"No files found matching: {pattern}"

        header = f"📁 Files matching '{pattern}' in {search_path}:\n"
        result = header + "\n".join(matches)
        if len(matches) >= max_results:
            result += f"\n\n... (showing first {max_results} results)"

        return result

    except Exception as e:
        return f"[error] Find failed: {e}"


def _has_command(cmd: str) -> bool:
    """Check if a command is available on PATH."""
    try:
        subprocess.run(
            [cmd, "--version"],
            capture_output=True,
            timeout=5,
        )
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _python_grep(
    query: str, path: Path, pattern: str, case_insensitive: bool, max_results: int
) -> str:
    """Pure Python grep fallback."""
    import re

    flags = re.IGNORECASE if case_insensitive else 0
    try:
        regex = re.compile(query, flags)
    except re.error:
        regex = re.compile(re.escape(query), flags)

    matches = []
    glob_pattern = pattern or "**/*"

    for file_path in path.rglob(glob_pattern if "**" in glob_pattern else f"**/{glob_pattern}"):
        if not file_path.is_file():
            continue
        parts = file_path.relative_to(path).parts
        if any(p.startswith(".") or p in ("node_modules", "__pycache__", ".git", "venv") for p in parts):
            continue

        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            for i, line in enumerate(content.split("\n"), 1):
                if regex.search(line):
                    rel = file_path.relative_to(path)
                    matches.append(f"{rel}:{i}: {line.strip()}")
                    if len(matches) >= max_results:
                        break
        except Exception:
            continue

        if len(matches) >= max_results:
            break

    if not matches:
        return f"No matches found for: {query}"

    return f"🔍 Search results for '{query}':\n\n" + "\n".join(matches)
