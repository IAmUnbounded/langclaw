"""Git integration tools — version control operations.

Provides the agent with git capabilities:
- Status, diff, log for understanding the repo state
- Commit, branch for making changes
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from langchain_core.tools import tool


def _run_git(args: list[str], cwd: str | None = None, timeout: int = 15) -> str:
    """Run a git command and return output."""
    try:
        result = subprocess.run(
            ["git"] + args,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        output = ""
        if result.stdout.strip():
            output = result.stdout.strip()
        if result.stderr.strip():
            stderr = result.stderr.strip()
            # Git sometimes writes non-error info to stderr
            if result.returncode != 0:
                output = f"{output}\n[stderr] {stderr}".strip()
        if result.returncode != 0 and not output:
            output = f"[exit code: {result.returncode}]"
        return output or "(no output)"
    except FileNotFoundError:
        return "[error] git is not installed or not in PATH"
    except subprocess.TimeoutExpired:
        return f"[error] git command timed out after {timeout}s"
    except Exception as e:
        return f"[error] {e}"


@tool
def git_status_tool(path: str = ".") -> str:
    """Show the working tree status of a git repository.

    Args:
        path: Path to the repository (default: current directory).

    Returns:
        Git status output with branch info and changed files.
    """
    branch = _run_git(["branch", "--show-current"], cwd=path)
    status = _run_git(["status", "--short", "--branch"], cwd=path)
    return f"📌 Branch: {branch}\n\n{status}"


@tool
def git_diff_tool(path: str = ".", staged: bool = False, commit: str = "") -> str:
    """Show git diff for changed files.

    Args:
        path: Path to the repository (default: current directory).
        staged: If True, show staged changes. If False, show unstaged.
        commit: Optional commit hash to diff against (e.g. HEAD~3).

    Returns:
        Diff output showing file changes.
    """
    args = ["diff"]
    if staged:
        args.append("--staged")
    if commit:
        args.append(commit)
    args.extend(["--stat"])

    stat = _run_git(args, cwd=path)

    # Also get the actual diff (truncated)
    diff_args = ["diff"]
    if staged:
        diff_args.append("--staged")
    if commit:
        diff_args.append(commit)

    full_diff = _run_git(diff_args, cwd=path)

    if len(full_diff) > 8000:
        full_diff = full_diff[:8000] + "\n\n... (diff truncated)"

    return f"📊 Diff summary:\n{stat}\n\n{full_diff}"


@tool
def git_log_tool(path: str = ".", count: int = 10, oneline: bool = True) -> str:
    """Show recent git commit history.

    Args:
        path: Path to the repository (default: current directory).
        count: Number of commits to show (default 10).
        oneline: If True, compact one-line format.

    Returns:
        Commit history.
    """
    args = ["log", f"-{count}"]
    if oneline:
        args.append("--oneline")
    args.append("--decorate")

    return _run_git(args, cwd=path)


@tool
def git_commit_tool(message: str, path: str = ".", add_all: bool = True) -> str:
    """Create a git commit.

    Args:
        message: Commit message.
        path: Path to the repository (default: current directory).
        add_all: If True, stage all changes before committing.

    Returns:
        Commit result.
    """
    if add_all:
        add_result = _run_git(["add", "-A"], cwd=path)
        if "[error]" in add_result:
            return f"Failed to stage: {add_result}"

    return _run_git(["commit", "-m", message], cwd=path)


@tool
def git_branch_tool(
    path: str = ".",
    create: str = "",
    switch: str = "",
    delete: str = "",
) -> str:
    """List, create, switch, or delete git branches.

    Args:
        path: Path to the repository (default: current directory).
        create: Name of a new branch to create.
        switch: Name of a branch to switch to.
        delete: Name of a branch to delete.

    Returns:
        Branch operation result.
    """
    if create:
        return _run_git(["checkout", "-b", create], cwd=path)
    elif switch:
        return _run_git(["checkout", switch], cwd=path)
    elif delete:
        return _run_git(["branch", "-d", delete], cwd=path)
    else:
        return _run_git(["branch", "-a", "--list"], cwd=path)
