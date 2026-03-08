"""Shell command execution tool — equivalent to OpenClaw's bash tool."""

from __future__ import annotations

import asyncio
import subprocess
import sys

from langchain_core.tools import tool


@tool
def bash_tool(command: str, timeout: int = 30) -> str:
    """Execute a shell command and return stdout + stderr.

    Use this to run any shell command on the host system.
    Examples: list files, install packages, run scripts, check system info.

    Args:
        command: The shell command to execute.
        timeout: Maximum seconds to wait (default 30).

    Returns:
        Combined stdout and stderr output, or error message.
    """
    try:
        # Use powershell on Windows, bash on Unix
        if sys.platform == "win32":
            shell_cmd = ["powershell", "-Command", command]
        else:
            shell_cmd = ["bash", "-c", command]

        result = subprocess.run(
            shell_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=None,  # Use current working directory
        )

        output_parts = []
        if result.stdout.strip():
            output_parts.append(result.stdout.strip())
        if result.stderr.strip():
            output_parts.append(f"[stderr] {result.stderr.strip()}")

        output = "\n".join(output_parts) if output_parts else "(no output)"

        if result.returncode != 0:
            output = f"[exit code: {result.returncode}]\n{output}"

        # Truncate very long output
        if len(output) > 10000:
            output = output[:10000] + "\n... (output truncated)"

        return output

    except subprocess.TimeoutExpired:
        return f"[error] Command timed out after {timeout}s: {command}"
    except Exception as e:
        return f"[error] Failed to execute command: {e}"
