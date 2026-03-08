"""Terminal management tools — persistent subprocess sessions.

Allows the agent to run long-running commands, send input to them,
read their output, and manage multiple concurrent terminal sessions.

Each terminal is a managed subprocess tracked by a unique ID.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from langchain_core.tools import tool


@dataclass
class ManagedTerminal:
    """A managed terminal subprocess."""

    terminal_id: str
    command: str
    process: subprocess.Popen
    started_at: float
    output_buffer: list[str] = field(default_factory=list)
    error_buffer: list[str] = field(default_factory=list)
    _reader_thread: threading.Thread | None = None
    _error_thread: threading.Thread | None = None

    @property
    def is_running(self) -> bool:
        return self.process.poll() is None

    @property
    def exit_code(self) -> int | None:
        return self.process.poll()

    @property
    def elapsed(self) -> float:
        return time.time() - self.started_at


class TerminalManager:
    """Singleton manager for all active terminal sessions."""

    _instance: "TerminalManager | None" = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._terminals = {}
            return cls._instance

    @property
    def terminals(self) -> dict[str, ManagedTerminal]:
        return self._terminals

    def create(self, command: str, cwd: str | None = None) -> ManagedTerminal:
        """Create and start a new terminal session."""
        terminal_id = str(uuid.uuid4())[:8]

        if sys.platform == "win32":
            shell_cmd = ["powershell", "-Command", command]
        else:
            shell_cmd = ["bash", "-c", command]

        process = subprocess.Popen(
            shell_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,
            text=True,
            cwd=cwd,
            bufsize=1,
        )

        terminal = ManagedTerminal(
            terminal_id=terminal_id,
            command=command,
            process=process,
            started_at=time.time(),
        )

        # Start output reader threads
        def read_stdout():
            try:
                for line in process.stdout:
                    terminal.output_buffer.append(line)
                    # Keep buffer manageable
                    if len(terminal.output_buffer) > 1000:
                        terminal.output_buffer = terminal.output_buffer[-500:]
            except Exception:
                pass

        def read_stderr():
            try:
                for line in process.stderr:
                    terminal.error_buffer.append(line)
                    if len(terminal.error_buffer) > 500:
                        terminal.error_buffer = terminal.error_buffer[-250:]
            except Exception:
                pass

        stdout_thread = threading.Thread(target=read_stdout, daemon=True)
        stderr_thread = threading.Thread(target=read_stderr, daemon=True)
        stdout_thread.start()
        stderr_thread.start()

        terminal._reader_thread = stdout_thread
        terminal._error_thread = stderr_thread

        self._terminals[terminal_id] = terminal
        return terminal

    def get(self, terminal_id: str) -> ManagedTerminal | None:
        return self._terminals.get(terminal_id)

    def kill(self, terminal_id: str) -> bool:
        terminal = self._terminals.get(terminal_id)
        if not terminal:
            return False
        try:
            terminal.process.terminate()
            terminal.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            terminal.process.kill()
        except Exception:
            pass
        return True

    def list_all(self) -> list[dict[str, Any]]:
        results = []
        for tid, t in self._terminals.items():
            results.append({
                "terminal_id": tid,
                "command": t.command[:80],
                "running": t.is_running,
                "elapsed": round(t.elapsed, 1),
                "exit_code": t.exit_code,
                "output_lines": len(t.output_buffer),
            })
        return results

    def cleanup_dead(self):
        """Remove terminals that have been dead for > 10 minutes."""
        to_remove = []
        for tid, t in self._terminals.items():
            if not t.is_running and t.elapsed > 600:
                to_remove.append(tid)
        for tid in to_remove:
            del self._terminals[tid]


@tool
def terminal_run_tool(command: str, cwd: str = "") -> str:
    """Start a long-running command in a managed terminal session.

    The command runs in the background. Use terminal_read to check output
    and terminal_send to provide input.

    Args:
        command: The command to run.
        cwd: Working directory (default: current directory).

    Returns:
        Terminal ID and initial status.
    """
    mgr = TerminalManager()
    terminal = mgr.create(command, cwd=cwd or None)

    # Wait a moment for initial output
    time.sleep(0.5)

    initial_output = "".join(terminal.output_buffer[-10:]) if terminal.output_buffer else "(no output yet)"

    return (
        f"🖥️ Terminal started\n"
        f"  🆔 ID: {terminal.terminal_id}\n"
        f"  📝 Command: {command}\n"
        f"  {'🟢 Running' if terminal.is_running else '🔴 Exited'}\n"
        f"\nInitial output:\n{initial_output}"
    )


@tool
def terminal_read_tool(terminal_id: str, last_n_lines: int = 50) -> str:
    """Read output from a managed terminal session.

    Args:
        terminal_id: The terminal ID returned by terminal_run.
        last_n_lines: Number of recent output lines to return (default 50).

    Returns:
        Recent output from the terminal.
    """
    mgr = TerminalManager()
    terminal = mgr.get(terminal_id)

    if not terminal:
        available = [t["terminal_id"] for t in mgr.list_all()]
        return f"❌ Terminal not found: {terminal_id}\nAvailable: {', '.join(available) or 'none'}"

    status = "🟢 Running" if terminal.is_running else f"🔴 Exited (code: {terminal.exit_code})"
    elapsed = f"{terminal.elapsed:.1f}s"

    stdout_lines = terminal.output_buffer[-last_n_lines:]
    stderr_lines = terminal.error_buffer[-10:]

    output = "".join(stdout_lines) if stdout_lines else "(no output)"
    errors = "".join(stderr_lines) if stderr_lines else ""

    result = f"🖥️ Terminal `{terminal_id}` — {status} ({elapsed})\n\n{output}"
    if errors:
        result += f"\n\n[stderr]\n{errors}"

    return result


@tool
def terminal_send_tool(terminal_id: str, input_text: str) -> str:
    """Send input to a running terminal session's stdin.

    Args:
        terminal_id: The terminal ID.
        input_text: Text to send to stdin (newline added automatically).

    Returns:
        Confirmation and recent output.
    """
    mgr = TerminalManager()
    terminal = mgr.get(terminal_id)

    if not terminal:
        return f"❌ Terminal not found: {terminal_id}"
    if not terminal.is_running:
        return f"❌ Terminal {terminal_id} is not running (exit code: {terminal.exit_code})"

    try:
        terminal.process.stdin.write(input_text + "\n")
        terminal.process.stdin.flush()
    except Exception as e:
        return f"❌ Failed to send input: {e}"

    # Wait a moment for response
    time.sleep(0.5)

    recent = "".join(terminal.output_buffer[-10:]) if terminal.output_buffer else "(no output)"
    return f"✅ Sent input to terminal {terminal_id}\n\nRecent output:\n{recent}"


@tool
def terminal_kill_tool(terminal_id: str) -> str:
    """Terminate a running terminal session.

    Args:
        terminal_id: The terminal ID to kill.

    Returns:
        Confirmation or error.
    """
    mgr = TerminalManager()
    terminal = mgr.get(terminal_id)

    if not terminal:
        return f"❌ Terminal not found: {terminal_id}"

    if mgr.kill(terminal_id):
        return f"✅ Terminal {terminal_id} terminated"
    else:
        return f"❌ Failed to terminate terminal {terminal_id}"
