"""Tool result summarization — smart summaries of tool outputs.

Mirrors openclaw's tool-summaries.ts:
- Builds a summary map from tool descriptions
- Provides per-tool summarization strategies
- Integrates with the compressor for smarter context reduction

Architecture:
    Each tool has a known output pattern. When compressing context,
    the summarizer uses these patterns to create concise summaries
    instead of just truncating. For example:
    - bash output → keep exit code + first/last N lines
    - file read → keep file path + line count
    - web search → keep result titles + URLs
    - git log → keep commit hashes + messages
"""

from __future__ import annotations

import re
import logging
from typing import Any

from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)

# Maximum summary length for a tool result
MAX_SUMMARY_CHARS = 500


def build_tool_summary_map(tools: list[BaseTool]) -> dict[str, str]:
    """Build a map of tool names to their descriptions.

    Used for quick lookup when summarizing tool results in compaction.

    Args:
        tools: List of available tools.

    Returns:
        Dict mapping tool name (lowercase) to description.
    """
    return {t.name.lower(): (t.description or "")[:200] for t in tools}


# Per-tool summarization strategies
def _summarize_bash_output(content: str) -> str:
    """Summarize bash tool output: keep exit info + first/last lines."""
    lines = content.strip().split("\n")
    if len(lines) <= 6:
        return content

    head = lines[:3]
    tail = lines[-3:]
    return "\n".join(head) + f"\n... ({len(lines) - 6} lines omitted) ...\n" + "\n".join(tail)


def _summarize_file_read(content: str) -> str:
    """Summarize file read: keep file header + line count."""
    lines = content.strip().split("\n")
    if len(lines) <= 10:
        return content

    head = lines[:5]
    return "\n".join(head) + f"\n... ({len(lines) - 5} more lines, {len(content)} chars total)"


def _summarize_search_results(content: str) -> str:
    """Summarize web/grep search: keep result titles/matches."""
    lines = content.strip().split("\n")
    if len(lines) <= 8:
        return content

    # Keep first 6 result lines
    results = lines[:6]
    return "\n".join(results) + f"\n... ({len(lines) - 6} more results)"


def _summarize_git_output(content: str) -> str:
    """Summarize git output: keep commit lines."""
    lines = content.strip().split("\n")
    if len(lines) <= 10:
        return content

    # Keep first 8 lines (usually commit hashes + messages)
    return "\n".join(lines[:8]) + f"\n... ({len(lines) - 8} more entries)"


def _summarize_generic(content: str) -> str:
    """Generic summarization: head + tail."""
    if len(content) <= MAX_SUMMARY_CHARS:
        return content

    half = MAX_SUMMARY_CHARS // 2
    return content[:half] + f"\n... (truncated {len(content) - MAX_SUMMARY_CHARS} chars) ...\n" + content[-half:]


# Map tool names to their summarization strategies
TOOL_SUMMARIZERS: dict[str, Any] = {
    "bash": _summarize_bash_output,
    "terminal_run": _summarize_bash_output,
    "terminal_read": _summarize_bash_output,
    "read_file": _summarize_file_read,
    "web_search": _summarize_search_results,
    "web_browse": _summarize_search_results,
    "grep_search": _summarize_search_results,
    "find_files": _summarize_search_results,
    "git_log": _summarize_git_output,
    "git_diff": _summarize_git_output,
    "git_status": _summarize_git_output,
}


def summarize_tool_result(tool_name: str, content: str) -> str:
    """Summarize a tool result using the appropriate strategy.

    Args:
        tool_name: Name of the tool that produced the output.
        content: The tool's output content.

    Returns:
        Summarized content (possibly unchanged if already short).
    """
    if len(content) <= MAX_SUMMARY_CHARS:
        return content

    summarizer = TOOL_SUMMARIZERS.get(tool_name, _summarize_generic)
    result = summarizer(content)

    # Final length guard
    if len(result) > MAX_SUMMARY_CHARS * 2:
        result = result[:MAX_SUMMARY_CHARS] + f"\n... (summarized from {len(content)} chars)"

    return result


def summarize_tool_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Apply per-tool summarization to all ToolMessages in a list.

    This is used during compaction to create smarter summaries
    than simple truncation.

    Args:
        messages: Messages possibly containing ToolMessages.

    Returns:
        Messages with ToolMessage content summarized.
    """
    result: list[BaseMessage] = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            tool_name = getattr(msg, "name", "") or ""
            summarized = summarize_tool_result(tool_name, content)
            if summarized != content:
                result.append(ToolMessage(
                    content=summarized,
                    tool_call_id=msg.tool_call_id,
                    name=tool_name,
                ))
                continue
        result.append(msg)
    return result


def extract_key_facts_from_tool_output(tool_name: str, content: str) -> dict[str, str]:
    """Extract structured key facts from tool output for memory injection.

    Mirrors openclaw's tool result detail extraction for context summaries.
    Returns a dict of {fact_type: value} for the most important facts.
    """
    facts: dict[str, str] = {}
    lines = content.strip().split("\n") if content else []

    if tool_name in ("bash", "terminal_run"):
        # Capture exit code and first line of output
        for line in lines:
            if "exit code" in line.lower() or "returncode" in line.lower():
                facts["exit_code"] = line.strip()[:80]
                break
        if lines:
            facts["first_line"] = lines[0][:100]

    elif tool_name == "read_file":
        # Capture file header (shebang, docstring, etc.)
        for line in lines[:3]:
            if line.strip():
                facts["file_start"] = line.strip()[:100]
                break

    elif tool_name in ("web_search", "grep_search"):
        # First result
        if lines:
            facts["first_result"] = lines[0][:150]
        facts["total_results"] = str(len(lines))

    elif tool_name in ("git_status", "git_diff", "git_log"):
        # Summary line
        if lines:
            facts["summary"] = lines[0][:100]
        facts["line_count"] = str(len(lines))

    elif tool_name == "memory_search":
        try:
            import json
            data = json.loads(content)
            results = data.get("results", [])
            if results:
                facts["top_result"] = str(results[0].get("content", ""))[:150]
            facts["result_count"] = str(len(results))
        except Exception:
            facts["output_preview"] = content[:150]

    return facts


def build_tool_call_summary(
    tool_name: str,
    args: dict,
    output: str,
    elapsed_ms: float | None = None,
) -> str:
    """Build a one-line summary of a tool call for context injection.

    Used in compaction summaries to convey what happened without
    including the full output.
    """
    args_str = ", ".join(
        f"{k}={str(v)[:40]}"
        for k, v in (args or {}).items()
        if k not in ("content",)  # Skip large content params
    )
    call_str = f"{tool_name}({args_str})" if args_str else tool_name

    output_preview = output[:120].strip().replace("\n", " ") if output else ""
    elapsed_str = f" [{elapsed_ms:.0f}ms]" if elapsed_ms is not None else ""

    return f"→ {call_str}: {output_preview}{elapsed_str}"
