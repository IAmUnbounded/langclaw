"""Session management tools — cross-session interaction.

Mirrors OpenClaw's sessions_spawn, sessions_list, sessions_send, sessions_history,
sessions_status, and subagents tools:
- List all active sessions with kind/channel classification
- View session history with content sanitization
- Send messages to other sessions
- Spawn background sub-agent sessions with announcement
- Check session status
- Access control via visibility guards
"""

from __future__ import annotations

import datetime
import json
import logging
import threading
from typing import Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# History limits
HISTORY_MAX_CHARS_PER_CONTENT = 4000
HISTORY_MAX_MESSAGES = 100


def _sanitize_content(content: Any, max_chars: int = HISTORY_MAX_CHARS_PER_CONTENT) -> str:
    """Sanitize message content for history display."""
    if content is None:
        return ""
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                btype = block.get("type", "")
                if btype == "text":
                    parts.append(block.get("text", ""))
                elif btype == "thinking":
                    # Truncate thinking blocks — they can be enormous
                    thinking = block.get("thinking", "")
                    if thinking:
                        parts.append(f"[thinking: {thinking[:200]}...]")
                elif btype == "image":
                    w = block.get("width", "?")
                    h = block.get("height", "?")
                    parts.append(f"[image {w}x{h}]")
                elif btype == "tool_use":
                    name = block.get("name", "tool")
                    parts.append(f"[tool_use: {name}]")
                elif btype == "tool_result":
                    parts.append("[tool_result]")
            elif isinstance(block, str):
                parts.append(block)
        text = " ".join(p for p in parts if p)
    else:
        text = str(content)

    if len(text) > max_chars:
        text = text[:max_chars] + f"... (truncated {len(text) - max_chars} chars)"
    return text


def _classify_session_kind(session_id: str, sender_id: str, channel: str) -> str:
    """Classify session kind from session metadata."""
    if channel == "subagent" or "subagent:" in session_id:
        return "subagent"
    if channel == "cron":
        return "cron"
    if channel in ("slack", "telegram", "discord", "whatsapp"):
        if "group" in session_id.lower() or "channel" in session_id.lower():
            return "group"
        return "direct"
    if channel == "cli":
        return "main"
    return "other"


@tool
def sessions_list_tool(
    kinds: str = "",
    limit: int = 0,
    active_minutes: int = 0,
) -> str:
    """List conversation sessions with optional filters.

    Args:
        kinds: Comma-separated session kinds to include: main, subagent, cron, direct, group, other.
               Empty = all kinds.
        limit: Maximum number of sessions to return (0 = no limit).
        active_minutes: Only show sessions active within this many minutes (0 = no filter).

    Returns:
        JSON-formatted list of sessions with metadata.
    """
    from openclaw.agent.session import SessionManager

    sm = SessionManager()
    all_sessions = sm.list_sessions()

    # Apply kind filter
    kind_set = {k.strip().lower() for k in kinds.split(",") if k.strip()} if kinds else set()

    now = datetime.datetime.now().timestamp()
    active_cutoff = now - active_minutes * 60 if active_minutes > 0 else 0

    rows = []
    for meta in all_sessions:
        # Active time filter
        if active_cutoff > 0 and meta.last_active < active_cutoff:
            continue

        kind = _classify_session_kind(meta.session_id, meta.sender_id, meta.channel)
        if kind_set and kind not in kind_set:
            continue

        rows.append({
            "session_id": meta.session_id,
            "kind": kind,
            "channel": meta.channel,
            "sender_id": meta.sender_id,
            "message_count": meta.message_count,
            "last_active": datetime.datetime.fromtimestamp(meta.last_active).strftime("%Y-%m-%d %H:%M"),
            "title": meta.title or None,
        })

    if limit > 0:
        rows = rows[:limit]

    if not rows:
        return json.dumps({"count": 0, "sessions": [], "text": "No sessions found."})

    # Format text summary
    lines = [f"Sessions ({len(rows)}):"]
    for r in rows:
        title_part = f" | {r['title']}" if r["title"] else ""
        lines.append(
            f"- {r['session_id']} [{r['kind']}] {r['channel']}:{r['sender_id']} "
            f"| {r['message_count']} msgs | {r['last_active']}{title_part}"
        )

    return json.dumps({
        "count": len(rows),
        "sessions": rows,
        "text": "\n".join(lines),
    }, indent=2)


@tool
def sessions_history_tool(
    session_id: str,
    limit: int = 20,
    include_tools: bool = False,
) -> str:
    """Get conversation history for a specific session.

    Args:
        session_id: The session ID to retrieve history for.
        limit: Maximum number of recent messages to return (default 20, max 100).
        include_tools: Include tool call/result messages (default False).

    Returns:
        JSON with sanitized message history.
    """
    from openclaw.agent.session import SessionManager

    sm = SessionManager()
    history = sm.get_session_history(session_id)

    if history is None:
        return json.dumps({"status": "not_found", "error": f"Session '{session_id}' not found."})

    limit = max(1, min(HISTORY_MAX_MESSAGES, limit))
    messages = history[-limit:]

    sanitized = []
    truncated = False
    for msg in messages:
        role = msg.get("type", "unknown")
        if not include_tools and role in ("tool",):
            continue

        raw_content = msg.get("content", "")
        content_str = _sanitize_content(raw_content)
        if len(content_str) < len(str(raw_content)):
            truncated = True

        entry: dict[str, Any] = {"role": role, "content": content_str}

        # Include tool_calls for AI messages
        if role == "ai" and msg.get("tool_calls"):
            entry["tool_calls"] = [
                {"name": tc.get("name", ""), "args": tc.get("args", {})}
                for tc in msg.get("tool_calls", [])
            ]

        sanitized.append(entry)

    return json.dumps({
        "session_id": session_id,
        "messages": sanitized,
        "total_messages": len(history),
        "returned": len(sanitized),
        "truncated": truncated,
    }, indent=2)


@tool
def sessions_send_tool(session_id: str, text: str) -> str:
    """Send a text message to another session.

    Injects the message into the target session's history so it will be
    processed when that session next runs the agent.

    Args:
        session_id: Target session ID.
        text: Message text to send.

    Returns:
        Confirmation or error message.
    """
    from langchain_core.messages import HumanMessage
    from openclaw.agent.session import SessionManager

    sm = SessionManager()
    session = sm._load_session(session_id)
    if not session:
        return json.dumps({
            "status": "error",
            "error": f"Session '{session_id}' not found.",
        })

    sm.append_message(session, HumanMessage(content=text))
    return json.dumps({
        "status": "ok",
        "session_id": session_id,
        "channel": session.metadata.channel,
        "sender_id": session.metadata.sender_id,
        "text": f"Message sent to session {session_id}",
    })


@tool
async def sessions_spawn_tool(
    task: str,
    label: str = "",
    model: str = "",
    thinking: str = "",
    allowed_tools: str = "",
    run_timeout_seconds: int = 300,
    cleanup: str = "keep",
    session_key: str = "",
) -> str:
    """Spawn a background sub-agent session and announce the result when done.

    The sub-agent runs asynchronously in the background. When it completes,
    it announces its result back to this session. Use subagents_tool(action='list')
    to check progress.

    Args:
        task: Clear task description for the sub-agent.
        label: Short human-readable label (shown in subagents list).
        model: Optional model override (e.g., 'anthropic/claude-sonnet-4-20250514').
        thinking: Optional thinking level: off, low, mid, high, max.
        allowed_tools: Comma-separated tool names (empty = all tools).
        run_timeout_seconds: Maximum seconds for the background run (default 300).
        cleanup: "keep" to retain session after completion, "delete" to remove it.
        session_key: Requester session key (auto-detected if empty).

    Returns:
        JSON with run_id, session key, and status.
    """
    from openclaw.agent.subagent_spawn import SpawnOptions, spawn_subagent_background
    from openclaw.agent.subagent_registry import get_subagent_registry

    tools_list = [t.strip() for t in allowed_tools.split(",") if t.strip()] or None

    options = SpawnOptions(
        task=task,
        label=label,
        model=model or None,
        thinking=thinking or None,
        allowed_tools=tools_list,
        run_timeout_seconds=run_timeout_seconds,
        cleanup=cleanup,
    )

    # Resolve requester session key from caller context
    requester_key = session_key.strip() or "main"

    # Get current depth
    registry = get_subagent_registry()
    current_depth = registry.get_depth_for_session(requester_key)

    result = await spawn_subagent_background(
        options=options,
        requester_session_key=requester_key,
        depth=current_depth,
    )

    return json.dumps(result, indent=2)


@tool
def sessions_status_tool(session_id: str = "", model: str = "") -> str:
    """Show a rich status card for a session, or list recent sub-agent runs.

    Mirrors openclaw's session_status tool: usage stats, time, model override.
    If session_id is empty, shows all sub-agent run statuses.
    If session_id is provided, shows that session's status card.
    Optional: set per-session model override (model=default resets to config default).

    Args:
        session_id: Session ID to check (empty = show sub-agent statuses).
        model: Optional model override to set for this session (e.g. 'anthropic/claude-sonnet-4-20250514').
               Use 'default' to reset to the configured model.

    Returns:
        Rich status card text or JSON with sub-agent statuses.
    """
    from openclaw.agent.subagent_registry import get_subagent_registry

    if not session_id:
        # Show sub-agent registry status
        registry = get_subagent_registry()
        all_runs = []
        with registry._lock:
            for requester, runs in registry._runs.items():
                for run in runs:
                    all_runs.append({
                        "run_id": run.run_id,
                        "label": run.display_label(),
                        "task": run.task[:100],
                        "status": run.status,
                        "requester": requester,
                        "child_session_key": run.child_session_key,
                        "runtime_ms": run.runtime_ms,
                        "model": run.model,
                    })

        return json.dumps({
            "total_runs": len(all_runs),
            "runs": sorted(all_runs, key=lambda r: r.get("run_id", ""), reverse=True)[:20],
        }, indent=2)

    # Show specific session status card
    from openclaw.agent.session import SessionManager
    from openclaw.config import OpenClawConfig

    sm = SessionManager()
    session = sm._load_session(session_id)

    if not session:
        # Try by sender_id match too
        all_sessions = sm.list_sessions()
        match = next((s for s in all_sessions if s.sender_id == session_id), None)
        if match:
            session = sm._load_session(match.session_id)
    if not session:
        return json.dumps({"status": "not_found", "error": f"Session '{session_id}' not found."})

    meta = session.metadata

    # Handle model override request
    changed_model = False
    if model:
        cfg = OpenClawConfig.load()
        if model.lower() == "default":
            sm.set_model_override(session, "")
            changed_model = True
        else:
            # Validate the model string
            if "/" not in model:
                return json.dumps({"status": "error", "error": f"Invalid model format '{model}'. Use 'provider/model-name'."})
            sm.set_model_override(session, model)
            changed_model = True
        meta = session.metadata  # Reload

    # Build status card text (mirrors openclaw's buildStatusMessage)
    cfg = OpenClawConfig.load()
    default_model = cfg.agent.model
    effective_model = meta.model_override or default_model

    now = datetime.datetime.now()
    created = datetime.datetime.fromtimestamp(meta.created_at).strftime("%Y-%m-%d %H:%M:%S")
    last = datetime.datetime.fromtimestamp(meta.last_active).strftime("%Y-%m-%d %H:%M:%S")

    # Timezone display
    try:
        import zoneinfo
        tz_name = getattr(cfg, "timezone", None) or "UTC"
        tz = zoneinfo.ZoneInfo(tz_name)
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M %Z")
        time_line = f"🕒 Time: {local_time} ({tz_name})"
    except Exception:
        time_line = f"🕒 Time: {now.strftime('%Y-%m-%d %H:%M')} (local)"

    # Model display
    model_line = f"🤖 Model: {effective_model}"
    if meta.model_override:
        model_line += " *(override)*"
    if changed_model:
        model_line += " ← updated"

    # Usage display
    usage_parts = []
    if meta.total_tokens:
        usage_parts.append(f"total={meta.total_tokens:,} tokens")
    if meta.context_tokens:
        usage_parts.append(f"context={meta.context_tokens:,}")
    usage_line = f"📊 Usage: {', '.join(usage_parts)}" if usage_parts else ""

    # Queue depth from sub-agent registry
    active_runs = 0
    try:
        registry = get_subagent_registry()
        with registry._lock:
            for runs in registry._runs.values():
                active_runs += sum(1 for r in runs if r.status in ("pending", "running"))
    except Exception:
        pass

    lines = [
        f"📋 Session: {meta.session_id}",
        f"📡 Channel: {meta.channel} / sender: {meta.sender_id}",
        model_line,
        time_line,
    ]
    if usage_line:
        lines.append(usage_line)
    lines += [
        f"💬 Messages: {meta.message_count}",
        f"🕐 Created: {created}",
        f"🕐 Last active: {last}",
    ]
    if meta.title:
        lines.append(f"📝 Title: {meta.title}")
    if active_runs:
        lines.append(f"⚙️  Active sub-agents: {active_runs}")

    status_card = "\n".join(lines)

    return json.dumps({
        "status": "ok",
        "session_id": meta.session_id,
        "channel": meta.channel,
        "sender_id": meta.sender_id,
        "message_count": meta.message_count,
        "model": effective_model,
        "model_override": meta.model_override or None,
        "changed_model": changed_model,
        "total_tokens": meta.total_tokens,
        "context_tokens": meta.context_tokens,
        "created_at": created,
        "last_active": last,
        "title": meta.title or None,
        "status_card": status_card,
        "text": status_card,
    }, indent=2)
