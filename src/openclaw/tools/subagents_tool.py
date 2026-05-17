"""Subagents management tool — list, kill, and steer spawned sub-agents.

Mirrors openclaw's subagents-tool.ts:
- list: show active and recently completed sub-agents
- kill: abort a running sub-agent (and its children, recursively)
- steer: interrupt current work and send a new message to a sub-agent
"""

from __future__ import annotations

import asyncio
import logging
import time

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

DEFAULT_RECENT_MINUTES = 30
MAX_RECENT_MINUTES = 24 * 60


def _format_duration(ms: int) -> str:
    """Format milliseconds as a compact duration string."""
    if ms < 1000:
        return f"{ms}ms"
    secs = ms // 1000
    if secs < 60:
        return f"{secs}s"
    mins = secs // 60
    secs = secs % 60
    if mins < 60:
        return f"{mins}m{secs}s" if secs else f"{mins}m"
    hours = mins // 60
    mins = mins % 60
    return f"{hours}h{mins}m" if mins else f"{hours}h"


def _build_list_text(
    active: list[dict],
    recent: list[dict],
    recent_minutes: int,
) -> str:
    """Format the list output."""
    lines: list[str] = []
    lines.append("active subagents:")
    if not active:
        lines.append("(none)")
    else:
        for i, entry in enumerate(active, 1):
            model_disp = (entry.get("model") or "").split("/")[-1] or "default"
            runtime = _format_duration(entry.get("runtime_ms", 0))
            label = entry.get("label", "subagent")
            task = entry.get("task", "")
            task_short = task[:72] if task else ""
            extra = f" - {task_short}" if task_short.lower() != label.lower() else ""
            lines.append(f"{i}. {label} ({model_disp}, {runtime}) running{extra}")

    lines.append("")
    lines.append(f"recent (last {recent_minutes}m):")
    if not recent:
        lines.append("(none)")
    else:
        for i, entry in enumerate(recent, len(active) + 1):
            model_disp = (entry.get("model") or "").split("/")[-1] or "default"
            runtime = _format_duration(entry.get("runtime_ms", 0))
            label = entry.get("label", "subagent")
            status = entry.get("status", "done")
            task = entry.get("task", "")
            task_short = task[:72] if task else ""
            extra = f" - {task_short}" if task_short.lower() != label.lower() else ""
            lines.append(f"{i}. {label} ({model_disp}, {runtime}) {status}{extra}")

    return "\n".join(lines)


@tool
async def subagents_tool(
    action: str = "list",
    target: str = "",
    message: str = "",
    recent_minutes: int = DEFAULT_RECENT_MINUTES,
    session_key: str = "",
) -> str:
    """List, kill, or steer spawned sub-agents.

    Actions:
    - list: Show active and recently completed sub-agents.
    - kill: Abort a running sub-agent (kills its children too). Use target="all" to kill all.
    - steer: Interrupt a sub-agent and send it a new message to redirect its work.

    Target formats (for kill/steer):
    - "last" → most recently started sub-agent
    - "1", "2", ... → numeric index from the list output
    - Run ID prefix (e.g., "abc123")
    - Label or label prefix (e.g., "researcher")

    Args:
        action: One of "list", "kill", "steer".
        target: Sub-agent target for kill/steer operations.
        message: New message for steer action.
        recent_minutes: How far back to look for recent sub-agents (default 30).
        session_key: Requester session key (auto-detected if empty).

    Returns:
        JSON-formatted result with status and relevant data.
    """
    import json
    from openclaw.agent.subagent_registry import (
        get_subagent_registry, MAX_STEER_MESSAGE_CHARS, SubagentOutcome,
    )

    registry = get_subagent_registry()

    # Resolve requester session key
    requester_key = session_key.strip() or "main"

    minutes = max(1, min(MAX_RECENT_MINUTES, recent_minutes))
    now = time.time()
    recent_cutoff = now - minutes * 60

    if action == "list":
        runs = registry.list_for_requester(requester_key)
        active_records = [r for r in runs if r.is_running]
        recent_records = [
            r for r in runs
            if r.ended_at and r.ended_at >= recent_cutoff
        ]

        active_views = [
            {
                "index": i + 1,
                "run_id": r.run_id,
                "label": r.display_label(),
                "task": r.task,
                "status": r.status,
                "runtime_ms": r.runtime_ms,
                "model": r.model,
                "started_at": r.started_at,
            }
            for i, r in enumerate(active_records)
        ]
        recent_views = [
            {
                "index": len(active_records) + i + 1,
                "run_id": r.run_id,
                "label": r.display_label(),
                "task": r.task,
                "status": r.status,
                "runtime_ms": r.runtime_ms,
                "model": r.model,
                "started_at": r.started_at,
                "ended_at": r.ended_at,
            }
            for i, r in enumerate(recent_records)
        ]

        text = _build_list_text(active_views, recent_views, minutes)
        return json.dumps({
            "status": "ok",
            "action": "list",
            "requester_session_key": requester_key,
            "total": len(runs),
            "active": active_views,
            "recent": recent_views,
            "text": text,
        }, indent=2)

    if action == "kill":
        target = target.strip()
        if not target:
            return json.dumps({
                "status": "error",
                "action": "kill",
                "error": "target is required for kill action",
            })

        if target in ("all", "*"):
            runs = registry.list_for_requester(requester_key)
            killed_labels: list[str] = []
            for run in runs:
                if run.is_running:
                    cascade_killed = registry.cascade_kill(run.child_session_key)
                    killed_labels.extend(cascade_killed)
                    if not cascade_killed:
                        # Kill this one directly
                        was_killed = registry.mark_killed(run.run_id)
                        if was_killed:
                            killed_labels.append(run.display_label())

            killed_count = len(killed_labels)
            text = (
                f"killed {killed_count} subagent{'s' if killed_count != 1 else ''}."
                if killed_count > 0 else "no running subagents to kill."
            )
            return json.dumps({
                "status": "ok",
                "action": "kill",
                "target": "all",
                "killed": killed_count,
                "labels": killed_labels,
                "text": text,
            })

        record = registry.resolve_target(requester_key, target, minutes)
        if not record:
            return json.dumps({
                "status": "error",
                "action": "kill",
                "target": target,
                "error": f"Unknown subagent target: {target!r}",
            })

        if not record.is_running:
            return json.dumps({
                "status": "done",
                "action": "kill",
                "target": target,
                "run_id": record.run_id,
                "text": f"{record.display_label()} is already finished.",
            })

        killed_labels = registry.cascade_kill(record.child_session_key)
        killed_count = len(killed_labels)
        cascade_count = max(0, killed_count - 1)
        cascade_text = (
            f" (+ {cascade_count} descendant{'s' if cascade_count != 1 else ''})"
            if cascade_count > 0 else ""
        )
        return json.dumps({
            "status": "ok",
            "action": "kill",
            "target": target,
            "run_id": record.run_id,
            "label": record.display_label(),
            "cascade_killed": cascade_count,
            "text": f"killed {record.display_label()}{cascade_text}.",
        })

    if action == "steer":
        target = target.strip()
        if not target:
            return json.dumps({"status": "error", "action": "steer", "error": "target required"})
        if not message:
            return json.dumps({"status": "error", "action": "steer", "error": "message required"})
        if len(message) > MAX_STEER_MESSAGE_CHARS:
            return json.dumps({
                "status": "error",
                "action": "steer",
                "error": f"Message too long ({len(message)} chars, max {MAX_STEER_MESSAGE_CHARS})",
            })

        record = registry.resolve_target(requester_key, target, minutes)
        if not record:
            return json.dumps({
                "status": "error",
                "action": "steer",
                "target": target,
                "error": f"Unknown subagent target: {target!r}",
            })

        if not record.is_running:
            return json.dumps({
                "status": "done",
                "action": "steer",
                "target": target,
                "run_id": record.run_id,
                "text": f"{record.display_label()} is already finished.",
            })

        # Self-steer protection
        if record.child_session_key == requester_key:
            return json.dumps({
                "status": "forbidden",
                "action": "steer",
                "error": "Subagents cannot steer themselves.",
            })

        # Rate limit check
        if not registry.check_steer_rate_limit(record.child_session_key):
            return json.dumps({
                "status": "rate_limited",
                "action": "steer",
                "target": target,
                "run_id": record.run_id,
                "error": "Steer rate limit exceeded. Wait a moment before sending another steer.",
            })

        # Suppress current run announcement
        registry.suppress_steer_announce(record.run_id)

        # Signal the cancel event to interrupt current work
        cancel_event = registry.get_cancel_event(record.run_id)
        if cancel_event:
            cancel_event.set()

        # Brief settle time
        try:
            await asyncio.wait_for(
                asyncio.sleep(1.0),
                timeout=2.0,
            )
        except asyncio.TimeoutError:
            pass

        # Mark old run as ended, start a new steer run
        registry.mark_ended(
            record.run_id,
            SubagentOutcome(status="steered", response="(steered to new message)"),
        )

        # Register a new run for the steer message
        new_record = registry.register(
            requester_session_key=requester_key,
            child_session_key=record.child_session_key,
            task=message,
            label=record.label,
            model=record.model,
            depth=record.depth,
            requester_channel=record.requester_channel,
        )

        # Launch the steered run via the background spawn system
        try:
            from openclaw.agent.subagent_spawn import launch_background_subagent
            asyncio.create_task(
                launch_background_subagent(
                    run_record=new_record,
                    message=message,
                    requester_session_key=requester_key,
                )
            )
        except Exception as e:
            registry.clear_steer_suppress(record.run_id)
            logger.error(f"Steer launch failed: {e}")
            return json.dumps({
                "status": "error",
                "action": "steer",
                "target": target,
                "run_id": record.run_id,
                "error": str(e),
            })

        return json.dumps({
            "status": "accepted",
            "action": "steer",
            "target": target,
            "run_id": new_record.run_id,
            "label": record.display_label(),
            "text": f"steered {record.display_label()}.",
        })

    return json.dumps({
        "status": "error",
        "error": f"Unknown action: {action!r}. Use list, kill, or steer.",
    })
