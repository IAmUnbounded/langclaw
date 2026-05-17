"""Background sub-agent spawning with announcement.

Mirrors openclaw's subagent-spawn.ts:
- Spawn a sub-agent in a background asyncio task
- The sub-agent gets its own scoped LangGraph graph
- On completion, announce the result back to the requester session
- Support cascade cancellation via SubagentRegistry cancel events
- Idempotency checking to prevent duplicate spawns
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage

from openclaw.agent.subagent_registry import (
    SubagentOutcome,
    SubagentRunRecord,
    get_subagent_registry,
)

logger = logging.getLogger(__name__)

# Maximum depth for sub-agent nesting
MAX_SUBAGENT_DEPTH = 3

# Default timeout for background sub-agents (seconds)
DEFAULT_RUN_TIMEOUT = 300  # 5 minutes


@dataclass
class SpawnOptions:
    """Options for spawning a background sub-agent."""

    task: str
    label: str = ""
    agent_id: str | None = None
    model: str | None = None
    thinking: str | None = None
    run_timeout_seconds: int = DEFAULT_RUN_TIMEOUT
    cleanup: str = "keep"  # "keep" or "delete"
    allowed_tools: list[str] | None = None
    context: str = ""


async def spawn_subagent_background(
    options: SpawnOptions,
    requester_session_key: str,
    depth: int = 0,
    requester_channel: str = "cli",
) -> dict[str, Any]:
    """Spawn a background sub-agent and return immediately.

    The sub-agent runs in an asyncio task. When complete, it announces
    the result back to the requester session.

    Args:
        options: Spawn configuration.
        requester_session_key: Session key of the requesting agent.
        depth: Current nesting depth (0 = top-level).

    Returns:
        Dict with run_id, child_session_key, status, and display text.
    """
    if depth >= MAX_SUBAGENT_DEPTH:
        return {
            "status": "error",
            "error": f"Maximum sub-agent depth ({MAX_SUBAGENT_DEPTH}) exceeded.",
        }

    # Generate unique session key for this sub-agent
    child_session_key = f"subagent:{requester_session_key}:{uuid.uuid4().hex[:8]}"

    registry = get_subagent_registry()
    record = registry.register(
        requester_session_key=requester_session_key,
        child_session_key=child_session_key,
        task=options.task,
        label=options.label,
        model=options.model,
        depth=depth + 1,
        requester_channel=requester_channel,
    )

    logger.info(
        f"Spawning sub-agent {record.run_id} (depth={depth + 1}): "
        f"{options.task[:80]}..."
    )

    # Launch in background
    asyncio.create_task(
        launch_background_subagent(
            run_record=record,
            message=options.task,
            requester_session_key=requester_session_key,
            options=options,
        ),
        name=f"subagent-{record.run_id}",
    )

    label_display = options.label or options.task[:48]

    # Notify the requester immediately that the sub-agent has been spawned
    if requester_channel == "whatsapp":
        try:
            from openclaw.agent.announce import notify_subagent_spawned
            asyncio.create_task(notify_subagent_spawned(
                requester_session_key=requester_session_key,
                channel=requester_channel,
                label=label_display,
                run_id=record.run_id,
            ))
        except Exception as e:
            logger.debug(f"Spawn notification skipped: {e}")

    return {
        "status": "spawned",
        "run_id": record.run_id,
        "child_session_key": child_session_key,
        "label": label_display,
        "text": (
            f"Sub-agent '{label_display}' spawned (run_id={record.run_id}). "
            f"It will announce its result when complete."
        ),
    }


async def launch_background_subagent(
    run_record: SubagentRunRecord,
    message: str,
    requester_session_key: str,
    options: SpawnOptions | None = None,
) -> None:
    """Execute a sub-agent in the background and announce results.

    This is the actual background task that runs the LangGraph agent,
    monitors cancellation, and announces results.
    """
    registry = get_subagent_registry()
    registry.mark_started(run_record.run_id)

    start_time = time.time()
    cancel_event = registry.get_cancel_event(run_record.run_id)

    try:
        result_response, tool_call_count = await _run_with_cancellation(
            run_record=run_record,
            message=message,
            options=options,
            cancel_event=cancel_event,
        )

        elapsed = time.time() - start_time
        outcome = SubagentOutcome(
            status="ok",
            response=result_response,
            tool_calls=tool_call_count,
            elapsed_seconds=elapsed,
        )

    except asyncio.CancelledError:
        elapsed = time.time() - start_time
        logger.info(f"Sub-agent {run_record.run_id} cancelled after {elapsed:.1f}s")
        outcome = SubagentOutcome(
            status="killed",
            elapsed_seconds=elapsed,
        )

    except asyncio.TimeoutError:
        elapsed = time.time() - start_time
        logger.warning(f"Sub-agent {run_record.run_id} timed out after {elapsed:.1f}s")
        outcome = SubagentOutcome(
            status="error",
            error=f"Timed out after {elapsed:.0f}s",
            elapsed_seconds=elapsed,
        )

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Sub-agent {run_record.run_id} failed: {e}")
        outcome = SubagentOutcome(
            status="error",
            error=str(e),
            elapsed_seconds=elapsed,
        )

    # Mark run as ended
    registry.mark_ended(run_record.run_id, outcome)

    # Announce result unless steer-suppressed
    if not run_record.steer_restart_suppressed:
        await _announce_result(
            run_record=run_record,
            outcome=outcome,
            requester_session_key=requester_session_key,
            channel=getattr(run_record, "requester_channel", "cli"),
        )


async def _run_with_cancellation(
    run_record: SubagentRunRecord,
    message: str,
    options: SpawnOptions | None,
    cancel_event: asyncio.Event | None,
) -> tuple[str, int]:
    """Run the sub-agent graph with cancellation support."""
    from openclaw.config import OpenClawConfig
    from openclaw.agent.graph import build_agent_graph
    from openclaw.tools import get_all_tools, get_tools_by_name

    cfg = OpenClawConfig.load()

    # Apply model override
    if options and options.model:
        cfg.agent.model = options.model

    # Apply thinking override
    if options and options.thinking:
        cfg.agent.thinking_level = options.thinking

    # Timeout
    timeout = (options.run_timeout_seconds if options else DEFAULT_RUN_TIMEOUT) or DEFAULT_RUN_TIMEOUT

    # Build scoped sub-agent system prompt
    system_prompt = _build_subagent_system_prompt(
        task=message,
        label=run_record.label,
        context=options.context if options else "",
    )

    # Tool selection
    if options and options.allowed_tools:
        tools = get_tools_by_name(*options.allowed_tools)
    else:
        tools = get_all_tools()

    # Remove spawn_subagent at max depth to prevent infinite nesting
    if run_record.depth >= MAX_SUBAGENT_DEPTH - 1:
        tools = [t for t in tools if t.name not in ("spawn_subagent", "sessions_spawn")]

    graph = build_agent_graph(
        cfg,
        system_prompt=system_prompt,
        session_id=run_record.child_session_key,
    )

    # Create the graph invocation task
    graph_task = asyncio.create_task(graph.ainvoke({"messages": [HumanMessage(content=message)]}))

    # Create cancellation monitor task
    async def _watch_cancel() -> None:
        if cancel_event:
            await cancel_event.wait()
            graph_task.cancel()

    cancel_task = asyncio.create_task(_watch_cancel())

    try:
        result = await asyncio.wait_for(graph_task, timeout=timeout)
        cancel_task.cancel()
    except (asyncio.CancelledError, asyncio.TimeoutError):
        cancel_task.cancel()
        raise

    # Extract response
    messages = result.get("messages", [])
    response_text = ""
    tool_call_count = 0

    for msg in messages:
        if isinstance(msg, AIMessage):
            if msg.content:
                response_text = str(msg.content)
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tool_call_count += len(msg.tool_calls)

    return response_text, tool_call_count


def _build_subagent_system_prompt(task: str, label: str = "", context: str = "") -> str:
    """Build a scoped system prompt for a background sub-agent."""
    parts = [
        "You are a focused sub-agent spawned to complete a specific task.",
        "Complete the task thoroughly and return your findings/results.",
        "Be concise but comprehensive in your response.",
    ]

    if label:
        parts.append(f"\n## IDENTITY\nYour label is: {label}")

    parts.append(f"\n## YOUR TASK\n{task}")

    if context:
        parts.append(f"\n## ADDITIONAL CONTEXT\n{context}")

    parts.append(
        "\n## RULES\n"
        "- Focus exclusively on the assigned task.\n"
        "- Use tools as needed to complete the task.\n"
        "- Return a clear, structured response when done.\n"
        "- Do NOT spawn further sub-agents unless absolutely necessary.\n"
        "- Keep your final response concise and action-oriented."
    )

    return "\n".join(parts)


async def _announce_result(
    run_record: SubagentRunRecord,
    outcome: SubagentOutcome,
    requester_session_key: str,
    channel: str = "cli",
) -> None:
    """Announce the sub-agent result back to the requester session.

    Uses the AnnounceRegistry for idempotent delivery — ensures the same
    run is not announced twice even if multiple completion events fire.
    """
    try:
        from openclaw.agent.announce import AnnouncePayload, announce_subagent_result

        payload = AnnouncePayload(
            run_id=run_record.run_id,
            label=run_record.display_label(),
            task=run_record.task,
            status=outcome.status,
            response=outcome.response or "",
            error=outcome.error,
            elapsed_seconds=outcome.elapsed_seconds or 0.0,
            tool_calls=outcome.tool_calls or 0,
            model=run_record.model,
            depth=run_record.depth,
        )

        sent = await announce_subagent_result(
            payload=payload,
            requester_session_key=requester_session_key,
            channel=channel,
        )
        if sent:
            logger.info(
                f"Sub-agent {run_record.run_id} announced: status={outcome.status} "
                f"channel={channel} requester={requester_session_key}"
            )
    except Exception as e:
        logger.warning(f"Failed to announce sub-agent result {run_record.run_id}: {e}")
