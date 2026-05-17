"""Sub-agent system — spawn child agents with scoped tasks.

Mirrors the pattern used by advanced agentic systems:
- Parent agent delegates a focused task to a child agent
- Child runs in a separate thread with its own LangGraph graph
- Child has a scoped system prompt and limited tool set
- Parent receives the child's result as a string

Architecture:
    Parent Agent ──spawn_subagent()──► SubAgent (thread pool)
                                            │
                                            ▼
                                     Scoped ReAct loop
                                            │
                                            ▼
                  ◄──── SubAgentResult ──────┘
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

# Maximum recursion depth for nested sub-agents
MAX_SUBAGENT_DEPTH = 3

# Default timeout for sub-agents (seconds)
DEFAULT_TIMEOUT = 120


@dataclass
class SubAgentConfig:
    """Configuration for spawning a sub-agent."""

    task: str
    """The task description for the sub-agent."""

    allowed_tools: list[str] | None = None
    """Tool names the sub-agent can use. None = all tools."""

    model: str | None = None
    """Model override for the sub-agent. None = use parent's model."""

    timeout: int = DEFAULT_TIMEOUT
    """Maximum seconds for the sub-agent to run."""

    max_iterations: int = 10
    """Maximum ReAct loop iterations."""

    context: str = ""
    """Additional context to inject into the sub-agent's system prompt."""

    parent_run_id: str = ""
    """Run ID of the parent agent (for tracking)."""

    depth: int = 0
    """Current nesting depth (0 = top-level agent)."""


@dataclass
class SubAgentResult:
    """Result from a sub-agent execution."""

    success: bool
    response: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    run_id: str = ""
    elapsed_seconds: float = 0.0
    error: str | None = None


def _build_subagent_prompt(config: SubAgentConfig) -> str:
    """Build a focused system prompt for the sub-agent."""
    prompt_parts = [
        "You are a focused sub-agent spawned to complete a specific task.",
        "Complete the task thoroughly and return your findings/results.",
        "Be concise but comprehensive in your response.",
        f"\n## YOUR TASK\n{config.task}",
    ]

    if config.context:
        prompt_parts.append(f"\n## ADDITIONAL CONTEXT\n{config.context}")

    prompt_parts.append(
        "\n## RULES\n"
        "- Focus exclusively on the assigned task.\n"
        "- Use tools as needed to complete the task.\n"
        "- Return a clear, structured response when done.\n"
        "- Do NOT spawn further sub-agents unless absolutely necessary."
    )

    return "\n".join(prompt_parts)


async def run_subagent(config: SubAgentConfig) -> SubAgentResult:
    """Run a sub-agent with a scoped task.

    This creates a new LangGraph graph with a scoped system prompt
    and runs it in a thread pool to avoid blocking the parent agent.

    Args:
        config: Sub-agent configuration.

    Returns:
        SubAgentResult with the sub-agent's response.
    """
    from openclaw.config import OpenClawConfig
    from openclaw.tools import get_all_tools, get_tools_by_name

    # Check depth limit
    if config.depth >= MAX_SUBAGENT_DEPTH:
        return SubAgentResult(
            success=False,
            response="",
            error=f"Maximum sub-agent depth ({MAX_SUBAGENT_DEPTH}) exceeded.",
        )

    run_id = f"sub-{str(uuid.uuid4())[:8]}"
    start_time = time.time()

    logger.info(
        f"🔀 Spawning sub-agent {run_id} (depth={config.depth + 1}): "
        f"{config.task[:80]}..."
    )

    try:
        cfg = OpenClawConfig.load()

        # Override model if specified
        if config.model:
            cfg.agent.model = config.model
        cfg.agent.max_iterations = config.max_iterations

        # Get tools (filtered if specified)
        if config.allowed_tools:
            tools = get_tools_by_name(*config.allowed_tools)
        else:
            tools = get_all_tools()

        # Remove the spawn_subagent tool from sub-agents at max depth - 1
        if config.depth >= MAX_SUBAGENT_DEPTH - 1:
            tools = [t for t in tools if t.name != "spawn_subagent"]

        # Build the sub-agent graph using the deepagents harness
        from openclaw.agent.graph import build_agent_graph, _extract_text
        from langchain_core.messages import AIMessage

        system_prompt = _build_subagent_prompt(config)
        compiled = build_agent_graph(cfg, system_prompt=system_prompt, session_id=run_id)

        result = await asyncio.wait_for(
            compiled.ainvoke({"messages": [HumanMessage(content=config.task)]}),
            timeout=config.timeout,
        )

        # Extract results
        response_text = ""
        tool_calls_made = []
        for msg in result["messages"]:
            if isinstance(msg, AIMessage):
                if msg.content:
                    response_text = _extract_text(msg.content)
                if msg.tool_calls:
                    tool_calls_made.extend(msg.tool_calls)

        elapsed = time.time() - start_time
        logger.info(f"✅ Sub-agent {run_id} completed in {elapsed:.1f}s")

        return SubAgentResult(
            success=True,
            response=response_text,
            tool_calls=tool_calls_made,
            run_id=run_id,
            elapsed_seconds=elapsed,
        )

    except asyncio.TimeoutError:
        elapsed = time.time() - start_time
        logger.warning(f"⏰ Sub-agent {run_id} timed out after {elapsed:.1f}s")
        return SubAgentResult(
            success=False,
            response="",
            run_id=run_id,
            elapsed_seconds=elapsed,
            error=f"Sub-agent timed out after {config.timeout}s",
        )

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Sub-agent {run_id} failed: {e}")
        return SubAgentResult(
            success=False,
            response="",
            run_id=run_id,
            elapsed_seconds=elapsed,
            error=str(e),
        )


def format_subagent_announcement(result: SubAgentResult, task: str = "") -> str:
    """Format a sub-agent result as a structured announcement.

    This mirrors OpenClaw's subagent-announce.ts — produces a readable
    summary of what the sub-agent did for the parent agent to consume.
    """
    parts = []

    status = "completed successfully" if result.success else "failed"
    parts.append(f"**Sub-agent {result.run_id}** {status} in {result.elapsed_seconds:.1f}s")

    if task:
        parts.append(f"**Task:** {task[:200]}")

    if result.error:
        parts.append(f"**Error:** {result.error}")

    if result.tool_calls:
        tool_names = list({tc.get("name", "unknown") for tc in result.tool_calls})
        parts.append(f"**Tools used:** {', '.join(tool_names[:10])}")

    if result.response:
        # Truncate long responses
        response = result.response
        if len(response) > 2000:
            response = response[:2000] + "\n\n... (response truncated)"
        parts.append(f"**Result:**\n{response}")

    return "\n".join(parts)
