"""Sub-agent tool — lets the main agent spawn child agents for focused tasks.

Examples of use by the agent:
- "Research this topic and summarize findings" → spawn a research sub-agent
- "Check API status while I write the docs" → parallel delegation
- "Analyze this codebase structure" → scoped code analysis sub-agent
"""

from __future__ import annotations

import asyncio
from typing import Optional

from langchain_core.tools import tool


@tool
async def spawn_subagent(
    task: str,
    allowed_tools: Optional[str] = None,
    timeout: int = 120,
    context: str = "",
) -> str:
    """Spawn a sub-agent to perform a focused task independently.

    The sub-agent runs in its own execution context with a scoped task.
    Use this to delegate work, parallelize research, or break down complex tasks.

    Args:
        task: Clear description of what the sub-agent should do.
        allowed_tools: Comma-separated tool names the sub-agent can use.
                       Leave empty to allow all tools.
        timeout: Maximum seconds for the sub-agent (default 120).
        context: Additional context or data to pass to the sub-agent.

    Returns:
        The sub-agent's response with its findings/results.
    """
    from openclaw.agent.subagent import SubAgentConfig, run_subagent

    tool_list = None
    if allowed_tools:
        tool_list = [t.strip() for t in allowed_tools.split(",") if t.strip()]

    config = SubAgentConfig(
        task=task,
        allowed_tools=tool_list,
        timeout=timeout,
        context=context,
    )

    result = await run_subagent(config)

    if result.success:
        output_parts = [f"✅ Sub-agent completed in {result.elapsed_seconds:.1f}s"]
        if result.tool_calls:
            output_parts.append(f"Tools used: {len(result.tool_calls)}")
        output_parts.append(f"\n{result.response}")
        return "\n".join(output_parts)
    else:
        return f"❌ Sub-agent failed: {result.error or 'Unknown error'}"
