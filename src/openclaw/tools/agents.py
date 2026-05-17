"""Agent profile management tools.

Expose the AgentRegistry through LangChain tools so the agent can
list, create, inspect, and delete agent profiles during a conversation.
"""

from __future__ import annotations

import re

from langchain_core.tools import tool


def _slugify(name: str) -> str:
    """Turn a human-readable name into a safe agent_id slug."""
    slug = name.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    return slug.strip("-")


@tool
def agents_list_tool() -> str:
    """List all registered agent profiles with their configurations."""
    from openclaw.agents import AgentRegistry

    registry = AgentRegistry()
    profiles = registry.list_all()

    if not profiles:
        return "No agent profiles registered."

    lines = [f"**Registered Agents** ({len(profiles)})\n"]
    for p in profiles:
        model_str = p.model or "(default)"
        tools_str = ", ".join(p.tools) if p.tools else "all"
        lines.append(
            f"- **{p.name}** (`{p.agent_id}`)\n"
            f"  {p.description}\n"
            f"  Model: {model_str} | Temp: {p.temperature} | "
            f"Tools: {tools_str} | Thinking: {p.thinking_level}\n"
        )

    return "\n".join(lines)


@tool
def agents_create_tool(
    name: str,
    description: str,
    model: str = "",
    tools: str = "",
    system_prompt_extra: str = "",
    thinking_level: str = "off",
) -> str:
    """Create a new agent profile. Tools should be comma-separated tool names.

    Args:
        name: Human-readable name for the agent (e.g. "Researcher").
        description: What this agent specialises in.
        model: Optional model override (empty string uses the default model).
        tools: Comma-separated list of tool names, or empty for all tools.
        system_prompt_extra: Additional system prompt text appended to the base prompt.
        thinking_level: Thinking budget — one of off, low, mid, high, max.

    Returns:
        Confirmation with the new agent's details.
    """
    from openclaw.agents import AgentProfile, AgentRegistry

    agent_id = _slugify(name)
    if not agent_id:
        return "Error: could not derive a valid agent_id from the name."

    tool_list = [t.strip() for t in tools.split(",") if t.strip()] if tools else []

    profile = AgentProfile(
        agent_id=agent_id,
        name=name,
        description=description,
        model=model,
        tools=tool_list,
        system_prompt_extra=system_prompt_extra,
        thinking_level=thinking_level,
    )

    registry = AgentRegistry()
    registered = registry.register(profile)

    return (
        f"Agent created!\n"
        f"  ID: {registered.agent_id}\n"
        f"  Name: {registered.name}\n"
        f"  Description: {registered.description}\n"
        f"  Model: {registered.model or '(default)'}\n"
        f"  Tools: {', '.join(registered.tools) or 'all'}\n"
        f"  Thinking: {registered.thinking_level}"
    )


@tool
def agents_get_tool(agent_id: str) -> str:
    """Get details of a specific agent profile.

    Args:
        agent_id: The unique identifier of the agent profile.

    Returns:
        Detailed agent profile information, or an error if not found.
    """
    from openclaw.agents import AgentRegistry

    registry = AgentRegistry()
    profile = registry.get(agent_id)

    if profile is None:
        return f"Agent '{agent_id}' not found."

    tools_str = ", ".join(profile.tools) if profile.tools else "all"
    extra = (
        f"\n  System prompt extra: {profile.system_prompt_extra[:200]}..."
        if len(profile.system_prompt_extra) > 200
        else (
            f"\n  System prompt extra: {profile.system_prompt_extra}"
            if profile.system_prompt_extra
            else ""
        )
    )

    from datetime import datetime, timezone

    created = datetime.fromtimestamp(profile.created_at, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC"
    )
    updated = datetime.fromtimestamp(profile.updated_at, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC"
    )

    return (
        f"**{profile.name}** (`{profile.agent_id}`)\n"
        f"  Description: {profile.description}\n"
        f"  Model: {profile.model or '(default)'}\n"
        f"  Temperature: {profile.temperature}\n"
        f"  Tools: {tools_str}\n"
        f"  Max iterations: {profile.max_iterations}\n"
        f"  Thinking: {profile.thinking_level}\n"
        f"  Created: {created}\n"
        f"  Updated: {updated}"
        f"{extra}"
    )


@tool
def agents_delete_tool(agent_id: str) -> str:
    """Delete an agent profile.

    Args:
        agent_id: The unique identifier of the agent profile to delete.

    Returns:
        Confirmation or error message.
    """
    from openclaw.agents import AgentRegistry

    if agent_id == "main":
        return "Cannot delete the built-in 'main' agent profile."

    registry = AgentRegistry()
    deleted = registry.delete(agent_id)

    if deleted:
        return f"Agent '{agent_id}' has been deleted."
    return f"Agent '{agent_id}' not found."
