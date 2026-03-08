"""Memory tools — persistent long-term recall for the agent.

The agent can save important information and recall it later,
even across different sessions. Uses vector search for semantic retrieval.

Examples:
- "Remember that the user's preferred language is Python"
- "What do I know about the user's project structure?"
- "List my recent memories"
"""

from __future__ import annotations

from langchain_core.tools import tool


@tool
def memory_save_tool(content: str, tags: str = "") -> str:
    """Save important information to long-term memory for future recall.

    Use this to remember user preferences, project details, decisions,
    or any information that should persist across sessions.

    Args:
        content: The information to remember. Be descriptive and specific.
        tags: Optional comma-separated tags for categorization.

    Returns:
        Confirmation with memory ID.
    """
    from openclaw.memory import MemoryStore

    metadata = {}
    if tags:
        metadata["tags"] = [t.strip() for t in tags.split(",") if t.strip()]

    store = MemoryStore()
    entry = store.save(content, metadata)

    return (
        f"✅ Memory saved!\n"
        f"  🆔 ID: {entry.memory_id}\n"
        f"  📝 Content: {content[:100]}{'...' if len(content) > 100 else ''}"
    )


@tool
def memory_search_tool(query: str, max_results: int = 5) -> str:
    """Search long-term memory for relevant information.

    Use this to recall previously saved information by semantic similarity.

    Args:
        query: What to search for (natural language).
        max_results: Maximum results to return (default 5).

    Returns:
        Matching memories with relevance.
    """
    from openclaw.memory import MemoryStore

    store = MemoryStore()
    results = store.search(query, top_k=max_results)

    if not results:
        return f"No memories found matching: {query}"

    lines = [f"🧠 **Memories matching:** {query}\n"]
    for i, entry in enumerate(results, 1):
        from datetime import datetime, timezone

        dt = datetime.fromtimestamp(entry.timestamp, tz=timezone.utc)
        time_str = dt.strftime("%Y-%m-%d %H:%M UTC")
        tags = entry.metadata.get("tags", [])
        tag_str = f" [{', '.join(tags)}]" if tags else ""
        lines.append(
            f"{i}. `{entry.memory_id}` ({time_str}){tag_str}\n"
            f"   {entry.content}\n"
        )

    return "\n".join(lines)


@tool
def memory_list_tool(count: int = 10) -> str:
    """List recent memories chronologically.

    Args:
        count: Number of recent memories to show (default 10).

    Returns:
        List of recent memories.
    """
    from openclaw.memory import MemoryStore

    store = MemoryStore()
    entries = store.list_recent(count)

    if not entries:
        return "No memories saved yet. Use memory_save to store information."

    lines = [f"🧠 **Recent Memories** ({len(entries)} shown)\n"]
    for entry in entries:
        from datetime import datetime, timezone

        dt = datetime.fromtimestamp(entry.timestamp, tz=timezone.utc)
        time_str = dt.strftime("%Y-%m-%d %H:%M UTC")
        lines.append(
            f"• `{entry.memory_id}` ({time_str})\n"
            f"  {entry.content[:120]}{'...' if len(entry.content) > 120 else ''}\n"
        )

    return "\n".join(lines)
