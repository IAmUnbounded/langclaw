"""Web search tool using DuckDuckGo (no API key required)."""

from __future__ import annotations

from langchain_core.tools import tool


@tool
def web_search_tool(query: str, max_results: int = 5) -> str:
    """Search the web using DuckDuckGo and return results.

    Args:
        query: The search query.
        max_results: Maximum number of results to return (default 5).

    Returns:
        Formatted search results with titles, URLs, and snippets.
    """
    try:
        from duckduckgo_search import DDGS

        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(r)

        if not results:
            return f"No results found for: {query}"

        lines = [f"🔍 Search results for: {query}\n"]
        for i, r in enumerate(results, 1):
            title = r.get("title", "Untitled")
            url = r.get("href", r.get("link", ""))
            snippet = r.get("body", r.get("snippet", ""))
            lines.append(f"{i}. **{title}**")
            lines.append(f"   {url}")
            if snippet:
                lines.append(f"   {snippet}")
            lines.append("")

        return "\n".join(lines)

    except ImportError:
        return "[error] duckduckgo-search package not installed. Run: pip install duckduckgo-search"
    except Exception as e:
        return f"[error] Search failed: {e}"
