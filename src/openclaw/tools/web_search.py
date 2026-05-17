"""Web search tool using Tinyfish Search API.

Falls back to DuckDuckGo if no TINYFISH_API_KEY is configured.
"""

from __future__ import annotations

import os

from langchain_core.tools import tool


def _get_tinyfish_key() -> str:
    key = os.environ.get("TINYFISH_API_KEY", "")
    if not key:
        try:
            from openclaw.config import OpenClawConfig
            key = OpenClawConfig.load().api_keys.tinyfish
        except Exception:
            pass
    return key


@tool
def web_search_tool(query: str, max_results: int = 5) -> str:
    """Search the web and return results with titles, URLs, and snippets.

    Args:
        query: The search query.
        max_results: Maximum number of results to return (default 5).

    Returns:
        Formatted search results with titles, URLs, and snippets.
    """
    api_key = _get_tinyfish_key()

    if api_key:
        return _search_tinyfish(query, max_results, api_key)
    return _search_duckduckgo(query, max_results)


def _search_tinyfish(query: str, max_results: int, api_key: str) -> str:
    try:
        import httpx

        resp = httpx.get(
            "https://api.search.tinyfish.ai",
            headers={"X-API-Key": api_key},
            params={"query": query},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        results = data.get("results", [])[:max_results]
        if not results:
            return f"No results found for: {query}"

        lines = [f"🔍 Search results for: {query}\n"]
        for r in results:
            lines.append(f"{r.get('position', '')}. **{r.get('title', 'Untitled')}**")
            lines.append(f"   {r.get('url', '')}")
            snippet = r.get("snippet", "")
            if snippet:
                lines.append(f"   {snippet}")
            lines.append("")

        return "\n".join(lines)

    except Exception as e:
        return f"[error] Tinyfish search failed: {e}"


def _search_duckduckgo(query: str, max_results: int) -> str:
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
            lines.append(f"{i}. **{r.get('title', 'Untitled')}**")
            lines.append(f"   {r.get('href', r.get('link', ''))}")
            snippet = r.get("body", r.get("snippet", ""))
            if snippet:
                lines.append(f"   {snippet}")
            lines.append("")

        return "\n".join(lines)

    except ImportError:
        return "[error] No search backend available. Set TINYFISH_API_KEY or install duckduckgo-search."
    except Exception as e:
        return f"[error] Search failed: {e}"
