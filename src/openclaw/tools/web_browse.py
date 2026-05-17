"""Web fetch tool using Tinyfish Fetch API.

Falls back to httpx + BeautifulSoup if no TINYFISH_API_KEY is configured.
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
def web_browse_tool(url: str, max_length: int = 8000) -> str:
    """Fetch a web page and return its clean text content.

    Uses Tinyfish Fetch API (renders JS, handles anti-bot) when configured,
    otherwise falls back to plain httpx + BeautifulSoup.

    Tips:
    - For Reddit: use old.reddit.com URLs (e.g. old.reddit.com/r/...) for better results.
    - For Reddit JSON data: append .json to any Reddit URL.
    - For JS-heavy sites without Tinyfish: try the mobile or AMP version.

    Args:
        url: The URL to fetch.
        max_length: Maximum characters to return (default 8000).

    Returns:
        Extracted text content from the page in markdown format.
    """
    api_key = _get_tinyfish_key()

    if api_key:
        return _fetch_tinyfish(url, max_length, api_key)
    return _fetch_plain(url, max_length)


def _fetch_tinyfish(url: str, max_length: int, api_key: str) -> str:
    try:
        import httpx

        resp = httpx.post(
            "https://api.fetch.tinyfish.ai",
            headers={"X-API-Key": api_key, "Content-Type": "application/json"},
            json={"urls": [url], "format": "markdown"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        errors = data.get("errors", [])
        results = data.get("results", [])

        if errors and not results:
            return f"[error] Tinyfish fetch failed for {url}: {errors}"

        if not results:
            return f"[error] No content returned for {url}"

        result = results[0]
        title = result.get("title", "")
        text = result.get("text", "")

        if not text:
            return f"[error] Empty content returned for {url}"

        if len(text) > max_length:
            text = text[:max_length] + "\n\n... (content truncated)"

        header = f"📄 **{title}**\n{url}\n\n" if title else f"📄 {url}\n\n"
        return header + text

    except Exception as e:
        return f"[error] Tinyfish fetch failed: {e}"


def _rewrite_for_plain_fetch(url: str) -> str:
    """Rewrite URLs to friendlier versions for the plain httpx fallback."""
    import re
    # www.reddit.com → old.reddit.com (no JS, no anti-bot)
    url = re.sub(r"https?://(www\.)?reddit\.com", "https://old.reddit.com", url)
    return url


def _fetch_plain(url: str, max_length: int) -> str:
    url = _rewrite_for_plain_fetch(url)
    try:
        import httpx
        from bs4 import BeautifulSoup

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }

        response = httpx.get(url, headers=headers, follow_redirects=True, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        for el in soup(["script", "style", "nav", "footer", "header", "aside"]):
            el.decompose()

        text = soup.get_text(separator="\n", strip=True)
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        text = "\n".join(lines)

        if len(text) > max_length:
            text = text[:max_length] + "\n\n... (content truncated)"

        return f"📄 Content from {url}:\n\n{text}"

    except ImportError:
        return "[error] httpx and beautifulsoup4 packages required."
    except Exception as e:
        return f"[error] Failed to fetch {url}: {e}"
