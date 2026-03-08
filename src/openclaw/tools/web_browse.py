"""Web browse tool — fetch and extract text content from a URL."""

from __future__ import annotations

from langchain_core.tools import tool


@tool
def web_browse_tool(url: str, max_length: int = 8000) -> str:
    """Fetch a web page and extract its text content.

    Args:
        url: The URL to fetch.
        max_length: Maximum characters to return (default 8000).

    Returns:
        Extracted text content from the page.
    """
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

        # Remove script and style elements
        for el in soup(["script", "style", "nav", "footer", "header", "aside"]):
            el.decompose()

        # Get text
        text = soup.get_text(separator="\n", strip=True)

        # Clean up whitespace
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        text = "\n".join(lines)

        if len(text) > max_length:
            text = text[:max_length] + "\n\n... (content truncated)"

        return f"📄 Content from {url}:\n\n{text}"

    except ImportError:
        return "[error] httpx and beautifulsoup4 packages required. Run: pip install httpx beautifulsoup4"
    except Exception as e:
        return f"[error] Failed to fetch {url}: {e}"
