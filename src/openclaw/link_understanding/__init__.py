"""Link understanding — URL detection, extraction, and transformation.

Mirrors OpenClaw's link-understanding module:
- URL detection: find URLs in message text
- Link extraction: parse and normalize URLs
- Link metadata: fetch page titles, descriptions, images
- Link transformation: format links for display in different channels
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse, urljoin

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# URL detection patterns
# ---------------------------------------------------------------------------

# Comprehensive URL regex supporting http(s), ftp, and bare domains
URL_PATTERN = re.compile(
    r"(?:https?://|ftp://)"                # Protocol
    r"(?:[a-zA-Z0-9\-._~:/?#\[\]@!$&'()*+,;=%]+)"  # URL characters
    r"|(?:www\.[a-zA-Z0-9\-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?)",  # www.domain
    re.IGNORECASE,
)

# Markdown link pattern [text](url)
MARKDOWN_LINK_PATTERN = re.compile(
    r"\[([^\]]+)\]\(([^)]+)\)",
)

# Common file extensions for rich preview
RICH_PREVIEW_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg",  # Images
    ".mp4", ".webm", ".mov",  # Video
    ".mp3", ".wav", ".ogg",  # Audio
    ".pdf",  # Documents
}


@dataclass
class DetectedLink:
    """A URL detected in message text."""

    url: str
    start: int  # Start position in text
    end: int  # End position in text
    display_text: str = ""  # Markdown link text, if any
    is_markdown: bool = False
    domain: str = ""
    path: str = ""
    extension: str = ""
    is_media: bool = False


@dataclass
class LinkMetadata:
    """Metadata fetched for a URL."""

    url: str
    title: str = ""
    description: str = ""
    image_url: str = ""
    site_name: str = ""
    content_type: str = ""
    status_code: int = 0
    error: str = ""


# ---------------------------------------------------------------------------
# URL detection
# ---------------------------------------------------------------------------

def detect_links(text: str) -> list[DetectedLink]:
    """Detect all URLs in a text string.

    Handles both bare URLs and markdown-formatted links.
    Returns links in order of appearance.
    """
    links: list[DetectedLink] = []
    seen_urls: set[str] = set()

    # First, find markdown links
    for match in MARKDOWN_LINK_PATTERN.finditer(text):
        url = match.group(2).strip()
        if url in seen_urls:
            continue
        seen_urls.add(url)

        parsed = urlparse(url)
        ext = _get_extension(parsed.path)
        links.append(DetectedLink(
            url=url,
            start=match.start(),
            end=match.end(),
            display_text=match.group(1),
            is_markdown=True,
            domain=parsed.hostname or "",
            path=parsed.path,
            extension=ext,
            is_media=ext.lower() in RICH_PREVIEW_EXTENSIONS,
        ))

    # Then find bare URLs (not already captured in markdown)
    for match in URL_PATTERN.finditer(text):
        url = match.group(0)

        # Skip if already found in a markdown link
        if url in seen_urls:
            continue

        # Check if this URL is inside a markdown link
        is_inside_markdown = False
        for md_match in MARKDOWN_LINK_PATTERN.finditer(text):
            if md_match.start() <= match.start() and match.end() <= md_match.end():
                is_inside_markdown = True
                break
        if is_inside_markdown:
            continue

        seen_urls.add(url)

        # Normalize www URLs
        normalized = url
        if url.startswith("www."):
            normalized = "https://" + url

        parsed = urlparse(normalized)
        ext = _get_extension(parsed.path)
        links.append(DetectedLink(
            url=normalized,
            start=match.start(),
            end=match.end(),
            domain=parsed.hostname or "",
            path=parsed.path,
            extension=ext,
            is_media=ext.lower() in RICH_PREVIEW_EXTENSIONS,
        ))

    # Sort by position
    links.sort(key=lambda l: l.start)
    return links


def _get_extension(path: str) -> str:
    """Extract file extension from URL path."""
    if "." in path.split("/")[-1]:
        return "." + path.split(".")[-1].split("?")[0].split("#")[0].lower()
    return ""


# ---------------------------------------------------------------------------
# Link metadata fetching
# ---------------------------------------------------------------------------

async def fetch_link_metadata(url: str, timeout: float = 10.0) -> LinkMetadata:
    """Fetch metadata for a URL (title, description, image).

    Uses OpenGraph tags and HTML meta tags for extraction.
    """
    metadata = LinkMetadata(url=url)

    try:
        import httpx
        from bs4 import BeautifulSoup

        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=timeout,
            headers={"User-Agent": "OpenClaw-Lang/0.1 (Link Preview)"},
        ) as client:
            response = await client.get(url)
            metadata.status_code = response.status_code
            metadata.content_type = response.headers.get("content-type", "")

            if response.status_code != 200:
                metadata.error = f"HTTP {response.status_code}"
                return metadata

            # Only parse HTML content
            if "text/html" not in metadata.content_type:
                return metadata

            soup = BeautifulSoup(response.text, "html.parser")

            # OpenGraph tags
            og_title = soup.find("meta", property="og:title")
            og_desc = soup.find("meta", property="og:description")
            og_image = soup.find("meta", property="og:image")
            og_site = soup.find("meta", property="og:site_name")

            metadata.title = _meta_content(og_title) or _get_title(soup)
            metadata.description = _meta_content(og_desc) or _get_description(soup)
            metadata.image_url = _meta_content(og_image) or ""
            metadata.site_name = _meta_content(og_site) or ""

            # Resolve relative image URL
            if metadata.image_url and not metadata.image_url.startswith("http"):
                metadata.image_url = urljoin(url, metadata.image_url)

    except ImportError:
        metadata.error = "httpx/beautifulsoup4 not installed"
    except Exception as e:
        metadata.error = str(e)

    return metadata


def _meta_content(tag: Any) -> str:
    """Extract content attribute from a meta tag."""
    if tag and tag.get("content"):
        return tag["content"].strip()
    return ""


def _get_title(soup: Any) -> str:
    """Extract title from HTML."""
    title = soup.find("title")
    if title and title.string:
        return title.string.strip()
    return ""


def _get_description(soup: Any) -> str:
    """Extract description from HTML meta tags."""
    desc = soup.find("meta", attrs={"name": "description"})
    if desc and desc.get("content"):
        return desc["content"].strip()
    return ""


# ---------------------------------------------------------------------------
# Link transformation (formatting for channels)
# ---------------------------------------------------------------------------

def format_link_for_channel(link: DetectedLink, channel: str) -> str:
    """Format a link for display in a specific channel.

    Different channels have different link rendering:
    - Markdown channels: [title](url) or bare URL
    - Plain text channels: url
    - HTML channels: <a href="url">title</a>
    """
    display = link.display_text or link.domain or link.url

    if channel in ("webchat", "discord", "slack"):
        # Markdown format
        return f"[{display}]({link.url})"
    elif channel == "telegram":
        # Telegram supports markdown
        return f"[{display}]({link.url})"
    elif channel == "whatsapp":
        # WhatsApp auto-renders URLs, just pass through
        return link.url
    else:
        # Plain text
        return f"{display}: {link.url}" if display != link.url else link.url


def replace_links_for_channel(text: str, channel: str) -> str:
    """Replace all links in text with channel-appropriate formatting.

    Only modifies bare URLs — markdown links are left as-is.
    """
    links = detect_links(text)
    if not links:
        return text

    # Process in reverse order to preserve positions
    result = text
    for link in reversed(links):
        if not link.is_markdown:
            formatted = format_link_for_channel(link, channel)
            result = result[:link.start] + formatted + result[link.end:]

    return result


# ---------------------------------------------------------------------------
# Batch link processing
# ---------------------------------------------------------------------------

async def enrich_links(text: str, max_links: int = 5) -> list[LinkMetadata]:
    """Detect links in text and fetch metadata for each.

    Returns metadata for up to max_links URLs.
    """
    links = detect_links(text)
    if not links:
        return []

    import asyncio

    tasks = [
        fetch_link_metadata(link.url)
        for link in links[:max_links]
        if not link.is_media  # Skip direct media links
    ]

    if not tasks:
        return []

    results = await asyncio.gather(*tasks, return_exceptions=True)
    metadata = []
    for r in results:
        if isinstance(r, LinkMetadata):
            metadata.append(r)
    return metadata
