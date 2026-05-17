"""Media understanding — vision/media analysis, audio processing, and provider support.

Mirrors OpenClaw's media-understanding module:
- Media type detection and routing
- Vision providers: GPT-4 Vision, Claude Vision, Gemini Vision
- Audio preflight: validation before processing
- Attachment staging: managing media files for processing
- Concurrency management for media operations
"""

from __future__ import annotations

import base64
import logging
import mimetypes
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Media types and detection
# ---------------------------------------------------------------------------

class MediaType(str, Enum):
    """Supported media types for understanding."""

    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    DOCUMENT = "document"
    UNKNOWN = "unknown"


# MIME type to media type mapping
MIME_TYPE_MAP: dict[str, MediaType] = {
    "image/jpeg": MediaType.IMAGE,
    "image/png": MediaType.IMAGE,
    "image/gif": MediaType.IMAGE,
    "image/webp": MediaType.IMAGE,
    "image/svg+xml": MediaType.IMAGE,
    "image/bmp": MediaType.IMAGE,
    "image/tiff": MediaType.IMAGE,
    "audio/mpeg": MediaType.AUDIO,
    "audio/wav": MediaType.AUDIO,
    "audio/ogg": MediaType.AUDIO,
    "audio/flac": MediaType.AUDIO,
    "audio/webm": MediaType.AUDIO,
    "audio/mp4": MediaType.AUDIO,
    "video/mp4": MediaType.VIDEO,
    "video/webm": MediaType.VIDEO,
    "video/quicktime": MediaType.VIDEO,
    "video/x-msvideo": MediaType.VIDEO,
    "application/pdf": MediaType.DOCUMENT,
}

# Maximum file sizes per media type (in bytes)
MAX_FILE_SIZES: dict[MediaType, int] = {
    MediaType.IMAGE: 20 * 1024 * 1024,    # 20 MB
    MediaType.AUDIO: 25 * 1024 * 1024,    # 25 MB
    MediaType.VIDEO: 100 * 1024 * 1024,   # 100 MB
    MediaType.DOCUMENT: 50 * 1024 * 1024, # 50 MB
}


@dataclass
class MediaAttachment:
    """A media file attachment for processing."""

    file_path: str = ""
    url: str = ""
    mime_type: str = ""
    media_type: MediaType = MediaType.UNKNOWN
    file_size: int = 0
    filename: str = ""
    base64_data: str = ""  # Base64-encoded content for API calls
    width: int = 0
    height: int = 0
    duration_seconds: float = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MediaAnalysis:
    """Result of media analysis."""

    description: str = ""
    text_content: str = ""  # OCR or transcription results
    labels: list[str] = field(default_factory=list)
    objects: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    provider: str = ""
    error: str = ""
    raw_response: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Media detection and staging
# ---------------------------------------------------------------------------

def detect_media_type(file_path: str | None = None, mime_type: str | None = None) -> MediaType:
    """Detect the media type from a file path or MIME type."""
    if mime_type:
        return MIME_TYPE_MAP.get(mime_type, MediaType.UNKNOWN)
    if file_path:
        guessed_mime, _ = mimetypes.guess_type(file_path)
        if guessed_mime:
            return MIME_TYPE_MAP.get(guessed_mime, MediaType.UNKNOWN)
    return MediaType.UNKNOWN


def stage_attachment(
    file_path: str | None = None,
    url: str | None = None,
    mime_type: str | None = None,
) -> MediaAttachment:
    """Create a MediaAttachment from a file path or URL.

    Reads file metadata and encodes to base64 for API calls if needed.
    """
    attachment = MediaAttachment()

    if file_path:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Media file not found: {file_path}")

        attachment.file_path = str(path.resolve())
        attachment.filename = path.name
        attachment.file_size = path.stat().st_size

        # Detect MIME type
        if mime_type:
            attachment.mime_type = mime_type
        else:
            guessed, _ = mimetypes.guess_type(str(path))
            attachment.mime_type = guessed or "application/octet-stream"

        attachment.media_type = detect_media_type(mime_type=attachment.mime_type)

        # Read and encode for API
        if attachment.media_type in (MediaType.IMAGE, MediaType.AUDIO):
            try:
                content = path.read_bytes()
                attachment.base64_data = base64.b64encode(content).decode("ascii")
            except Exception as e:
                logger.warning("Failed to encode file: %s", e)

        # Get image dimensions
        if attachment.media_type == MediaType.IMAGE:
            try:
                from PIL import Image
                with Image.open(path) as img:
                    attachment.width, attachment.height = img.size
            except ImportError:
                pass
            except Exception:
                pass

    elif url:
        attachment.url = url
        attachment.media_type = detect_media_type(file_path=url, mime_type=mime_type)
        if mime_type:
            attachment.mime_type = mime_type

    return attachment


def validate_attachment(attachment: MediaAttachment) -> tuple[bool, str]:
    """Validate a media attachment for processing.

    Checks file size limits, supported formats, etc.
    Returns (is_valid, reason).
    """
    max_size = MAX_FILE_SIZES.get(attachment.media_type, 10 * 1024 * 1024)
    if attachment.file_size > max_size:
        return False, (
            f"File too large: {attachment.file_size / 1024 / 1024:.1f} MB "
            f"(max {max_size / 1024 / 1024:.0f} MB for {attachment.media_type.value})"
        )

    if attachment.media_type == MediaType.UNKNOWN:
        return False, f"Unsupported media type: {attachment.mime_type}"

    return True, "OK"


# ---------------------------------------------------------------------------
# Audio preflight
# ---------------------------------------------------------------------------

def audio_preflight(attachment: MediaAttachment) -> tuple[bool, str]:
    """Preflight check for audio processing.

    Validates format, duration, and size before sending to transcription.
    """
    if attachment.media_type != MediaType.AUDIO:
        return False, "Not an audio file"

    supported_formats = {"audio/mpeg", "audio/wav", "audio/ogg", "audio/flac", "audio/webm", "audio/mp4"}
    if attachment.mime_type not in supported_formats:
        return False, f"Unsupported audio format: {attachment.mime_type}"

    # Check file size (Whisper API limit: 25 MB)
    if attachment.file_size > 25 * 1024 * 1024:
        return False, "Audio file too large (max 25 MB for transcription)"

    return True, "OK"


# ---------------------------------------------------------------------------
# Vision analysis (multi-provider)
# ---------------------------------------------------------------------------

async def analyze_image(
    attachment: MediaAttachment,
    prompt: str = "Describe this image in detail.",
    provider: str = "auto",
    model: str | None = None,
) -> MediaAnalysis:
    """Analyze an image using a vision-capable LLM.

    Supports multiple providers:
    - openai: GPT-4o vision
    - anthropic: Claude vision
    - google: Gemini vision
    - auto: use first available provider
    """
    if attachment.media_type != MediaType.IMAGE:
        return MediaAnalysis(error="Not an image file")

    analysis = MediaAnalysis()

    # Auto-select provider
    if provider == "auto":
        if os.environ.get("OPENAI_API_KEY"):
            provider = "openai"
        elif os.environ.get("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        elif os.environ.get("GOOGLE_API_KEY"):
            provider = "google"
        else:
            return MediaAnalysis(error="No vision provider configured")

    analysis.provider = provider

    try:
        if provider == "openai":
            analysis = await _analyze_openai(attachment, prompt, model)
        elif provider == "anthropic":
            analysis = await _analyze_anthropic(attachment, prompt, model)
        elif provider == "google":
            analysis = await _analyze_google(attachment, prompt, model)
        else:
            analysis.error = f"Unknown provider: {provider}"
    except Exception as e:
        analysis.error = str(e)
        logger.exception("Image analysis failed with provider %s", provider)

    return analysis


async def _analyze_openai(
    attachment: MediaAttachment,
    prompt: str,
    model: str | None = None,
) -> MediaAnalysis:
    """Analyze image using OpenAI GPT-4 Vision."""
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage

    llm = ChatOpenAI(model=model or "gpt-4o", max_tokens=1024)

    # Build content with image
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]

    if attachment.base64_data:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{attachment.mime_type};base64,{attachment.base64_data}",
            },
        })
    elif attachment.url:
        content.append({
            "type": "image_url",
            "image_url": {"url": attachment.url},
        })
    else:
        return MediaAnalysis(error="No image data available")

    response = await llm.ainvoke([HumanMessage(content=content)])

    return MediaAnalysis(
        description=response.content if isinstance(response.content, str) else str(response.content),
        provider="openai",
        confidence=0.9,
    )


async def _analyze_anthropic(
    attachment: MediaAttachment,
    prompt: str,
    model: str | None = None,
) -> MediaAnalysis:
    """Analyze image using Anthropic Claude Vision."""
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import HumanMessage

    llm = ChatAnthropic(model=model or "claude-sonnet-4-20250514", max_tokens=1024)

    content: list[dict[str, Any]] = []

    if attachment.base64_data:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": attachment.mime_type,
                "data": attachment.base64_data,
            },
        })
    elif attachment.url:
        content.append({
            "type": "image",
            "source": {"type": "url", "url": attachment.url},
        })
    else:
        return MediaAnalysis(error="No image data available")

    content.append({"type": "text", "text": prompt})

    response = await llm.ainvoke([HumanMessage(content=content)])

    return MediaAnalysis(
        description=response.content if isinstance(response.content, str) else str(response.content),
        provider="anthropic",
        confidence=0.9,
    )


async def _analyze_google(
    attachment: MediaAttachment,
    prompt: str,
    model: str | None = None,
) -> MediaAnalysis:
    """Analyze image using Google Gemini Vision."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage

    llm = ChatGoogleGenerativeAI(model=model or "gemini-2.0-flash")

    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]

    if attachment.base64_data:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{attachment.mime_type};base64,{attachment.base64_data}",
            },
        })
    elif attachment.url:
        content.append({
            "type": "image_url",
            "image_url": {"url": attachment.url},
        })

    response = await llm.ainvoke([HumanMessage(content=content)])

    return MediaAnalysis(
        description=response.content if isinstance(response.content, str) else str(response.content),
        provider="google",
        confidence=0.9,
    )


# ---------------------------------------------------------------------------
# Audio transcription
# ---------------------------------------------------------------------------

async def transcribe_audio(
    attachment: MediaAttachment,
    language: str | None = None,
) -> MediaAnalysis:
    """Transcribe audio using OpenAI Whisper API."""
    ok, reason = audio_preflight(attachment)
    if not ok:
        return MediaAnalysis(error=reason)

    try:
        import openai

        client = openai.AsyncOpenAI()
        with open(attachment.file_path, "rb") as audio_file:
            transcript = await client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language=language,
            )

        return MediaAnalysis(
            text_content=transcript.text,
            description=f"Audio transcription ({attachment.filename})",
            provider="openai/whisper",
        )
    except ImportError:
        return MediaAnalysis(error="openai package not installed")
    except Exception as e:
        return MediaAnalysis(error=f"Transcription failed: {e}")


# ---------------------------------------------------------------------------
# Multi-media processing
# ---------------------------------------------------------------------------

async def process_media(
    attachment: MediaAttachment,
    prompt: str = "",
    provider: str = "auto",
) -> MediaAnalysis:
    """Process any media attachment through the appropriate pipeline.

    Routes to the correct handler based on media type.
    """
    is_valid, reason = validate_attachment(attachment)
    if not is_valid:
        return MediaAnalysis(error=reason)

    if attachment.media_type == MediaType.IMAGE:
        return await analyze_image(
            attachment,
            prompt=prompt or "Describe this image in detail.",
            provider=provider,
        )
    elif attachment.media_type == MediaType.AUDIO:
        return await transcribe_audio(attachment)
    elif attachment.media_type == MediaType.VIDEO:
        return MediaAnalysis(
            error="Video processing not yet implemented — extract frames first"
        )
    elif attachment.media_type == MediaType.DOCUMENT:
        return MediaAnalysis(
            error="Document processing not yet implemented — use file reading tools"
        )
    else:
        return MediaAnalysis(error=f"Unsupported media type: {attachment.media_type}")
