"""TTS core — provider abstraction and audio generation.

Supports:
- OpenAI TTS (gpt-4o-mini-tts, tts-1, tts-1-hd)
- ElevenLabs (multilingual_v2, turbo_v2_5)
- Edge TTS (Microsoft free, via edge-tts package)

Each provider is called via its HTTP API or SDK.
Audio is saved to ~/.langclaw/tts/ and the path is returned.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

TTS_OUTPUT_DIR = Path.home() / ".langclaw" / "tts"

# Defaults
DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_MAX_TEXT_LENGTH = 1500

DEFAULT_OPENAI_MODEL = "gpt-4o-mini-tts"
DEFAULT_OPENAI_VOICE = "alloy"

DEFAULT_ELEVENLABS_MODEL = "eleven_multilingual_v2"
DEFAULT_ELEVENLABS_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # Rachel

DEFAULT_EDGE_VOICE = "en-US-MichelleNeural"


class TtsProvider(str, Enum):
    """Supported TTS providers."""

    OPENAI = "openai"
    ELEVENLABS = "elevenlabs"
    EDGE = "edge"


@dataclass
class TtsConfig:
    """Resolved TTS configuration."""

    provider: TtsProvider = TtsProvider.OPENAI
    model: str = DEFAULT_OPENAI_MODEL
    voice: str = DEFAULT_OPENAI_VOICE
    output_format: str = "mp3"  # mp3, opus, aac, flac
    speed: float = 1.0
    timeout: int = DEFAULT_TIMEOUT_SECONDS
    max_text_length: int = DEFAULT_MAX_TEXT_LENGTH

    # Provider-specific
    openai_api_key: str = ""
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = DEFAULT_ELEVENLABS_VOICE_ID
    elevenlabs_model_id: str = DEFAULT_ELEVENLABS_MODEL
    elevenlabs_stability: float = 0.5
    elevenlabs_similarity_boost: float = 0.75

    edge_voice: str = DEFAULT_EDGE_VOICE


@dataclass
class TtsDirectiveOverrides:
    """Overrides parsed from [[tts:...]] directives in text."""

    voice: str | None = None
    provider: str | None = None
    speed: float | None = None
    model: str | None = None


@dataclass
class TtsResult:
    """Result from TTS generation."""

    success: bool
    audio_path: str = ""
    duration_seconds: float = 0.0
    provider: str = ""
    error: str | None = None
    format: str = "mp3"


def resolve_tts_config(
    provider: str | None = None,
    voice: str | None = None,
    model: str | None = None,
    output_format: str | None = None,
    channel: str = "",
) -> TtsConfig:
    """Resolve TTS config from parameters and environment.

    Reads API keys from environment variables:
    - OPENAI_API_KEY for OpenAI TTS
    - ELEVENLABS_API_KEY for ElevenLabs
    """
    config = TtsConfig()

    # Resolve provider
    if provider:
        try:
            config.provider = TtsProvider(provider.lower())
        except ValueError:
            config.provider = TtsProvider.OPENAI

    # API keys from environment
    config.openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    config.elevenlabs_api_key = os.environ.get("ELEVENLABS_API_KEY", "")

    # Provider-specific defaults
    if config.provider == TtsProvider.OPENAI:
        config.model = model or DEFAULT_OPENAI_MODEL
        config.voice = voice or DEFAULT_OPENAI_VOICE
    elif config.provider == TtsProvider.ELEVENLABS:
        config.elevenlabs_voice_id = voice or DEFAULT_ELEVENLABS_VOICE_ID
        config.elevenlabs_model_id = model or DEFAULT_ELEVENLABS_MODEL
    elif config.provider == TtsProvider.EDGE:
        config.edge_voice = voice or DEFAULT_EDGE_VOICE

    # Channel-specific output format
    if output_format:
        config.output_format = output_format
    elif channel in ("telegram",):
        config.output_format = "opus"  # Telegram voice messages need Opus/OGG
    else:
        config.output_format = "mp3"

    return config


def strip_markdown_for_audio(text: str) -> str:
    """Strip markdown formatting from text for cleaner audio.

    Removes: headers, bold, italic, links, code blocks, images.
    Preserves: plain text content.
    """
    # Remove code blocks
    text = re.sub(r"```[\s\S]*?```", " (code block omitted) ", text)
    text = re.sub(r"`[^`]+`", "", text)

    # Remove headers
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

    # Remove bold/italic
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"__(.+?)__", r"\1", text)
    text = re.sub(r"_(.+?)_", r"\1", text)

    # Remove links, keep text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    # Remove images
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)

    # Remove horizontal rules
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)

    # Clean up whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_tts_directives(text: str) -> tuple[str, TtsDirectiveOverrides]:
    """Parse [[tts:...]] directives from text.

    Supported directives:
    - [[tts:voice=alloy]]
    - [[tts:provider=openai]]
    - [[tts:speed=1.5]]
    - [[tts:model=tts-1-hd]]

    Returns:
        Tuple of (cleaned text, parsed overrides).
    """
    overrides = TtsDirectiveOverrides()

    directive_pattern = r"\[\[tts:([^\]]+)\]\]"
    matches = re.findall(directive_pattern, text)

    for match in matches:
        pairs = match.split(",")
        for pair in pairs:
            if "=" not in pair:
                continue
            key, value = pair.strip().split("=", 1)
            key = key.strip().lower()
            value = value.strip()

            if key == "voice":
                overrides.voice = value
            elif key == "provider":
                overrides.provider = value
            elif key == "speed":
                try:
                    overrides.speed = float(value)
                except ValueError:
                    pass
            elif key == "model":
                overrides.model = value

    # Remove directives from text
    cleaned = re.sub(directive_pattern, "", text).strip()
    return cleaned, overrides


async def _tts_openai(text: str, config: TtsConfig) -> TtsResult:
    """Generate speech using OpenAI TTS API."""
    try:
        import httpx
    except ImportError:
        return TtsResult(success=False, error="httpx not installed for OpenAI TTS")

    if not config.openai_api_key:
        return TtsResult(success=False, error="OPENAI_API_KEY not set")

    TTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"tts_{hashlib.md5(text[:100].encode()).hexdigest()[:8]}_{int(time.time())}.{config.output_format}"
    output_path = TTS_OUTPUT_DIR / filename

    start = time.time()
    try:
        async with httpx.AsyncClient(timeout=config.timeout) as client:
            response = await client.post(
                "https://api.openai.com/v1/audio/speech",
                headers={
                    "Authorization": f"Bearer {config.openai_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": config.model,
                    "input": text,
                    "voice": config.voice,
                    "response_format": config.output_format,
                    "speed": config.speed,
                },
            )

            if response.status_code != 200:
                return TtsResult(
                    success=False,
                    error=f"OpenAI TTS API error: {response.status_code} {response.text[:200]}",
                    provider="openai",
                )

            output_path.write_bytes(response.content)
            elapsed = time.time() - start

            return TtsResult(
                success=True,
                audio_path=str(output_path),
                duration_seconds=elapsed,
                provider="openai",
                format=config.output_format,
            )

    except Exception as e:
        return TtsResult(success=False, error=f"OpenAI TTS failed: {e}", provider="openai")


async def _tts_elevenlabs(text: str, config: TtsConfig) -> TtsResult:
    """Generate speech using ElevenLabs API."""
    try:
        import httpx
    except ImportError:
        return TtsResult(success=False, error="httpx not installed for ElevenLabs TTS")

    if not config.elevenlabs_api_key:
        return TtsResult(success=False, error="ELEVENLABS_API_KEY not set")

    TTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"tts_{hashlib.md5(text[:100].encode()).hexdigest()[:8]}_{int(time.time())}.mp3"
    output_path = TTS_OUTPUT_DIR / filename

    start = time.time()
    try:
        voice_id = config.elevenlabs_voice_id
        async with httpx.AsyncClient(timeout=config.timeout) as client:
            response = await client.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                headers={
                    "xi-api-key": config.elevenlabs_api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "text": text,
                    "model_id": config.elevenlabs_model_id,
                    "voice_settings": {
                        "stability": config.elevenlabs_stability,
                        "similarity_boost": config.elevenlabs_similarity_boost,
                    },
                },
            )

            if response.status_code != 200:
                return TtsResult(
                    success=False,
                    error=f"ElevenLabs API error: {response.status_code} {response.text[:200]}",
                    provider="elevenlabs",
                )

            output_path.write_bytes(response.content)
            elapsed = time.time() - start

            return TtsResult(
                success=True,
                audio_path=str(output_path),
                duration_seconds=elapsed,
                provider="elevenlabs",
                format="mp3",
            )

    except Exception as e:
        return TtsResult(success=False, error=f"ElevenLabs TTS failed: {e}", provider="elevenlabs")


async def _tts_edge(text: str, config: TtsConfig) -> TtsResult:
    """Generate speech using Edge TTS (free Microsoft neural voices)."""
    try:
        import edge_tts
    except ImportError:
        return TtsResult(success=False, error="edge-tts package not installed (pip install edge-tts)")

    TTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"tts_{hashlib.md5(text[:100].encode()).hexdigest()[:8]}_{int(time.time())}.mp3"
    output_path = TTS_OUTPUT_DIR / filename

    start = time.time()
    try:
        communicate = edge_tts.Communicate(text, config.edge_voice)
        await communicate.save(str(output_path))
        elapsed = time.time() - start

        return TtsResult(
            success=True,
            audio_path=str(output_path),
            duration_seconds=elapsed,
            provider="edge",
            format="mp3",
        )

    except Exception as e:
        return TtsResult(success=False, error=f"Edge TTS failed: {e}", provider="edge")


# Provider dispatch
_PROVIDERS = {
    TtsProvider.OPENAI: _tts_openai,
    TtsProvider.ELEVENLABS: _tts_elevenlabs,
    TtsProvider.EDGE: _tts_edge,
}

# Fallback chain: try primary, then these
_FALLBACK_ORDER = [TtsProvider.OPENAI, TtsProvider.EDGE, TtsProvider.ELEVENLABS]


async def text_to_speech(
    text: str,
    config: TtsConfig | None = None,
    channel: str = "",
) -> TtsResult:
    """Convert text to speech using the configured provider with fallback.

    1. Strips markdown
    2. Parses directives
    3. Truncates to max length
    4. Tries primary provider
    5. Falls back through chain on failure

    Args:
        text: Text to convert to speech.
        config: TTS configuration (resolved from defaults if None).
        channel: Channel name for format selection.

    Returns:
        TtsResult with audio path or error.
    """
    cfg = config or resolve_tts_config(channel=channel)

    # Preprocess text
    clean_text, directives = parse_tts_directives(text)
    clean_text = strip_markdown_for_audio(clean_text)

    # Apply directives
    if directives.voice:
        cfg.voice = directives.voice
        cfg.edge_voice = directives.voice
    if directives.provider:
        try:
            cfg.provider = TtsProvider(directives.provider.lower())
        except ValueError:
            pass
    if directives.speed:
        cfg.speed = directives.speed
    if directives.model:
        cfg.model = directives.model

    # Truncate
    if len(clean_text) > cfg.max_text_length:
        clean_text = clean_text[: cfg.max_text_length] + "..."
        logger.info(f"TTS text truncated to {cfg.max_text_length} chars")

    if not clean_text.strip():
        return TtsResult(success=False, error="No text to convert")

    # Try primary provider
    provider_fn = _PROVIDERS.get(cfg.provider)
    if provider_fn:
        result = await provider_fn(clean_text, cfg)
        if result.success:
            logger.info(f"TTS generated: {result.provider}, {result.duration_seconds:.1f}s")
            return result
        logger.warning(f"TTS primary ({cfg.provider.value}) failed: {result.error}")

    # Fallback chain
    for fallback_provider in _FALLBACK_ORDER:
        if fallback_provider == cfg.provider:
            continue
        provider_fn = _PROVIDERS.get(fallback_provider)
        if provider_fn:
            result = await provider_fn(clean_text, cfg)
            if result.success:
                logger.info(f"TTS fallback success: {result.provider}")
                return result

    return TtsResult(success=False, error="All TTS providers failed")
