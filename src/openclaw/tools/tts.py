"""TTS tool — text-to-speech conversion for the agent.

Mirrors openclaw's tts-tool.ts:
- Converts text to audio using configured TTS provider
- Supports provider/voice/speed overrides
- Returns audio file path with MEDIA: tag for channel delivery
- Channel-specific formats (Opus for Telegram voice messages)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def tts_tool(text: str, provider: str = "", voice: str = "", channel: str = "") -> str:
    """Convert text to speech audio.

    Generates an audio file from the given text using TTS (text-to-speech).
    Supports multiple providers: OpenAI, ElevenLabs, Edge TTS (free).

    You can include directives in the text like [[tts:voice=nova]] or
    [[tts:provider=edge,speed=1.2]] to override settings.

    Args:
        text: The text to convert to speech.
        provider: TTS provider (openai, elevenlabs, edge). Default: openai.
        voice: Voice name/ID. Default: alloy (OpenAI), MichelleNeural (Edge).
        channel: Target channel for format selection (e.g., telegram for Opus).

    Returns:
        MEDIA: path to the audio file, or error message.
    """
    try:
        from openclaw.tts import text_to_speech, resolve_tts_config

        config = resolve_tts_config(
            provider=provider or None,
            voice=voice or None,
            channel=channel,
        )

        # Run async TTS in event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    result = pool.submit(
                        lambda: asyncio.run(text_to_speech(text, config=config, channel=channel))
                    ).result(timeout=60)
            else:
                result = loop.run_until_complete(
                    text_to_speech(text, config=config, channel=channel)
                )
        except RuntimeError:
            result = asyncio.run(text_to_speech(text, config=config, channel=channel))

        if result.success:
            media_tag = f"MEDIA:{result.audio_path}"
            # For Telegram, add voice message hint
            if channel == "telegram" and result.format == "opus":
                media_tag += " [[audio_as_voice]]"
            return (
                f"{media_tag}\n\n"
                f"Audio generated successfully ({result.provider}, "
                f"{result.duration_seconds:.1f}s, {result.format})"
            )
        else:
            return f"[error] TTS generation failed: {result.error}"

    except ImportError as e:
        return f"[error] TTS dependencies not available: {e}. Install with: pip install edge-tts httpx"
    except Exception as e:
        logger.error(f"TTS tool error: {e}")
        return f"[error] TTS failed: {e}"
