"""Text-to-Speech system — multi-provider audio generation.

Mirrors openclaw's TTS system:
- Multi-provider: OpenAI TTS, ElevenLabs, Edge TTS
- Automatic fallback chain
- Channel-specific output formats (Opus for Telegram, MP3 default)
- Directive parsing ([[tts:voice=alloy]] syntax)
- Text preprocessing (markdown stripping, length limits)
"""

from openclaw.tts.core import (
    TtsProvider,
    TtsConfig,
    TtsResult,
    resolve_tts_config,
    text_to_speech,
    strip_markdown_for_audio,
)

__all__ = [
    "TtsProvider",
    "TtsConfig",
    "TtsResult",
    "resolve_tts_config",
    "text_to_speech",
    "strip_markdown_for_audio",
]
