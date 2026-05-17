"""Model catalog — context window lookup and model capability registry.

Mirrors openclaw's context.ts model discovery and model-catalog.ts:
- Known context windows for popular models
- Dynamic config-based overrides
- Provider-aware capability flags (vision, thinking, tools)
- Used by compressor and context window guard for token budgeting
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Default context window when model is unknown
DEFAULT_CONTEXT_TOKENS = 128_000

# Safety margin: assume 20% of context window is used by system prompt overhead
SAFETY_MARGIN = 0.8


@dataclass
class ModelCapabilities:
    """Capabilities of a specific model."""

    context_window: int = DEFAULT_CONTEXT_TOKENS
    max_output_tokens: int = 8192
    supports_vision: bool = False
    supports_thinking: bool = False
    supports_tools: bool = True
    supports_streaming: bool = True
    provider: str = ""
    model_id: str = ""


# Known model catalog: provider/model -> capabilities
# Sources: Anthropic, OpenAI, Google docs (as of 2025)
_KNOWN_MODELS: dict[str, ModelCapabilities] = {
    # ---- Anthropic ----
    "anthropic/claude-opus-4-5": ModelCapabilities(
        context_window=200_000, max_output_tokens=32_768,
        supports_vision=True, supports_thinking=True, provider="anthropic", model_id="claude-opus-4-5",
    ),
    "anthropic/claude-opus-4-7": ModelCapabilities(
        context_window=200_000, max_output_tokens=32_768,
        supports_vision=True, supports_thinking=True, provider="anthropic", model_id="claude-opus-4-7",
    ),
    "anthropic/claude-sonnet-4-5": ModelCapabilities(
        context_window=200_000, max_output_tokens=16_000,
        supports_vision=True, supports_thinking=True, provider="anthropic", model_id="claude-sonnet-4-5",
    ),
    "anthropic/claude-sonnet-4-6": ModelCapabilities(
        context_window=200_000, max_output_tokens=16_000,
        supports_vision=True, supports_thinking=True, provider="anthropic", model_id="claude-sonnet-4-6",
    ),
    "anthropic/claude-haiku-4-5": ModelCapabilities(
        context_window=200_000, max_output_tokens=8_192,
        supports_vision=True, supports_thinking=False, provider="anthropic", model_id="claude-haiku-4-5",
    ),
    # Legacy aliases
    "anthropic/claude-sonnet-4-20250514": ModelCapabilities(
        context_window=200_000, max_output_tokens=16_000,
        supports_vision=True, supports_thinking=True, provider="anthropic",
    ),
    "anthropic/claude-3-7-sonnet-20250219": ModelCapabilities(
        context_window=200_000, max_output_tokens=16_000,
        supports_vision=True, supports_thinking=True, provider="anthropic",
    ),
    "anthropic/claude-3-5-sonnet-20241022": ModelCapabilities(
        context_window=200_000, max_output_tokens=8_192,
        supports_vision=True, supports_thinking=False, provider="anthropic",
    ),
    "anthropic/claude-3-5-haiku-20241022": ModelCapabilities(
        context_window=200_000, max_output_tokens=8_192,
        supports_vision=True, supports_thinking=False, provider="anthropic",
    ),
    "anthropic/claude-3-opus-20240229": ModelCapabilities(
        context_window=200_000, max_output_tokens=4_096,
        supports_vision=True, supports_thinking=False, provider="anthropic",
    ),
    # ---- OpenAI ----
    "openai/gpt-4o": ModelCapabilities(
        context_window=128_000, max_output_tokens=16_384,
        supports_vision=True, supports_thinking=False, provider="openai", model_id="gpt-4o",
    ),
    "openai/gpt-4o-mini": ModelCapabilities(
        context_window=128_000, max_output_tokens=16_384,
        supports_vision=True, supports_thinking=False, provider="openai", model_id="gpt-4o-mini",
    ),
    "openai/gpt-4.1": ModelCapabilities(
        context_window=1_000_000, max_output_tokens=32_768,
        supports_vision=True, supports_thinking=False, provider="openai", model_id="gpt-4.1",
    ),
    "openai/gpt-4.1-mini": ModelCapabilities(
        context_window=1_000_000, max_output_tokens=32_768,
        supports_vision=True, supports_thinking=False, provider="openai", model_id="gpt-4.1-mini",
    ),
    "openai/o1": ModelCapabilities(
        context_window=200_000, max_output_tokens=100_000,
        supports_vision=True, supports_thinking=True, provider="openai", model_id="o1",
    ),
    "openai/o1-mini": ModelCapabilities(
        context_window=128_000, max_output_tokens=65_536,
        supports_vision=False, supports_thinking=True, provider="openai", model_id="o1-mini",
    ),
    "openai/o3": ModelCapabilities(
        context_window=200_000, max_output_tokens=100_000,
        supports_vision=True, supports_thinking=True, provider="openai", model_id="o3",
    ),
    "openai/o3-mini": ModelCapabilities(
        context_window=200_000, max_output_tokens=65_536,
        supports_vision=False, supports_thinking=True, provider="openai", model_id="o3-mini",
    ),
    "openai/o4-mini": ModelCapabilities(
        context_window=200_000, max_output_tokens=100_000,
        supports_vision=True, supports_thinking=True, provider="openai", model_id="o4-mini",
    ),
    # ---- Google ----
    "google/gemini-2.5-pro": ModelCapabilities(
        context_window=1_000_000, max_output_tokens=65_536,
        supports_vision=True, supports_thinking=True, provider="google", model_id="gemini-2.5-pro",
    ),
    "google/gemini-2.5-flash": ModelCapabilities(
        context_window=1_000_000, max_output_tokens=65_536,
        supports_vision=True, supports_thinking=True, provider="google", model_id="gemini-2.5-flash",
    ),
    "google/gemini-2.0-flash": ModelCapabilities(
        context_window=1_048_576, max_output_tokens=8_192,
        supports_vision=True, supports_thinking=False, provider="google", model_id="gemini-2.0-flash",
    ),
    "google/gemini-1.5-pro": ModelCapabilities(
        context_window=2_000_000, max_output_tokens=8_192,
        supports_vision=True, supports_thinking=False, provider="google", model_id="gemini-1.5-pro",
    ),
    "google/gemini-1.5-flash": ModelCapabilities(
        context_window=1_000_000, max_output_tokens=8_192,
        supports_vision=True, supports_thinking=False, provider="google", model_id="gemini-1.5-flash",
    ),
}

# Runtime override cache (populated from config)
_runtime_overrides: dict[str, int] = {}


def _normalize_model_key(model_str: str) -> str:
    """Normalize a model string to canonical 'provider/model' form."""
    model_str = model_str.strip()
    if "/" not in model_str:
        # Try to infer provider from known model names
        lower = model_str.lower()
        if lower.startswith("claude"):
            return f"anthropic/{model_str}"
        elif lower.startswith("gpt") or lower.startswith("o1") or lower.startswith("o3") or lower.startswith("o4"):
            return f"openai/{model_str}"
        elif lower.startswith("gemini"):
            return f"google/{model_str}"
        return model_str
    return model_str


def lookup_model(model_str: str) -> ModelCapabilities:
    """Look up model capabilities by model string.

    Falls back to default capabilities if the model is unknown.
    """
    key = _normalize_model_key(model_str)

    # Exact match
    if key in _KNOWN_MODELS:
        caps = _KNOWN_MODELS[key]
        # Apply runtime override if present
        if key in _runtime_overrides:
            caps = ModelCapabilities(
                **{**caps.__dict__, "context_window": _runtime_overrides[key]}
            )
        return caps

    # Prefix match (handle version-specific model IDs)
    for catalog_key, caps in _KNOWN_MODELS.items():
        if key.startswith(catalog_key) or catalog_key.startswith(key):
            return caps

    # Provider-level defaults
    provider = key.split("/")[0] if "/" in key else "unknown"
    defaults = {
        "anthropic": ModelCapabilities(context_window=200_000, max_output_tokens=8_192, supports_vision=True, provider="anthropic"),
        "openai": ModelCapabilities(context_window=128_000, max_output_tokens=16_384, supports_vision=True, provider="openai"),
        "google": ModelCapabilities(context_window=1_000_000, max_output_tokens=8_192, supports_vision=True, provider="google"),
    }

    if provider in defaults:
        logger.debug(f"Model '{model_str}' not in catalog, using {provider} defaults")
        return defaults[provider]

    logger.debug(f"Unknown model '{model_str}', using global defaults")
    return ModelCapabilities()


def lookup_context_window(model_str: str) -> int:
    """Get the context window size for a model in tokens."""
    return lookup_model(model_str).context_window


def apply_config_overrides(config_overrides: dict[str, int]) -> None:
    """Apply runtime context window overrides from configuration."""
    for model_id, context_window in config_overrides.items():
        if context_window > 0:
            key = _normalize_model_key(model_id)
            _runtime_overrides[key] = context_window
            logger.debug(f"Context window override: {key} → {context_window:,}")


def get_effective_context_budget(model_str: str, fraction: float = SAFETY_MARGIN) -> int:
    """Get the effective token budget for a model, after safety margin.

    Args:
        model_str: Model identifier.
        fraction: Fraction of context window to use (default 0.8 = 80%).

    Returns:
        Token budget in tokens.
    """
    window = lookup_context_window(model_str)
    return max(1000, int(window * fraction))


def list_models(provider: str | None = None) -> list[dict[str, Any]]:
    """List all known models, optionally filtered by provider."""
    result = []
    for key, caps in sorted(_KNOWN_MODELS.items()):
        if provider and not key.startswith(provider + "/"):
            continue
        result.append({
            "model": key,
            "context_window": caps.context_window,
            "max_output_tokens": caps.max_output_tokens,
            "supports_vision": caps.supports_vision,
            "supports_thinking": caps.supports_thinking,
            "supports_tools": caps.supports_tools,
        })
    return result
