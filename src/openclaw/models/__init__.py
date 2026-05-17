"""Model discovery and management — listing, aliases, status, and fallback chains.

Mirrors OpenClaw's model management system:
- Model catalog: known models with metadata (context window, pricing, capabilities)
- Model aliases: short names for commonly used models
- Provider probing: check which providers have valid API keys
- Model status: verify model availability
- Fallback chains: ordered list of models to try on failure
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Metadata about a known LLM model."""

    provider: str  # openai, anthropic, google
    model_id: str  # Full model ID (e.g. gpt-4o, claude-sonnet-4-20250514)
    display_name: str  # Human-friendly name
    context_window: int = 128000
    max_output_tokens: int = 4096
    supports_vision: bool = False
    supports_tools: bool = True
    supports_streaming: bool = True
    supports_thinking: bool = False  # Extended thinking (Claude)
    input_price_per_1m: float = 0.0  # USD per 1M input tokens
    output_price_per_1m: float = 0.0  # USD per 1M output tokens
    is_deprecated: bool = False
    notes: str = ""


# ---------------------------------------------------------------------------
# Model catalog — known models and their capabilities
# ---------------------------------------------------------------------------

MODEL_CATALOG: dict[str, ModelInfo] = {
    # OpenAI
    "openai/gpt-4o": ModelInfo(
        provider="openai", model_id="gpt-4o",
        display_name="GPT-4o",
        context_window=128000, max_output_tokens=16384,
        supports_vision=True,
        input_price_per_1m=2.50, output_price_per_1m=10.00,
    ),
    "openai/gpt-4o-mini": ModelInfo(
        provider="openai", model_id="gpt-4o-mini",
        display_name="GPT-4o Mini",
        context_window=128000, max_output_tokens=16384,
        supports_vision=True,
        input_price_per_1m=0.15, output_price_per_1m=0.60,
    ),
    "openai/gpt-4.1": ModelInfo(
        provider="openai", model_id="gpt-4.1",
        display_name="GPT-4.1",
        context_window=1047576, max_output_tokens=32768,
        supports_vision=True,
        input_price_per_1m=2.00, output_price_per_1m=8.00,
    ),
    "openai/gpt-4.1-mini": ModelInfo(
        provider="openai", model_id="gpt-4.1-mini",
        display_name="GPT-4.1 Mini",
        context_window=1047576, max_output_tokens=32768,
        supports_vision=True,
        input_price_per_1m=0.40, output_price_per_1m=1.60,
    ),
    "openai/gpt-4.1-nano": ModelInfo(
        provider="openai", model_id="gpt-4.1-nano",
        display_name="GPT-4.1 Nano",
        context_window=1047576, max_output_tokens=32768,
        supports_vision=True,
        input_price_per_1m=0.10, output_price_per_1m=0.40,
    ),
    "openai/o3": ModelInfo(
        provider="openai", model_id="o3",
        display_name="O3",
        context_window=200000, max_output_tokens=100000,
        supports_vision=True, supports_thinking=True,
        input_price_per_1m=2.00, output_price_per_1m=8.00,
    ),
    "openai/o3-mini": ModelInfo(
        provider="openai", model_id="o3-mini",
        display_name="O3 Mini",
        context_window=200000, max_output_tokens=100000,
        supports_thinking=True,
        input_price_per_1m=1.10, output_price_per_1m=4.40,
    ),
    "openai/o4-mini": ModelInfo(
        provider="openai", model_id="o4-mini",
        display_name="O4 Mini",
        context_window=200000, max_output_tokens=100000,
        supports_vision=True, supports_thinking=True,
        input_price_per_1m=1.10, output_price_per_1m=4.40,
    ),
    # Anthropic
    "anthropic/claude-opus-4-20250514": ModelInfo(
        provider="anthropic", model_id="claude-opus-4-20250514",
        display_name="Claude Opus 4",
        context_window=200000, max_output_tokens=32000,
        supports_vision=True, supports_thinking=True,
        input_price_per_1m=15.00, output_price_per_1m=75.00,
    ),
    "anthropic/claude-sonnet-4-20250514": ModelInfo(
        provider="anthropic", model_id="claude-sonnet-4-20250514",
        display_name="Claude Sonnet 4",
        context_window=200000, max_output_tokens=16000,
        supports_vision=True, supports_thinking=True,
        input_price_per_1m=3.00, output_price_per_1m=15.00,
    ),
    "anthropic/claude-haiku-4-20250414": ModelInfo(
        provider="anthropic", model_id="claude-haiku-4-20250414",
        display_name="Claude Haiku 4",
        context_window=200000, max_output_tokens=8192,
        supports_vision=True,
        input_price_per_1m=0.80, output_price_per_1m=4.00,
    ),
    "anthropic/claude-3-5-sonnet-20241022": ModelInfo(
        provider="anthropic", model_id="claude-3-5-sonnet-20241022",
        display_name="Claude 3.5 Sonnet",
        context_window=200000, max_output_tokens=8192,
        supports_vision=True, supports_thinking=True,
        input_price_per_1m=3.00, output_price_per_1m=15.00,
    ),
    # Google
    "google/gemini-2.5-pro": ModelInfo(
        provider="google", model_id="gemini-2.5-pro",
        display_name="Gemini 2.5 Pro",
        context_window=1048576, max_output_tokens=65536,
        supports_vision=True, supports_thinking=True,
        input_price_per_1m=1.25, output_price_per_1m=10.00,
    ),
    "google/gemini-2.5-flash": ModelInfo(
        provider="google", model_id="gemini-2.5-flash",
        display_name="Gemini 2.5 Flash",
        context_window=1048576, max_output_tokens=65536,
        supports_vision=True, supports_thinking=True,
        input_price_per_1m=0.15, output_price_per_1m=0.60,
    ),
    "google/gemini-2.0-flash": ModelInfo(
        provider="google", model_id="gemini-2.0-flash",
        display_name="Gemini 2.0 Flash",
        context_window=1048576, max_output_tokens=8192,
        supports_vision=True,
        input_price_per_1m=0.10, output_price_per_1m=0.40,
    ),
}

# ---------------------------------------------------------------------------
# Model aliases — short names
# ---------------------------------------------------------------------------

MODEL_ALIASES: dict[str, str] = {
    # OpenAI shortcuts
    "gpt4o": "openai/gpt-4o",
    "gpt4o-mini": "openai/gpt-4o-mini",
    "4o": "openai/gpt-4o",
    "4o-mini": "openai/gpt-4o-mini",
    "4.1": "openai/gpt-4.1",
    "4.1-mini": "openai/gpt-4.1-mini",
    "4.1-nano": "openai/gpt-4.1-nano",
    "o3": "openai/o3",
    "o3-mini": "openai/o3-mini",
    "o4-mini": "openai/o4-mini",
    # Anthropic shortcuts
    "opus": "anthropic/claude-opus-4-20250514",
    "opus4": "anthropic/claude-opus-4-20250514",
    "sonnet": "anthropic/claude-sonnet-4-20250514",
    "sonnet4": "anthropic/claude-sonnet-4-20250514",
    "haiku": "anthropic/claude-haiku-4-20250414",
    "haiku4": "anthropic/claude-haiku-4-20250414",
    "claude": "anthropic/claude-sonnet-4-20250514",
    "sonnet-3.5": "anthropic/claude-3-5-sonnet-20241022",
    # Google shortcuts
    "gemini": "google/gemini-2.5-flash",
    "gemini-pro": "google/gemini-2.5-pro",
    "gemini-flash": "google/gemini-2.5-flash",
    "flash": "google/gemini-2.5-flash",
}

# ---------------------------------------------------------------------------
# Default fallback chains
# ---------------------------------------------------------------------------

DEFAULT_FALLBACK_CHAINS: dict[str, list[str]] = {
    "openai/gpt-4o": ["openai/gpt-4o-mini", "google/gemini-2.0-flash"],
    "anthropic/claude-sonnet-4-20250514": [
        "anthropic/claude-haiku-4-20250414",
        "openai/gpt-4o",
        "google/gemini-2.0-flash",
    ],
    "anthropic/claude-opus-4-20250514": [
        "anthropic/claude-sonnet-4-20250514",
        "openai/gpt-4o",
    ],
    "google/gemini-2.5-pro": ["google/gemini-2.5-flash", "openai/gpt-4o"],
}


# ---------------------------------------------------------------------------
# Provider probing
# ---------------------------------------------------------------------------

def get_available_providers() -> dict[str, bool]:
    """Check which LLM providers have API keys configured."""
    return {
        "openai": bool(os.environ.get("OPENAI_API_KEY")),
        "anthropic": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "google": bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")),
    }


def get_available_models(provider: str | None = None) -> list[ModelInfo]:
    """List models that are available (provider has API key).

    If provider is specified, only list models for that provider.
    """
    available_providers = get_available_providers()
    models = []
    for key, info in MODEL_CATALOG.items():
        if info.is_deprecated:
            continue
        if provider and info.provider != provider:
            continue
        if available_providers.get(info.provider, False):
            models.append(info)
    return models


async def probe_model(model_key: str) -> dict[str, Any]:
    """Probe a model to check if it's available and responding.

    Returns status dict with availability and latency.
    """
    import time as _time

    info = MODEL_CATALOG.get(model_key)
    if not info:
        return {"model": model_key, "available": False, "error": "Unknown model"}

    providers = get_available_providers()
    if not providers.get(info.provider):
        return {
            "model": model_key,
            "available": False,
            "error": f"No API key for {info.provider}",
        }

    start = _time.monotonic()
    try:
        if info.provider == "openai":
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model=info.model_id, max_tokens=5)
            await llm.ainvoke("Hi")
        elif info.provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            llm = ChatAnthropic(model=info.model_id, max_tokens=5)
            await llm.ainvoke("Hi")
        elif info.provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(model=info.model_id, max_output_tokens=5)
            await llm.ainvoke("Hi")

        latency = _time.monotonic() - start
        return {
            "model": model_key,
            "available": True,
            "latency_ms": round(latency * 1000),
        }
    except Exception as e:
        return {
            "model": model_key,
            "available": False,
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# Resolve helpers
# ---------------------------------------------------------------------------

def resolve_model(model_str: str) -> str:
    """Resolve a model string (possibly an alias) to a full provider/model key.

    Examples:
        "sonnet" -> "anthropic/claude-sonnet-4-20250514"
        "openai/gpt-4o" -> "openai/gpt-4o"
        "gpt-4o" -> "openai/gpt-4o"
    """
    # Check direct alias
    if model_str in MODEL_ALIASES:
        return MODEL_ALIASES[model_str]

    # Already a full key
    if model_str in MODEL_CATALOG:
        return model_str

    # Try with provider prefix
    for prefix in ("openai/", "anthropic/", "google/"):
        full = prefix + model_str
        if full in MODEL_CATALOG:
            return full

    # Return as-is (custom/unknown model)
    logger.warning("Unknown model: %s — passing through as-is", model_str)
    return model_str


def get_model_info(model_str: str) -> ModelInfo | None:
    """Get model info, resolving aliases first."""
    resolved = resolve_model(model_str)
    return MODEL_CATALOG.get(resolved)


def get_fallback_chain(model_str: str) -> list[str]:
    """Get the fallback chain for a model."""
    resolved = resolve_model(model_str)
    return list(DEFAULT_FALLBACK_CHAINS.get(resolved, []))


def estimate_cost(
    model_str: str,
    input_tokens: int,
    output_tokens: int,
) -> float | None:
    """Estimate the cost of a request in USD.

    Returns None if model pricing is not known.
    """
    info = get_model_info(model_str)
    if not info or (info.input_price_per_1m == 0 and info.output_price_per_1m == 0):
        return None

    input_cost = (input_tokens / 1_000_000) * info.input_price_per_1m
    output_cost = (output_tokens / 1_000_000) * info.output_price_per_1m
    return round(input_cost + output_cost, 6)


def list_aliases() -> dict[str, str]:
    """Return a copy of all model aliases."""
    return dict(MODEL_ALIASES)


def add_alias(alias: str, model_key: str) -> None:
    """Add a custom model alias."""
    MODEL_ALIASES[alias] = model_key


def remove_alias(alias: str) -> bool:
    """Remove a model alias."""
    if alias in MODEL_ALIASES:
        del MODEL_ALIASES[alias]
        return True
    return False
