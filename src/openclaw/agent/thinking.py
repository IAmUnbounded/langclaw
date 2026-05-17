"""Thinking Levels — configurable extended thinking / reasoning depth.

Mirrors OpenClaw's thinking level system:
- off: Standard mode, no extended thinking
- low: Light reasoning (~1K tokens)
- mid: Moderate reasoning (~4K tokens)
- high: Deep reasoning (~16K tokens)
- max: Maximum reasoning (~64K tokens)

Provider support:
- Anthropic (claude-sonnet-4+, claude-opus-4+): Extended thinking
- OpenAI (o1, o3): Reasoning effort
- Google (gemini-2.5-pro): Thinking config
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


class ThinkingLevel(str, Enum):
    OFF = "off"
    LOW = "low"
    MID = "mid"
    HIGH = "high"
    MAX = "max"


# Map levels to token budgets
_LEVEL_BUDGETS = {
    ThinkingLevel.OFF: 0,
    ThinkingLevel.LOW: 1024,
    ThinkingLevel.MID: 4096,
    ThinkingLevel.HIGH: 16384,
    ThinkingLevel.MAX: 65536,
}

_LEVEL_DESCRIPTIONS = {
    ThinkingLevel.OFF: "No extended thinking (standard mode)",
    ThinkingLevel.LOW: "Light reasoning (~1K tokens budget)",
    ThinkingLevel.MID: "Moderate reasoning (~4K tokens budget)",
    ThinkingLevel.HIGH: "Deep reasoning (~16K tokens budget)",
    ThinkingLevel.MAX: "Maximum reasoning (~64K tokens budget)",
}

# OpenAI reasoning effort mapping
_OPENAI_EFFORT = {
    ThinkingLevel.OFF: None,
    ThinkingLevel.LOW: "low",
    ThinkingLevel.MID: "medium",
    ThinkingLevel.HIGH: "high",
    ThinkingLevel.MAX: "high",
}


@dataclass
class ThinkingConfig:
    """Configuration for extended thinking/reasoning."""

    level: ThinkingLevel = ThinkingLevel.OFF
    budget_tokens: int = 0  # 0 = auto based on level
    stream_thinking: bool = False

    @classmethod
    def from_level(cls, level: str | ThinkingLevel) -> "ThinkingConfig":
        if isinstance(level, str):
            level = ThinkingLevel(level)
        return cls(
            level=level,
            budget_tokens=_LEVEL_BUDGETS.get(level, 0),
        )


def supports_thinking(provider: str, model_name: str) -> bool:
    """Check if a model supports extended thinking/reasoning."""
    if provider == "anthropic":
        return any(x in model_name for x in [
            "claude-sonnet-4", "claude-opus-4",
        ])
    elif provider == "openai":
        return any(x in model_name for x in ["o1", "o3"])
    elif provider == "google":
        return "gemini-2.5" in model_name
    return False


def get_thinking_level_description(level: ThinkingLevel) -> str:
    """Human-readable description of a thinking level."""
    return _LEVEL_DESCRIPTIONS.get(level, "Unknown")


def apply_thinking_to_llm(
    llm: BaseChatModel,
    thinking_config: ThinkingConfig,
    provider: str,
    model_name: str,
) -> BaseChatModel:
    """Apply thinking/reasoning configuration to an LLM instance.

    Returns a new or reconfigured LLM. Does not mutate the original.
    """
    if thinking_config.level == ThinkingLevel.OFF:
        return llm

    if not supports_thinking(provider, model_name):
        logger.warning(
            f"Model {provider}/{model_name} does not support extended thinking. "
            f"Ignoring thinking level '{thinking_config.level.value}'."
        )
        return llm

    budget = thinking_config.budget_tokens

    if provider == "anthropic":
        # Anthropic extended thinking requires temperature=1 and
        # the 'thinking' parameter via model_kwargs.
        try:
            from langchain_anthropic import ChatAnthropic

            new_llm = ChatAnthropic(
                model=model_name,
                temperature=1,  # Required for extended thinking
                api_key=getattr(llm, "anthropic_api_key", None) or getattr(llm, "api_key", None),
                model_kwargs={
                    "thinking": {
                        "type": "enabled",
                        "budget_tokens": budget,
                    }
                },
            )
            logger.info(f"Anthropic extended thinking enabled: {budget} token budget")
            return new_llm
        except Exception as e:
            logger.warning(f"Failed to apply Anthropic thinking: {e}")
            return llm

    elif provider == "openai":
        # OpenAI reasoning models use reasoning_effort
        effort = _OPENAI_EFFORT.get(thinking_config.level)
        if not effort:
            return llm
        try:
            from langchain_openai import ChatOpenAI

            new_llm = ChatOpenAI(
                model=model_name,
                temperature=getattr(llm, "temperature", 1),
                api_key=getattr(llm, "openai_api_key", None) or getattr(llm, "api_key", None),
                model_kwargs={"reasoning_effort": effort},
            )
            logger.info(f"OpenAI reasoning effort set to '{effort}'")
            return new_llm
        except Exception as e:
            logger.warning(f"Failed to apply OpenAI reasoning: {e}")
            return llm

    elif provider == "google":
        # Google Gemini thinking config
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            new_llm = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=getattr(llm, "temperature", 0.7),
                google_api_key=getattr(llm, "google_api_key", None),
                model_kwargs={"thinking_config": {"thinking_budget": budget}},
            )
            logger.info(f"Google thinking config enabled: {budget} token budget")
            return new_llm
        except Exception as e:
            logger.warning(f"Failed to apply Google thinking: {e}")
            return llm

    return llm
