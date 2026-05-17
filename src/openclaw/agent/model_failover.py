"""Model Failover & Auth Profiles — multi-auth rotation and model fallback.

Mirrors OpenClaw's model-auth.ts / model-fallback.ts:
- Multiple auth profiles per provider with cooldown
- Automatic rotation on auth/rate-limit/billing errors
- Model fallback chain (primary -> fallback models)
- Error classification for smart retry decisions
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)

# Cooldown durations per error type (seconds)
COOLDOWNS: dict[str, float] = {
    "auth": 300.0,        # 5 minutes
    "rate_limit": 60.0,   # 1 minute
    "billing": 3600.0,    # 1 hour
    "context_overflow": 0.0,  # Not transient — caller must compress
    "unavailable": 120.0,    # 2 minutes
    "unknown": 30.0,
}


@dataclass
class AuthProfile:
    """A single authentication profile for an LLM provider."""

    profile_id: str
    provider: str  # "openai", "anthropic", "google"
    method: str = "api-key"  # "api-key" or "oauth"
    api_key: str = ""
    scope: str = ""  # e.g. "Claude Pro", "Claude Max"
    last_used: float = 0.0
    failure_count: int = 0
    cooldown_until: float = 0.0
    is_active: bool = True

    def is_available(self) -> bool:
        """Check if profile is available (active and not in cooldown)."""
        return self.is_active and time.time() >= self.cooldown_until

    def record_failure(self, cooldown_seconds: float = 60.0) -> None:
        self.failure_count += 1
        self.cooldown_until = time.time() + cooldown_seconds

    def record_success(self) -> None:
        self.last_used = time.time()
        self.failure_count = 0
        self.cooldown_until = 0.0


@dataclass
class ModelFallbackChain:
    """Ordered chain of models to try on failure."""

    primary: str
    fallbacks: list[str] = field(default_factory=list)

    def get_chain(self) -> list[str]:
        return [self.primary] + self.fallbacks


class FailoverManager:
    """Manages auth profile rotation and model failover."""

    def __init__(self) -> None:
        self._profiles: dict[str, list[AuthProfile]] = {}  # provider -> profiles
        self._fallback_chain: ModelFallbackChain | None = None
        self._lock = threading.Lock()

    def add_profile(self, profile: AuthProfile) -> None:
        with self._lock:
            self._profiles.setdefault(profile.provider, []).append(profile)

    def get_available_profile(self, provider: str) -> AuthProfile | None:
        """Get the next available auth profile for a provider."""
        with self._lock:
            profiles = self._profiles.get(provider, [])
            # Sort by least recently used
            available = [p for p in profiles if p.is_available()]
            if not available:
                return None
            available.sort(key=lambda p: p.last_used)
            return available[0]

    def record_failure(self, profile_id: str, error_type: str) -> None:
        cooldown = COOLDOWNS.get(error_type, COOLDOWNS["unknown"])
        with self._lock:
            for profiles in self._profiles.values():
                for p in profiles:
                    if p.profile_id == profile_id:
                        p.record_failure(cooldown)
                        logger.warning(
                            f"Auth profile {profile_id} failed ({error_type}), "
                            f"cooldown {cooldown}s"
                        )
                        return

    def record_success(self, profile_id: str) -> None:
        with self._lock:
            for profiles in self._profiles.values():
                for p in profiles:
                    if p.profile_id == profile_id:
                        p.record_success()
                        return

    def set_fallback_chain(self, chain: ModelFallbackChain) -> None:
        self._fallback_chain = chain

    def get_next_model(self, failed_model: str) -> str | None:
        """Get next model in fallback chain after a failure."""
        if not self._fallback_chain:
            return None
        chain = self._fallback_chain.get_chain()
        try:
            idx = chain.index(failed_model)
            if idx + 1 < len(chain):
                return chain[idx + 1]
        except ValueError:
            pass
        return None

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            status: dict[str, Any] = {"profiles": {}, "fallback_chain": None}
            for provider, profiles in self._profiles.items():
                status["profiles"][provider] = [
                    {
                        "profile_id": p.profile_id,
                        "method": p.method,
                        "scope": p.scope,
                        "is_available": p.is_available(),
                        "failure_count": p.failure_count,
                        "cooldown_remaining": max(0, p.cooldown_until - time.time()),
                    }
                    for p in profiles
                ]
            if self._fallback_chain:
                status["fallback_chain"] = self._fallback_chain.get_chain()
            return status

    @classmethod
    def from_config(cls, config: Any) -> "FailoverManager":
        """Build from OpenClawConfig.

        Creates auth profiles from available API keys and sets up
        fallback chain from config.
        """
        fm = cls()

        # Create profiles from API keys
        keys = {
            "openai": config.api_keys.openai or os.environ.get("OPENAI_API_KEY", ""),
            "anthropic": config.api_keys.anthropic or os.environ.get("ANTHROPIC_API_KEY", ""),
            "google": config.api_keys.google or os.environ.get("GOOGLE_API_KEY", ""),
        }
        for provider, key in keys.items():
            if key:
                fm.add_profile(AuthProfile(
                    profile_id=f"{provider}-default",
                    provider=provider,
                    api_key=key,
                    scope="default",
                ))

        # Set up fallback chain
        primary = config.agent.model
        fallbacks = getattr(config.agent, "fallback_models", [])
        if not fallbacks:
            # Auto-generate fallback chain from available keys
            provider, _ = config.parse_model()
            fallback_models = []
            defaults = {
                "openai": "openai/gpt-4o",
                "anthropic": "anthropic/claude-sonnet-4-20250514",
                "google": "google/gemini-2.0-flash",
            }
            for p, model in defaults.items():
                if p != provider and keys.get(p):
                    fallback_models.append(model)
            fallbacks = fallback_models

        fm.set_fallback_chain(ModelFallbackChain(primary=primary, fallbacks=fallbacks))
        return fm


def classify_llm_error(error: Exception) -> str:
    """Classify an LLM error into a category for failover decisions."""
    msg = str(error).lower()
    if any(s in msg for s in ["authentication", "invalid api key", "unauthorized", "401", "invalid x-api-key"]):
        return "auth"
    if any(s in msg for s in ["rate limit", "429", "too many requests", "rate_limit"]):
        return "rate_limit"
    if any(s in msg for s in ["billing", "quota", "insufficient", "402", "payment"]):
        return "billing"
    if any(s in msg for s in ["context", "too long", "maximum context", "token limit", "max_tokens"]):
        return "context_overflow"
    if any(s in msg for s in ["unavailable", "503", "502", "overloaded", "capacity"]):
        return "unavailable"
    return "unknown"


def _build_llm(model_str: str, api_key: str, temperature: float) -> BaseChatModel:
    """Build an LLM instance from a model string."""
    if "/" in model_str:
        provider, model_name = model_str.split("/", 1)
    else:
        provider, model_name = "openai", model_str

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model_name, temperature=temperature, api_key=api_key or None)
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model_name, temperature=temperature, api_key=api_key or None)
    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model_name, temperature=temperature, google_api_key=api_key or None)
    else:
        raise ValueError(f"Unsupported provider: {provider}")


async def resilient_llm_call(
    config: Any,
    messages: list[BaseMessage],
    tools: list | None = None,
    failover_manager: FailoverManager | None = None,
    max_retries: int = 3,
) -> Any:
    """Make an LLM call with automatic failover.

    1. Try primary model with available auth profile
    2. On auth/rate_limit/billing error: try next auth profile
    3. If all profiles exhausted: try next model in fallback chain
    4. On context overflow: raise (caller should compress)
    5. On unavailable: retry with backoff
    """
    fm = failover_manager or FailoverManager.from_config(config)
    chain = fm._fallback_chain.get_chain() if fm._fallback_chain else [config.agent.model]

    last_error: Exception | None = None

    for model_str in chain:
        provider = model_str.split("/")[0] if "/" in model_str else "openai"

        for attempt in range(max_retries):
            profile = fm.get_available_profile(provider)
            api_key = profile.api_key if profile else ""

            try:
                llm = _build_llm(model_str, api_key, config.agent.temperature)
                if tools:
                    llm = llm.bind_tools(tools)

                response = await llm.ainvoke(messages)

                if profile:
                    fm.record_success(profile.profile_id)
                logger.info(f"LLM call succeeded: {model_str}")
                return response

            except Exception as e:
                last_error = e
                error_type = classify_llm_error(e)
                logger.warning(f"LLM call failed ({model_str}): {error_type} — {e}")

                if profile:
                    fm.record_failure(profile.profile_id, error_type)

                if error_type == "context_overflow":
                    raise  # Caller must compress

                if error_type == "unavailable" and attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.info(f"Retrying in {wait}s...")
                    await asyncio.sleep(wait)
                    continue

                # Try next profile or model
                break

    raise last_error or RuntimeError("All models and auth profiles exhausted")
