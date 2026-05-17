"""Auth profile health checking — validate API keys and track profile status.

Mirrors openclaw's auth-health.ts and live-auth-keys.ts:
- Validate API keys by making lightweight health check calls
- Track which profiles are healthy vs failed
- Report auth profile status for debugging
- Support multiple profiles per provider with cooldown tracking
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Health check timeout
HEALTH_CHECK_TIMEOUT = 10.0  # seconds

# Health check cache TTL (don't re-check too often)
HEALTH_CACHE_TTL = 300.0  # 5 minutes


@dataclass
class AuthHealthResult:
    """Result of an API key health check."""

    provider: str
    profile_id: str
    healthy: bool
    error: str | None = None
    latency_ms: float | None = None
    checked_at: float = field(default_factory=time.time)
    model: str | None = None  # Model used for health check


@dataclass
class AuthProfileStatus:
    """Current status of all auth profiles."""

    profiles: list[AuthHealthResult] = field(default_factory=list)
    checked_at: float = field(default_factory=time.time)

    @property
    def healthy_count(self) -> int:
        return sum(1 for p in self.profiles if p.healthy)

    @property
    def total_count(self) -> int:
        return len(self.profiles)

    def has_healthy_provider(self, provider: str) -> bool:
        return any(p.provider == provider and p.healthy for p in self.profiles)

    def summary(self) -> str:
        healthy = [p.profile_id for p in self.profiles if p.healthy]
        failed = [p.profile_id for p in self.profiles if not p.healthy]

        lines = [f"Auth profiles: {self.healthy_count}/{self.total_count} healthy"]
        if healthy:
            lines.append(f"  Healthy: {', '.join(healthy)}")
        if failed:
            lines.append(f"  Failed: {', '.join(failed)}")
        return "\n".join(lines)


# Health check cache
_health_cache: dict[str, AuthHealthResult] = {}
_cache_lock = asyncio.Lock()


async def check_api_key_health(
    provider: str,
    api_key: str,
    profile_id: str = "default",
) -> AuthHealthResult:
    """Check if an API key is valid by making a lightweight test call.

    Uses a minimal, cheap request to validate the key without incurring
    significant cost.
    """
    start = time.time()

    # Check cache first
    cache_key = f"{provider}:{api_key[:8]}:{profile_id}"
    async with _cache_lock:
        cached = _health_cache.get(cache_key)
        if cached and (time.time() - cached.checked_at) < HEALTH_CACHE_TTL:
            return cached

    try:
        if provider == "openai":
            result = await _check_openai(api_key, profile_id)
        elif provider == "anthropic":
            result = await _check_anthropic(api_key, profile_id)
        elif provider == "google":
            result = await _check_google(api_key, profile_id)
        else:
            result = AuthHealthResult(
                provider=provider,
                profile_id=profile_id,
                healthy=False,
                error=f"Unknown provider: {provider}",
            )

        result.latency_ms = (time.time() - start) * 1000

    except asyncio.TimeoutError:
        result = AuthHealthResult(
            provider=provider,
            profile_id=profile_id,
            healthy=False,
            error=f"Health check timed out after {HEALTH_CHECK_TIMEOUT}s",
            latency_ms=(time.time() - start) * 1000,
        )
    except Exception as e:
        result = AuthHealthResult(
            provider=provider,
            profile_id=profile_id,
            healthy=False,
            error=str(e),
            latency_ms=(time.time() - start) * 1000,
        )

    # Cache the result
    async with _cache_lock:
        _health_cache[cache_key] = result

    if result.healthy:
        logger.debug(f"Auth health OK: {profile_id} ({result.latency_ms:.0f}ms)")
    else:
        logger.warning(f"Auth health FAIL: {profile_id} — {result.error}")

    return result


async def _check_openai(api_key: str, profile_id: str) -> AuthHealthResult:
    """Check OpenAI API key via models list."""
    import httpx

    async with httpx.AsyncClient(timeout=HEALTH_CHECK_TIMEOUT) as client:
        resp = await client.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        if resp.status_code == 200:
            return AuthHealthResult(
                provider="openai",
                profile_id=profile_id,
                healthy=True,
            )
        elif resp.status_code == 401:
            return AuthHealthResult(
                provider="openai",
                profile_id=profile_id,
                healthy=False,
                error=f"Invalid API key (401)",
            )
        elif resp.status_code == 429:
            # Rate limited but key is valid
            return AuthHealthResult(
                provider="openai",
                profile_id=profile_id,
                healthy=True,
                error="Rate limited (key is valid)",
            )
        else:
            return AuthHealthResult(
                provider="openai",
                profile_id=profile_id,
                healthy=False,
                error=f"HTTP {resp.status_code}",
            )


async def _check_anthropic(api_key: str, profile_id: str) -> AuthHealthResult:
    """Check Anthropic API key via a minimal messages call."""
    import httpx

    async with httpx.AsyncClient(timeout=HEALTH_CHECK_TIMEOUT) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 1,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        if resp.status_code in (200, 400):
            # 400 might be validation error but key is valid
            return AuthHealthResult(
                provider="anthropic",
                profile_id=profile_id,
                healthy=True,
                model="claude-haiku-4-5",
            )
        elif resp.status_code == 401:
            return AuthHealthResult(
                provider="anthropic",
                profile_id=profile_id,
                healthy=False,
                error="Invalid API key (401)",
            )
        elif resp.status_code == 529:
            # Overloaded but key is valid
            return AuthHealthResult(
                provider="anthropic",
                profile_id=profile_id,
                healthy=True,
                error="API overloaded (key is valid)",
            )
        else:
            data = {}
            try:
                data = resp.json()
            except Exception:
                pass
            error_msg = data.get("error", {}).get("message", f"HTTP {resp.status_code}")
            return AuthHealthResult(
                provider="anthropic",
                profile_id=profile_id,
                healthy=resp.status_code < 500,
                error=error_msg,
            )


async def _check_google(api_key: str, profile_id: str) -> AuthHealthResult:
    """Check Google AI API key via models list."""
    import httpx

    async with httpx.AsyncClient(timeout=HEALTH_CHECK_TIMEOUT) as client:
        resp = await client.get(
            f"https://generativelanguage.googleapis.com/v1beta/models",
            params={"key": api_key},
        )
        if resp.status_code == 200:
            return AuthHealthResult(
                provider="google",
                profile_id=profile_id,
                healthy=True,
            )
        elif resp.status_code in (400, 401, 403):
            return AuthHealthResult(
                provider="google",
                profile_id=profile_id,
                healthy=False,
                error=f"Invalid API key ({resp.status_code})",
            )
        else:
            return AuthHealthResult(
                provider="google",
                profile_id=profile_id,
                healthy=resp.status_code < 500,
                error=f"HTTP {resp.status_code}",
            )


async def check_all_configured_keys(config: Any) -> AuthProfileStatus:
    """Check all API keys configured in the system.

    Args:
        config: OpenClawConfig instance.

    Returns:
        AuthProfileStatus with health check results for all profiles.
    """
    results: list[AuthHealthResult] = []

    # Read API keys from config and environment
    keys = {
        "openai": getattr(config.api_keys, "openai", "") or os.environ.get("OPENAI_API_KEY", ""),
        "anthropic": getattr(config.api_keys, "anthropic", "") or os.environ.get("ANTHROPIC_API_KEY", ""),
        "google": getattr(config.api_keys, "google", "") or os.environ.get("GOOGLE_API_KEY", ""),
    }

    tasks = []
    for provider, key in keys.items():
        if key:
            tasks.append(
                check_api_key_health(provider, key, f"{provider}-default")
            )

    if not tasks:
        return AuthProfileStatus(profiles=[])

    check_results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in check_results:
        if isinstance(result, AuthHealthResult):
            results.append(result)
        elif isinstance(result, Exception):
            logger.warning(f"Health check exception: {result}")

    return AuthProfileStatus(profiles=results)


def clear_health_cache() -> None:
    """Clear the health check cache to force fresh checks."""
    _health_cache.clear()
    logger.debug("Auth health cache cleared")
