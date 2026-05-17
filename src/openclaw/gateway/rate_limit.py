"""Rate limiting — sliding window rate limiter for the gateway.

Mirrors OpenClaw's auth-rate-limit system:
- Sliding window algorithm
- Per-IP and per-sender rate limiting
- Configurable windows and thresholds
- Lockout on excessive failures
- Automatic entry pruning
- Loopback exemption
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""

    # Request rate limiting
    requests_per_minute: int = 60
    requests_per_hour: int = 1000

    # Auth failure limiting
    max_auth_failures: int = 5
    auth_lockout_seconds: float = 300  # 5 minutes

    # Per-sender limits (more generous than per-IP)
    sender_requests_per_minute: int = 30

    # Exempt loopback addresses
    exempt_loopback: bool = True


@dataclass
class RateLimitEntry:
    """Tracking entry for a rate-limited entity."""

    timestamps: list[float] = field(default_factory=list)
    auth_failures: int = 0
    locked_until: float = 0
    last_request: float = 0

    @property
    def is_locked(self) -> bool:
        return self.locked_until > time.time()


@dataclass
class RateLimitResult:
    """Result of a rate limit check."""

    allowed: bool = True
    reason: str = ""
    retry_after_seconds: float = 0
    remaining_requests: int = 0


class SlidingWindowRateLimiter:
    """Sliding window rate limiter with lockout support.

    Tracks request timestamps per IP and per sender, using a sliding
    window to enforce rate limits. Supports auth failure tracking
    with automatic lockout.
    """

    def __init__(self, config: RateLimitConfig | None = None) -> None:
        self.config = config or RateLimitConfig()
        self._ip_entries: dict[str, RateLimitEntry] = defaultdict(RateLimitEntry)
        self._sender_entries: dict[str, RateLimitEntry] = defaultdict(RateLimitEntry)
        self._last_prune = time.time()

    def check_request(
        self,
        client_ip: str,
        sender_id: str = "",
    ) -> RateLimitResult:
        """Check if a request should be allowed.

        Evaluates rate limits for both IP and sender.
        """
        # Exempt loopback
        if self.config.exempt_loopback and client_ip in ("127.0.0.1", "::1", "localhost"):
            return RateLimitResult(allowed=True)

        now = time.time()

        # Auto-prune every 5 minutes
        if now - self._last_prune > 300:
            self._prune_old_entries()
            self._last_prune = now

        # Check IP lockout
        ip_entry = self._ip_entries[client_ip]
        if ip_entry.is_locked:
            retry_after = ip_entry.locked_until - now
            return RateLimitResult(
                allowed=False,
                reason=f"IP locked due to excessive auth failures (retry in {retry_after:.0f}s)",
                retry_after_seconds=retry_after,
            )

        # Check IP rate limit (per minute)
        one_minute_ago = now - 60
        ip_entry.timestamps = [t for t in ip_entry.timestamps if t > one_minute_ago]
        if len(ip_entry.timestamps) >= self.config.requests_per_minute:
            oldest = ip_entry.timestamps[0]
            retry_after = 60 - (now - oldest)
            return RateLimitResult(
                allowed=False,
                reason="Rate limit exceeded (per minute)",
                retry_after_seconds=max(0, retry_after),
                remaining_requests=0,
            )

        # Check IP rate limit (per hour)
        one_hour_ago = now - 3600
        hourly_count = sum(1 for t in ip_entry.timestamps if t > one_hour_ago)
        if hourly_count >= self.config.requests_per_hour:
            return RateLimitResult(
                allowed=False,
                reason="Rate limit exceeded (per hour)",
                retry_after_seconds=60,
                remaining_requests=0,
            )

        # Check sender rate limit
        if sender_id:
            sender_entry = self._sender_entries[sender_id]
            sender_entry.timestamps = [t for t in sender_entry.timestamps if t > one_minute_ago]
            if len(sender_entry.timestamps) >= self.config.sender_requests_per_minute:
                oldest = sender_entry.timestamps[0]
                retry_after = 60 - (now - oldest)
                return RateLimitResult(
                    allowed=False,
                    reason="Sender rate limit exceeded",
                    retry_after_seconds=max(0, retry_after),
                    remaining_requests=0,
                )
            sender_entry.timestamps.append(now)
            sender_entry.last_request = now

        # Record the request
        ip_entry.timestamps.append(now)
        ip_entry.last_request = now

        remaining = self.config.requests_per_minute - len(ip_entry.timestamps)
        return RateLimitResult(
            allowed=True,
            remaining_requests=remaining,
        )

    def record_auth_failure(self, client_ip: str) -> None:
        """Record an authentication failure for an IP.

        Locks the IP after max_auth_failures consecutive failures.
        """
        if self.config.exempt_loopback and client_ip in ("127.0.0.1", "::1"):
            return

        entry = self._ip_entries[client_ip]
        entry.auth_failures += 1

        if entry.auth_failures >= self.config.max_auth_failures:
            entry.locked_until = time.time() + self.config.auth_lockout_seconds
            logger.warning(
                "IP %s locked for %ds after %d auth failures",
                client_ip, self.config.auth_lockout_seconds, entry.auth_failures,
            )

    def reset_auth_failures(self, client_ip: str) -> None:
        """Reset auth failure count after successful authentication."""
        if client_ip in self._ip_entries:
            self._ip_entries[client_ip].auth_failures = 0

    def _prune_old_entries(self) -> None:
        """Remove entries with no recent activity."""
        cutoff = time.time() - 3600  # 1 hour
        stale_ips = [
            ip for ip, entry in self._ip_entries.items()
            if entry.last_request < cutoff and not entry.is_locked
        ]
        for ip in stale_ips:
            del self._ip_entries[ip]

        stale_senders = [
            sid for sid, entry in self._sender_entries.items()
            if entry.last_request < cutoff
        ]
        for sid in stale_senders:
            del self._sender_entries[sid]

        if stale_ips or stale_senders:
            logger.debug(
                "Rate limiter pruned: %d IPs, %d senders",
                len(stale_ips), len(stale_senders),
            )

    def get_stats(self) -> dict[str, Any]:
        """Return rate limiter statistics."""
        now = time.time()
        return {
            "tracked_ips": len(self._ip_entries),
            "tracked_senders": len(self._sender_entries),
            "locked_ips": sum(1 for e in self._ip_entries.values() if e.is_locked),
            "config": {
                "requests_per_minute": self.config.requests_per_minute,
                "requests_per_hour": self.config.requests_per_hour,
                "max_auth_failures": self.config.max_auth_failures,
                "lockout_seconds": self.config.auth_lockout_seconds,
            },
        }
