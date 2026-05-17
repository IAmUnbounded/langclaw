"""Routing system — session key resolution and channel-to-session binding.

Mirrors OpenClaw's routing module:
- Route resolution: determines which session handles a message
- Session key continuity: maintains sender-to-session mappings
- Route bindings: channel-specific session routing rules
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RouteBinding:
    """A binding between a channel+sender and a target session."""

    channel: str
    sender_id: str
    session_id: str
    agent_id: str = "main"
    group_id: str = ""
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    sticky: bool = False  # If True, always route to this session


@dataclass
class ResolvedRoute:
    """Result of route resolution — tells the system where to send a message."""

    session_id: str
    agent_id: str = "main"
    is_new_session: bool = False
    binding: RouteBinding | None = None


def make_session_key(channel: str, sender_id: str, group_id: str = "") -> str:
    """Generate a deterministic session key from channel + sender + group.

    This mirrors OpenClaw's resolve-session-key logic. Group messages
    get a shared session per group; DMs get a per-sender session.
    """
    if group_id:
        raw = f"{channel}:group:{group_id}"
    else:
        raw = f"{channel}:dm:{sender_id}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class RouteResolver:
    """Resolves inbound messages to target sessions.

    Manages a registry of route bindings that map (channel, sender)
    pairs to specific sessions. This enables:
    - Persistent conversations per sender
    - Shared group sessions
    - Agent-specific routing (different agents for different channels)
    - Sticky bindings (e.g. pairing a WhatsApp number to a session)
    """

    def __init__(self) -> None:
        self._bindings: dict[str, RouteBinding] = {}
        self._agent_routes: dict[str, str] = {}  # channel -> agent_id

    def bind(
        self,
        channel: str,
        sender_id: str,
        session_id: str,
        agent_id: str = "main",
        group_id: str = "",
        sticky: bool = False,
    ) -> RouteBinding:
        """Create or update a route binding."""
        key = make_session_key(channel, sender_id, group_id)
        binding = RouteBinding(
            channel=channel,
            sender_id=sender_id,
            session_id=session_id,
            agent_id=agent_id,
            group_id=group_id,
            sticky=sticky,
        )
        self._bindings[key] = binding
        logger.debug("Route bound: %s -> session=%s agent=%s", key, session_id, agent_id)
        return binding

    def unbind(self, channel: str, sender_id: str, group_id: str = "") -> bool:
        """Remove a route binding."""
        key = make_session_key(channel, sender_id, group_id)
        if key in self._bindings:
            del self._bindings[key]
            logger.debug("Route unbound: %s", key)
            return True
        return False

    def set_channel_agent(self, channel: str, agent_id: str) -> None:
        """Set the default agent for a channel."""
        self._agent_routes[channel] = agent_id

    def resolve(
        self,
        channel: str,
        sender_id: str,
        group_id: str = "",
    ) -> ResolvedRoute:
        """Resolve a message to a session.

        Resolution order:
        1. Check for existing sticky binding
        2. Check for existing non-sticky binding
        3. Create a new session key and return as new session
        """
        key = make_session_key(channel, sender_id, group_id)

        # Check existing bindings
        binding = self._bindings.get(key)
        if binding:
            binding.last_used = time.time()
            return ResolvedRoute(
                session_id=binding.session_id,
                agent_id=binding.agent_id,
                is_new_session=False,
                binding=binding,
            )

        # No binding — create new route
        agent_id = self._agent_routes.get(channel, "main")
        session_id = key  # Use the session key as the session ID

        return ResolvedRoute(
            session_id=session_id,
            agent_id=agent_id,
            is_new_session=True,
        )

    def list_bindings(self, channel: str | None = None) -> list[RouteBinding]:
        """List all route bindings, optionally filtered by channel."""
        bindings = list(self._bindings.values())
        if channel:
            bindings = [b for b in bindings if b.channel == channel]
        return sorted(bindings, key=lambda b: b.last_used, reverse=True)

    def cleanup_stale(self, max_age_seconds: float = 86400 * 30) -> int:
        """Remove bindings not used within max_age_seconds (default 30 days)."""
        cutoff = time.time() - max_age_seconds
        stale_keys = [
            k for k, b in self._bindings.items()
            if not b.sticky and b.last_used < cutoff
        ]
        for key in stale_keys:
            del self._bindings[key]
        if stale_keys:
            logger.info("Cleaned up %d stale route bindings", len(stale_keys))
        return len(stale_keys)

    def get_stats(self) -> dict[str, Any]:
        """Return routing statistics."""
        channels: dict[str, int] = {}
        for b in self._bindings.values():
            channels[b.channel] = channels.get(b.channel, 0) + 1
        return {
            "total_bindings": len(self._bindings),
            "by_channel": channels,
            "sticky_count": sum(1 for b in self._bindings.values() if b.sticky),
            "agent_routes": dict(self._agent_routes),
        }


# Module-level singleton
_resolver: RouteResolver | None = None


def get_resolver() -> RouteResolver:
    """Return the global route resolver singleton."""
    global _resolver
    if _resolver is None:
        _resolver = RouteResolver()
    return _resolver
