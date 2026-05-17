"""Agent identity system — per-agent name, emoji, ack reactions, and message prefixes.

Mirrors openclaw's identity.ts:
- Per-agent identity: name, emoji, ack reaction
- Message prefix for outbound messages (e.g., "[AgentName]")
- Response prefix for formatted responses
- Channel-level and account-level overrides
- Default ack reaction per agent
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_ACK_REACTION = "👀"
DEFAULT_AGENT_ID = "main"


@dataclass
class IdentityConfig:
    """Per-agent identity configuration."""

    agent_id: str = DEFAULT_AGENT_ID
    name: str = ""           # Human-readable name (e.g., "ResearchBot")
    emoji: str = ""          # Emoji avatar (e.g., "🔬")
    ack_reaction: str = ""   # Reaction to add when acknowledging a message
    message_prefix: str = "" # Prefix for outbound messages (e.g., "[Bot]")
    response_prefix: str = ""# Prefix for text responses

    @property
    def display_name(self) -> str:
        return self.name or self.agent_id

    @property
    def effective_ack_reaction(self) -> str:
        if self.ack_reaction:
            return self.ack_reaction
        if self.emoji:
            return self.emoji
        return DEFAULT_ACK_REACTION

    @property
    def name_prefix(self) -> str | None:
        if self.name:
            return f"[{self.name}]"
        return None

    def format_message(self, text: str) -> str:
        """Format an outbound message with identity prefix."""
        prefix = self.message_prefix or self.name_prefix or ""
        if prefix:
            return f"{prefix} {text}"
        return text

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "emoji": self.emoji,
            "ack_reaction": self.ack_reaction,
            "message_prefix": self.message_prefix,
            "response_prefix": self.response_prefix,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IdentityConfig":
        return cls(
            agent_id=data.get("agent_id", DEFAULT_AGENT_ID),
            name=data.get("name", ""),
            emoji=data.get("emoji", ""),
            ack_reaction=data.get("ack_reaction", ""),
            message_prefix=data.get("message_prefix", ""),
            response_prefix=data.get("response_prefix", ""),
        )

    @classmethod
    def from_profile(cls, profile: Any) -> "IdentityConfig":
        """Build IdentityConfig from an AgentProfile."""
        identity_data = getattr(profile, "identity", {}) or {}
        if isinstance(identity_data, dict):
            return cls.from_dict({
                "agent_id": getattr(profile, "agent_id", DEFAULT_AGENT_ID),
                **identity_data,
            })
        return cls(agent_id=getattr(profile, "agent_id", DEFAULT_AGENT_ID))


def resolve_agent_identity(agent_id: str) -> IdentityConfig:
    """Resolve identity config for a given agent ID.

    Reads from the agent registry profile or returns defaults.
    """
    try:
        from openclaw.agents import AgentRegistry
        registry = AgentRegistry()
        profile = registry.get(agent_id)
        if profile:
            return IdentityConfig.from_profile(profile)
    except Exception as e:
        logger.debug(f"Could not load identity for {agent_id}: {e}")

    return IdentityConfig(agent_id=agent_id)


def resolve_ack_reaction(
    agent_id: str,
    channel: str = "",
    account_id: str = "",
    config: Any = None,
) -> str:
    """Resolve the ack reaction for an agent.

    Priority (highest to lowest):
    1. Channel account-level config
    2. Channel-level config
    3. Global messages config
    4. Agent identity emoji fallback
    5. Default (👀)
    """
    # Check config overrides
    if config is not None:
        # Channel account level
        if channel and account_id:
            channels = getattr(config, "channels", {}) or {}
            ch_cfg = channels.get(channel, {}) or {}
            accounts = ch_cfg.get("accounts", {}) or {}
            acc_reaction = accounts.get(account_id, {}).get("ackReaction")
            if acc_reaction is not None:
                return acc_reaction.strip()

        # Channel level
        if channel:
            channels = getattr(config, "channels", {}) or {}
            ch_cfg = channels.get(channel, {}) or {}
            ch_reaction = ch_cfg.get("ackReaction")
            if ch_reaction is not None:
                return ch_reaction.strip()

        # Global level
        messages_cfg = getattr(config, "messages", {}) or {}
        global_reaction = messages_cfg.get("ackReaction") if isinstance(messages_cfg, dict) else None
        if global_reaction is not None:
            return global_reaction.strip()

    # Agent identity fallback
    identity = resolve_agent_identity(agent_id)
    return identity.effective_ack_reaction


def resolve_message_prefix(
    agent_id: str,
    channel: str = "",
    config: Any = None,
    has_allow_from: bool = False,
) -> str:
    """Resolve the message prefix for outbound messages.

    Returns empty string for channels where the agent is already identified,
    or a [Name] prefix for channels where multiple agents may send.
    """
    if config is not None:
        messages_cfg = getattr(config, "messages", {}) or {}
        configured = messages_cfg.get("messagePrefix") if isinstance(messages_cfg, dict) else None
        if configured is not None:
            return configured

    if has_allow_from:
        return ""

    identity = resolve_agent_identity(agent_id)
    return identity.name_prefix or f"[{DEFAULT_AGENT_ID}]"


def resolve_response_prefix(
    agent_id: str,
    channel: str = "",
    account_id: str = "",
    config: Any = None,
) -> str | None:
    """Resolve an optional prefix for response messages.

    Returns None if no prefix should be applied.
    """
    if config is not None:
        # Channel account level
        if channel and account_id:
            channels = getattr(config, "channels", {}) or {}
            ch_cfg = channels.get(channel, {}) or {}
            accounts = ch_cfg.get("accounts", {}) or {}
            resp_prefix = accounts.get(account_id, {}).get("responsePrefix")
            if resp_prefix is not None:
                return resp_prefix.strip() or None

        # Channel level
        if channel:
            channels = getattr(config, "channels", {}) or {}
            ch_cfg = channels.get(channel, {}) or {}
            resp_prefix = ch_cfg.get("responsePrefix")
            if resp_prefix is not None:
                return resp_prefix.strip() or None

        # Global level
        messages_cfg = getattr(config, "messages", {}) or {}
        global_prefix = messages_cfg.get("responsePrefix") if isinstance(messages_cfg, dict) else None
        if global_prefix is not None:
            return global_prefix.strip() or None

    return None
