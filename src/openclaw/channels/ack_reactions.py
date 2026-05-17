"""ACK reactions — configurable per-channel message acknowledgment.

Mirrors openclaw's ack-reactions.ts:
- Configurable reaction scopes per channel (all, direct, group-all, group-mentions, off)
- WhatsApp-specific reaction modes (always, mentions, never)
- Automatic reaction removal after agent replies

Architecture:
    When a message arrives, the gateway checks shouldAckReaction() to
    decide whether to send an immediate "thinking" reaction (e.g., a
    thumbs-up emoji). After the agent replies, removeAckReactionAfterReply()
    cleans it up.

    Message in ──► shouldAckReaction() ──► add reaction (if yes)
                                                │
                        Agent processes...       │
                                                │
    Reply out ──► removeAckReactionAfterReply() ◄┘
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

logger = logging.getLogger(__name__)


# Type aliases
AckReactionScope = Literal["all", "direct", "group-all", "group-mentions", "off", "none"]
WhatsAppAckReactionMode = Literal["always", "mentions", "never"]


@dataclass
class AckReactionGateParams:
    """Parameters for deciding whether to ACK-react to a message."""

    scope: AckReactionScope = "direct"
    is_direct_message: bool = True
    is_group_message: bool = False
    is_mentioned: bool = False
    channel: str = ""
    sender_id: str = ""
    message_content: str = ""
    bot_user_id: str = ""


@dataclass
class AckReactionConfig:
    """Per-channel ACK reaction configuration."""

    enabled: bool = True
    scope: AckReactionScope = "direct"
    emoji: str = "eyes"  # Default thinking emoji
    remove_after_reply: bool = True
    whatsapp_mode: WhatsAppAckReactionMode = "mentions"


# Default configs per channel
DEFAULT_ACK_CONFIGS: dict[str, AckReactionConfig] = {
    "cli": AckReactionConfig(enabled=False),
    "webchat": AckReactionConfig(enabled=False),
    "discord": AckReactionConfig(
        enabled=True, scope="group-mentions", emoji="eyes",
    ),
    "telegram": AckReactionConfig(
        enabled=True, scope="direct", emoji="eyes",
    ),
    "whatsapp": AckReactionConfig(
        enabled=True, scope="all", emoji="eyes",
    ),
    "slack": AckReactionConfig(
        enabled=True, scope="group-mentions", emoji="eyes",
    ),
}


def get_ack_config(channel: str) -> AckReactionConfig:
    """Get the ACK reaction config for a channel, with defaults."""
    return DEFAULT_ACK_CONFIGS.get(channel, AckReactionConfig(enabled=False))


def should_ack_reaction(params: AckReactionGateParams) -> bool:
    """Determine if an ACK reaction should be sent for this message.

    Evaluates the reaction scope against message context:
    - "all": Always react
    - "direct": React only to DMs
    - "group-all": React to all group messages
    - "group-mentions": React only when bot is mentioned in groups
    - "off" / "none": Never react

    Args:
        params: Gate parameters with scope and message context.

    Returns:
        True if an ACK reaction should be sent.
    """
    scope = params.scope

    if scope in ("off", "none"):
        return False

    if scope == "all":
        return True

    if scope == "direct":
        return params.is_direct_message

    if scope == "group-all":
        return params.is_group_message

    if scope == "group-mentions":
        if not params.is_group_message:
            return params.is_direct_message
        return params.is_mentioned

    return False


def should_ack_reaction_for_whatsapp(
    *,
    mode: WhatsAppAckReactionMode = "mentions",
    is_mentioned: bool = False,
    is_group: bool = False,
) -> bool:
    """WhatsApp-specific ACK reaction logic.

    WhatsApp has different reaction semantics:
    - "always": React to every message
    - "mentions": React only when mentioned (in groups)
    - "never": Don't react

    Args:
        mode: WhatsApp reaction mode.
        is_mentioned: Whether the bot was mentioned.
        is_group: Whether this is a group chat.

    Returns:
        True if ACK reaction should be sent.
    """
    if mode == "never":
        return False

    if mode == "always":
        return True

    if mode == "mentions":
        if is_group:
            return is_mentioned
        return True  # Always react in DMs

    return False


def remove_ack_reaction_after_reply(
    *,
    channel: str,
    message_id: str,
    reaction_emoji: str = "eyes",
    remove_callback: Any = None,
) -> bool:
    """Remove the ACK reaction after the agent has replied.

    This is called by the gateway after sending the agent's response.
    The actual removal is channel-specific and handled by the callback.

    Args:
        channel: Channel name.
        message_id: Original message ID that was reacted to.
        reaction_emoji: The emoji that was used for the reaction.
        remove_callback: Channel-specific callback to remove the reaction.

    Returns:
        True if removal was attempted.
    """
    config = get_ack_config(channel)
    if not config.remove_after_reply:
        return False

    if remove_callback is not None:
        try:
            remove_callback(message_id, reaction_emoji)
            return True
        except Exception as e:
            logger.debug(f"Failed to remove ACK reaction: {e}")
            return False

    return False


def build_ack_reaction_params(
    *,
    channel: str,
    sender_id: str,
    message_content: str,
    is_dm: bool = False,
    is_group: bool = False,
    is_mentioned: bool = False,
    bot_user_id: str = "",
) -> AckReactionGateParams:
    """Helper to build AckReactionGateParams from common message fields."""
    config = get_ack_config(channel)
    return AckReactionGateParams(
        scope=config.scope,
        is_direct_message=is_dm,
        is_group_message=is_group,
        is_mentioned=is_mentioned,
        channel=channel,
        sender_id=sender_id,
        message_content=message_content,
        bot_user_id=bot_user_id,
    )
