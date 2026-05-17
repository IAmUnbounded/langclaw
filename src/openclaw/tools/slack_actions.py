"""Slack actions tool — channel-specific operations for Slack.

Mirrors openclaw's slack-actions.ts:
- Send, edit, delete messages
- React with emoji, list reactions
- Pin/unpin messages, list pins
- Read channel messages
- Auto-threading based on context

Actions:
- sendMessage: Send a message to a channel
- editMessage: Edit a previously sent message
- deleteMessage: Delete a message
- readMessages: Read recent messages from a channel
- react: Add an emoji reaction
- reactions: List reactions on a message
- pinMessage: Pin a message
- unpinMessage: Unpin a message
- listPins: List pinned messages
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def slack_actions_tool(
    action: str,
    channel: str = "",
    text: str = "",
    message_ts: str = "",
    thread_ts: str = "",
    emoji: str = "",
    limit: int = 10,
) -> str:
    """Perform Slack-specific actions.

    Available actions:
    - sendMessage: Send a message (requires channel, text; optional thread_ts)
    - editMessage: Edit a message (requires channel, message_ts, text)
    - deleteMessage: Delete a message (requires channel, message_ts)
    - readMessages: Read recent messages (requires channel; optional limit)
    - react: Add emoji reaction (requires channel, message_ts, emoji)
    - reactions: Get reactions on a message (requires channel, message_ts)
    - pinMessage: Pin a message (requires channel, message_ts)
    - unpinMessage: Unpin a message (requires channel, message_ts)
    - listPins: List pinned items (requires channel)

    Args:
        action: The action to perform.
        channel: Slack channel ID (e.g., C01234567).
        text: Message text (for send/edit).
        message_ts: Message timestamp (for edit/delete/react/pin).
        thread_ts: Thread timestamp (for threaded replies).
        emoji: Emoji name without colons (for react).
        limit: Number of messages to read (for readMessages).

    Returns:
        Result of the action as a string.
    """
    try:
        return _handle_action(
            action=action,
            channel=channel,
            text=text,
            message_ts=message_ts,
            thread_ts=thread_ts,
            emoji=emoji,
            limit=limit,
        )
    except Exception as e:
        return f"[error] Slack action failed: {e}"


def _handle_action(
    action: str,
    channel: str,
    text: str,
    message_ts: str,
    thread_ts: str,
    emoji: str,
    limit: int,
) -> str:
    """Dispatch and execute a Slack action.

    Uses the Slack Web API client from the adapter singleton.
    Falls back to direct API calls if adapter is not available.
    """
    import asyncio

    try:
        from slack_sdk import WebClient
    except ImportError:
        return "[error] slack-sdk not installed. Run: pip install slack-bolt"

    import os
    token = os.environ.get("SLACK_BOT_TOKEN", "")
    if not token:
        return "[error] SLACK_BOT_TOKEN not set in environment"

    client = WebClient(token=token)

    if action == "sendMessage":
        if not channel or not text:
            return "[error] sendMessage requires 'channel' and 'text'"
        kwargs: dict[str, Any] = {"channel": channel, "text": text}
        if thread_ts:
            kwargs["thread_ts"] = thread_ts
        resp = client.chat_postMessage(**kwargs)
        ts = resp.get("ts", "")
        return f"Message sent to {channel} (ts: {ts})"

    elif action == "editMessage":
        if not channel or not message_ts or not text:
            return "[error] editMessage requires 'channel', 'message_ts', and 'text'"
        client.chat_update(channel=channel, ts=message_ts, text=text)
        return f"Message {message_ts} updated in {channel}"

    elif action == "deleteMessage":
        if not channel or not message_ts:
            return "[error] deleteMessage requires 'channel' and 'message_ts'"
        client.chat_delete(channel=channel, ts=message_ts)
        return f"Message {message_ts} deleted from {channel}"

    elif action == "readMessages":
        if not channel:
            return "[error] readMessages requires 'channel'"
        kwargs_hist: dict[str, Any] = {"channel": channel, "limit": min(limit, 50)}
        if thread_ts:
            resp = client.conversations_replies(
                channel=channel, ts=thread_ts, limit=min(limit, 50)
            )
        else:
            resp = client.conversations_history(**kwargs_hist)
        messages = resp.get("messages", [])
        lines = []
        for msg in messages:
            user = msg.get("user", "bot")
            ts = msg.get("ts", "")
            txt = msg.get("text", "")[:200]
            lines.append(f"[{ts}] <{user}>: {txt}")
        return "\n".join(lines) if lines else "No messages found"

    elif action == "react":
        if not channel or not message_ts or not emoji:
            return "[error] react requires 'channel', 'message_ts', and 'emoji'"
        client.reactions_add(channel=channel, timestamp=message_ts, name=emoji)
        return f"Added :{emoji}: to {message_ts}"

    elif action == "reactions":
        if not channel or not message_ts:
            return "[error] reactions requires 'channel' and 'message_ts'"
        resp = client.reactions_get(channel=channel, timestamp=message_ts)
        msg = resp.get("message", {})
        reactions = msg.get("reactions", [])
        if not reactions:
            return "No reactions on this message"
        lines = [f":{r['name']}: ({r['count']})" for r in reactions]
        return "\n".join(lines)

    elif action == "pinMessage":
        if not channel or not message_ts:
            return "[error] pinMessage requires 'channel' and 'message_ts'"
        client.pins_add(channel=channel, timestamp=message_ts)
        return f"Pinned message {message_ts} in {channel}"

    elif action == "unpinMessage":
        if not channel or not message_ts:
            return "[error] unpinMessage requires 'channel' and 'message_ts'"
        client.pins_remove(channel=channel, timestamp=message_ts)
        return f"Unpinned message {message_ts} in {channel}"

    elif action == "listPins":
        if not channel:
            return "[error] listPins requires 'channel'"
        resp = client.pins_list(channel=channel)
        items = resp.get("items", [])
        if not items:
            return "No pinned items"
        lines = []
        for item in items[:20]:
            msg = item.get("message", {})
            txt = msg.get("text", "")[:100]
            ts = msg.get("ts", "")
            lines.append(f"[{ts}] {txt}")
        return "\n".join(lines)

    else:
        return (
            f"[error] Unknown action '{action}'. "
            f"Available: sendMessage, editMessage, deleteMessage, readMessages, "
            f"react, reactions, pinMessage, unpinMessage, listPins"
        )
