"""Telegram action tools — Telegram-specific operations.

Mirrors OpenClaw's Telegram action tools:
- Message actions: send, edit, delete, forward, reply
- Reaction management: react, get reactions
- Sticker support
- Inline keyboard/buttons
- Chat management
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Module-level Telegram bot reference (set by channel adapter)
_telegram_bot: Any = None


def set_telegram_bot(bot: Any) -> None:
    """Set the Telegram bot instance for action tools."""
    global _telegram_bot
    _telegram_bot = bot


def _get_bot() -> Any:
    """Get the Telegram bot instance."""
    if _telegram_bot is None:
        raise RuntimeError("Telegram bot not initialized. Start the Telegram channel first.")
    return _telegram_bot


@tool
async def telegram_actions_tool(
    action: str,
    chat_id: str = "",
    message_id: str = "",
    content: str = "",
    emoji: str = "",
    sticker_id: str = "",
    parse_mode: str = "Markdown",
    reply_to_message_id: str = "",
    buttons: Optional[List[Dict[str, str]]] = None,
    params: Optional[Dict[str, Any]] = None,
) -> str:
    """Perform Telegram-specific operations.

    Actions:
    **Messaging:**
    - send: Send a message (requires chat_id, content)
    - edit: Edit a message (requires chat_id, message_id, content)
    - delete: Delete a message (requires chat_id, message_id)
    - forward: Forward a message (requires chat_id, message_id, params.to_chat_id)
    - reply: Reply to a message (requires chat_id, reply_to_message_id, content)

    **Reactions:**
    - react: React to a message (requires chat_id, message_id, emoji)
    - get_reactions: Get reactions on a message (requires chat_id, message_id)

    **Stickers:**
    - send_sticker: Send a sticker (requires chat_id, sticker_id)

    **Keyboard/Buttons:**
    - send_buttons: Send message with inline buttons
      (requires chat_id, content, buttons: [{"text": "...", "callback_data": "..."}])

    **Chat:**
    - get_chat: Get chat info (requires chat_id)
    - get_members_count: Get member count (requires chat_id)
    - pin: Pin a message (requires chat_id, message_id)
    - unpin: Unpin a message (requires chat_id, message_id)

    Args:
        action: The action to perform.
        chat_id: Telegram chat ID.
        message_id: Message ID for edit/delete/react.
        content: Message text content.
        emoji: Reaction emoji.
        sticker_id: Sticker file_id.
        parse_mode: Message parse mode (Markdown, HTML, MarkdownV2).
        reply_to_message_id: Message to reply to.
        buttons: Inline keyboard buttons.
        params: Additional parameters.
    """
    try:
        bot = _get_bot()
    except RuntimeError:
        return _simulate_action(action, chat_id, content, message_id, emoji)

    params = params or {}

    try:
        if action == "send":
            msg = await bot.send_message(
                chat_id=int(chat_id),
                text=content,
                parse_mode=parse_mode,
            )
            return json.dumps({"status": "sent", "message_id": msg.message_id})

        elif action == "edit":
            await bot.edit_message_text(
                chat_id=int(chat_id),
                message_id=int(message_id),
                text=content,
                parse_mode=parse_mode,
            )
            return json.dumps({"status": "edited", "message_id": message_id})

        elif action == "delete":
            await bot.delete_message(
                chat_id=int(chat_id),
                message_id=int(message_id),
            )
            return json.dumps({"status": "deleted", "message_id": message_id})

        elif action == "forward":
            to_chat = params.get("to_chat_id", "")
            if not to_chat:
                return json.dumps({"error": "to_chat_id required in params"})
            msg = await bot.forward_message(
                chat_id=int(to_chat),
                from_chat_id=int(chat_id),
                message_id=int(message_id),
            )
            return json.dumps({"status": "forwarded", "message_id": msg.message_id})

        elif action == "reply":
            msg = await bot.send_message(
                chat_id=int(chat_id),
                text=content,
                reply_to_message_id=int(reply_to_message_id) if reply_to_message_id else None,
                parse_mode=parse_mode,
            )
            return json.dumps({"status": "replied", "message_id": msg.message_id})

        elif action == "react":
            from telegram import ReactionTypeEmoji
            await bot.set_message_reaction(
                chat_id=int(chat_id),
                message_id=int(message_id),
                reaction=[ReactionTypeEmoji(emoji=emoji)],
            )
            return json.dumps({"status": "reacted", "emoji": emoji})

        elif action == "get_reactions":
            return json.dumps({
                "note": "Telegram Bot API does not support fetching reactions directly. "
                        "Reactions are received via updates.",
            })

        elif action == "send_sticker":
            msg = await bot.send_sticker(
                chat_id=int(chat_id),
                sticker=sticker_id,
            )
            return json.dumps({"status": "sticker_sent", "message_id": msg.message_id})

        elif action == "send_buttons":
            from telegram import InlineKeyboardButton, InlineKeyboardMarkup
            keyboard = []
            for btn in (buttons or []):
                keyboard.append([InlineKeyboardButton(
                    text=btn.get("text", ""),
                    callback_data=btn.get("callback_data", ""),
                    url=btn.get("url", None),
                )])
            reply_markup = InlineKeyboardMarkup(keyboard)
            msg = await bot.send_message(
                chat_id=int(chat_id),
                text=content,
                reply_markup=reply_markup,
                parse_mode=parse_mode,
            )
            return json.dumps({"status": "sent_with_buttons", "message_id": msg.message_id})

        elif action == "get_chat":
            chat = await bot.get_chat(int(chat_id))
            return json.dumps({
                "id": chat.id,
                "type": chat.type,
                "title": chat.title or "",
                "username": chat.username or "",
                "description": chat.description or "",
            })

        elif action == "get_members_count":
            count = await bot.get_chat_member_count(int(chat_id))
            return json.dumps({"chat_id": chat_id, "members_count": count})

        elif action == "pin":
            await bot.pin_chat_message(
                chat_id=int(chat_id),
                message_id=int(message_id),
            )
            return json.dumps({"status": "pinned"})

        elif action == "unpin":
            await bot.unpin_chat_message(
                chat_id=int(chat_id),
                message_id=int(message_id),
            )
            return json.dumps({"status": "unpinned"})

        else:
            return f"Unknown Telegram action: {action}"

    except Exception as e:
        logger.exception("Telegram action '%s' failed", action)
        return json.dumps({"error": str(e), "action": action})


def _simulate_action(
    action: str, chat_id: str, content: str,
    message_id: str, emoji: str,
) -> str:
    """Return a simulated response when the Telegram bot is not connected."""
    return json.dumps({
        "status": "simulated",
        "action": action,
        "note": "Telegram bot is not connected. Action was not executed.",
        "params": {
            "chat_id": chat_id,
            "content": content[:100] if content else "",
            "message_id": message_id,
            "emoji": emoji,
        },
    })
