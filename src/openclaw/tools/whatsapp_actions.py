"""WhatsApp action tools — WhatsApp-specific operations.

Mirrors OpenClaw's WhatsApp action tools:
- Message actions: send, reply, forward
- Reaction management: react, unreact
- Group management: info, participants
- Status: connection status, linked phone
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Module-level WhatsApp adapter reference (set by gateway)
_whatsapp_adapter: Any = None


def set_whatsapp_adapter(adapter: Any) -> None:
    """Set the WhatsApp adapter instance for action tools."""
    global _whatsapp_adapter
    _whatsapp_adapter = adapter


def _get_adapter() -> Any:
    """Get the WhatsApp adapter instance."""
    if _whatsapp_adapter is None:
        raise RuntimeError("WhatsApp adapter not initialized.")
    return _whatsapp_adapter


@tool
async def whatsapp_actions_tool(
    action: str,
    jid: str = "",
    message_id: str = "",
    content: str = "",
    emoji: str = "",
    media_path: str = "",
    params: Optional[Dict[str, Any]] = None,
) -> str:
    """Perform WhatsApp-specific operations.

    Actions:
    **Messaging:**
    - send: Send a text message (requires jid, content)
    - reply: Reply to a message (requires jid, message_id, content)
    - forward: Forward a message (requires jid, message_id, params.to_jid)
    - send_media: Send a media file (requires jid, media_path)

    **Reactions:**
    - react: React to a message (requires jid, message_id, emoji)
    - unreact: Remove reaction (requires jid, message_id)

    **Group:**
    - group_info: Get group info (requires jid — must be a group JID)
    - group_participants: List group participants (requires jid)

    **Status:**
    - connection_status: Check WhatsApp connection status
    - linked_phone: Get linked phone number

    Args:
        action: The action to perform.
        jid: WhatsApp JID (phone@s.whatsapp.net or group@g.us).
        message_id: Message ID for reply/react.
        content: Message text content.
        emoji: Reaction emoji.
        media_path: Path to media file for send_media.
        params: Additional parameters.
    """
    try:
        adapter = _get_adapter()
    except RuntimeError:
        return _simulate_action(action, jid, content, message_id, emoji)

    params = params or {}

    try:
        if action == "send":
            if not jid or not content:
                return json.dumps({"error": "jid and content are required"})
            await adapter.send_message(jid, content)
            return json.dumps({"status": "sent", "jid": jid})

        elif action == "reply":
            if not jid or not content:
                return json.dumps({"error": "jid, message_id, and content are required"})
            await adapter.send_message(jid, content, reply_to=message_id)
            return json.dumps({"status": "replied", "jid": jid})

        elif action == "forward":
            to_jid = params.get("to_jid", "")
            if not to_jid:
                return json.dumps({"error": "to_jid required in params"})
            # Forward by sending the content to another chat
            await adapter.send_message(to_jid, content or f"[Forwarded from {jid}]")
            return json.dumps({"status": "forwarded", "to_jid": to_jid})

        elif action == "send_media":
            if not jid or not media_path:
                return json.dumps({"error": "jid and media_path are required"})
            if hasattr(adapter, "send_media"):
                await adapter.send_media(jid, media_path, caption=content)
                return json.dumps({"status": "media_sent", "jid": jid})
            return json.dumps({"error": "Media sending not supported by this adapter"})

        elif action == "react":
            if not jid or not message_id or not emoji:
                return json.dumps({"error": "jid, message_id, and emoji are required"})
            if hasattr(adapter, "react"):
                await adapter.react(jid, message_id, emoji)
                return json.dumps({"status": "reacted", "emoji": emoji})
            return json.dumps({"error": "Reactions not supported by this adapter"})

        elif action == "unreact":
            if not jid or not message_id:
                return json.dumps({"error": "jid and message_id are required"})
            if hasattr(adapter, "react"):
                await adapter.react(jid, message_id, "")  # Empty emoji = remove reaction
                return json.dumps({"status": "unreacted"})
            return json.dumps({"error": "Reactions not supported by this adapter"})

        elif action == "group_info":
            if hasattr(adapter, "get_group_info"):
                info = await adapter.get_group_info(jid)
                return json.dumps(info)
            return json.dumps({
                "jid": jid,
                "note": "Group info fetching not fully supported",
            })

        elif action == "group_participants":
            if hasattr(adapter, "get_group_participants"):
                participants = await adapter.get_group_participants(jid)
                return json.dumps({"participants": participants})
            return json.dumps({
                "jid": jid,
                "note": "Participant listing not fully supported",
            })

        elif action == "connection_status":
            return json.dumps({
                "connected": adapter.is_connected if hasattr(adapter, "is_connected") else False,
                "phone": getattr(adapter, "phone_number", None),
            })

        elif action == "linked_phone":
            return json.dumps({
                "phone": getattr(adapter, "phone_number", None),
                "connected": getattr(adapter, "is_connected", False),
            })

        else:
            return f"Unknown WhatsApp action: {action}"

    except Exception as e:
        logger.exception("WhatsApp action '%s' failed", action)
        return json.dumps({"error": str(e), "action": action})


def _simulate_action(
    action: str, jid: str, content: str,
    message_id: str, emoji: str,
) -> str:
    """Return a simulated response when WhatsApp is not connected."""
    return json.dumps({
        "status": "simulated",
        "action": action,
        "note": "WhatsApp is not connected. Action was not executed.",
        "params": {
            "jid": jid,
            "content": content[:100] if content else "",
            "message_id": message_id,
            "emoji": emoji,
        },
    })
