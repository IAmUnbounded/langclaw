"""Discord action tools — Discord-specific operations.

Mirrors OpenClaw's Discord action tools:
- Message actions: send, edit, delete, react, pin, search
- Thread management: create, archive, rename
- Guild management: roles, channels, members
- Moderation: timeout, kick, ban
- Presence management
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Module-level Discord bot reference (set by channel adapter)
_discord_bot: Any = None


def set_discord_bot(bot: Any) -> None:
    """Set the Discord bot instance for action tools."""
    global _discord_bot
    _discord_bot = bot


def _get_bot() -> Any:
    """Get the Discord bot instance."""
    if _discord_bot is None:
        raise RuntimeError("Discord bot not initialized. Start the Discord channel first.")
    return _discord_bot


@tool
async def discord_actions_tool(
    action: str,
    channel_id: str = "",
    message_id: str = "",
    content: str = "",
    user_id: str = "",
    guild_id: str = "",
    role_id: str = "",
    emoji: str = "",
    thread_name: str = "",
    reason: str = "",
    duration_seconds: int = 0,
    params: Optional[Dict[str, Any]] = None,
) -> str:
    """Perform Discord-specific operations.

    Actions:
    **Messaging:**
    - send: Send a message to a channel (requires channel_id, content)
    - edit: Edit a message (requires channel_id, message_id, content)
    - delete: Delete a message (requires channel_id, message_id)
    - react: Add a reaction (requires channel_id, message_id, emoji)
    - unreact: Remove a reaction (requires channel_id, message_id, emoji)
    - pin: Pin a message (requires channel_id, message_id)
    - unpin: Unpin a message (requires channel_id, message_id)
    - fetch: Fetch recent messages (requires channel_id, optional params.limit)
    - search: Search messages (requires channel_id, content as query)

    **Threads:**
    - thread_create: Create a thread (requires channel_id, thread_name, optional message_id)
    - thread_archive: Archive a thread (requires channel_id as thread_id)

    **Guild/Server:**
    - add_role: Add a role to a member (requires guild_id, user_id, role_id)
    - remove_role: Remove a role (requires guild_id, user_id, role_id)
    - list_members: List guild members (requires guild_id)
    - list_roles: List guild roles (requires guild_id)
    - list_channels: List guild channels (requires guild_id)

    **Moderation:**
    - timeout: Timeout a member (requires guild_id, user_id, duration_seconds)
    - kick: Kick a member (requires guild_id, user_id, optional reason)
    - ban: Ban a member (requires guild_id, user_id, optional reason)

    **Status:**
    - set_presence: Set bot presence/status (requires content as status text)
    """
    try:
        bot = _get_bot()
    except RuntimeError:
        return _simulate_action(action, channel_id, content, message_id, emoji, user_id, guild_id, reason)

    params = params or {}

    try:
        if action == "send":
            channel = bot.get_channel(int(channel_id))
            if not channel:
                return f"Channel {channel_id} not found"
            msg = await channel.send(content)
            return json.dumps({"status": "sent", "message_id": str(msg.id)})

        elif action == "edit":
            channel = bot.get_channel(int(channel_id))
            if not channel:
                return f"Channel {channel_id} not found"
            msg = await channel.fetch_message(int(message_id))
            await msg.edit(content=content)
            return json.dumps({"status": "edited", "message_id": message_id})

        elif action == "delete":
            channel = bot.get_channel(int(channel_id))
            if not channel:
                return f"Channel {channel_id} not found"
            msg = await channel.fetch_message(int(message_id))
            await msg.delete()
            return json.dumps({"status": "deleted", "message_id": message_id})

        elif action == "react":
            channel = bot.get_channel(int(channel_id))
            if not channel:
                return f"Channel {channel_id} not found"
            msg = await channel.fetch_message(int(message_id))
            await msg.add_reaction(emoji)
            return json.dumps({"status": "reacted", "emoji": emoji})

        elif action == "unreact":
            channel = bot.get_channel(int(channel_id))
            if not channel:
                return f"Channel {channel_id} not found"
            msg = await channel.fetch_message(int(message_id))
            await msg.remove_reaction(emoji, bot.user)
            return json.dumps({"status": "unreacted", "emoji": emoji})

        elif action == "pin":
            channel = bot.get_channel(int(channel_id))
            if not channel:
                return f"Channel {channel_id} not found"
            msg = await channel.fetch_message(int(message_id))
            await msg.pin()
            return json.dumps({"status": "pinned"})

        elif action == "unpin":
            channel = bot.get_channel(int(channel_id))
            if not channel:
                return f"Channel {channel_id} not found"
            msg = await channel.fetch_message(int(message_id))
            await msg.unpin()
            return json.dumps({"status": "unpinned"})

        elif action == "fetch":
            channel = bot.get_channel(int(channel_id))
            if not channel:
                return f"Channel {channel_id} not found"
            limit = params.get("limit", 10)
            messages = []
            async for msg in channel.history(limit=limit):
                messages.append({
                    "id": str(msg.id),
                    "author": str(msg.author),
                    "content": msg.content[:500],
                    "timestamp": msg.created_at.isoformat(),
                })
            return json.dumps({"messages": messages})

        elif action == "search":
            return json.dumps({
                "note": "Discord does not have a native search API for bots. "
                        "Use fetch with filtering instead.",
            })

        elif action == "thread_create":
            channel = bot.get_channel(int(channel_id))
            if not channel:
                return f"Channel {channel_id} not found"
            if message_id:
                msg = await channel.fetch_message(int(message_id))
                thread = await msg.create_thread(name=thread_name)
            else:
                thread = await channel.create_thread(
                    name=thread_name,
                    type=getattr(bot, "ChannelType", type("", (), {"public_thread": 11})).public_thread,
                )
            return json.dumps({"status": "thread_created", "thread_id": str(thread.id)})

        elif action == "thread_archive":
            thread = bot.get_channel(int(channel_id))
            if thread:
                await thread.edit(archived=True)
            return json.dumps({"status": "thread_archived"})

        elif action == "add_role":
            guild = bot.get_guild(int(guild_id))
            if not guild:
                return f"Guild {guild_id} not found"
            member = guild.get_member(int(user_id))
            role = guild.get_role(int(role_id))
            if member and role:
                await member.add_roles(role, reason=reason)
            return json.dumps({"status": "role_added"})

        elif action == "remove_role":
            guild = bot.get_guild(int(guild_id))
            if not guild:
                return f"Guild {guild_id} not found"
            member = guild.get_member(int(user_id))
            role = guild.get_role(int(role_id))
            if member and role:
                await member.remove_roles(role, reason=reason)
            return json.dumps({"status": "role_removed"})

        elif action == "list_members":
            guild = bot.get_guild(int(guild_id))
            if not guild:
                return f"Guild {guild_id} not found"
            members = [
                {"id": str(m.id), "name": str(m), "roles": [r.name for r in m.roles[1:]]}
                for m in guild.members[:100]
            ]
            return json.dumps({"members": members})

        elif action == "list_roles":
            guild = bot.get_guild(int(guild_id))
            if not guild:
                return f"Guild {guild_id} not found"
            roles = [
                {"id": str(r.id), "name": r.name, "position": r.position, "members": len(r.members)}
                for r in guild.roles
            ]
            return json.dumps({"roles": roles})

        elif action == "list_channels":
            guild = bot.get_guild(int(guild_id))
            if not guild:
                return f"Guild {guild_id} not found"
            channels = [
                {"id": str(c.id), "name": c.name, "type": str(c.type)}
                for c in guild.channels
            ]
            return json.dumps({"channels": channels})

        elif action == "timeout":
            import datetime
            guild = bot.get_guild(int(guild_id))
            if not guild:
                return f"Guild {guild_id} not found"
            member = guild.get_member(int(user_id))
            if member:
                until = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=duration_seconds)
                await member.timeout(until, reason=reason)
            return json.dumps({"status": "timed_out", "duration": duration_seconds})

        elif action == "kick":
            guild = bot.get_guild(int(guild_id))
            if not guild:
                return f"Guild {guild_id} not found"
            member = guild.get_member(int(user_id))
            if member:
                await member.kick(reason=reason)
            return json.dumps({"status": "kicked"})

        elif action == "ban":
            guild = bot.get_guild(int(guild_id))
            if not guild:
                return f"Guild {guild_id} not found"
            member = guild.get_member(int(user_id))
            if member:
                await member.ban(reason=reason)
            return json.dumps({"status": "banned"})

        elif action == "set_presence":
            import discord
            await bot.change_presence(activity=discord.Game(name=content))
            return json.dumps({"status": "presence_set", "activity": content})

        else:
            return f"Unknown Discord action: {action}"

    except Exception as e:
        logger.exception("Discord action '%s' failed", action)
        return json.dumps({"error": str(e), "action": action})


def _simulate_action(
    action: str, channel_id: str, content: str,
    message_id: str, emoji: str, user_id: str,
    guild_id: str, reason: str,
) -> str:
    """Return a simulated response when the Discord bot is not connected."""
    return json.dumps({
        "status": "simulated",
        "action": action,
        "note": "Discord bot is not connected. Action was not executed.",
        "params": {
            "channel_id": channel_id,
            "content": content[:100] if content else "",
            "message_id": message_id,
            "emoji": emoji,
            "user_id": user_id,
            "guild_id": guild_id,
        },
    })
