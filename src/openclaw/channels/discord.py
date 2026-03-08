"""Discord channel adapter using discord.py."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from openclaw.channels.base import ChannelAdapter, InboundMessage

logger = logging.getLogger(__name__)


class DiscordAdapter(ChannelAdapter):
    """Discord bot adapter.

    Requires: pip install discord.py
    Config: channels.discord.token, channels.discord.allow_from
    """

    def __init__(
        self,
        token: str,
        allow_from: list[str] | None = None,
        dm_policy: str = "pairing",
        **kwargs: Any,
    ):
        super().__init__(name="discord", **kwargs)
        self.token = token
        self.allow_from = allow_from or []
        self.dm_policy = dm_policy
        self._client = None

    async def start(self) -> None:
        """Start the Discord bot."""
        try:
            import discord
        except ImportError:
            logger.error("discord.py not installed. Run: pip install discord.py")
            return

        intents = discord.Intents.default()
        intents.message_content = True
        self._client = discord.Client(intents=intents)

        adapter = self

        @self._client.event
        async def on_ready():
            logger.info(f"Discord adapter logged in as {self._client.user}")

        @self._client.event
        async def on_message(message: discord.Message):
            # Ignore own messages
            if message.author == self._client.user:
                return

            sender_id = str(message.author.id)
            username = str(message.author)

            # Check allowlist
            if adapter.allow_from and username not in adapter.allow_from and sender_id not in adapter.allow_from:
                return

            # Check if it's a DM or a channel mention
            is_dm = isinstance(message.channel, discord.DMChannel)
            is_mentioned = self._client.user in message.mentions if not is_dm else False

            # Only respond to DMs or mentions
            if not is_dm and not is_mentioned:
                return

            content = message.content
            # Remove bot mention from content
            if is_mentioned and self._client.user:
                content = content.replace(f"<@{self._client.user.id}>", "").strip()

            inbound = InboundMessage(
                content=content,
                sender_id=sender_id,
                channel="discord",
                raw={"username": username, "channel_id": message.channel.id},
                group_id=str(message.guild.id) if message.guild else None,
            )

            response = await adapter.handle_inbound(inbound)
            if response:
                # Split long messages for Discord's 2000 char limit
                chunks = [response[i:i+1900] for i in range(0, len(response), 1900)]
                for chunk in chunks:
                    await message.channel.send(chunk)

        # Start in background
        asyncio.create_task(self._client.start(self.token))
        logger.info("Discord adapter starting...")

    async def stop(self) -> None:
        """Stop the Discord bot."""
        if self._client:
            await self._client.close()

    async def send_message(self, recipient_id: str, content: str, **kwargs: Any) -> None:
        """Send a message to a Discord channel."""
        if self._client:
            channel = self._client.get_channel(int(recipient_id))
            if channel:
                await channel.send(content)
