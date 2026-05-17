"""Slack channel adapter using slack_bolt.

Mirrors openclaw's Slack integration:
- Bot and user token support
- App mention and DM handling
- Threading (reply_to_mode: off, first, all)
- Slash command support
- Block Kit message formatting
- File/media upload support
- Emoji reactions

Requires: pip install slack-bolt
Config: channels.slack.bot_token, channels.slack.app_token, channels.slack.signing_secret
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from openclaw.channels.base import ChannelAdapter, InboundMessage

logger = logging.getLogger(__name__)


class SlackAdapter(ChannelAdapter):
    """Slack bot adapter using Bolt framework.

    Supports two modes:
    1. Socket Mode (recommended): Uses app-level token for WebSocket connection
    2. HTTP Mode: Receives events via webhook (needs public URL)

    Config keys:
    - bot_token: xoxb-... Bot User OAuth Token
    - app_token: xapp-... App-Level Token (for Socket Mode)
    - signing_secret: Slack signing secret (for HTTP mode)
    - allow_from: List of user IDs to accept messages from
    - reply_to_mode: "off" | "first" | "all" (threading behavior)
    """

    def __init__(
        self,
        bot_token: str,
        app_token: str = "",
        signing_secret: str = "",
        allow_from: list[str] | None = None,
        reply_to_mode: str = "first",
        **kwargs: Any,
    ):
        super().__init__(name="slack", **kwargs)
        self.bot_token = bot_token
        self.app_token = app_token
        self.signing_secret = signing_secret
        self.allow_from = allow_from or []
        self.reply_to_mode = reply_to_mode
        self._app = None
        self._handler = None
        self._bot_user_id: str = ""
        self._thread_ts_map: dict[str, str] = {}  # channel:ts → first reply thread_ts

    async def start(self) -> None:
        """Start the Slack bot."""
        try:
            from slack_bolt.async_app import AsyncApp
            from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
        except ImportError:
            logger.error("slack-bolt not installed. Run: pip install slack-bolt")
            return

        self._app = AsyncApp(
            token=self.bot_token,
            signing_secret=self.signing_secret or None,
        )

        # Get bot user ID
        try:
            auth = await self._app.client.auth_test()
            self._bot_user_id = auth.get("user_id", "")
            logger.info(f"Slack bot user: {self._bot_user_id}")
        except Exception as e:
            logger.warning(f"Could not get bot user ID: {e}")

        adapter = self

        # Handle app mentions
        @self._app.event("app_mention")
        async def handle_mention(event: dict, say: Any):
            await adapter._process_event(event, say)

        # Handle direct messages
        @self._app.event("message")
        async def handle_message(event: dict, say: Any):
            # Skip bot messages and message_changed events
            if event.get("bot_id") or event.get("subtype"):
                return
            # Only process DMs (channel type "im")
            channel_type = event.get("channel_type", "")
            if channel_type == "im":
                await adapter._process_event(event, say)

        # Handle slash commands (generic)
        @self._app.command(re.compile(r".*"))
        async def handle_command(ack: Any, command: dict, say: Any):
            await ack()
            text = command.get("text", "")
            cmd = command.get("command", "")
            user_id = command.get("user_id", "")
            channel_id = command.get("channel_id", "")

            if adapter.allow_from and user_id not in adapter.allow_from:
                return

            inbound = InboundMessage(
                content=f"{cmd} {text}".strip(),
                sender_id=user_id,
                channel="slack",
                raw={
                    "command": cmd,
                    "channel_id": channel_id,
                    "response_url": command.get("response_url", ""),
                },
            )
            response = await adapter.handle_inbound(inbound)
            if response:
                await say(text=response, channel=channel_id)

        # Start Socket Mode or just register handlers for HTTP mode
        if self.app_token:
            self._socket_handler = AsyncSocketModeHandler(self._app, self.app_token)
            asyncio.create_task(self._socket_handler.start_async())
            logger.info("Slack adapter started (Socket Mode)")
        else:
            logger.info("Slack adapter ready (HTTP mode — mount app at /slack/events)")

    async def _process_event(self, event: dict, say: Any) -> None:
        """Process a Slack event (mention or DM)."""
        user_id = event.get("user", "")
        text = event.get("text", "")
        channel_id = event.get("channel", "")
        thread_ts = event.get("thread_ts", "")
        message_ts = event.get("ts", "")

        # Check allowlist
        if self.allow_from and user_id not in self.allow_from:
            return

        # Skip own messages
        if user_id == self._bot_user_id:
            return

        # Remove bot mention from text
        if self._bot_user_id:
            text = text.replace(f"<@{self._bot_user_id}>", "").strip()

        is_dm = event.get("channel_type") == "im"
        is_thread = bool(thread_ts)

        inbound = InboundMessage(
            content=text,
            sender_id=user_id,
            channel="slack",
            raw={
                "channel_id": channel_id,
                "thread_ts": thread_ts,
                "message_ts": message_ts,
                "is_dm": is_dm,
                "is_thread": is_thread,
            },
            group_id=channel_id if not is_dm else None,
        )

        response = await self.handle_inbound(inbound)
        if response:
            # Determine threading behavior
            reply_thread_ts = self._resolve_thread_ts(
                channel_id, message_ts, thread_ts
            )

            # Send response (split if > 3000 chars for readability)
            chunks = _split_message(response, max_length=3000)
            for chunk in chunks:
                kwargs: dict[str, Any] = {"text": chunk, "channel": channel_id}
                if reply_thread_ts:
                    kwargs["thread_ts"] = reply_thread_ts
                await say(**kwargs)

    def _resolve_thread_ts(
        self, channel_id: str, message_ts: str, thread_ts: str
    ) -> str | None:
        """Resolve the thread_ts for replying based on reply_to_mode."""
        if self.reply_to_mode == "off":
            return None

        # If already in a thread, always reply in that thread
        if thread_ts:
            return thread_ts

        if self.reply_to_mode == "all":
            # Always create/continue a thread on the original message
            return message_ts

        if self.reply_to_mode == "first":
            # Reply in thread to first message, then continue in same thread
            key = f"{channel_id}:{message_ts}"
            if key not in self._thread_ts_map:
                self._thread_ts_map[key] = message_ts
            return self._thread_ts_map[key]

        return None

    async def stop(self) -> None:
        """Stop the Slack bot."""
        if hasattr(self, "_socket_handler") and self._socket_handler:
            await self._socket_handler.close_async()
        logger.info("Slack adapter stopped")

    async def send_message(
        self, recipient_id: str, content: str, **kwargs: Any
    ) -> None:
        """Send a message to a Slack channel or user.

        Args:
            recipient_id: Channel ID or user ID.
            content: Message text.
            **kwargs: Optional thread_ts, blocks, etc.
        """
        if not self._app:
            logger.warning("Slack adapter not started")
            return

        msg_kwargs: dict[str, Any] = {
            "channel": recipient_id,
            "text": content,
        }
        if "thread_ts" in kwargs:
            msg_kwargs["thread_ts"] = kwargs["thread_ts"]
        if "blocks" in kwargs:
            msg_kwargs["blocks"] = kwargs["blocks"]

        try:
            await self._app.client.chat_postMessage(**msg_kwargs)
        except Exception as e:
            logger.error(f"Failed to send Slack message: {e}")

    async def add_reaction(
        self, channel: str, timestamp: str, emoji: str
    ) -> bool:
        """Add an emoji reaction to a message."""
        if not self._app:
            return False
        try:
            await self._app.client.reactions_add(
                channel=channel, timestamp=timestamp, name=emoji
            )
            return True
        except Exception as e:
            logger.debug(f"Failed to add reaction: {e}")
            return False

    async def remove_reaction(
        self, channel: str, timestamp: str, emoji: str
    ) -> bool:
        """Remove an emoji reaction from a message."""
        if not self._app:
            return False
        try:
            await self._app.client.reactions_remove(
                channel=channel, timestamp=timestamp, name=emoji
            )
            return True
        except Exception as e:
            logger.debug(f"Failed to remove reaction: {e}")
            return False

    async def upload_file(
        self, channel: str, file_path: str, title: str = "", thread_ts: str = ""
    ) -> bool:
        """Upload a file to a Slack channel."""
        if not self._app:
            return False
        try:
            kwargs: dict[str, Any] = {
                "channels": channel,
                "file": file_path,
            }
            if title:
                kwargs["title"] = title
            if thread_ts:
                kwargs["thread_ts"] = thread_ts
            await self._app.client.files_upload_v2(**kwargs)
            return True
        except Exception as e:
            logger.error(f"Failed to upload file: {e}")
            return False


def _split_message(text: str, max_length: int = 3000) -> list[str]:
    """Split a long message into chunks, preferring line boundaries."""
    if len(text) <= max_length:
        return [text]

    chunks = []
    while text:
        if len(text) <= max_length:
            chunks.append(text)
            break

        # Find a good split point (newline near the limit)
        split_at = text.rfind("\n", 0, max_length)
        if split_at < max_length // 2:
            split_at = max_length

        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")

    return chunks


# Need to import re for the slash command regex
import re
