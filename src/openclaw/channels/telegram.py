"""Telegram channel adapter using python-telegram-bot."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from openclaw.channels.base import ChannelAdapter, InboundMessage

logger = logging.getLogger(__name__)


class TelegramAdapter(ChannelAdapter):
    """Telegram bot adapter.

    Requires: pip install python-telegram-bot
    Config: channels.telegram.token, channels.telegram.allow_from
    """

    def __init__(
        self,
        token: str,
        allow_from: list[str] | None = None,
        dm_policy: str = "pairing",
        **kwargs: Any,
    ):
        super().__init__(name="telegram", **kwargs)
        self.token = token
        self.allow_from = allow_from or []
        self.dm_policy = dm_policy
        self._app = None
        self._paired: set[str] = set()

    async def start(self) -> None:
        """Start the Telegram bot."""
        try:
            from telegram import Update
            from telegram.ext import (
                Application,
                CommandHandler,
                MessageHandler as TGMessageHandler,
                filters,
            )
        except ImportError:
            logger.error("python-telegram-bot not installed. Run: pip install python-telegram-bot")
            return

        self._app = Application.builder().token(self.token).build()

        async def handle_message(update: Update, context: Any) -> None:
            if not update.message or not update.message.text:
                return

            sender_id = str(update.message.from_user.id)
            username = update.message.from_user.username or sender_id

            # Check allowlist
            if self.allow_from and username not in self.allow_from and sender_id not in self.allow_from:
                if self.dm_policy == "pairing" and sender_id not in self._paired:
                    await update.message.reply_text(
                        "🔒 Pairing required. Ask the admin to approve your access."
                    )
                    return

            # Create inbound message
            inbound = InboundMessage(
                content=update.message.text,
                sender_id=sender_id,
                channel="telegram",
                raw={"username": username, "chat_id": update.message.chat_id},
            )

            response = await self.handle_inbound(inbound)
            if response:
                await update.message.reply_text(response)

        self._app.add_handler(TGMessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling()
        logger.info("Telegram adapter started")

    async def stop(self) -> None:
        """Stop the Telegram bot."""
        if self._app:
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()

    async def send_message(self, recipient_id: str, content: str, **kwargs: Any) -> None:
        """Send a message to a Telegram chat."""
        if self._app:
            await self._app.bot.send_message(chat_id=int(recipient_id), text=content)
