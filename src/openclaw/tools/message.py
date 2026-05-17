"""Cross-channel message delivery tool.

Mirrors OpenClaw's `message` tool — proactively send messages to any channel.

Architecture:
    Agent → message_tool → MessageBus → Channel Handler → Delivery

The MessageBus is a singleton that channel adapters register with.
When the agent calls message_tool, the bus routes to the correct handler.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Coroutine

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


class MessageBus:
    """Central message bus for cross-channel delivery.

    Channel adapters register themselves as handlers. When the agent
    sends a message, the bus routes it to the appropriate handler.
    """

    _instance: MessageBus | None = None

    def __init__(self) -> None:
        self._handlers: dict[str, Callable[..., Coroutine]] = {}
        self._outbox: list[dict[str, Any]] = []
        self._delivery_log: list[dict[str, Any]] = []

    @classmethod
    def get_instance(cls) -> "MessageBus":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register_handler(self, channel: str, handler: Callable[..., Coroutine]) -> None:
        """Register a channel handler.

        handler signature: async (recipient_id: str, content: str, **kwargs) -> dict
        """
        self._handlers[channel] = handler
        logger.info(f"MessageBus: registered handler for '{channel}'")

    def unregister_handler(self, channel: str) -> None:
        self._handlers.pop(channel, None)

    async def send(
        self,
        channel: str,
        recipient_id: str,
        content: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Send a message to a channel."""
        result: dict[str, Any] = {
            "channel": channel,
            "recipient_id": recipient_id,
            "timestamp": time.time(),
            "success": False,
        }

        handler = self._handlers.get(channel)
        if not handler:
            result["error"] = f"No handler registered for channel '{channel}'"
            # Queue for later delivery
            self._outbox.append({
                "channel": channel,
                "recipient_id": recipient_id,
                "content": content,
                "kwargs": kwargs,
                "queued_at": time.time(),
            })
            result["queued"] = True
            self._delivery_log.append(result)
            return result

        try:
            await handler(recipient_id, content, **kwargs)
            result["success"] = True
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"MessageBus delivery failed ({channel}): {e}")

        self._delivery_log.append(result)
        # Keep log bounded
        if len(self._delivery_log) > 100:
            self._delivery_log = self._delivery_log[-100:]

        return result

    async def broadcast(
        self,
        content: str,
        recipient_id: str = "",
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Send to all registered channels."""
        results = []
        for channel in list(self._handlers):
            r = await self.send(channel, recipient_id, content, **kwargs)
            results.append(r)
        return results

    def get_available_channels(self) -> list[str]:
        return list(self._handlers.keys())

    def get_delivery_log(self, count: int = 10) -> list[dict[str, Any]]:
        return self._delivery_log[-count:]

    def drain_outbox(self) -> list[dict[str, Any]]:
        """Return and clear queued messages."""
        queued = list(self._outbox)
        self._outbox.clear()
        return queued


def _run_async(coro: Coroutine) -> Any:
    """Bridge sync tool to async bus."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result(timeout=30)
    else:
        return asyncio.run(coro)


@tool
def message_tool(
    text: str,
    channel: str = "",
    recipient_id: str = "",
    image: str = "",
    reaction: str = "",
    reply_to: str = "",
    thread_id: str = "",
) -> str:
    """Send a message to a specific channel and recipient.

    Use this tool to proactively send messages to users on any connected channel.
    This is different from your regular response — use this for:
    - Sending updates to a different channel than the current conversation
    - Proactive notifications (e.g., from cron jobs)
    - Cross-channel messaging (e.g., forward info from webchat to Telegram)

    Args:
        text: The message text to send.
        channel: Target channel (webchat, telegram, discord, whatsapp). Empty = broadcast.
        recipient_id: Recipient identifier. Empty = default recipient.
        image: Optional image path or URL to attach.
        reaction: Optional emoji reaction to add.
        reply_to: Optional message ID to reply to.
        thread_id: Optional thread ID for threading.

    Returns:
        Delivery status message.
    """
    if not text.strip():
        return "[error] Message text cannot be empty."

    bus = MessageBus.get_instance()
    kwargs: dict[str, Any] = {}
    if image:
        kwargs["image"] = image
    if reaction:
        kwargs["reaction"] = reaction
    if reply_to:
        kwargs["reply_to"] = reply_to
    if thread_id:
        kwargs["thread_id"] = thread_id

    try:
        if channel:
            result = _run_async(
                bus.send(channel, recipient_id, text, **kwargs)
            )
            if result.get("success"):
                return f"Message delivered to {channel}" + (f":{recipient_id}" if recipient_id else "")
            elif result.get("queued"):
                return (
                    f"No handler for '{channel}' — message queued for later delivery. "
                    f"Available channels: {', '.join(bus.get_available_channels()) or 'none'}"
                )
            else:
                return f"[error] Delivery failed: {result.get('error', 'unknown')}"
        else:
            # Broadcast to all channels
            available = bus.get_available_channels()
            if not available:
                return (
                    "No channels available for broadcast. "
                    "Connect a channel first (webchat, telegram, discord, whatsapp)."
                )
            results = _run_async(
                bus.broadcast(text, recipient_id, **kwargs)
            )
            successes = sum(1 for r in results if r.get("success"))
            return (
                f"Broadcast complete: {successes}/{len(results)} channels delivered.\n"
                f"Channels: {', '.join(available)}"
            )

    except Exception as e:
        return f"[error] Message delivery failed: {e}"
