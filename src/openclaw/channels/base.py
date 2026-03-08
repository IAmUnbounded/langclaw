"""Base channel adapter — abstract interface for messaging platforms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable


@dataclass
class InboundMessage:
    """A normalized inbound message from any channel."""

    content: str
    sender_id: str
    channel: str
    raw: dict[str, Any] = field(default_factory=dict)
    media: list[str] = field(default_factory=list)  # URLs or file paths
    reply_to: str | None = None
    group_id: str | None = None


MessageHandler = Callable[[InboundMessage], Awaitable[str]]


class ChannelAdapter(ABC):
    """Abstract base class for channel adapters.

    Each adapter:
    1. Connects to a messaging platform
    2. Receives messages and converts them to InboundMessage
    3. Sends responses back to the platform
    """

    def __init__(self, name: str, handler: MessageHandler | None = None):
        self.name = name
        self._handler = handler

    def set_handler(self, handler: MessageHandler) -> None:
        """Set the message handler callback."""
        self._handler = handler

    @abstractmethod
    async def start(self) -> None:
        """Start the adapter and begin listening for messages."""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Stop the adapter and clean up resources."""
        ...

    @abstractmethod
    async def send_message(self, recipient_id: str, content: str, **kwargs: Any) -> None:
        """Send a message to a specific recipient."""
        ...

    async def handle_inbound(self, message: InboundMessage) -> str | None:
        """Process an inbound message through the handler."""
        if self._handler:
            return await self._handler(message)
        return None
