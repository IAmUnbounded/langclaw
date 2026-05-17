"""WhatsApp channel adapter using neonize (whatsmeow Python bindings).

Provides QR-code-based WhatsApp Web linking, similar to OpenClaw's
Baileys-based adapter. The gateway exposes a /whatsapp/link page
where users scan a QR code to pair their WhatsApp account.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Callable

from openclaw.channels.base import ChannelAdapter, InboundMessage

logger = logging.getLogger(__name__)

# Session store directory
WA_SESSION_DIR = Path.home() / ".langclaw" / "whatsapp"

# neonize event codes (from neonize.events.INT_TO_EVENT)
_EV_PAIR_STATUS = 2
_EV_CONNECTED = 3
_EV_LOGGED_OUT = 6
_EV_MESSAGE = 17


class WhatsAppAdapter(ChannelAdapter):
    """WhatsApp adapter with QR code linking.

    Requires: pip install neonize
    Config: channels.whatsapp.enabled, channels.whatsapp.session_name
    """

    def __init__(
        self,
        session_name: str = "openclaw-wa",
        allow_from: list[str] | None = None,
        dm_policy: str = "open",
        group_policy: str = "mention",
        group_history_count: int = 50,
        max_message_length: int = 4000,
        **kwargs: Any,
    ):
        super().__init__(name="whatsapp", **kwargs)
        self.session_name = session_name
        self.allow_from = allow_from or []
        self.dm_policy = dm_policy
        self.group_policy = group_policy
        self.group_history_count = group_history_count
        self.max_message_length = max_message_length

        self._client = None
        self._connected = False
        self._qr_data: str | None = None
        self._qr_listeners: list[Callable] = []
        self._status_listeners: list[Callable] = []
        self._phone_number: str | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        # Map from chat user string -> full JID protobuf for correct reply routing
        self._chat_jids: dict[str, Any] = {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def current_qr(self) -> str | None:
        return self._qr_data

    @property
    def phone_number(self) -> str | None:
        return self._phone_number

    def add_qr_listener(self, callback: Callable[[str], Any]) -> None:
        """Register a callback for QR code updates."""
        self._qr_listeners.append(callback)

    def remove_qr_listener(self, callback: Callable) -> None:
        self._qr_listeners = [cb for cb in self._qr_listeners if cb is not callback]

    def add_status_listener(self, callback: Callable[[dict], Any]) -> None:
        """Register a callback for connection status updates."""
        self._status_listeners.append(callback)

    def remove_status_listener(self, callback: Callable) -> None:
        self._status_listeners = [cb for cb in self._status_listeners if cb is not callback]

    def _schedule_coro(self, coro):
        """Schedule a coroutine on the event loop from a sync callback (thread-safe)."""
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(coro, self._loop)

    async def _notify_qr(self, qr_data: str) -> None:
        """Notify all QR listeners."""
        self._qr_data = qr_data
        for cb in self._qr_listeners:
            try:
                result = cb(qr_data)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.warning(f"QR listener error: {e}")

    async def _notify_status(self, status: dict) -> None:
        """Notify all status listeners."""
        for cb in self._status_listeners:
            try:
                result = cb(status)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.warning(f"Status listener error: {e}")

    async def start(self) -> None:
        """Start the WhatsApp client and begin listening for messages."""
        try:
            from neonize.client import NewClient
            from neonize.events import ConnectedEv, MessageEv, PairStatusEv, LoggedOutEv
        except ImportError:
            logger.error(
                "neonize not installed. Run: pip install neonize\n"
                "neonize provides Python bindings for whatsmeow (Go WhatsApp Web library)."
            )
            await self._notify_status({
                "state": "error",
                "message": "neonize not installed. Run: pip install neonize",
            })
            return

        # Capture the running event loop for thread-safe coroutine scheduling
        self._loop = asyncio.get_running_loop()

        # Ensure session directory exists
        WA_SESSION_DIR.mkdir(parents=True, exist_ok=True)
        db_path = str(WA_SESSION_DIR / f"{self.session_name}.db")

        self._client = NewClient(db_path)

        # --- QR code callback (called from neonize's internal thread) ---
        def on_qr(client, qr_bytes: bytes):
            qr_data = qr_bytes.decode("utf-8", errors="replace")
            logger.info("WhatsApp QR code received — scan to link")
            self._schedule_coro(self._notify_qr(qr_data))
            self._schedule_coro(self._notify_status({
                "state": "qr",
                "message": "Scan the QR code with WhatsApp",
            }))

        self._client.event.qr(on_qr)

        # --- Event callbacks registered by event code ---
        def on_pair_status(client, event):
            logger.info("WhatsApp pairing in progress...")
            self._schedule_coro(self._notify_status({
                "state": "pairing",
                "message": "Pairing in progress...",
            }))

        def on_connected(client, event):
            self._connected = True
            self._qr_data = None
            try:
                jid = client.get_me()
                self._phone_number = jid.User if jid else None
            except Exception:
                self._phone_number = None

            logger.info(f"WhatsApp connected! Phone: {self._phone_number or 'unknown'}")
            self._schedule_coro(self._notify_status({
                "state": "connected",
                "message": f"Connected as {self._phone_number or 'unknown'}",
                "phone": self._phone_number,
            }))

        def on_logged_out(client, event):
            self._connected = False
            self._phone_number = None
            logger.info("WhatsApp logged out")
            self._schedule_coro(self._notify_status({
                "state": "disconnected",
                "message": "Logged out from WhatsApp",
            }))

        def on_message(client, event):
            try:
                # Skip messages sent by us
                if event.Info.MessageSource.IsFromMe:
                    return
                logger.info(f"WhatsApp message received from {event.Info.MessageSource.Sender.User}")
                self._schedule_coro(self._handle_wa_message(client, event))
            except Exception as e:
                logger.error(f"Error dispatching WhatsApp message: {e}", exc_info=True)

        self._client.event.list_func[_EV_PAIR_STATUS] = on_pair_status
        self._client.event.list_func[_EV_CONNECTED] = on_connected
        self._client.event.list_func[_EV_LOGGED_OUT] = on_logged_out
        self._client.event.list_func[_EV_MESSAGE] = on_message

        # Connect in background thread (neonize's connect() is blocking)
        logger.info("WhatsApp adapter starting...")
        await self._notify_status({
            "state": "connecting",
            "message": "Starting WhatsApp connection...",
        })

        self._loop.run_in_executor(None, self._client.connect)

    async def _handle_wa_message(self, client, event) -> None:
        """Process an incoming WhatsApp message."""
        try:
            info = event.Info
            msg = event.Message

            # Extract text content
            text = ""
            if msg.conversation:
                text = msg.conversation
            elif msg.extendedTextMessage and msg.extendedTextMessage.text:
                text = msg.extendedTextMessage.text

            if not text:
                logger.debug("Skipping non-text WhatsApp message")
                return

            sender_jid = info.MessageSource.Sender.User
            sender_server = info.MessageSource.Sender.Server
            chat_jid_full = info.MessageSource.Chat
            chat_jid = chat_jid_full.User
            chat_server = chat_jid_full.Server
            is_group = info.MessageSource.IsGroup

            # Cache the full JID protobuf for reply routing
            self._chat_jids[chat_jid] = chat_jid_full

            logger.info(f"Processing WhatsApp message: '{text[:80]}' from={sender_jid}@{sender_server} chat={chat_jid}@{chat_server} group={is_group}")

            # Check allowlist — match against phone number, LID, or chat JID
            if self.allow_from:
                identifiers = {sender_jid, chat_jid}
                # Also try matching with/without country code prefix
                if not identifiers.intersection(set(self.allow_from)):
                    logger.info(f"Blocked message from {sender_jid} (not in allow_from: {self.allow_from})")
                    return

            # Group policy check
            if is_group and self.group_policy == "mention":
                if msg.extendedTextMessage and msg.extendedTextMessage.contextInfo:
                    mentioned = msg.extendedTextMessage.contextInfo.mentionedJid or []
                    me = client.get_me()
                    if me and f"{me.User}@s.whatsapp.net" not in mentioned:
                        return
                else:
                    return

            inbound = InboundMessage(
                content=text,
                sender_id=sender_jid,
                channel="whatsapp",
                group_id=chat_jid if is_group else None,
                raw={
                    "chat_jid": chat_jid,
                    "is_group": is_group,
                    "message_id": info.ID,
                    "timestamp": str(info.Timestamp),
                },
            )

            logger.info(f"Calling agent handler for WhatsApp message from {sender_jid}")
            response = await self.handle_inbound(inbound)
            if response:
                logger.info(f"Sending reply to {chat_jid}: '{response[:80]}...'")
                await self.send_message(chat_jid, response)
            else:
                logger.warning("No response from agent handler")
        except Exception as e:
            logger.error(f"Error in _handle_wa_message: {e}", exc_info=True)

    async def stop(self) -> None:
        """Stop the WhatsApp client."""
        if self._client:
            try:
                self._client.disconnect()
            except Exception as e:
                logger.warning(f"Error disconnecting WhatsApp: {e}")
        self._connected = False
        self._qr_data = None
        logger.info("WhatsApp adapter stopped")
        await self._notify_status({
            "state": "disconnected",
            "message": "WhatsApp adapter stopped",
        })

    async def send_message(self, recipient_id: str, content: str, **kwargs: Any) -> None:
        """Send a message to a WhatsApp chat."""
        if not self._client or not self._connected:
            logger.warning("Cannot send message: WhatsApp not connected")
            return

        try:
            from neonize.utils import build_jid

            # Use cached full JID if available (preserves correct server: lid vs s.whatsapp.net)
            if recipient_id in self._chat_jids:
                jid = self._chat_jids[recipient_id]
                logger.info(f"Using cached JID for {recipient_id}@{jid.Server}")
            else:
                jid = build_jid(recipient_id)
                logger.info(f"Using default JID for {recipient_id}@s.whatsapp.net")

            # Chunk long messages
            chunks = self._chunk_message(content)
            loop = asyncio.get_running_loop()
            for chunk in chunks:
                # neonize's send_message is blocking, run in executor
                await loop.run_in_executor(None, self._client.send_message, jid, chunk)
                logger.info(f"Sent WhatsApp message to {recipient_id} ({len(chunk)} chars)")

        except Exception as e:
            logger.error(f"Failed to send WhatsApp message: {e}", exc_info=True)

    def _chunk_message(self, text: str) -> list[str]:
        """Split a long message into chunks."""
        if len(text) <= self.max_message_length:
            return [text]

        chunks = []
        while text:
            if len(text) <= self.max_message_length:
                chunks.append(text)
                break

            # Try to split at a newline
            split_idx = text.rfind("\n", 0, self.max_message_length)
            if split_idx == -1:
                split_idx = text.rfind(" ", 0, self.max_message_length)
            if split_idx == -1:
                split_idx = self.max_message_length

            chunks.append(text[:split_idx])
            text = text[split_idx:].lstrip()

        return chunks

    async def restart_linking(self) -> None:
        """Restart the QR code linking process."""
        await self.stop()

        # Delete existing session to force new QR
        session_db = WA_SESSION_DIR / f"{self.session_name}.db"
        if session_db.exists():
            session_db.unlink()

        await self.start()

    async def unlink(self) -> None:
        """Unlink and remove the WhatsApp session."""
        if self._client and self._connected:
            try:
                self._client.logout()
            except Exception:
                pass

        await self.stop()

        # Remove session files
        session_db = WA_SESSION_DIR / f"{self.session_name}.db"
        if session_db.exists():
            session_db.unlink()

        await self._notify_status({
            "state": "unlinked",
            "message": "WhatsApp account unlinked",
        })
