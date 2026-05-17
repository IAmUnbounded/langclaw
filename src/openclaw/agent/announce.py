"""Announce system — idempotent sub-agent result broadcasting.

Mirrors openclaw's announce-idempotency.ts:
- Track which run IDs have already been announced
- Prevent duplicate announcements (idempotency)
- Format announcements as structured messages
- Route announcements to the appropriate channel/session
- Support multi-channel delivery (Slack, Telegram, Discord, WhatsApp, CLI)
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# How long to track announced run IDs (seconds)
ANNOUNCE_DEDUP_TTL = 3600  # 1 hour

# Maximum chars for response preview in announcements
MAX_RESPONSE_PREVIEW = 4000


@dataclass
class AnnounceRecord:
    """Record of a completed announcement."""

    run_id: str
    announced_at: float = field(default_factory=time.time)
    channel: str = ""
    success: bool = True


class AnnounceRegistry:
    """Registry tracking which sub-agent runs have been announced.

    Prevents duplicate announcements when multiple completion events
    fire for the same run.
    """

    _instance: "AnnounceRegistry | None" = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._announced: dict[str, AnnounceRecord] = {}

    @classmethod
    def get_instance(cls) -> "AnnounceRegistry":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def try_claim(self, run_id: str, channel: str = "") -> bool:
        """Attempt to claim announcement rights for a run.

        Returns True if this is the first announcement for this run_id
        (i.e., caller should proceed with announcing).
        Returns False if already announced.
        """
        self._prune_expired()
        with self._lock:
            if run_id in self._announced:
                return False
            self._announced[run_id] = AnnounceRecord(
                run_id=run_id,
                channel=channel,
            )
            return True

    def mark_failed(self, run_id: str) -> None:
        """Mark an announcement as failed so it can be retried."""
        with self._lock:
            if run_id in self._announced:
                del self._announced[run_id]

    def _prune_expired(self) -> None:
        """Remove expired entries to prevent memory growth."""
        now = time.time()
        cutoff = now - ANNOUNCE_DEDUP_TTL
        with self._lock:
            expired = [
                run_id for run_id, rec in self._announced.items()
                if rec.announced_at < cutoff
            ]
            for run_id in expired:
                del self._announced[run_id]


def get_announce_registry() -> AnnounceRegistry:
    return AnnounceRegistry.get_instance()


@dataclass
class AnnouncePayload:
    """Structured announcement payload for a completed sub-agent run."""

    run_id: str
    label: str
    task: str
    status: str  # "ok", "error", "killed"
    response: str = ""
    error: str | None = None
    elapsed_seconds: float = 0.0
    tool_calls: int = 0
    model: str | None = None
    depth: int = 0


def format_text_announcement(payload: AnnouncePayload) -> str:
    """Format a sub-agent completion as a human-readable text message.

    Mirrors openclaw's subagents-format.ts formatSubagentAnnouncement.
    """
    status_emoji = {
        "ok": "✅",
        "error": "❌",
        "killed": "🛑",
        "steered": "↩️",
    }.get(payload.status, "ℹ️")

    status_text = {
        "ok": "completed",
        "error": "failed",
        "killed": "was killed",
        "steered": "was redirected",
    }.get(payload.status, payload.status)

    elapsed = f"{payload.elapsed_seconds:.1f}s" if payload.elapsed_seconds else "n/a"
    model_str = payload.model.split("/")[-1] if payload.model else "default"

    lines = [
        f"{status_emoji} **Sub-agent '{payload.label}'** {status_text} in {elapsed} ({model_str})",
    ]

    # Task preview
    task_preview = payload.task[:120] + ("..." if len(payload.task) > 120 else "")
    lines.append(f"**Task:** {task_preview}")

    # Stats
    if payload.tool_calls:
        lines.append(f"**Tool calls:** {payload.tool_calls}")

    # Error
    if payload.error:
        lines.append(f"**Error:** {payload.error}")

    # Response
    if payload.response and payload.status == "ok":
        response = payload.response
        if len(response) > MAX_RESPONSE_PREVIEW:
            response = response[:MAX_RESPONSE_PREVIEW] + "\n... *(response truncated)*"
        lines.append(f"\n**Result:**\n{response}")

    return "\n".join(lines)


async def announce_subagent_result(
    payload: AnnouncePayload,
    requester_session_key: str,
    channel: str = "cli",
    channel_client: Any = None,
) -> bool:
    """Announce a sub-agent result through the appropriate channel.

    Idempotent: will not announce the same run_id twice.

    Args:
        payload: The announcement payload.
        requester_session_key: Session key of the requesting agent.
        channel: Channel to deliver to (cli, slack, telegram, discord, whatsapp).
        channel_client: Optional channel client for direct delivery.

    Returns:
        True if announcement was sent, False if skipped (already announced).
    """
    registry = get_announce_registry()

    if not registry.try_claim(payload.run_id, channel):
        logger.debug(f"Announcement already sent for run {payload.run_id}")
        return False

    try:
        text = format_text_announcement(payload)
        await _deliver_announcement(
            text=text,
            requester_session_key=requester_session_key,
            channel=channel,
            channel_client=channel_client,
        )
        logger.info(f"Announced sub-agent result: run={payload.run_id} status={payload.status}")
        return True

    except Exception as e:
        logger.error(f"Failed to announce sub-agent {payload.run_id}: {e}")
        registry.mark_failed(payload.run_id)
        return False


def _get_sender_id_from_session(session_id: str) -> str | None:
    """Look up the sender's ID (e.g. WhatsApp JID) from a session UUID."""
    try:
        from openclaw.agent.session import SessionManager
        session = SessionManager()._load_session(session_id)
        if session:
            return session.metadata.sender_id
    except Exception:
        pass
    return None


async def _notify_whatsapp(sender_id: str, text: str) -> None:
    """Send a direct WhatsApp message via the gateway's adapter."""
    try:
        from openclaw.gateway.server import _whatsapp_adapter
        if _whatsapp_adapter and _whatsapp_adapter.is_connected:
            await _whatsapp_adapter.send_message(sender_id, text)
    except Exception as e:
        logger.warning(f"WhatsApp notification failed: {e}")


async def notify_subagent_spawned(
    requester_session_key: str,
    channel: str,
    label: str,
    run_id: str,
) -> None:
    """Send an immediate 'sub-agent spawned' notice to the requester."""
    if channel != "whatsapp":
        return
    sender_id = _get_sender_id_from_session(requester_session_key)
    if not sender_id:
        return
    text = f"🔀 *Sub-agent spawned:* {label}\n_I'll message you when it completes._ (id: {run_id})"
    await _notify_whatsapp(sender_id, text)


async def _deliver_announcement(
    text: str,
    requester_session_key: str,
    channel: str = "cli",
    channel_client: Any = None,
) -> None:
    """Deliver the announcement text through the appropriate channel."""
    if channel == "cli":
        try:
            from langchain_core.messages import HumanMessage
            from openclaw.agent.session import SessionManager

            sm = SessionManager()
            session = sm.get_or_create_session(
                sender_id=requester_session_key,
                channel="cli",
            )
            sm.append_message(
                session,
                HumanMessage(content=f"[subagent-result]\n{text}"),
            )
        except Exception as e:
            logger.warning(f"CLI announcement delivery failed: {e}")

    elif channel == "whatsapp":
        sender_id = _get_sender_id_from_session(requester_session_key)
        if sender_id:
            await _notify_whatsapp(sender_id, text)
        elif channel_client:
            try:
                await channel_client.send_message(text)
            except Exception as e:
                logger.warning(f"WhatsApp announcement delivery failed: {e}")
        else:
            logger.info(f"Sub-agent announcement (whatsapp, no sender):\n{text}")

    elif channel in ("slack", "telegram", "discord") and channel_client:
        try:
            await channel_client.send_message(text)
        except Exception as e:
            logger.warning(f"{channel} announcement delivery failed: {e}")

    else:
        logger.info(f"Sub-agent announcement:\n{text}")
