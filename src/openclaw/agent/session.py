"""Session manager — per-sender conversation history persistence.

Mirrors OpenClaw's session model:
- Sessions stored as JSONL files under ~/.langclaw/sessions/
- Per-sender session isolation (channel:sender_id key)
- Session metadata tracking
- File-based write locking for concurrent safety
- Auto-compaction when sessions exceed threshold
- Session TTL with auto-archival
"""

from __future__ import annotations

import fcntl
import json
import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

logger = logging.getLogger(__name__)

DEFAULT_SESSIONS_DIR = Path.home() / ".langclaw" / "sessions"

# Session auto-compaction threshold
MAX_SESSION_MESSAGES = 200

# Session TTL (30 days)
SESSION_TTL_SECONDS = 30 * 24 * 3600


@dataclass
class SessionMetadata:
    """Metadata for a session."""

    session_id: str
    sender_id: str
    channel: str
    created_at: float
    last_active: float
    message_count: int = 0
    title: str = ""
    model_override: str = ""   # Per-session model override (provider/model)
    total_tokens: int = 0      # Cumulative token usage across all runs
    context_tokens: int = 0    # Last-known context window token count


@dataclass
class Session:
    """A single conversation session."""

    metadata: SessionMetadata
    messages: list[BaseMessage] = field(default_factory=list)


def _message_to_dict(msg: BaseMessage) -> dict[str, Any]:
    """Serialize a LangChain message to a dict for JSONL storage."""
    data: dict[str, Any] = {
        "type": msg.type,
        "content": msg.content,
        "timestamp": time.time(),
    }
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        data["tool_calls"] = [
            {
                "id": tc.get("id", ""),
                "name": tc.get("name", ""),
                "args": tc.get("args", {}),
            }
            for tc in msg.tool_calls
        ]
    if hasattr(msg, "tool_call_id") and msg.tool_call_id:
        data["tool_call_id"] = msg.tool_call_id
    if hasattr(msg, "name") and msg.name:
        data["name"] = msg.name
    if msg.additional_kwargs:
        data["additional_kwargs"] = msg.additional_kwargs
    return data


def _dict_to_message(data: dict[str, Any]) -> BaseMessage:
    """Deserialize a dict from JSONL to a LangChain message."""
    msg_type = data.get("type", "human")
    content = data.get("content", "")

    if msg_type == "human":
        return HumanMessage(content=content)
    elif msg_type == "ai":
        kwargs: dict[str, Any] = {"content": content}
        if "tool_calls" in data:
            kwargs["tool_calls"] = data["tool_calls"]
        if "additional_kwargs" in data:
            kwargs["additional_kwargs"] = data["additional_kwargs"]
        return AIMessage(**kwargs)
    elif msg_type == "tool":
        return ToolMessage(
            content=content,
            tool_call_id=data.get("tool_call_id", ""),
            name=data.get("name", ""),
        )
    elif msg_type == "system":
        return SystemMessage(content=content)
    else:
        return HumanMessage(content=content)


@contextmanager
def _file_lock(path: Path) -> Generator[None, None, None]:
    """File-based lock for concurrent write protection.

    Uses fcntl advisory locks on a .lock file alongside the target.
    """
    lock_path = path.parent / f".{path.name}.lock"
    lock_file = None
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_file = open(lock_path, "w")
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        if lock_file:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                lock_file.close()
            except Exception:
                pass


class SessionManager:
    """Manages conversation sessions with JSONL persistence.

    Session key format: <channel>:<sender_id>
    File layout: <sessions_dir>/<session_id>.jsonl
    Metadata file: <sessions_dir>/<session_id>.meta.json

    Features:
    - File-based locking for concurrent write safety
    - Auto-compaction when sessions exceed MAX_SESSION_MESSAGES
    - Session TTL with cleanup
    """

    def __init__(self, sessions_dir: Path | None = None):
        self.sessions_dir = sessions_dir or DEFAULT_SESSIONS_DIR
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self._active_sessions: dict[str, Session] = {}
        self._key_to_session_id: dict[str, str] = {}
        self._load_index()

    def _load_index(self) -> None:
        """Load session index from disk."""
        known = {f.name for f in SessionMetadata.__dataclass_fields__.values()}
        for meta_file in self.sessions_dir.glob("*.meta.json"):
            try:
                data = json.loads(meta_file.read_text(encoding="utf-8"))
                data = {k: v for k, v in data.items() if k in known}
                meta = SessionMetadata(**data)
                key = f"{meta.channel}:{meta.sender_id}"
                self._key_to_session_id[key] = meta.session_id
            except (json.JSONDecodeError, TypeError, KeyError):
                continue

    def _session_file(self, session_id: str) -> Path:
        return self.sessions_dir / f"{session_id}.jsonl"

    def _meta_file(self, session_id: str) -> Path:
        return self.sessions_dir / f"{session_id}.meta.json"

    def get_or_create_session(
        self, sender_id: str, channel: str = "cli"
    ) -> Session:
        """Get existing session for a sender/channel pair, or create a new one."""
        key = f"{channel}:{sender_id}"

        # Check in-memory cache
        if key in self._active_sessions:
            return self._active_sessions[key]

        # Check disk
        if key in self._key_to_session_id:
            session_id = self._key_to_session_id[key]
            session = self._load_session(session_id)
            if session:
                self._active_sessions[key] = session
                return session

        # Create new session
        session_id = str(uuid.uuid4())[:12]
        now = time.time()
        meta = SessionMetadata(
            session_id=session_id,
            sender_id=sender_id,
            channel=channel,
            created_at=now,
            last_active=now,
        )
        session = Session(metadata=meta, messages=[])
        self._active_sessions[key] = session
        self._key_to_session_id[key] = session_id
        self._save_metadata(session)
        return session

    def _load_session(self, session_id: str) -> Session | None:
        """Load a session from its JSONL file."""
        meta_path = self._meta_file(session_id)
        jsonl_path = self._session_file(session_id)

        if not meta_path.exists():
            return None

        try:
            meta_data = json.loads(meta_path.read_text(encoding="utf-8"))
            # Strip unknown keys for forward-compat
            known = {f.name for f in SessionMetadata.__dataclass_fields__.values()}
            meta_data = {k: v for k, v in meta_data.items() if k in known}
            meta = SessionMetadata(**meta_data)
        except (json.JSONDecodeError, TypeError):
            return None

        messages: list[BaseMessage] = []
        if jsonl_path.exists():
            for line in jsonl_path.read_text(encoding="utf-8").strip().split("\n"):
                if line.strip():
                    try:
                        data = json.loads(line)
                        messages.append(_dict_to_message(data))
                    except json.JSONDecodeError:
                        continue

        return Session(metadata=meta, messages=messages)

    def append_message(self, session: Session, message: BaseMessage) -> None:
        """Append a message to a session and persist it (thread-safe)."""
        session.messages.append(message)
        session.metadata.last_active = time.time()
        session.metadata.message_count = len(session.messages)

        jsonl_path = self._session_file(session.metadata.session_id)

        # File-locked write
        with _file_lock(jsonl_path):
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(_message_to_dict(message)) + "\n")

        self._save_metadata(session)

        # Auto-compact if session grows too large
        if len(session.messages) > MAX_SESSION_MESSAGES:
            self.compact_session(session)

    def append_messages(self, session: Session, messages: list[BaseMessage]) -> None:
        """Append multiple messages to a session."""
        for msg in messages:
            self.append_message(session, msg)

    def _save_metadata(self, session: Session) -> None:
        """Save session metadata to disk."""
        meta_path = self._meta_file(session.metadata.session_id)
        with _file_lock(meta_path):
            meta_path.write_text(
                json.dumps(
                    {
                        "session_id": session.metadata.session_id,
                        "sender_id": session.metadata.sender_id,
                        "channel": session.metadata.channel,
                        "created_at": session.metadata.created_at,
                        "last_active": session.metadata.last_active,
                        "message_count": session.metadata.message_count,
                        "title": session.metadata.title,
                        "model_override": session.metadata.model_override,
                        "total_tokens": session.metadata.total_tokens,
                        "context_tokens": session.metadata.context_tokens,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    def list_sessions(self) -> list[SessionMetadata]:
        """List all sessions with metadata."""
        sessions: list[SessionMetadata] = []
        for meta_file in sorted(self.sessions_dir.glob("*.meta.json")):
            try:
                data = json.loads(meta_file.read_text(encoding="utf-8"))
                sessions.append(SessionMetadata(**data))
            except (json.JSONDecodeError, TypeError):
                continue
        return sorted(sessions, key=lambda s: s.last_active, reverse=True)

    def get_session_history(self, session_id: str) -> list[dict[str, Any]]:
        """Get session history as a list of dicts (for API responses)."""
        session = self._load_session(session_id)
        if not session:
            return []
        return [_message_to_dict(msg) for msg in session.messages]

    def compact_session(self, session: Session, keep_recent: int = 50) -> None:
        """Compact a session by summarizing old messages.

        Keeps the most recent `keep_recent` messages and replaces
        older messages with a summary.
        """
        if len(session.messages) <= keep_recent:
            return

        logger.info(
            f"Compacting session {session.metadata.session_id}: "
            f"{len(session.messages)} → ~{keep_recent} messages"
        )

        dropped = session.messages[:-keep_recent]
        recent = session.messages[-keep_recent:]

        # Build a compact summary of dropped messages
        user_count = sum(1 for m in dropped if isinstance(m, HumanMessage))
        ai_count = sum(1 for m in dropped if isinstance(m, AIMessage))
        tool_count = sum(1 for m in dropped if isinstance(m, ToolMessage))

        summary_text = (
            f"[Session compacted: {len(dropped)} older messages summarized. "
            f"Contained {user_count} user messages, {ai_count} assistant responses, "
            f"{tool_count} tool results. Conversation has been ongoing.]"
        )

        session.messages = [HumanMessage(content=summary_text)] + recent
        session.metadata.message_count = len(session.messages)

        # Rewrite JSONL
        jsonl_path = self._session_file(session.metadata.session_id)
        with _file_lock(jsonl_path):
            with open(jsonl_path, "w", encoding="utf-8") as f:
                for msg in session.messages:
                    f.write(json.dumps(_message_to_dict(msg)) + "\n")
        self._save_metadata(session)

    def prune_session(self, session: Session, max_messages: int = 100) -> None:
        """Prune old messages from a session, keeping the most recent."""
        if len(session.messages) <= max_messages:
            return
        self.compact_session(session, keep_recent=max_messages)

    def cleanup_expired_sessions(self) -> int:
        """Remove sessions older than SESSION_TTL_SECONDS.

        Returns number of sessions cleaned up.
        """
        now = time.time()
        cleaned = 0

        for meta_file in self.sessions_dir.glob("*.meta.json"):
            try:
                data = json.loads(meta_file.read_text(encoding="utf-8"))
                last_active = data.get("last_active", 0)
                if now - last_active > SESSION_TTL_SECONDS:
                    session_id = data.get("session_id", "")
                    # Remove files
                    jsonl_path = self._session_file(session_id)
                    if jsonl_path.exists():
                        jsonl_path.unlink()
                    meta_file.unlink()
                    # Remove lock files
                    for lock in self.sessions_dir.glob(f".{session_id}*.lock"):
                        lock.unlink(missing_ok=True)
                    cleaned += 1
                    logger.info(f"Cleaned up expired session: {session_id}")
            except Exception:
                continue

        return cleaned

    def set_model_override(self, session: Session, model: str) -> None:
        """Set or clear a per-session model override and persist it."""
        session.metadata.model_override = model.strip()
        self._save_metadata(session)

    def record_token_usage(self, session: Session, tokens_used: int, context_tokens: int = 0) -> None:
        """Accumulate token usage for a session."""
        session.metadata.total_tokens += tokens_used
        if context_tokens > 0:
            session.metadata.context_tokens = context_tokens
        self._save_metadata(session)

    def get_session_by_id(self, session_id: str) -> Session | None:
        """Load a session by ID (public API)."""
        return self._load_session(session_id)
