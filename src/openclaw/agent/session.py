"""Session manager — per-sender conversation history persistence.

Mirrors OpenClaw's session model:
- Sessions stored as JSONL files under ~/.openclaw/agents/<agent_id>/sessions/<session_id>.jsonl
- Per-sender session isolation
- Session metadata tracking
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

DEFAULT_SESSIONS_DIR = Path.home() / ".openclaw" / "sessions"


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


class SessionManager:
    """Manages conversation sessions with JSONL persistence.

    Session key format: <channel>:<sender_id>
    File layout: <sessions_dir>/<session_id>.jsonl
    Metadata file: <sessions_dir>/<session_id>.meta.json
    """

    def __init__(self, sessions_dir: Path | None = None):
        self.sessions_dir = sessions_dir or DEFAULT_SESSIONS_DIR
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self._active_sessions: dict[str, Session] = {}
        self._key_to_session_id: dict[str, str] = {}
        self._load_index()

    def _load_index(self) -> None:
        """Load session index from disk."""
        for meta_file in self.sessions_dir.glob("*.meta.json"):
            try:
                data = json.loads(meta_file.read_text(encoding="utf-8"))
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
        """Append a message to a session and persist it."""
        session.messages.append(message)
        session.metadata.last_active = time.time()
        session.metadata.message_count = len(session.messages)

        # Append to JSONL
        jsonl_path = self._session_file(session.metadata.session_id)
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(_message_to_dict(message)) + "\n")

        self._save_metadata(session)

    def append_messages(self, session: Session, messages: list[BaseMessage]) -> None:
        """Append multiple messages to a session."""
        for msg in messages:
            self.append_message(session, msg)

    def _save_metadata(self, session: Session) -> None:
        """Save session metadata to disk."""
        meta_path = self._meta_file(session.metadata.session_id)
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

    def prune_session(self, session: Session, max_messages: int = 100) -> None:
        """Prune old messages from a session, keeping the most recent."""
        if len(session.messages) <= max_messages:
            return
        session.messages = session.messages[-max_messages:]
        # Rewrite JSONL
        jsonl_path = self._session_file(session.metadata.session_id)
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for msg in session.messages:
                f.write(json.dumps(_message_to_dict(msg)) + "\n")
        session.metadata.message_count = len(session.messages)
        self._save_metadata(session)
