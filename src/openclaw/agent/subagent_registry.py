"""Sub-agent registry — track spawned sub-agents for list/kill/steer.

Mirrors openclaw's subagent-registry.ts:
- In-process registry keyed by parent session key
- Records: run_id, child session key, task, label, status, timing
- Supports kill (abort), steer (redirect), and list operations
- Cascade kill: killing a sub-agent also kills its children
- Steer: interrupt current work and restart with a new message
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# How long to wait for a run to settle before steering (seconds)
STEER_ABORT_SETTLE_TIMEOUT = 5.0

# Rate limit for steer operations per target (seconds)
STEER_RATE_LIMIT = 2.0

# Maximum steer message length
MAX_STEER_MESSAGE_CHARS = 4_000


@dataclass
class SubagentOutcome:
    """Outcome of a sub-agent run."""

    status: str = "ok"  # "ok", "error", "killed"
    response: str = ""
    error: str | None = None
    tool_calls: int = 0
    elapsed_seconds: float = 0.0


@dataclass
class SubagentRunRecord:
    """Record of a spawned sub-agent run."""

    run_id: str
    requester_session_key: str  # parent session key
    child_session_key: str  # unique session key for this sub-agent
    task: str  # the task description
    label: str = ""  # short display label
    model: str | None = None

    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    ended_at: float | None = None

    outcome: SubagentOutcome | None = None
    cancel_event: asyncio.Event | None = field(default=None, repr=False)

    # Steer support
    steer_restart_suppressed: bool = False

    # Depth tracking
    depth: int = 0

    # Channel to announce results to (mirrors openclaw's delivery context)
    requester_channel: str = "cli"

    @property
    def is_running(self) -> bool:
        return self.started_at is not None and self.ended_at is None

    @property
    def status(self) -> str:
        if self.ended_at is None:
            return "running" if self.started_at else "pending"
        if self.outcome is None:
            return "done"
        return self.outcome.status

    @property
    def runtime_ms(self) -> int:
        if self.started_at is None:
            return 0
        end = self.ended_at or time.time()
        return int((end - self.started_at) * 1000)

    def display_label(self, fallback: str = "subagent") -> str:
        return (self.label or self.task[:48] or fallback).strip()


class SubagentRegistry:
    """In-process registry for tracking spawned sub-agents.

    Thread-safe: all mutations are protected by a lock.
    Keyed by requester session key so each parent can see its own children.
    """

    _instance: SubagentRegistry | None = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # requester_key -> list[SubagentRunRecord]
        self._runs: dict[str, list[SubagentRunRecord]] = {}
        # run_id -> SubagentRunRecord (for fast lookup)
        self._by_run_id: dict[str, SubagentRunRecord] = {}
        # steer rate limiting: target child_session_key -> last steer timestamp
        self._steer_timestamps: dict[str, float] = {}

    @classmethod
    def get_instance(cls) -> "SubagentRegistry":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def register(
        self,
        requester_session_key: str,
        child_session_key: str,
        task: str,
        label: str = "",
        model: str | None = None,
        depth: int = 0,
        requester_channel: str = "cli",
    ) -> SubagentRunRecord:
        """Register a new sub-agent run."""
        run_id = str(uuid.uuid4())[:12]
        record = SubagentRunRecord(
            run_id=run_id,
            requester_session_key=requester_session_key,
            child_session_key=child_session_key,
            task=task,
            label=label,
            model=model,
            cancel_event=asyncio.Event(),
            depth=depth,
            requester_channel=requester_channel,
        )
        with self._lock:
            if requester_session_key not in self._runs:
                self._runs[requester_session_key] = []
            self._runs[requester_session_key].append(record)
            self._by_run_id[run_id] = record
        return record

    def mark_started(self, run_id: str) -> None:
        with self._lock:
            record = self._by_run_id.get(run_id)
            if record:
                record.started_at = time.time()

    def mark_ended(
        self,
        run_id: str,
        outcome: SubagentOutcome,
    ) -> None:
        with self._lock:
            record = self._by_run_id.get(run_id)
            if record:
                record.ended_at = time.time()
                record.outcome = outcome

    def mark_killed(self, run_id: str) -> bool:
        """Mark a run as killed. Returns True if it was running."""
        with self._lock:
            record = self._by_run_id.get(run_id)
            if not record:
                return False
            was_running = record.is_running
            record.ended_at = time.time()
            record.outcome = SubagentOutcome(status="killed")
            if record.cancel_event:
                record.cancel_event.set()
            return was_running

    def get_cancel_event(self, run_id: str) -> asyncio.Event | None:
        with self._lock:
            record = self._by_run_id.get(run_id)
            return record.cancel_event if record else None

    def list_for_requester(
        self,
        requester_session_key: str,
    ) -> list[SubagentRunRecord]:
        """List all runs spawned by a requester session."""
        with self._lock:
            runs = self._runs.get(requester_session_key, [])
            return sorted(runs, key=lambda r: r.created_at, reverse=True)

    def get_by_run_id(self, run_id: str) -> SubagentRunRecord | None:
        with self._lock:
            return self._by_run_id.get(run_id)

    def get_by_child_key(self, child_session_key: str) -> SubagentRunRecord | None:
        """Find a record by child session key."""
        with self._lock:
            for record in self._by_run_id.values():
                if record.child_session_key == child_session_key:
                    return record
            return None

    def resolve_target(
        self,
        requester_key: str,
        token: str,
        recent_minutes: int = 30,
    ) -> SubagentRunRecord | None:
        """Resolve a target specification to a SubagentRunRecord.

        Supports:
        - "last" → most recently started
        - "1", "2", ... → index in active+recent list
        - run_id prefix (12-char hex)
        - label or label prefix
        """
        runs = self.list_for_requester(requester_key)
        if not runs:
            return None

        token = token.strip()
        now = time.time()
        recent_cutoff = now - recent_minutes * 60

        # Build ordered list: active first, then recent, newest first
        active = [r for r in runs if not r.ended_at]
        recent = [
            r for r in runs
            if r.ended_at and r.ended_at >= recent_cutoff
        ]
        ordered = active + recent

        if token == "last":
            return ordered[0] if ordered else None

        # Numeric index
        if token.isdigit():
            idx = int(token) - 1
            if 0 <= idx < len(ordered):
                return ordered[idx]
            return None

        # Run ID prefix
        by_run_id = [r for r in runs if r.run_id.startswith(token)]
        if len(by_run_id) == 1:
            return by_run_id[0]
        if len(by_run_id) > 1:
            logger.warning(f"Ambiguous run_id prefix: {token}")
            return None

        # Exact label match
        token_lower = token.lower()
        by_label = [r for r in runs if r.display_label().lower() == token_lower]
        if len(by_label) == 1:
            return by_label[0]

        # Label prefix match
        by_prefix = [r for r in runs if r.display_label().lower().startswith(token_lower)]
        if len(by_prefix) == 1:
            return by_prefix[0]

        return None

    def check_steer_rate_limit(self, child_session_key: str) -> bool:
        """Return True if steer is allowed (not rate limited)."""
        now = time.time()
        with self._lock:
            last = self._steer_timestamps.get(child_session_key, 0)
            if now - last < STEER_RATE_LIMIT:
                return False
            self._steer_timestamps[child_session_key] = now
            return True

    def suppress_steer_announce(self, run_id: str) -> None:
        """Suppress the completion announcement for a run being steered."""
        with self._lock:
            record = self._by_run_id.get(run_id)
            if record:
                record.steer_restart_suppressed = True

    def clear_steer_suppress(self, run_id: str) -> None:
        """Restore announcement for a steered run (on steer failure)."""
        with self._lock:
            record = self._by_run_id.get(run_id)
            if record:
                record.steer_restart_suppressed = False

    def get_depth_for_session(self, session_key: str) -> int:
        """Get the nesting depth of a sub-agent by its session key."""
        record = self.get_by_child_key(session_key)
        return record.depth if record else 0

    def get_active_children(self, parent_session_key: str) -> list[SubagentRunRecord]:
        """Get currently active (running) children of a session."""
        with self._lock:
            runs = self._runs.get(parent_session_key, [])
            return [r for r in runs if r.is_running]

    def cascade_kill(self, root_child_session_key: str) -> list[str]:
        """Recursively kill a sub-agent and all its descendants.

        Returns list of killed labels.
        """
        killed: list[str] = []
        seen: set[str] = set()

        def _kill_recursive(session_key: str) -> None:
            if session_key in seen:
                return
            seen.add(session_key)

            # Kill direct children first
            with self._lock:
                children = self._runs.get(session_key, [])
                child_copies = list(children)

            for child in child_copies:
                if child.child_session_key not in seen:
                    _kill_recursive(child.child_session_key)

            # Kill this session's run
            record = self.get_by_child_key(session_key)
            if record and record.is_running:
                was_killed = self.mark_killed(record.run_id)
                if was_killed:
                    killed.append(record.display_label())

        _kill_recursive(root_child_session_key)
        return killed


def get_subagent_registry() -> SubagentRegistry:
    """Get the global sub-agent registry instance."""
    return SubagentRegistry.get_instance()
