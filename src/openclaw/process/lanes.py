"""Execution lanes — separate queues for different command types.

Mirrors openclaw's CommandLane system. Different types of work run in
separate lanes to prevent blocking (e.g., cron jobs don't block user
messages, sub-agents don't starve the main loop).

Architecture:
    Main ────► user messages, interactive commands
    Cron ────► scheduled/heartbeat jobs
    Subagent ► child agent tasks
    Nested ──► nested tool calls within an agent turn
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)


class CommandLane(str, Enum):
    """Execution lane for a command."""

    MAIN = "main"
    CRON = "cron"
    SUBAGENT = "subagent"
    NESTED = "nested"


# Agent-specific lane aliases
AGENT_LANE_NESTED = CommandLane.NESTED
AGENT_LANE_SUBAGENT = CommandLane.SUBAGENT


@dataclass
class LaneCommand:
    """A command queued for execution in a specific lane."""

    lane: CommandLane
    payload: Any
    callback: Callable[..., Awaitable[Any]] | None = None
    priority: int = 0  # Lower = higher priority


@dataclass
class LaneStats:
    """Statistics for a lane."""

    pending: int = 0
    running: int = 0
    completed: int = 0
    failed: int = 0


class LaneManager:
    """Manages execution lanes with separate asyncio queues.

    Each lane has its own queue and worker, so different types of
    work don't block each other.
    """

    def __init__(self, max_concurrent_per_lane: int = 1):
        self._queues: dict[CommandLane, asyncio.Queue[LaneCommand]] = {}
        self._workers: dict[CommandLane, asyncio.Task[None]] = {}
        self._stats: dict[CommandLane, LaneStats] = {}
        self._max_concurrent = max_concurrent_per_lane
        self._running = False

        # Initialize all lanes
        for lane in CommandLane:
            self._queues[lane] = asyncio.Queue()
            self._stats[lane] = LaneStats()

    async def start(self) -> None:
        """Start lane workers."""
        if self._running:
            return
        self._running = True
        for lane in CommandLane:
            self._workers[lane] = asyncio.create_task(
                self._lane_worker(lane), name=f"lane-{lane.value}"
            )
        logger.info("Lane manager started with %d lanes", len(CommandLane))

    async def stop(self) -> None:
        """Stop all lane workers."""
        self._running = False
        for task in self._workers.values():
            task.cancel()
        await asyncio.gather(*self._workers.values(), return_exceptions=True)
        self._workers.clear()
        logger.info("Lane manager stopped")

    async def submit(self, command: LaneCommand) -> None:
        """Submit a command to the appropriate lane."""
        self._stats[command.lane].pending += 1
        await self._queues[command.lane].put(command)

    def get_stats(self) -> dict[str, LaneStats]:
        """Get statistics for all lanes."""
        return {lane.value: self._stats[lane] for lane in CommandLane}

    async def _lane_worker(self, lane: CommandLane) -> None:
        """Worker loop for a single lane."""
        queue = self._queues[lane]
        stats = self._stats[lane]

        while self._running:
            try:
                command = await asyncio.wait_for(queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            stats.pending -= 1
            stats.running += 1

            try:
                if command.callback:
                    await command.callback(command.payload)
                stats.completed += 1
            except Exception as e:
                stats.failed += 1
                logger.error("Lane %s command failed: %s", lane.value, e)
            finally:
                stats.running -= 1
                queue.task_done()


def classify_lane(
    *,
    is_cron: bool = False,
    is_subagent: bool = False,
    depth: int = 0,
) -> CommandLane:
    """Classify which lane a command should run in.

    Args:
        is_cron: Whether this is a cron/heartbeat job.
        is_subagent: Whether this is a sub-agent task.
        depth: Nesting depth (> 0 means nested).

    Returns:
        The appropriate CommandLane.
    """
    if is_cron:
        return CommandLane.CRON
    if is_subagent:
        return CommandLane.SUBAGENT
    if depth > 0:
        return CommandLane.NESTED
    return CommandLane.MAIN
