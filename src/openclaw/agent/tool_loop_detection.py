"""Tool loop detection — prevent infinite tool call loops.

Mirrors openclaw's tool-loop-detection.ts:
- Detects repeated identical tool calls (generic repeat)
- Identifies stuck polling loops (known poll patterns)
- Detects ping-pong alternating patterns
- Global circuit breaker stops execution after excessive iterations

Architecture:
    Each tool call is hashed (name + args) and recorded in a circular
    buffer. Detection runs before each tool call to identify patterns:

    generic_repeat:        A → A → A → A  (same call repeated)
    known_poll_no_progress: poll → result → poll → result (no change)
    ping_pong:             A → B → A → B  (alternating pair)
    global_circuit_breaker: total calls > threshold
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

# Buffer and threshold constants
TOOL_CALL_HISTORY_SIZE = 30
WARNING_THRESHOLD = 10
CRITICAL_THRESHOLD = 20
GLOBAL_CIRCUIT_BREAKER_THRESHOLD = 30

# Known polling tool patterns (tool names that commonly poll)
KNOWN_POLL_TOOLS = frozenset({
    "bash",
    "terminal_run",
    "terminal_read",
    "terminal_send",
})


class LoopLevel(str, Enum):
    """Severity level of a detected loop."""

    NONE = "none"
    WARNING = "warning"
    CRITICAL = "critical"
    CIRCUIT_BREAK = "circuit_break"


class LoopDetectorKind(str, Enum):
    """Type of loop detector that triggered."""

    GENERIC_REPEAT = "generic_repeat"
    KNOWN_POLL_NO_PROGRESS = "known_poll_no_progress"
    GLOBAL_CIRCUIT_BREAKER = "global_circuit_breaker"
    PING_PONG = "ping_pong"


@dataclass
class LoopDetectionResult:
    """Result from loop detection check."""

    stuck: bool = False
    level: LoopLevel = LoopLevel.NONE
    detector: LoopDetectorKind | None = None
    count: int = 0
    message: str = ""


@dataclass
class ToolCallRecord:
    """A recorded tool call for pattern detection."""

    tool_name: str
    call_hash: str
    tool_call_id: str = ""
    output_hash: str = ""


@dataclass
class ToolLoopState:
    """Mutable state for tool loop tracking within an agent run."""

    history: list[ToolCallRecord] = field(default_factory=list)
    total_calls: int = 0
    consecutive_identical: int = 0
    last_hash: str = ""


@dataclass
class ToolLoopDetectionConfig:
    """Configuration for loop detection thresholds."""

    warning_threshold: int = WARNING_THRESHOLD
    critical_threshold: int = CRITICAL_THRESHOLD
    circuit_breaker_threshold: int = GLOBAL_CIRCUIT_BREAKER_THRESHOLD
    history_size: int = TOOL_CALL_HISTORY_SIZE
    enabled: bool = True


def hash_tool_call(tool_name: str, params: Any) -> str:
    """Create a stable hash of a tool call (name + args).

    This allows detecting identical repeated calls regardless of
    argument ordering.
    """
    try:
        params_str = json.dumps(params, sort_keys=True, default=str)
    except (TypeError, ValueError):
        params_str = str(params)

    raw = f"{tool_name}:{params_str}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def record_tool_call(
    state: ToolLoopState,
    tool_name: str,
    params: Any,
    tool_call_id: str = "",
    config: ToolLoopDetectionConfig | None = None,
) -> None:
    """Record a tool call in the history buffer."""
    cfg = config or ToolLoopDetectionConfig()

    call_hash = hash_tool_call(tool_name, params)
    record = ToolCallRecord(
        tool_name=tool_name,
        call_hash=call_hash,
        tool_call_id=tool_call_id,
    )

    state.history.append(record)
    state.total_calls += 1

    # Trim history to buffer size
    if len(state.history) > cfg.history_size:
        state.history = state.history[-cfg.history_size :]

    # Track consecutive identical calls
    if call_hash == state.last_hash:
        state.consecutive_identical += 1
    else:
        state.consecutive_identical = 1
        state.last_hash = call_hash


def record_tool_call_outcome(
    state: ToolLoopState,
    tool_call_id: str,
    output: str,
) -> None:
    """Record the output of a tool call for no-progress detection."""
    output_hash = hashlib.md5(output.encode()).hexdigest()[:12]

    for record in reversed(state.history):
        if record.tool_call_id == tool_call_id:
            record.output_hash = output_hash
            break


def detect_tool_call_loop(
    state: ToolLoopState,
    tool_name: str,
    params: Any,
    config: ToolLoopDetectionConfig | None = None,
) -> LoopDetectionResult:
    """Detect if the agent is stuck in a tool call loop.

    Runs multiple detectors and returns the most severe result.

    Args:
        state: Current tool loop tracking state.
        tool_name: Name of the tool about to be called.
        params: Arguments for the tool call.
        config: Optional detection configuration.

    Returns:
        LoopDetectionResult indicating if/how the agent is stuck.
    """
    cfg = config or ToolLoopDetectionConfig()

    if not cfg.enabled:
        return LoopDetectionResult()

    call_hash = hash_tool_call(tool_name, params)
    results: list[LoopDetectionResult] = []

    # 1. Global circuit breaker
    if state.total_calls >= cfg.circuit_breaker_threshold:
        results.append(LoopDetectionResult(
            stuck=True,
            level=LoopLevel.CIRCUIT_BREAK,
            detector=LoopDetectorKind.GLOBAL_CIRCUIT_BREAKER,
            count=state.total_calls,
            message=(
                f"Global circuit breaker: {state.total_calls} tool calls "
                f"exceeds threshold of {cfg.circuit_breaker_threshold}."
            ),
        ))

    # 2. Generic repeat detection
    consecutive = state.consecutive_identical if call_hash == state.last_hash else 0
    # Add 1 because this call hasn't been recorded yet
    if call_hash == state.last_hash:
        consecutive += 1

    if consecutive >= cfg.critical_threshold:
        results.append(LoopDetectionResult(
            stuck=True,
            level=LoopLevel.CRITICAL,
            detector=LoopDetectorKind.GENERIC_REPEAT,
            count=consecutive,
            message=(
                f"Tool '{tool_name}' called {consecutive} times consecutively "
                f"with identical arguments. Agent appears stuck."
            ),
        ))
    elif consecutive >= cfg.warning_threshold:
        results.append(LoopDetectionResult(
            stuck=False,
            level=LoopLevel.WARNING,
            detector=LoopDetectorKind.GENERIC_REPEAT,
            count=consecutive,
            message=(
                f"Tool '{tool_name}' called {consecutive} times consecutively "
                f"with identical arguments. Consider a different approach."
            ),
        ))

    # 3. Known poll no-progress detection
    if tool_name in KNOWN_POLL_TOOLS and len(state.history) >= 4:
        recent = state.history[-4:]
        same_tool = all(r.tool_name == tool_name for r in recent)
        same_output = (
            len({r.output_hash for r in recent if r.output_hash}) <= 1
            and any(r.output_hash for r in recent)
        )
        if same_tool and same_output:
            results.append(LoopDetectionResult(
                stuck=True,
                level=LoopLevel.WARNING,
                detector=LoopDetectorKind.KNOWN_POLL_NO_PROGRESS,
                count=4,
                message=(
                    f"Tool '{tool_name}' polled 4+ times with identical output. "
                    f"No progress detected — consider waiting or trying a different approach."
                ),
            ))

    # 4. Ping-pong detection (A → B → A → B)
    if len(state.history) >= 4:
        h = state.history
        last_four_hashes = [r.call_hash for r in h[-4:]]
        if (
            last_four_hashes[0] == last_four_hashes[2]
            and last_four_hashes[1] == last_four_hashes[3]
            and last_four_hashes[0] != last_four_hashes[1]
            and call_hash == last_four_hashes[0]
        ):
            results.append(LoopDetectionResult(
                stuck=True,
                level=LoopLevel.WARNING,
                detector=LoopDetectorKind.PING_PONG,
                count=4,
                message=(
                    f"Ping-pong pattern detected: alternating between "
                    f"'{h[-4].tool_name}' and '{h[-3].tool_name}'. "
                    f"Agent may be stuck in an oscillating loop."
                ),
            ))

    if not results:
        return LoopDetectionResult()

    # Return most severe
    severity_order = {
        LoopLevel.CIRCUIT_BREAK: 3,
        LoopLevel.CRITICAL: 2,
        LoopLevel.WARNING: 1,
        LoopLevel.NONE: 0,
    }
    return max(results, key=lambda r: severity_order.get(r.level, 0))


def get_tool_call_stats(state: ToolLoopState) -> dict[str, Any]:
    """Get statistics about tool call patterns for debugging."""
    from collections import Counter

    tool_counts = Counter(r.tool_name for r in state.history)
    hash_counts = Counter(r.call_hash for r in state.history)
    most_frequent_hash = hash_counts.most_common(1)

    return {
        "total_calls": state.total_calls,
        "unique_patterns": len(hash_counts),
        "tool_distribution": dict(tool_counts),
        "most_frequent": {
            "hash": most_frequent_hash[0][0] if most_frequent_hash else "",
            "count": most_frequent_hash[0][1] if most_frequent_hash else 0,
        },
        "consecutive_identical": state.consecutive_identical,
    }


def format_loop_warning(result: LoopDetectionResult) -> str:
    """Format a loop detection result as a warning message for the LLM.

    This message gets injected into the conversation to help the agent
    break out of the loop.
    """
    if not result.stuck and result.level == LoopLevel.NONE:
        return ""

    parts = [f"⚠️ **Loop Detection ({result.level.value})**"]
    parts.append(result.message)

    if result.level == LoopLevel.CIRCUIT_BREAK:
        parts.append(
            "**Action required:** Stop making tool calls and provide your "
            "best response with the information gathered so far."
        )
    elif result.level == LoopLevel.CRITICAL:
        parts.append(
            "**Suggestion:** Try a completely different approach or tool. "
            "If you cannot make progress, explain what's blocking you."
        )
    elif result.level == LoopLevel.WARNING:
        parts.append(
            "**Suggestion:** Consider whether you're making progress. "
            "Try a different approach if the current one isn't working."
        )

    return "\n".join(parts)
