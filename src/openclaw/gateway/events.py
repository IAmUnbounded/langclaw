"""Gateway event types — mirroring OpenClaw's event streaming system.

Event types for real-time streaming to WebSocket clients:
- lifecycle: start/end/error phases
- assistant: streamed text deltas
- tool: tool start/end with args/output
"""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field


class GatewayEvent(BaseModel):
    """Base event model for gateway streaming."""

    type: str
    timestamp: float = Field(default_factory=time.time)
    run_id: str = ""
    data: dict[str, Any] = Field(default_factory=dict)


class LifecycleEvent(GatewayEvent):
    """Lifecycle event: start, end, or error."""

    type: str = "lifecycle"
    phase: str = "start"  # start | end | error
    error: str | None = None


class AssistantDeltaEvent(GatewayEvent):
    """Streamed text delta from the assistant."""

    type: str = "assistant_delta"
    content: str = ""


class AssistantMessageEvent(GatewayEvent):
    """Complete assistant message."""

    type: str = "assistant_message"
    content: str = ""


class ToolStartEvent(GatewayEvent):
    """Tool execution started."""

    type: str = "tool_start"
    tool_name: str = ""
    tool_args: dict[str, Any] = Field(default_factory=dict)


class ToolEndEvent(GatewayEvent):
    """Tool execution completed."""

    type: str = "tool_end"
    tool_name: str = ""
    output: str = ""


def event_to_dict(event: dict[str, Any]) -> dict[str, Any]:
    """Convert an agent stream event to a gateway event dict."""
    event_type = event.get("type", "")
    base = {
        "timestamp": time.time(),
        "run_id": event.get("run_id", ""),
    }

    if event_type == "lifecycle":
        return {
            **base,
            "type": "lifecycle",
            "phase": event.get("phase", ""),
            "error": event.get("error"),
        }
    elif event_type == "tool_start":
        return {
            **base,
            "type": "tool_start",
            "tool_name": event.get("name", ""),
            "tool_args": event.get("args", {}),
        }
    elif event_type == "tool_end":
        return {
            **base,
            "type": "tool_end",
            "tool_name": event.get("name", ""),
            "output": event.get("output", ""),
        }
    elif event_type == "assistant_delta":
        return {
            **base,
            "type": "assistant_delta",
            "content": event.get("content", ""),
        }
    elif event_type == "assistant_message":
        return {
            **base,
            "type": "assistant_message",
            "content": event.get("content", ""),
        }
    else:
        return {**base, "type": event_type, **event}
