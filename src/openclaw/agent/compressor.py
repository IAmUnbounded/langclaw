"""Context compressor — smart context window management.

Handles token counting, conversation summarization, and intelligent
truncation to keep the context window within budget.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

logger = logging.getLogger(__name__)

# Rough token estimates (chars per token varies by model)
CHARS_PER_TOKEN = 4

# Default max context tokens (can be overridden in config)
DEFAULT_MAX_CONTEXT_TOKENS = 100_000


def estimate_tokens(text: str) -> int:
    """Estimate token count from text length.

    Uses a rough approximation of ~4 chars per token.
    For more accurate counting, install tiktoken.
    """
    try:
        import tiktoken

        # Try to use tiktoken for accurate counts
        enc = tiktoken.encoding_for_model("gpt-4o")
        return len(enc.encode(text))
    except (ImportError, Exception):
        return len(text) // CHARS_PER_TOKEN


def estimate_messages_tokens(messages: list[BaseMessage]) -> int:
    """Estimate total tokens across all messages."""
    total = 0
    for msg in messages:
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        total += estimate_tokens(content)
        # Add overhead for message structure
        total += 4  # role, separators, etc.
    return total


def compress_messages(
    messages: list[BaseMessage],
    max_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
    keep_recent: int = 20,
) -> list[BaseMessage]:
    """Compress conversation history to fit within token budget.

    Strategy:
    1. Always keep the system message (first message if SystemMessage)
    2. Always keep the most recent `keep_recent` messages
    3. If still over budget, summarize the middle section
    4. Truncate tool outputs that are too long

    Args:
        messages: Full conversation history.
        max_tokens: Token budget.
        keep_recent: Number of recent messages to always keep.

    Returns:
        Compressed message list.
    """
    if not messages:
        return messages

    current_tokens = estimate_messages_tokens(messages)
    if current_tokens <= max_tokens:
        return messages

    logger.info(
        f"Context compression: {current_tokens} tokens → target {max_tokens}"
    )

    # Step 1: Truncate long tool outputs
    compressed = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if len(content) > 3000:
                truncated = content[:2500] + "\n\n... (output truncated for context)"
                compressed.append(ToolMessage(
                    content=truncated,
                    tool_call_id=msg.tool_call_id,
                    name=getattr(msg, 'name', ''),
                ))
                continue
        compressed.append(msg)

    current_tokens = estimate_messages_tokens(compressed)
    if current_tokens <= max_tokens:
        return compressed

    # Step 2: Keep system message + recent messages, summarize middle
    system_msgs = []
    conversation = []

    for msg in compressed:
        if isinstance(msg, SystemMessage):
            system_msgs.append(msg)
        else:
            conversation.append(msg)

    if len(conversation) <= keep_recent:
        return compressed

    # Keep only the most recent messages
    recent = conversation[-keep_recent:]

    # Summarize the dropped messages
    dropped = conversation[:-keep_recent]
    dropped_count = len(dropped)
    tool_calls_count = sum(
        1 for m in dropped
        if isinstance(m, AIMessage) and m.tool_calls
    )

    summary = (
        f"[Conversation summary: {dropped_count} earlier messages were compressed. "
        f"Included {tool_calls_count} tool calls. "
        f"The conversation has been ongoing.]"
    )

    result = system_msgs + [HumanMessage(content=summary)] + recent

    new_tokens = estimate_messages_tokens(result)
    logger.info(
        f"Context compressed: {current_tokens} → {new_tokens} tokens "
        f"(dropped {dropped_count} messages)"
    )

    return result
