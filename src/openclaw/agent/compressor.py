"""Context compressor — smart context window management.

Handles token counting, conversation summarization, and intelligent
truncation to keep the context window within budget.

Mirrors OpenClaw's compact.ts / compaction.ts:
- Detect context overflow before/after LLM calls
- Multi-stage LLM-based summarization (with heuristic fallback)
- Adaptive chunk ratio based on average message size
- Tool-use/result pairing repair after chunk dropping
- Truncate long tool outputs
- Keep system prompt + recent messages intact
- Model catalog integration for accurate context window limits
- Configurable keep_recent parameter
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
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

# Tool output truncation limit
TOOL_OUTPUT_MAX_CHARS = 5000

# Reserve tokens for the LLM response
RESERVE_TOKENS = 4096

# Base chunk ratio for multi-stage summarization
BASE_CHUNK_RATIO = 0.4
MIN_CHUNK_RATIO = 0.15

# Safety margin for token estimation inaccuracy
ESTIMATION_SAFETY_MARGIN = 1.2


def estimate_tokens(text: str) -> int:
    """Estimate token count from text length.

    Uses tiktoken for accurate counts if available,
    otherwise falls back to ~4 chars per token.
    """
    try:
        import tiktoken
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
        # Add overhead for message structure (role, separators, tool_calls metadata)
        total += 4
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                total += estimate_tokens(str(tc.get("args", {})))
    return total


def _truncate_tool_outputs(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Step 1: Truncate long tool outputs to reduce token count."""
    result = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if len(content) > TOOL_OUTPUT_MAX_CHARS:
                truncated = content[:TOOL_OUTPUT_MAX_CHARS - 500] + "\n\n... (output truncated for context)"
                result.append(ToolMessage(
                    content=truncated,
                    tool_call_id=msg.tool_call_id,
                    name=getattr(msg, 'name', ''),
                ))
                continue
        result.append(msg)
    return result


def _summarize_dropped_messages(dropped: list[BaseMessage]) -> str:
    """Create a meaningful summary of dropped messages.

    Extracts key information:
    - User questions/requests
    - Tool calls made and their outcomes
    - Key decisions or findings
    """
    parts = []
    user_messages = []
    tool_calls = []
    assistant_snippets = []

    for msg in dropped:
        if isinstance(msg, HumanMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            user_messages.append(content[:100])
        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append(tc.get("name", "unknown"))
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if content and len(content) > 20:
                assistant_snippets.append(content[:80])
        elif isinstance(msg, ToolMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            name = getattr(msg, 'name', 'tool')
            # Capture first line of tool output as key fact
            first_line = content.split('\n')[0][:80]
            if first_line and not first_line.startswith("[error]"):
                parts.append(f"  - {name}: {first_line}")

    summary_lines = [
        f"[Context compacted: {len(dropped)} earlier messages were summarized]",
    ]

    if user_messages:
        summary_lines.append(f"User asked about: {'; '.join(user_messages[:5])}")

    if tool_calls:
        # Count tool usage
        from collections import Counter
        counts = Counter(tool_calls)
        tool_summary = ", ".join(f"{name}({n}x)" if n > 1 else name for name, n in counts.most_common(8))
        summary_lines.append(f"Tools used: {tool_summary}")

    if assistant_snippets:
        summary_lines.append(f"Key findings: {'; '.join(assistant_snippets[:3])}")

    if parts:
        summary_lines.append("Notable results:")
        summary_lines.extend(parts[:5])

    return "\n".join(summary_lines)


def compress_messages(
    messages: list[BaseMessage],
    max_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
    keep_recent: int = 20,
) -> list[BaseMessage]:
    """Compress conversation history to fit within token budget.

    Strategy:
    1. Always keep the system message (first message if SystemMessage)
    2. Truncate long tool outputs (>3000 chars)
    3. Always keep the most recent `keep_recent` messages
    4. Summarize the dropped middle section with key topics/actions
    5. If still over budget, progressively reduce keep_recent

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
    effective_budget = max_tokens - RESERVE_TOKENS

    if current_tokens <= effective_budget:
        return messages

    logger.info(
        f"Context compression: {current_tokens} tokens > budget {effective_budget}"
    )

    # Step 1: Truncate long tool outputs
    compressed = _truncate_tool_outputs(messages)
    current_tokens = estimate_messages_tokens(compressed)
    if current_tokens <= effective_budget:
        return compressed

    # Step 1.5: Apply per-tool summarization (smarter than truncation)
    try:
        from openclaw.agent.tool_summaries import summarize_tool_messages
        compressed = summarize_tool_messages(compressed)
        current_tokens = estimate_messages_tokens(compressed)
        if current_tokens <= effective_budget:
            return compressed
    except ImportError:
        pass

    # Step 2: Separate system messages from conversation
    system_msgs: list[BaseMessage] = []
    conversation: list[BaseMessage] = []

    for msg in compressed:
        if isinstance(msg, SystemMessage):
            system_msgs.append(msg)
        else:
            conversation.append(msg)

    if len(conversation) <= keep_recent:
        # Can't compress further without losing recent context
        return compressed

    # Step 3: Progressive compression — reduce keep_recent if still over budget
    for attempt_keep in range(keep_recent, 4, -4):
        recent = conversation[-attempt_keep:]
        dropped = conversation[:-attempt_keep]

        if not dropped:
            continue

        summary_text = _summarize_dropped_messages(dropped)
        summary_msg = HumanMessage(content=summary_text)

        candidate = system_msgs + [summary_msg] + recent
        candidate_tokens = estimate_messages_tokens(candidate)

        if candidate_tokens <= effective_budget:
            logger.info(
                f"Context compressed: {current_tokens} → {candidate_tokens} tokens "
                f"(dropped {len(dropped)} messages, kept {attempt_keep} recent)"
            )
            return candidate

    # Step 4: Last resort — keep only system + summary + last 4 messages
    recent = conversation[-4:]
    dropped = conversation[:-4]
    summary_text = _summarize_dropped_messages(dropped)
    result = system_msgs + [HumanMessage(content=summary_text)] + recent

    new_tokens = estimate_messages_tokens(result)
    logger.warning(
        f"Aggressive compression: {current_tokens} → {new_tokens} tokens "
        f"(kept only 4 recent messages)"
    )
    return result


def needs_compression(messages: list[BaseMessage], max_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS) -> bool:
    """Check if messages need compression without actually compressing."""
    return estimate_messages_tokens(messages) > (max_tokens - RESERVE_TOKENS)


def repair_tool_use_result_pairing(messages: list[BaseMessage]) -> tuple[list[BaseMessage], int]:
    """Remove orphaned ToolMessages whose AIMessage tool_call was dropped.

    When we drop the first chunk of messages, an AI message with tool_calls
    might get dropped while its corresponding ToolMessages remain. These
    orphaned ToolMessages cause API errors ("unexpected tool_use_id").

    Returns:
        (repaired_messages, orphan_count)
    """
    # Collect all tool_call IDs that have corresponding AIMessage entries
    valid_tool_call_ids: set[str] = set()
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                tc_id = tc.get("id", "")
                if tc_id:
                    valid_tool_call_ids.add(tc_id)

    # Filter out ToolMessages with no corresponding tool_call
    result: list[BaseMessage] = []
    orphan_count = 0
    for msg in messages:
        if isinstance(msg, ToolMessage):
            tc_id = getattr(msg, "tool_call_id", "") or ""
            if tc_id and tc_id not in valid_tool_call_ids:
                orphan_count += 1
                continue
        result.append(msg)

    if orphan_count > 0:
        logger.debug(f"Repaired {orphan_count} orphaned tool_result messages")

    return result, orphan_count


def compute_adaptive_chunk_ratio(messages: list[BaseMessage], context_window: int) -> float:
    """Compute adaptive chunk ratio based on average message size.

    When messages are large on average, use smaller chunks to avoid
    exceeding model limits during summarization.

    Mirrors openclaw's computeAdaptiveChunkRatio.
    """
    if not messages:
        return BASE_CHUNK_RATIO

    total = estimate_messages_tokens(messages)
    avg = total / len(messages)
    safe_avg = avg * ESTIMATION_SAFETY_MARGIN
    avg_ratio = safe_avg / max(1, context_window)

    if avg_ratio > 0.1:
        reduction = min(avg_ratio * 2, BASE_CHUNK_RATIO - MIN_CHUNK_RATIO)
        return max(MIN_CHUNK_RATIO, BASE_CHUNK_RATIO - reduction)

    return BASE_CHUNK_RATIO


def split_messages_by_token_share(
    messages: list[BaseMessage],
    parts: int = 2,
) -> list[list[BaseMessage]]:
    """Split messages into N roughly equal token-sized chunks.

    Mirrors openclaw's splitMessagesByTokenShare.
    """
    if not messages or parts <= 1:
        return [messages]

    parts = max(1, min(parts, len(messages)))
    total = estimate_messages_tokens(messages)
    target = total / parts
    chunks: list[list[BaseMessage]] = []
    current: list[BaseMessage] = []
    current_tokens = 0

    for msg in messages:
        msg_tokens = estimate_messages_tokens([msg])
        if chunks.__len__() < parts - 1 and current and current_tokens + msg_tokens > target:
            chunks.append(current)
            current = []
            current_tokens = 0
        current.append(msg)
        current_tokens += msg_tokens

    if current:
        chunks.append(current)

    return chunks


async def _llm_summarize_messages(
    messages: list[BaseMessage],
    model_str: str,
    api_key: str = "",
    custom_instructions: str = "",
    previous_summary: str = "",
) -> str:
    """Use an LLM to summarize a chunk of messages.

    This mirrors openclaw's generateSummary / summarizeChunks approach.
    Falls back to heuristic summary on failure.
    """
    try:
        from openclaw.agent.model_catalog import lookup_model

        # Build summarization prompt
        conversation_text = []
        for msg in messages:
            role = msg.type
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            # Truncate very long tool outputs
            if isinstance(msg, ToolMessage) and len(content) > 1000:
                content = content[:1000] + "... (truncated)"
            conversation_text.append(f"[{role}]: {content[:500]}")

        history_str = "\n".join(conversation_text)

        system = "You are a conversation summarizer. Create a concise, factual summary."
        if custom_instructions:
            system += f"\n\nFocus: {custom_instructions}"
        if previous_summary:
            system += f"\n\nPrevious summary:\n{previous_summary}"

        prompt = (
            f"Summarize the following conversation, preserving:\n"
            f"- Key decisions and conclusions\n"
            f"- Open questions and TODOs\n"
            f"- Important facts and context\n"
            f"- Actions taken\n\n"
            f"Conversation:\n{history_str}\n\n"
            f"Provide a concise summary in 3-10 sentences:"
        )

        # Use a fast/cheap model for summarization
        summary_model = "openai/gpt-4o-mini"  # Default summarizer
        if model_str.startswith("anthropic"):
            summary_model = "anthropic/claude-haiku-4-5"
        elif model_str.startswith("google"):
            summary_model = "google/gemini-2.0-flash"

        provider = summary_model.split("/")[0]
        model_name = summary_model.split("/", 1)[1]

        import os
        if provider == "openai":
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model=model_name,
                temperature=0,
                api_key=api_key or os.environ.get("OPENAI_API_KEY") or None,
            )
        elif provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            llm = ChatAnthropic(
                model=model_name,
                temperature=0,
                api_key=api_key or os.environ.get("ANTHROPIC_API_KEY") or None,
            )
        elif provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=0,
                google_api_key=api_key or os.environ.get("GOOGLE_API_KEY") or None,
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")

        from langchain_core.messages import SystemMessage, HumanMessage
        response = await llm.ainvoke([
            SystemMessage(content=system),
            HumanMessage(content=prompt),
        ])

        summary = response.content if isinstance(response.content, str) else str(response.content)
        logger.info(f"LLM summarization: {len(messages)} messages → {len(summary)} chars")
        return summary

    except Exception as e:
        logger.warning(f"LLM summarization failed: {e}, falling back to heuristic")
        return _summarize_dropped_messages(messages)


async def compress_messages_with_llm(
    messages: list[BaseMessage],
    max_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
    keep_recent: int = 20,
    model_str: str = "",
    api_key: str = "",
    custom_instructions: str = "",
) -> list[BaseMessage]:
    """Compress conversation history using LLM-based summarization.

    Multi-stage approach:
    1. Truncate long tool outputs
    2. Apply per-tool summarization
    3. Drop oldest chunks, summarize them with LLM
    4. Repair tool_use/tool_result pairing after drops
    5. Progressive fallback to heuristic summary

    Mirrors openclaw's summarizeInStages / pruneHistoryForContextShare.
    """
    if not messages:
        return messages

    effective_budget = max_tokens - RESERVE_TOKENS
    current_tokens = estimate_messages_tokens(messages)

    if current_tokens <= effective_budget:
        return messages

    logger.info(f"LLM compaction: {current_tokens:,} → budget {effective_budget:,}")

    # Step 1: Truncate tool outputs
    messages = _truncate_tool_outputs(messages)

    # Step 2: Per-tool smart summarization
    try:
        from openclaw.agent.tool_summaries import summarize_tool_messages
        messages = summarize_tool_messages(messages)
    except ImportError:
        pass

    if estimate_messages_tokens(messages) <= effective_budget:
        return messages

    # Step 3: Separate system messages
    system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
    conversation = [m for m in messages if not isinstance(m, SystemMessage)]

    if len(conversation) <= keep_recent:
        return messages

    # Step 4: Multi-stage drop + LLM summarize
    previous_summary = ""
    while len(conversation) > keep_recent:
        chunks = split_messages_by_token_share(conversation, parts=2)
        if len(chunks) <= 1:
            break

        to_drop, remaining = chunks[0], chunks[1:]
        flat_remaining = [m for chunk in remaining for m in chunk]

        # Repair orphaned tool results
        flat_remaining, _ = repair_tool_use_result_pairing(flat_remaining)

        # LLM summarize dropped chunk
        try:
            summary_text = await _llm_summarize_messages(
                to_drop, model_str, api_key, custom_instructions, previous_summary
            )
            previous_summary = summary_text
        except Exception as e:
            logger.warning(f"Summarization failed: {e}")
            summary_text = _summarize_dropped_messages(to_drop)

        summary_msg = HumanMessage(
            content=f"[Context summary: earlier conversation]\n{summary_text}"
        )

        candidate = system_msgs + [summary_msg] + flat_remaining
        candidate_tokens = estimate_messages_tokens(candidate)

        if candidate_tokens <= effective_budget:
            logger.info(f"LLM compaction done: {candidate_tokens:,} tokens")
            return candidate

        # Still over budget — reduce the remaining messages
        conversation = flat_remaining

    # Fallback: heuristic compression
    logger.warning("LLM compaction insufficient, falling back to heuristic")
    return compress_messages(
        system_msgs + conversation,
        max_tokens=max_tokens,
        keep_recent=min(keep_recent, 8),
    )
