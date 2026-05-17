"""Context window guard — proactive context budget management.

Mirrors openclaw's context-window-guard.ts:
- Check token budget before LLM calls
- Determine if messages need compression
- Compute per-component token budgets (system, history, reserve)
- Detect when messages are approaching the limit (early warning)
- Integrate with model catalog for accurate limits
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, HumanMessage

from openclaw.agent.compressor import estimate_tokens, estimate_messages_tokens
from openclaw.agent.model_catalog import lookup_context_window, get_effective_context_budget

logger = logging.getLogger(__name__)

# How many tokens to reserve for the LLM's response
RESPONSE_RESERVE_TOKENS = 4096

# Warning threshold — start warning when usage exceeds this fraction
WARNING_THRESHOLD = 0.80

# Critical threshold — must compress before this is reached
CRITICAL_THRESHOLD = 0.90


@dataclass
class ContextBudget:
    """Token budget breakdown for a model call."""

    total_context_window: int
    system_tokens: int
    history_tokens: int
    response_reserve: int

    @property
    def used_tokens(self) -> int:
        return self.system_tokens + self.history_tokens

    @property
    def available_for_history(self) -> int:
        return max(0, self.total_context_window - self.system_tokens - self.response_reserve)

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.total_context_window - self.used_tokens - self.response_reserve)

    @property
    def usage_fraction(self) -> float:
        if self.total_context_window <= 0:
            return 1.0
        return (self.used_tokens + self.response_reserve) / self.total_context_window

    @property
    def needs_warning(self) -> bool:
        return self.usage_fraction >= WARNING_THRESHOLD

    @property
    def needs_compression(self) -> bool:
        return self.usage_fraction >= CRITICAL_THRESHOLD

    @property
    def is_overflow(self) -> bool:
        return self.used_tokens + self.response_reserve > self.total_context_window

    def describe(self) -> str:
        pct = self.usage_fraction * 100
        return (
            f"Context: {self.used_tokens:,}/{self.total_context_window:,} tokens "
            f"({pct:.0f}%) — "
            f"system={self.system_tokens:,}, history={self.history_tokens:,}, "
            f"reserve={self.response_reserve:,}"
        )


def compute_context_budget(
    messages: list[BaseMessage],
    model_str: str,
    system_prompt: str = "",
    response_reserve: int = RESPONSE_RESERVE_TOKENS,
) -> ContextBudget:
    """Compute the token budget for the current context.

    Args:
        messages: Conversation history.
        model_str: Model identifier string.
        system_prompt: System prompt text.
        response_reserve: Tokens to reserve for the model's response.

    Returns:
        ContextBudget with usage breakdown.
    """
    context_window = lookup_context_window(model_str)

    # Estimate system tokens
    sys_tokens = estimate_tokens(system_prompt) if system_prompt else 0
    # Add overhead for system message wrapper
    sys_tokens += 10

    # Estimate history tokens
    history_tokens = estimate_messages_tokens(messages)

    return ContextBudget(
        total_context_window=context_window,
        system_tokens=sys_tokens,
        history_tokens=history_tokens,
        response_reserve=response_reserve,
    )


def check_context_overflow(
    messages: list[BaseMessage],
    model_str: str,
    system_prompt: str = "",
) -> bool:
    """Quick check: does the current context exceed the model's limit?"""
    budget = compute_context_budget(messages, model_str, system_prompt)
    return budget.is_overflow


def get_history_budget(
    model_str: str,
    system_prompt: str = "",
    response_reserve: int = RESPONSE_RESERVE_TOKENS,
    max_history_share: float = 0.5,
) -> int:
    """Get the maximum tokens available for conversation history.

    Args:
        model_str: Model identifier.
        system_prompt: System prompt text (affects remaining budget).
        response_reserve: Tokens reserved for response.
        max_history_share: Maximum fraction of context window for history.

    Returns:
        Token budget for history in tokens.
    """
    context_window = lookup_context_window(model_str)
    sys_tokens = estimate_tokens(system_prompt) + 10 if system_prompt else 0

    # History budget = total - system - reserve
    raw_budget = context_window - sys_tokens - response_reserve

    # Apply max share limit
    max_by_share = int(context_window * max_history_share)

    return max(0, min(raw_budget, max_by_share))


def log_context_status(
    messages: list[BaseMessage],
    model_str: str,
    system_prompt: str = "",
    node_name: str = "",
) -> ContextBudget:
    """Log context window status and return the budget.

    Call this before LLM calls to get visibility into context usage.
    """
    budget = compute_context_budget(messages, model_str, system_prompt)

    if budget.is_overflow:
        logger.error(
            f"[{node_name or 'context'}] OVERFLOW: {budget.describe()}"
        )
    elif budget.needs_compression:
        logger.warning(
            f"[{node_name or 'context'}] Critical: {budget.describe()}"
        )
    elif budget.needs_warning:
        logger.info(
            f"[{node_name or 'context'}] Warning: {budget.describe()}"
        )
    else:
        logger.debug(
            f"[{node_name or 'context'}] OK: {budget.describe()}"
        )

    return budget


def select_messages_within_budget(
    messages: list[BaseMessage],
    token_budget: int,
    keep_first_n: int = 0,
) -> list[BaseMessage]:
    """Select messages that fit within a token budget, keeping the most recent.

    Args:
        messages: Full message list.
        token_budget: Maximum tokens for selected messages.
        keep_first_n: Always keep the first N messages (e.g., system prompt injections).

    Returns:
        Subset of messages fitting the budget.
    """
    if not messages:
        return messages

    if estimate_messages_tokens(messages) <= token_budget:
        return messages

    # Always keep the first N messages
    pinned = messages[:keep_first_n] if keep_first_n > 0 else []
    pinned_tokens = estimate_messages_tokens(pinned)
    remaining_budget = token_budget - pinned_tokens

    if remaining_budget <= 0:
        return pinned

    # Add messages from the end (most recent first) until budget is exhausted
    selected: list[BaseMessage] = []
    selected_tokens = 0

    for msg in reversed(messages[keep_first_n:]):
        msg_tokens = estimate_messages_tokens([msg])
        if selected_tokens + msg_tokens > remaining_budget:
            break
        selected.insert(0, msg)
        selected_tokens += msg_tokens

    return pinned + selected
