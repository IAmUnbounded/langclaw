"""Command poll backoff — exponential backoff for polling operations.

Mirrors openclaw's command-poll-backoff.ts and cli-runner/reliability.ts:
- Exponential backoff with jitter for polling operations
- Watchdog timer for stuck processes
- Configurable min/max delay
- Retry budget with abort on exhaustion
- Integration with tool loop detection for smart retry decisions
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Default backoff parameters
DEFAULT_MIN_DELAY_MS = 200
DEFAULT_MAX_DELAY_MS = 10_000
DEFAULT_JITTER_FACTOR = 0.3
DEFAULT_MAX_ATTEMPTS = 10
DEFAULT_WATCHDOG_TIMEOUT_MS = 120_000  # 2 minutes


@dataclass
class BackoffConfig:
    """Configuration for exponential backoff."""

    min_delay_ms: int = DEFAULT_MIN_DELAY_MS
    max_delay_ms: int = DEFAULT_MAX_DELAY_MS
    jitter_factor: float = DEFAULT_JITTER_FACTOR
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    multiplier: float = 2.0
    watchdog_timeout_ms: int = DEFAULT_WATCHDOG_TIMEOUT_MS

    def compute_delay(self, attempt: int) -> float:
        """Compute delay in seconds for the given attempt number (0-indexed)."""
        delay_ms = self.min_delay_ms * (self.multiplier ** attempt)
        delay_ms = min(delay_ms, self.max_delay_ms)

        # Add jitter to prevent thundering herd
        if self.jitter_factor > 0:
            jitter = delay_ms * self.jitter_factor * (random.random() * 2 - 1)
            delay_ms = max(self.min_delay_ms, delay_ms + jitter)

        return delay_ms / 1000.0  # Convert to seconds


@dataclass
class RetryState:
    """Mutable state for tracking retry progress."""

    attempt: int = 0
    total_delay_ms: float = 0.0
    last_error: Exception | None = None
    start_time: float = field(default_factory=time.time)

    @property
    def elapsed_ms(self) -> float:
        return (time.time() - self.start_time) * 1000


class RetryExhausted(Exception):
    """Raised when all retry attempts are exhausted."""

    def __init__(self, attempts: int, last_error: Exception | None = None):
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(
            f"Retry exhausted after {attempts} attempts"
            + (f": {last_error}" if last_error else "")
        )


class WatchdogTimeout(Exception):
    """Raised when the watchdog timer fires."""

    def __init__(self, timeout_ms: int, elapsed_ms: float):
        self.timeout_ms = timeout_ms
        self.elapsed_ms = elapsed_ms
        super().__init__(
            f"Watchdog timeout after {elapsed_ms:.0f}ms (limit {timeout_ms}ms)"
        )


async def retry_async(
    fn: Callable,
    config: BackoffConfig | None = None,
    label: str = "operation",
    should_retry: Callable[[Exception], bool] | None = None,
) -> Any:
    """Execute an async function with exponential backoff retry.

    Args:
        fn: Async callable to retry.
        config: Backoff configuration.
        label: Operation label for logging.
        should_retry: Optional predicate to decide if an error is retryable.
                     Default: retry on all exceptions except asyncio.CancelledError.

    Returns:
        The result of fn() on success.

    Raises:
        RetryExhausted: If all attempts fail.
        asyncio.CancelledError: If the operation is cancelled.
    """
    cfg = config or BackoffConfig()
    state = RetryState()

    default_retry = lambda e: not isinstance(e, (asyncio.CancelledError, KeyboardInterrupt))
    retry_check = should_retry or default_retry

    while state.attempt < cfg.max_attempts:
        # Check watchdog
        if state.elapsed_ms > cfg.watchdog_timeout_ms:
            raise WatchdogTimeout(cfg.watchdog_timeout_ms, state.elapsed_ms)

        try:
            result = await fn()
            if state.attempt > 0:
                logger.info(
                    f"{label}: succeeded on attempt {state.attempt + 1} "
                    f"after {state.elapsed_ms:.0f}ms"
                )
            return result

        except asyncio.CancelledError:
            raise

        except Exception as e:
            state.last_error = e
            state.attempt += 1

            if not retry_check(e):
                raise

            if state.attempt >= cfg.max_attempts:
                raise RetryExhausted(state.attempt, state.last_error) from e

            delay = cfg.compute_delay(state.attempt - 1)
            state.total_delay_ms += delay * 1000

            logger.warning(
                f"{label}: attempt {state.attempt}/{cfg.max_attempts} failed "
                f"({type(e).__name__}: {e}). Retrying in {delay:.1f}s..."
            )

            await asyncio.sleep(delay)

    raise RetryExhausted(state.attempt, state.last_error)


def retry_sync(
    fn: Callable,
    config: BackoffConfig | None = None,
    label: str = "operation",
    should_retry: Callable[[Exception], bool] | None = None,
) -> Any:
    """Synchronous version of retry_async using time.sleep."""
    import asyncio as _asyncio

    cfg = config or BackoffConfig()
    state = RetryState()

    default_retry = lambda e: not isinstance(e, KeyboardInterrupt)
    retry_check = should_retry or default_retry

    while state.attempt < cfg.max_attempts:
        if state.elapsed_ms > cfg.watchdog_timeout_ms:
            raise WatchdogTimeout(cfg.watchdog_timeout_ms, state.elapsed_ms)

        try:
            return fn()
        except Exception as e:
            state.last_error = e
            state.attempt += 1

            if not retry_check(e):
                raise

            if state.attempt >= cfg.max_attempts:
                raise RetryExhausted(state.attempt, state.last_error) from e

            delay = cfg.compute_delay(state.attempt - 1)
            state.total_delay_ms += delay * 1000

            logger.warning(
                f"{label}: attempt {state.attempt}/{cfg.max_attempts} failed "
                f"({type(e).__name__}: {e}). Retrying in {delay:.1f}s..."
            )
            time.sleep(delay)

    raise RetryExhausted(state.attempt, state.last_error)


class WatchdogTimer:
    """Fires a callback after a timeout if not cancelled.

    Used to detect stuck processes and force termination.
    """

    def __init__(self, timeout_ms: int, callback: Callable):
        self._timeout_ms = timeout_ms
        self._callback = callback
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the watchdog timer."""
        self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        await asyncio.sleep(self._timeout_ms / 1000.0)
        try:
            if asyncio.iscoroutinefunction(self._callback):
                await self._callback()
            else:
                self._callback()
        except Exception as e:
            logger.error(f"Watchdog callback failed: {e}")

    def cancel(self) -> None:
        """Cancel the watchdog timer."""
        if self._task and not self._task.done():
            self._task.cancel()
            self._task = None


class CommandPollBackoff:
    """Adaptive polling backoff for waiting on command results.

    Starts with fast polling, slows down as time passes,
    and resets on progress (changed output).

    Mirrors openclaw's command-poll-backoff.ts.
    """

    def __init__(
        self,
        initial_delay_ms: int = 100,
        max_delay_ms: int = 5000,
        slowdown_after_ms: int = 10000,
        multiplier: float = 1.5,
    ):
        self._initial_delay_ms = initial_delay_ms
        self._max_delay_ms = max_delay_ms
        self._slowdown_after_ms = slowdown_after_ms
        self._multiplier = multiplier

        self._current_delay_ms = initial_delay_ms
        self._start_time = time.time()
        self._poll_count = 0
        self._last_output_hash: str | None = None

    def record_progress(self, output: str) -> bool:
        """Record new output and check if progress was made.

        Returns True if the output changed (progress), False if stuck.
        """
        import hashlib
        new_hash = hashlib.md5(output.encode()).hexdigest()[:8]
        progress = new_hash != self._last_output_hash
        if progress:
            self._current_delay_ms = self._initial_delay_ms  # Reset on progress
        self._last_output_hash = new_hash
        return progress

    async def wait(self) -> None:
        """Wait for the current backoff delay."""
        elapsed_ms = (time.time() - self._start_time) * 1000

        # Slow down polling after the slowdown threshold
        if elapsed_ms > self._slowdown_after_ms:
            self._current_delay_ms = min(
                int(self._current_delay_ms * self._multiplier),
                self._max_delay_ms,
            )

        self._poll_count += 1
        await asyncio.sleep(self._current_delay_ms / 1000.0)

    def reset(self) -> None:
        """Reset backoff to initial state."""
        self._current_delay_ms = self._initial_delay_ms
        self._start_time = time.time()
        self._poll_count = 0
        self._last_output_hash = None

    @property
    def poll_count(self) -> int:
        return self._poll_count

    @property
    def elapsed_ms(self) -> float:
        return (time.time() - self._start_time) * 1000
