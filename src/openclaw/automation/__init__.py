"""Cron scheduler — proactive nudges and scheduled jobs.

Mirrors OpenClaw's cron/wakeup system:
- Schedule types: 'at' (one-shot), 'every' (interval), 'cron' (expression)
- Execution: main session (system event) or isolated (dedicated agent turn)
- Delivery: announce to channel, webhook, or none
- Persistence: jobs stored in ~/.langclaw/cron/jobs.json
- The agent can create/manage jobs via the cron tool
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Awaitable, Optional

logger = logging.getLogger(__name__)

CRON_DIR = Path.home() / ".langclaw" / "cron"
JOBS_FILE = CRON_DIR / "jobs.json"


@dataclass
class CronSchedule:
    """When a job should run."""

    kind: str = "at"  # "at" | "every" | "cron"
    at: str | None = None  # ISO 8601 timestamp for one-shot
    every_seconds: int | None = None  # Interval in seconds for repeating
    cron_expr: str | None = None  # Cron expression (5-field)
    timezone: str = "UTC"


@dataclass
class CronDelivery:
    """How a job's result should be delivered."""

    mode: str = "announce"  # "announce" | "webhook" | "none"
    channel: str = "webchat"
    to: str = ""  # recipient / channel ID / webhook URL


@dataclass
class CronJob:
    """A scheduled job."""

    job_id: str = ""
    name: str = ""
    enabled: bool = True
    schedule: CronSchedule = field(default_factory=CronSchedule)
    payload: str = ""  # Message or system event text
    payload_kind: str = "agentTurn"  # "agentTurn" | "systemEvent"
    delivery: CronDelivery = field(default_factory=CronDelivery)
    delete_after_run: bool = False
    created_at: float = 0.0
    last_run_at: float | None = None
    next_run_at: float | None = None
    run_count: int = 0

    def __post_init__(self):
        if not self.job_id:
            self.job_id = str(uuid.uuid4())[:8]
        if not self.created_at:
            self.created_at = time.time()
        if self.next_run_at is None:
            self._compute_next_run()

    def _compute_next_run(self) -> None:
        """Calculate the next run timestamp."""
        now = time.time()

        if self.schedule.kind == "at" and self.schedule.at:
            try:
                dt = datetime.fromisoformat(self.schedule.at)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                self.next_run_at = dt.timestamp()
            except ValueError:
                self.next_run_at = now + 60

        elif self.schedule.kind == "every" and self.schedule.every_seconds:
            base = self.last_run_at or now
            self.next_run_at = base + self.schedule.every_seconds

        elif self.schedule.kind == "cron" and self.schedule.cron_expr:
            # Simple cron parsing for common patterns
            self.next_run_at = _next_cron_time(self.schedule.cron_expr, now)

        else:
            self.next_run_at = now + 60

    def is_due(self) -> bool:
        """Check if this job should run now."""
        if not self.enabled or self.next_run_at is None:
            return False
        return time.time() >= self.next_run_at

    def mark_ran(self) -> None:
        """Update after running."""
        self.last_run_at = time.time()
        self.run_count += 1

        if self.schedule.kind == "at":
            # One-shot: disable after running
            self.enabled = False
            self.next_run_at = None
        else:
            self._compute_next_run()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "job_id": self.job_id,
            "name": self.name,
            "enabled": self.enabled,
            "schedule": asdict(self.schedule),
            "payload": self.payload,
            "payload_kind": self.payload_kind,
            "delivery": asdict(self.delivery),
            "delete_after_run": self.delete_after_run,
            "created_at": self.created_at,
            "last_run_at": self.last_run_at,
            "next_run_at": self.next_run_at,
            "run_count": self.run_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CronJob":
        """Deserialize from dict."""
        schedule_data = data.get("schedule", {})
        delivery_data = data.get("delivery", {})
        return cls(
            job_id=data.get("job_id", ""),
            name=data.get("name", ""),
            enabled=data.get("enabled", True),
            schedule=CronSchedule(**schedule_data),
            payload=data.get("payload", ""),
            payload_kind=data.get("payload_kind", "agentTurn"),
            delivery=CronDelivery(**delivery_data),
            delete_after_run=data.get("delete_after_run", False),
            created_at=data.get("created_at", 0.0),
            last_run_at=data.get("last_run_at"),
            next_run_at=data.get("next_run_at"),
            run_count=data.get("run_count", 0),
        )


def _next_cron_time(expr: str, now: float) -> float:
    """Simple cron next-run calculator for common patterns.

    Supports: minute hour day month weekday (5-field standard cron)
    Uses a basic forward-scan approach.
    """
    parts = expr.strip().split()
    if len(parts) < 5:
        return now + 3600  # Fallback: 1 hour

    minute_spec, hour_spec, day_spec, month_spec, dow_spec = parts[:5]

    dt = datetime.fromtimestamp(now, tz=timezone.utc) + timedelta(minutes=1)
    dt = dt.replace(second=0, microsecond=0)

    # Simple: try the next 1440 minutes (24 hours)
    for _ in range(1440):
        if _matches_cron_field(minute_spec, dt.minute) and \
           _matches_cron_field(hour_spec, dt.hour) and \
           _matches_cron_field(day_spec, dt.day) and \
           _matches_cron_field(month_spec, dt.month) and \
           _matches_cron_field(dow_spec, dt.weekday()):
            return dt.timestamp()
        dt += timedelta(minutes=1)

    # If nothing matched in 24 hours, default to 1 hour from now
    return now + 3600


def _matches_cron_field(spec: str, value: int) -> bool:
    """Check if a value matches a cron field spec."""
    if spec == "*":
        return True
    if spec.startswith("*/"):
        try:
            step = int(spec[2:])
            return value % step == 0
        except ValueError:
            return True
    try:
        return value == int(spec)
    except ValueError:
        pass
    # Handle comma-separated values
    if "," in spec:
        values = [int(v.strip()) for v in spec.split(",") if v.strip().isdigit()]
        return value in values
    return True


# --- Job Store ---

class CronStore:
    """Persistent cron job storage."""

    def __init__(self):
        CRON_DIR.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, CronJob] = {}
        self._load()

    def _load(self) -> None:
        """Load jobs from disk."""
        if JOBS_FILE.exists():
            try:
                data = json.loads(JOBS_FILE.read_text(encoding="utf-8"))
                # Support both flat list and {version, jobs} format
                if isinstance(data, dict):
                    data = data.get("jobs", [])
                if isinstance(data, list):
                    for job_data in data:
                        if isinstance(job_data, dict):
                            job = CronJob.from_dict(job_data)
                            self.jobs[job.job_id] = job
            except (json.JSONDecodeError, TypeError, KeyError, AttributeError):
                pass

    def _save(self) -> None:
        """Save jobs to disk."""
        data = [job.to_dict() for job in self.jobs.values()]
        JOBS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def add(self, job: CronJob) -> CronJob:
        """Add a new job."""
        self.jobs[job.job_id] = job
        self._save()
        logger.info(f"Cron job added: {job.name} ({job.job_id})")
        return job

    def remove(self, job_id: str) -> bool:
        """Remove a job."""
        if job_id in self.jobs:
            del self.jobs[job_id]
            self._save()
            return True
        return False

    def update(self, job_id: str, **kwargs: Any) -> CronJob | None:
        """Update a job's fields."""
        job = self.jobs.get(job_id)
        if not job:
            return None
        for key, value in kwargs.items():
            if hasattr(job, key):
                setattr(job, key, value)
        self._save()
        return job

    def list_all(self) -> list[CronJob]:
        """List all jobs."""
        return sorted(self.jobs.values(), key=lambda j: j.created_at, reverse=True)

    def get_due_jobs(self) -> list[CronJob]:
        """Get all jobs that are due to run."""
        return [j for j in self.jobs.values() if j.is_due()]

    def mark_ran(self, job_id: str) -> None:
        """Mark a job as having run."""
        job = self.jobs.get(job_id)
        if job:
            job.mark_ran()
            if job.delete_after_run:
                del self.jobs[job_id]
            self._save()


# --- Scheduler ---

# Type for the callback the scheduler calls when a job fires
JobCallback = Callable[[CronJob], Awaitable[Optional[str]]]


class CronScheduler:
    """Background scheduler that checks for due jobs and executes them.

    Runs inside the Gateway process, checking every 30 seconds for due jobs.
    When a job fires, it calls the callback (typically: run the agent with
    the job's payload and deliver the result).
    """

    def __init__(self, callback: JobCallback, check_interval: float = 30.0):
        self.store = CronStore()
        self._callback = callback
        self._check_interval = check_interval
        self._task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        """Start the scheduler background loop."""
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info(f"Cron scheduler started (check every {self._check_interval}s, {len(self.store.jobs)} jobs loaded)")

    async def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _loop(self) -> None:
        """Main scheduler loop."""
        while self._running:
            try:
                due_jobs = self.store.get_due_jobs()
                for job in due_jobs:
                    logger.info(f"Cron job firing: {job.name} ({job.job_id})")
                    try:
                        result = await self._callback(job)
                        self.store.mark_ran(job.job_id)
                        logger.info(f"Cron job completed: {job.name} ({job.job_id})")
                    except Exception as e:
                        logger.error(f"Cron job failed: {job.name} ({job.job_id}): {e}")
                        # Still mark as ran to avoid infinite retries
                        self.store.mark_ran(job.job_id)
            except Exception as e:
                logger.error(f"Cron scheduler error: {e}")

            await asyncio.sleep(self._check_interval)
