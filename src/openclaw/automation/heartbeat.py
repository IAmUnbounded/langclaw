"""Heartbeat integration — parse HEARTBEAT.md and manage the heartbeat daemon.

Parses HEARTBEAT.md workspace file to extract:
- Check interval
- Active hours (start/end)
- Timezone

Creates/manages a system cron job for the heartbeat.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, time as dt_time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class HeartbeatConfig:
    """Parsed heartbeat configuration from HEARTBEAT.md."""

    check_interval_seconds: int = 30
    active_start: dt_time = dt_time(6, 0)
    active_end: dt_time = dt_time(23, 0)
    timezone: str = "auto"
    enabled: bool = True


def parse_heartbeat_md(content: str) -> HeartbeatConfig:
    """Parse HEARTBEAT.md content into a HeartbeatConfig.

    Extracts structured data from the markdown format:
    - **Check interval**: Every N seconds
    - **Active hours**: HH:MM - HH:MM
    - **Timezone**: auto-detect | UTC | timezone name
    """
    config = HeartbeatConfig()

    # Parse check interval
    interval_match = re.search(r"Check interval.*?Every\s+(\d+)\s+(seconds?|minutes?)", content, re.IGNORECASE)
    if interval_match:
        value = int(interval_match.group(1))
        unit = interval_match.group(2).lower()
        if "minute" in unit:
            value *= 60
        config.check_interval_seconds = value

    # Parse active hours
    hours_match = re.search(r"Active hours.*?(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})", content, re.IGNORECASE)
    if hours_match:
        try:
            start_parts = hours_match.group(1).split(":")
            end_parts = hours_match.group(2).split(":")
            config.active_start = dt_time(int(start_parts[0]), int(start_parts[1]))
            config.active_end = dt_time(int(end_parts[0]), int(end_parts[1]))
        except (ValueError, IndexError):
            pass

    # Parse timezone
    tz_match = re.search(r"Timezone.*?:\s*(.+)", content, re.IGNORECASE)
    if tz_match:
        tz_val = tz_match.group(1).strip()
        if tz_val.lower() not in ("auto-detect", "auto"):
            config.timezone = tz_val

    return config


def is_within_active_hours(config: HeartbeatConfig) -> bool:
    """Check if the current time falls within the configured active hours."""
    now = datetime.now().time()
    if config.active_start <= config.active_end:
        return config.active_start <= now <= config.active_end
    else:
        # Wraps midnight (e.g., 22:00 - 06:00)
        return now >= config.active_start or now <= config.active_end


def ensure_heartbeat_job(workspace: Path) -> dict[str, Any] | None:
    """Parse HEARTBEAT.md and create/update the system heartbeat cron job.

    Called during gateway startup. Creates a repeating cron job that
    fires on the configured interval during active hours.

    Returns the job dict if created/updated, None if HEARTBEAT.md missing.
    """
    heartbeat_file = workspace / "HEARTBEAT.md"
    if not heartbeat_file.is_file():
        return None

    try:
        content = heartbeat_file.read_text(encoding="utf-8")
    except Exception:
        return None

    config = parse_heartbeat_md(content)
    if not config.enabled:
        return None

    from openclaw.automation import CronJob, CronSchedule, CronDelivery, CronStore

    store = CronStore()

    # Check if heartbeat job already exists
    HEARTBEAT_JOB_NAME = "__system_heartbeat__"
    existing = None
    for job in store.jobs.values():
        if job.name == HEARTBEAT_JOB_NAME:
            existing = job
            break

    if existing:
        # Update interval if changed
        if existing.schedule.every_seconds != config.check_interval_seconds:
            existing.schedule.every_seconds = config.check_interval_seconds
            store._save()
            logger.info(f"Heartbeat interval updated to {config.check_interval_seconds}s")
        return existing.to_dict()

    # Create new heartbeat job
    job = CronJob(
        name=HEARTBEAT_JOB_NAME,
        schedule=CronSchedule(
            kind="every",
            every_seconds=config.check_interval_seconds,
        ),
        payload=(
            "This is a heartbeat check. Review pending tasks, "
            "check for completed background jobs, and report anything "
            "important. Only respond if there's something useful to share."
        ),
        payload_kind="systemEvent",
        delivery=CronDelivery(mode="announce", channel="webchat"),
    )
    store.add(job)
    logger.info(
        f"Heartbeat job created: every {config.check_interval_seconds}s, "
        f"active {config.active_start.strftime('%H:%M')}-{config.active_end.strftime('%H:%M')}"
    )
    return job.to_dict()
