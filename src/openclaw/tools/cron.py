"""Cron tool — lets the agent create, list, and manage scheduled jobs.

The agent can proactively set up reminders, recurring checks, and
automated nudges that fire without user input.

Examples the agent can do:
- "Remind me in 30 minutes to check the build"
- "Every morning at 7am, summarize my emails"
- "Check the server status every 2 hours"
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import tool

from openclaw.automation import CronJob, CronSchedule, CronDelivery, CronStore


@tool
def cron_add_tool(
    name: str,
    payload: str,
    schedule_kind: str = "at",
    schedule_at: str = "",
    every_seconds: int = 0,
    cron_expr: str = "",
    channel: str = "webchat",
    delete_after_run: bool = False,
) -> str:
    """Create a scheduled job that will fire automatically.

    Use this to set reminders, recurring tasks, or proactive nudges.

    Schedule types:
    - "at": One-shot at a specific time. Provide schedule_at as ISO 8601 (e.g. "2026-03-08T10:00:00")
    - "every": Repeat at a fixed interval. Provide every_seconds (e.g. 3600 for hourly)
    - "cron": Cron expression (5-field). Provide cron_expr (e.g. "0 7 * * *" for daily at 7am)

    Args:
        name: Descriptive name for the job (e.g. "Morning briefing", "Server health check").
        payload: The message/instruction to run when the job fires.
        schedule_kind: "at", "every", or "cron".
        schedule_at: ISO 8601 timestamp for one-shot jobs.
        every_seconds: Interval in seconds for repeating jobs.
        cron_expr: Cron expression for cron-type jobs.
        channel: Channel to deliver results to (default "webchat").
        delete_after_run: Whether to delete the job after it runs (for one-shot reminders).

    Returns:
        Confirmation with job ID and schedule details.
    """
    schedule = CronSchedule(
        kind=schedule_kind,
        at=schedule_at if schedule_at else None,
        every_seconds=every_seconds if every_seconds else None,
        cron_expr=cron_expr if cron_expr else None,
    )

    delivery = CronDelivery(
        mode="announce",
        channel=channel,
    )

    job = CronJob(
        name=name,
        schedule=schedule,
        payload=payload,
        delivery=delivery,
        delete_after_run=delete_after_run,
    )

    store = CronStore()
    store.add(job)

    # Build human-readable schedule description
    schedule_desc = ""
    if schedule_kind == "at" and schedule_at:
        schedule_desc = f"at {schedule_at}"
    elif schedule_kind == "every" and every_seconds:
        if every_seconds >= 3600:
            schedule_desc = f"every {every_seconds // 3600}h"
        elif every_seconds >= 60:
            schedule_desc = f"every {every_seconds // 60}m"
        else:
            schedule_desc = f"every {every_seconds}s"
    elif schedule_kind == "cron" and cron_expr:
        schedule_desc = f"cron: {cron_expr}"

    return (
        f"✅ Scheduled job created!\n"
        f"  📋 Name: {name}\n"
        f"  🆔 ID: {job.job_id}\n"
        f"  ⏰ Schedule: {schedule_desc}\n"
        f"  📨 Delivery: {channel}\n"
        f"  📝 Payload: {payload[:100]}\n"
        f"  {'🗑️ Will delete after running' if delete_after_run else '🔄 Persistent job'}"
    )


@tool
def cron_list_tool() -> str:
    """List all scheduled cron jobs.

    Returns a formatted list of all jobs with their schedule, status, and next run time.
    """
    store = CronStore()
    jobs = store.list_all()

    if not jobs:
        return "No scheduled jobs. Use cron_add_tool to create one."

    lines = [f"📋 **Scheduled Jobs** ({len(jobs)} total)\n"]
    for job in jobs:
        status = "✅ enabled" if job.enabled else "⏸️ disabled"
        next_run = ""
        if job.next_run_at:
            from datetime import datetime, timezone
            dt = datetime.fromtimestamp(job.next_run_at, tz=timezone.utc)
            next_run = dt.strftime("%Y-%m-%d %H:%M UTC")

        schedule_desc = ""
        if job.schedule.kind == "at" and job.schedule.at:
            schedule_desc = f"at {job.schedule.at}"
        elif job.schedule.kind == "every" and job.schedule.every_seconds:
            schedule_desc = f"every {job.schedule.every_seconds}s"
        elif job.schedule.kind == "cron" and job.schedule.cron_expr:
            schedule_desc = f"cron: {job.schedule.cron_expr}"

        lines.append(
            f"**{job.name}** (`{job.job_id}`)\n"
            f"   Status: {status} | Schedule: {schedule_desc}\n"
            f"   Next run: {next_run} | Runs: {job.run_count}\n"
            f"   Payload: {job.payload[:80]}{'...' if len(job.payload) > 80 else ''}\n"
        )

    return "\n".join(lines)


@tool
def cron_remove_tool(job_id: str) -> str:
    """Remove a scheduled cron job.

    Args:
        job_id: The job ID to remove.

    Returns:
        Confirmation or error message.
    """
    store = CronStore()
    if store.remove(job_id):
        return f"✅ Removed job {job_id}"
    else:
        return f"❌ Job not found: {job_id}"


@tool
def cron_run_tool(job_id: str) -> str:
    """Manually trigger a cron job to run immediately.

    Args:
        job_id: The job ID to run now.

    Returns:
        The job's payload (which will be executed).
    """
    store = CronStore()
    job = store.jobs.get(job_id)
    if not job:
        return f"❌ Job not found: {job_id}"

    return f"🔄 Triggering job: {job.name}\nPayload: {job.payload}"
