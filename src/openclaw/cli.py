"""CLI entry point — mirrors OpenClaw's CLI commands.

Commands:
    openclaw gateway    — Start the Gateway server
    openclaw chat       — Interactive REPL chat
    openclaw agent      — Single-shot agent message
    openclaw setup      — Create workspace with default files
    openclaw doctor     — Check configuration and dependencies
"""

from __future__ import annotations

import asyncio
import sys

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="openclaw-lang")
def main():
    """🦞 OpenClaw-Lang — Personal AI Assistant built with LangGraph."""
    pass


@main.command()
@click.option("--port", default=None, type=int, help="Gateway port (default: 18789)")
@click.option("--host", default=None, help="Bind host (default: 127.0.0.1)")
def gateway(port: int | None, host: str | None):
    """Start the Gateway server."""
    from openclaw.config import OpenClawConfig
    from openclaw.gateway.server import run_gateway
    from openclaw.workspace import ensure_workspace

    config = OpenClawConfig.load()
    if port:
        config.gateway.port = port
    if host:
        config.gateway.host = host

    # Ensure workspace exists
    ensure_workspace(config.get_workspace_path())

    run_gateway(config)


@main.command()
@click.option("--message", "-m", default=None, help="Message to send (non-interactive)")
@click.option("--model", default=None, help="Model override (e.g. openai/gpt-4o)")
def agent(message: str | None, model: str | None):
    """Send a single message to the agent."""
    from openclaw.config import OpenClawConfig
    from openclaw.agent.graph import run_agent
    from openclaw.workspace import ensure_workspace

    config = OpenClawConfig.load()
    if model:
        config.agent.model = model

    ensure_workspace(config.get_workspace_path())

    if not message:
        message = click.prompt("Message")

    console.print(f"\n🦞 [bold]Sending to {config.agent.model}...[/bold]\n")

    result = asyncio.run(run_agent(
        message=message,
        config=config,
    ))

    # Display tool calls
    for tc in result.get("tool_calls", []):
        console.print(Panel(
            f"[dim]{tc.get('args', {})}[/dim]",
            title=f"🔧 {tc.get('name', 'tool')}",
            border_style="blue",
        ))

    # Display response
    if result.get("response"):
        console.print()
        console.print(Markdown(result["response"]))
    else:
        console.print("[dim]No response[/dim]")


@main.command()
@click.option("--model", default=None, help="Model override")
def chat(model: str | None):
    """Interactive chat REPL."""
    from openclaw.config import OpenClawConfig
    from openclaw.agent.graph import run_agent
    from openclaw.agent.session import SessionManager
    from openclaw.workspace import ensure_workspace
    from langchain_core.messages import HumanMessage

    config = OpenClawConfig.load()
    if model:
        config.agent.model = model

    ensure_workspace(config.get_workspace_path())

    sm = SessionManager()
    session = sm.get_or_create_session("cli-user", "cli")

    console.print(Panel(
        f"[bold]🦞 OpenClaw-Lang Chat[/bold]\n"
        f"Model: {config.agent.model}\n"
        f"Session: {session.metadata.session_id}\n"
        f"Type 'exit' or 'quit' to leave, 'clear' to reset session.",
        border_style="bright_magenta",
    ))

    while True:
        try:
            user_input = console.input("\n[bold green]You:[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye! 🦞[/dim]")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            console.print("[dim]Goodbye! 🦞[/dim]")
            break
        if user_input.lower() == "clear":
            session = sm.get_or_create_session(f"cli-user-{id(session)}", "cli")
            console.print("[dim]Session cleared.[/dim]")
            continue

        # Persist user message
        sm.append_message(session, HumanMessage(content=user_input))

        console.print(f"\n[dim]Thinking...[/dim]", end="")

        try:
            result = asyncio.run(run_agent(
                message=user_input,
                session_id=session.metadata.session_id,
                config=config,
                history=session.messages[:-1],
            ))

            console.print("\r", end="")  # Clear "Thinking..."

            # Show tool calls
            for tc in result.get("tool_calls", []):
                console.print(Panel(
                    f"[dim]{str(tc.get('args', {}))[:500]}[/dim]",
                    title=f"🔧 {tc.get('name', 'tool')}",
                    border_style="blue",
                    width=80,
                ))

            # Show response
            response_text = result.get("response", "")
            if response_text:
                console.print(f"\n[bold cyan]🦞 Assistant:[/bold cyan]")
                console.print(Markdown(response_text))

                # Persist AI messages
                sm.append_messages(session, result.get("messages", []))
            else:
                console.print("[dim]No response[/dim]")

        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")


@main.command()
def setup():
    """Create workspace with default bootstrap files."""
    from openclaw.config import OpenClawConfig
    from openclaw.workspace import ensure_workspace

    config = OpenClawConfig.load()
    ws = ensure_workspace(config.get_workspace_path())

    console.print(Panel(
        f"[bold green]✅ Workspace created![/bold green]\n\n"
        f"📁 {ws}\n"
        f"   AGENTS.md — operating instructions\n"
        f"   SOUL.md — persona & tone\n"
        f"   TOOLS.md — tool notes\n"
        f"   IDENTITY.md — agent identity\n"
        f"   USER.md — your profile\n"
        f"   skills/ — workspace skills\n\n"
        f"Edit these files to customize your assistant!",
        title="🦞 OpenClaw-Lang Setup",
        border_style="bright_magenta",
    ))

    # Create default config if needed
    from pathlib import Path
    config_path = Path.home() / ".openclaw" / "openclaw.json"
    if not config_path.exists():
        config.save(config_path)
        console.print(f"\n📄 Config: {config_path}")
        console.print("[dim]Edit openclaw.json to set your API keys and model.[/dim]")


@main.command()
def doctor():
    """Check configuration, dependencies, and API keys."""
    from openclaw.config import OpenClawConfig
    from pathlib import Path
    import importlib

    console.print(Panel("[bold]🩺 OpenClaw-Lang Doctor[/bold]", border_style="bright_magenta"))

    # Check config
    try:
        config = OpenClawConfig.load()
        console.print("[green]✅[/green] Configuration loaded")
        console.print(f"   Model: {config.agent.model}")
        console.print(f"   Gateway: {config.gateway.host}:{config.gateway.port}")
    except Exception as e:
        console.print(f"[red]❌[/red] Config error: {e}")
        config = OpenClawConfig()

    # Check workspace
    ws = config.get_workspace_path()
    if ws.exists():
        files = list(ws.glob("*.md"))
        console.print(f"[green]✅[/green] Workspace: {ws} ({len(files)} files)")
    else:
        console.print(f"[yellow]⚠️[/yellow] Workspace not found: {ws}")
        console.print("   Run: openclaw setup")

    # Check API keys
    import os
    providers = {
        "OpenAI": config.api_keys.openai or os.environ.get("OPENAI_API_KEY", ""),
        "Anthropic": config.api_keys.anthropic or os.environ.get("ANTHROPIC_API_KEY", ""),
        "Google": config.api_keys.google or os.environ.get("GOOGLE_API_KEY", ""),
    }
    for name, key in providers.items():
        if key:
            console.print(f"[green]✅[/green] {name} API key: {key[:8]}...")
        else:
            console.print(f"[dim]⬜[/dim] {name} API key: not set")

    # Check dependencies
    console.print()
    deps = {
        "langgraph": "langgraph",
        "langchain_core": "langchain-core",
        "langchain_openai": "langchain-openai",
        "langchain_anthropic": "langchain-anthropic",
        "langchain_google_genai": "langchain-google-genai",
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
        "rich": "rich",
        "duckduckgo_search": "duckduckgo-search",
        "httpx": "httpx",
        "bs4": "beautifulsoup4",
    }
    for module, pkg in deps.items():
        try:
            importlib.import_module(module)
            console.print(f"[green]✅[/green] {pkg}")
        except ImportError:
            console.print(f"[red]❌[/red] {pkg} — not installed")

    # Optional deps
    optional = {
        "telegram": "python-telegram-bot",
        "discord": "discord.py",
        "chromadb": "chromadb",
    }
    console.print("\n[dim]Optional:[/dim]")
    for module, pkg in optional.items():
        try:
            importlib.import_module(module)
            console.print(f"[green]  ✅[/green] {pkg}")
        except ImportError:
            console.print(f"[dim]  ⬜[/dim] {pkg}")

    # Check cron jobs
    from openclaw.automation import CronStore
    store = CronStore()
    jobs = store.list_all()
    if jobs:
        enabled = sum(1 for j in jobs if j.enabled)
        console.print(f"\n[green]✅[/green] Cron: {len(jobs)} jobs ({enabled} enabled)")
    else:
        console.print(f"\n[dim]⬜[/dim] Cron: no scheduled jobs")


@main.group()
def cron():
    """Manage scheduled jobs (proactive nudges)."""
    pass


@cron.command("list")
def cron_list():
    """List all scheduled cron jobs."""
    from openclaw.automation import CronStore
    from datetime import datetime, timezone

    store = CronStore()
    jobs = store.list_all()

    if not jobs:
        console.print("[dim]No scheduled jobs. Use 'openclaw cron add' to create one.[/dim]")
        return

    from rich.table import Table
    table = Table(title="⏰ Cron Jobs", border_style="bright_magenta")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="bold")
    table.add_column("Schedule")
    table.add_column("Next Run")
    table.add_column("Runs", justify="right")
    table.add_column("Status")

    for job in jobs:
        schedule = ""
        if job.schedule.kind == "at" and job.schedule.at:
            schedule = f"at {job.schedule.at}"
        elif job.schedule.kind == "every" and job.schedule.every_seconds:
            schedule = f"every {job.schedule.every_seconds}s"
        elif job.schedule.kind == "cron" and job.schedule.cron_expr:
            schedule = job.schedule.cron_expr

        next_run = "—"
        if job.next_run_at:
            dt = datetime.fromtimestamp(job.next_run_at, tz=timezone.utc)
            next_run = dt.strftime("%Y-%m-%d %H:%M UTC")

        status = "[green]enabled[/green]" if job.enabled else "[dim]disabled[/dim]"
        table.add_row(job.job_id, job.name, schedule, next_run, str(job.run_count), status)

    console.print(table)


@cron.command("add")
@click.option("--name", "-n", required=True, help="Job name")
@click.option("--payload", "-p", required=True, help="Message/instruction to run")
@click.option("--at", "at_time", default=None, help="One-shot ISO timestamp (e.g. 2026-03-08T10:00:00)")
@click.option("--every", "every_sec", default=None, type=int, help="Repeat interval in seconds")
@click.option("--cron-expr", default=None, help="Cron expression (e.g. '0 7 * * *')")
@click.option("--channel", default="webchat", help="Delivery channel")
@click.option("--delete-after-run", is_flag=True, help="Delete after running (one-shot)")
def cron_add(name, payload, at_time, every_sec, cron_expr, channel, delete_after_run):
    """Add a new scheduled job."""
    from openclaw.automation import CronJob, CronSchedule, CronDelivery, CronStore

    if at_time:
        kind = "at"
    elif every_sec:
        kind = "every"
    elif cron_expr:
        kind = "cron"
    else:
        console.print("[red]Error: Specify --at, --every, or --cron-expr[/red]")
        return

    schedule = CronSchedule(
        kind=kind,
        at=at_time,
        every_seconds=every_sec,
        cron_expr=cron_expr,
    )

    job = CronJob(
        name=name,
        schedule=schedule,
        payload=payload,
        delivery=CronDelivery(mode="announce", channel=channel),
        delete_after_run=delete_after_run,
    )

    store = CronStore()
    store.add(job)

    console.print(Panel(
        f"[bold green]✅ Job created![/bold green]\n\n"
        f"🆔 ID: {job.job_id}\n"
        f"📋 Name: {name}\n"
        f"⏰ Schedule: {kind} ({at_time or every_sec or cron_expr})\n"
        f"📝 Payload: {payload[:80]}\n"
        f"📨 Channel: {channel}",
        title="🦞 Cron Job",
        border_style="bright_magenta",
    ))


@cron.command("remove")
@click.argument("job_id")
def cron_remove(job_id):
    """Remove a scheduled job."""
    from openclaw.automation import CronStore
    store = CronStore()
    if store.remove(job_id):
        console.print(f"[green]✅ Removed job {job_id}[/green]")
    else:
        console.print(f"[red]❌ Job not found: {job_id}[/red]")


@cron.command("run")
@click.argument("job_id")
def cron_run(job_id):
    """Manually trigger a job now."""
    from openclaw.automation import CronStore
    store = CronStore()
    job = store.jobs.get(job_id)
    if not job:
        console.print(f"[red]❌ Job not found: {job_id}[/red]")
        return

    console.print(f"[bold]🔄 Running job: {job.name}[/bold]")
    console.print(f"[dim]Payload: {job.payload}[/dim]")

    from openclaw.agent.graph import run_agent
    result = asyncio.run(run_agent(message=f"[Scheduled job: {job.name}] {job.payload}"))

    store.mark_ran(job_id)

    if result.get("response"):
        console.print(f"\n[bold cyan]🦞 Response:[/bold cyan]")
        console.print(Markdown(result["response"]))


if __name__ == "__main__":
    main()

