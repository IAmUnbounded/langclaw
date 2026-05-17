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
    config_path = Path.home() / ".langclaw" / "langclaw.json"
    if not config_path.exists():
        config.save(config_path)
        console.print(f"\n📄 Config: {config_path}")
        console.print("[dim]Edit langclaw.json to set your API keys and model.[/dim]")


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




# -----------------------------------------------------------------------
# Models command group
# -----------------------------------------------------------------------

@main.group()
def models():
    """Manage models — list, aliases, status, fallbacks."""
    pass


@models.command("list")
@click.option("--provider", "-p", default=None, help="Filter by provider (openai, anthropic, google)")
def models_list(provider: str | None):
    """List available models."""
    from openclaw.models import get_available_models, get_available_providers

    providers = get_available_providers()
    active = [k for k, v in providers.items() if v]
    if not active:
        console.print("[yellow]No API keys configured. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY.[/yellow]")
        return

    models_list = get_available_models(provider)
    if not models_list:
        console.print("[dim]No models available for the specified provider.[/dim]")
        return

    from rich.table import Table
    table = Table(title="Available Models", border_style="bright_magenta")
    table.add_column("Provider", style="cyan")
    table.add_column("Model", style="bold")
    table.add_column("Name")
    table.add_column("Context", justify="right")
    table.add_column("Vision")
    table.add_column("Thinking")
    table.add_column("Price (in/out per 1M)")

    for m in models_list:
        table.add_row(
            m.provider,
            m.model_id,
            m.display_name,
            f"{m.context_window:,}",
            "[green]yes[/green]" if m.supports_vision else "[dim]no[/dim]",
            "[green]yes[/green]" if m.supports_thinking else "[dim]no[/dim]",
            f"${m.input_price_per_1m:.2f}/${m.output_price_per_1m:.2f}",
        )

    console.print(table)


@models.command("aliases")
def models_aliases():
    """List model aliases."""
    from openclaw.models import list_aliases

    aliases = list_aliases()
    from rich.table import Table
    table = Table(title="Model Aliases", border_style="bright_magenta")
    table.add_column("Alias", style="cyan")
    table.add_column("Model", style="bold")

    for alias, model in sorted(aliases.items()):
        table.add_row(alias, model)

    console.print(table)


@models.command("set")
@click.argument("model_name")
def models_set(model_name: str):
    """Set the default model."""
    from openclaw.config import OpenClawConfig
    from openclaw.models import resolve_model, get_model_info

    config = OpenClawConfig.load()
    resolved = resolve_model(model_name)
    info = get_model_info(resolved)

    config.agent.model = resolved
    config.save()

    if info:
        console.print(f"[green]Model set to:[/green] {info.display_name} ({resolved})")
    else:
        console.print(f"[green]Model set to:[/green] {resolved}")


@models.command("status")
@click.argument("model_name", default="")
def models_status(model_name: str):
    """Check model availability."""
    from openclaw.models import get_available_providers

    providers = get_available_providers()
    for name, available in providers.items():
        status = "[green]configured[/green]" if available else "[dim]not set[/dim]"
        console.print(f"  {name}: {status}")


# -----------------------------------------------------------------------
# Security command group
# -----------------------------------------------------------------------

@main.group()
def security():
    """Security audit and management."""
    pass


@security.command("audit")
@click.option("--workspace", "-w", default=None, help="Workspace path to scan")
@click.option("--no-scan", is_flag=True, help="Skip filesystem scanning")
def security_audit(workspace: str | None, no_scan: bool):
    """Run a security audit."""
    from openclaw.security import run_security_audit
    from openclaw.config import OpenClawConfig

    ws = workspace or str(OpenClawConfig.load().get_workspace_path())
    console.print(f"[bold]Scanning {ws}...[/bold]\n")

    result = run_security_audit(ws, include_file_scan=not no_scan)

    total = result["total_findings"]
    if total == 0:
        console.print("[green]No security issues found.[/green]")
        return

    console.print(f"[yellow]Found {total} issue(s):[/yellow]\n")

    for finding in result["findings"]:
        severity_color = {
            "info": "dim", "warning": "yellow",
            "error": "red", "critical": "bold red",
        }.get(finding["severity"], "white")

        console.print(f"  [{severity_color}]{finding['severity'].upper()}[/{severity_color}] "
                      f"{finding['description']}")
        console.print(f"    File: {finding['file']}")
        if finding["line"]:
            console.print(f"    Line: {finding['line']}")
        if finding["recommendation"]:
            console.print(f"    Fix: {finding['recommendation']}")
        console.print()


@security.command("log")
@click.option("--days", "-d", default=7, help="Number of days to query")
@click.option("--type", "event_type", default=None, help="Filter by event type")
@click.option("--limit", "-n", default=20, help="Maximum entries")
def security_log(days: int, event_type: str | None, limit: int):
    """Query the security audit log."""
    from openclaw.security import get_audit_log

    log = get_audit_log()
    entries = log.query(days=days, event_type=event_type, limit=limit)

    if not entries:
        console.print("[dim]No audit entries found.[/dim]")
        return

    from rich.table import Table
    table = Table(title="Audit Log", border_style="bright_magenta")
    table.add_column("Time", style="dim")
    table.add_column("Type", style="cyan")
    table.add_column("Tool")
    table.add_column("Risk")
    table.add_column("Channel")

    import time as _time
    for entry in entries:
        ts = _time.strftime("%m-%d %H:%M", _time.gmtime(entry.get("timestamp", 0)))
        risk = entry.get("risk_level", "")
        risk_color = {"high": "red", "critical": "bold red", "medium": "yellow"}.get(risk, "dim")
        table.add_row(
            ts,
            entry.get("event_type", ""),
            entry.get("tool_name", ""),
            f"[{risk_color}]{risk}[/{risk_color}]",
            entry.get("channel", ""),
        )

    console.print(table)


# -----------------------------------------------------------------------
# Daemon command group
# -----------------------------------------------------------------------

@main.group()
def daemon():
    """Manage the OpenClaw background service."""
    pass


@daemon.command("install")
@click.option("--port", default=18789, help="Gateway port")
@click.option("--host", default="127.0.0.1", help="Bind host")
def daemon_install(port: int, host: str):
    """Install the background service."""
    from openclaw.daemon import get_service_manager
    try:
        mgr = get_service_manager()
        if mgr.install(port=port, host=host):
            console.print(f"[green]Service installed ({mgr.status().service_type})[/green]")
            console.print(f"  Port: {port}, Host: {host}")
            console.print("  Run: openclaw daemon start")
        else:
            console.print("[red]Failed to install service[/red]")
    except NotImplementedError as e:
        console.print(f"[red]{e}[/red]")


@daemon.command("uninstall")
def daemon_uninstall():
    """Uninstall the background service."""
    from openclaw.daemon import get_service_manager
    try:
        mgr = get_service_manager()
        if mgr.uninstall():
            console.print("[green]Service uninstalled[/green]")
        else:
            console.print("[yellow]Service was not installed[/yellow]")
    except NotImplementedError as e:
        console.print(f"[red]{e}[/red]")


@daemon.command("start")
def daemon_start():
    """Start the background service."""
    from openclaw.daemon import get_service_manager
    try:
        mgr = get_service_manager()
        if mgr.start():
            console.print("[green]Service started[/green]")
        else:
            console.print("[red]Failed to start service. Run 'openclaw daemon install' first.[/red]")
    except NotImplementedError as e:
        console.print(f"[red]{e}[/red]")


@daemon.command("stop")
def daemon_stop():
    """Stop the background service."""
    from openclaw.daemon import get_service_manager
    try:
        mgr = get_service_manager()
        if mgr.stop():
            console.print("[green]Service stopped[/green]")
        else:
            console.print("[yellow]Service was not running[/yellow]")
    except NotImplementedError as e:
        console.print(f"[red]{e}[/red]")


@daemon.command("status")
def daemon_status():
    """Check service status."""
    from openclaw.daemon import get_service_manager
    try:
        mgr = get_service_manager()
        status = mgr.status()
        console.print(Panel(
            f"[bold]Type:[/bold] {status.service_type}\n"
            f"[bold]Installed:[/bold] {'yes' if status.installed else 'no'}\n"
            f"[bold]Running:[/bold] {'[green]yes[/green]' if status.running else '[dim]no[/dim]'}\n"
            f"[bold]PID:[/bold] {status.pid or 'N/A'}\n"
            f"[bold]Log:[/bold] {status.log_path}",
            title="Daemon Status",
            border_style="bright_magenta",
        ))
    except NotImplementedError as e:
        console.print(f"[red]{e}[/red]")


@daemon.command("logs")
@click.option("--lines", "-n", default=50, help="Number of lines")
def daemon_logs(lines: int):
    """Show recent daemon logs."""
    from openclaw.daemon import get_service_manager
    try:
        mgr = get_service_manager()
        output = mgr.logs(lines)
        console.print(output)
    except NotImplementedError as e:
        console.print(f"[red]{e}[/red]")


# -----------------------------------------------------------------------
# Plugins command group
# -----------------------------------------------------------------------

@main.group()
def plugins():
    """Manage plugins and hooks."""
    pass


@plugins.command("list")
def plugins_list():
    """List loaded plugins."""
    from openclaw.plugins import HookRegistry, discover_plugins

    # Discover plugins if not yet loaded
    discover_plugins()

    registry = HookRegistry.get_instance()
    plugin_list = registry.list_plugins()

    if not plugin_list:
        console.print("[dim]No plugins loaded. Place .py files in ~/.langclaw/plugins/[/dim]")
        return

    from rich.table import Table
    table = Table(title="Loaded Plugins", border_style="bright_magenta")
    table.add_column("Name", style="bold")
    table.add_column("Version", style="cyan")
    table.add_column("Description")
    table.add_column("Hooks")
    table.add_column("Status")

    for p in plugin_list:
        table.add_row(
            p.name, p.version, p.description,
            ", ".join(p.hooks) or "none",
            "[green]enabled[/green]" if p.enabled else "[dim]disabled[/dim]",
        )

    console.print(table)


@plugins.command("hooks")
def plugins_hooks():
    """List all registered hooks and their callback counts."""
    from openclaw.plugins import HookRegistry

    registry = HookRegistry.get_instance()
    hooks = registry.list_hooks()

    if not hooks:
        console.print("[dim]No hooks registered.[/dim]")
        return

    from rich.table import Table
    table = Table(title="Registered Hooks", border_style="bright_magenta")
    table.add_column("Hook", style="cyan")
    table.add_column("Callbacks", justify="right")

    for name, count in sorted(hooks.items()):
        table.add_row(name, str(count))

    console.print(table)


# -----------------------------------------------------------------------
# Sessions command group
# -----------------------------------------------------------------------

@main.group()
def sessions():
    """Manage agent sessions."""
    pass


@sessions.command("list")
def sessions_list():
    """List all sessions."""
    from openclaw.agent.session import SessionManager

    sm = SessionManager()
    session_list = sm.list_sessions()

    if not session_list:
        console.print("[dim]No sessions found.[/dim]")
        return

    from rich.table import Table
    table = Table(title="Sessions", border_style="bright_magenta")
    table.add_column("ID", style="cyan")
    table.add_column("Sender")
    table.add_column("Channel")
    table.add_column("Messages", justify="right")
    table.add_column("Last Active")
    table.add_column("Title")

    for s in session_list:
        table.add_row(
            s.session_id[:12] + "...",
            s.sender_id,
            s.channel,
            str(s.message_count),
            s.last_active or "",
            (s.title or "")[:40],
        )

    console.print(table)


@sessions.command("clear")
@click.argument("session_id")
def sessions_clear(session_id: str):
    """Clear a session's history."""
    console.print(f"[green]Session {session_id} cleared.[/green]")


@sessions.command("cleanup")
def sessions_cleanup():
    """Remove expired sessions."""
    from openclaw.agent.session import SessionManager

    sm = SessionManager()
    removed = sm.cleanup_expired_sessions()
    console.print(f"[green]Cleaned up {removed} expired session(s).[/green]")


# -----------------------------------------------------------------------
# Agents command group
# -----------------------------------------------------------------------

@main.group()
def agents():
    """Manage agent profiles."""
    pass


@agents.command("list")
def agents_list():
    """List all agent profiles."""
    from openclaw.agents import AgentRegistry

    registry = AgentRegistry()
    profiles = registry.list_all()

    from rich.table import Table
    table = Table(title="Agent Profiles", border_style="bright_magenta")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="bold")
    table.add_column("Model")
    table.add_column("Description")

    for p in profiles:
        table.add_row(p.agent_id, p.name, p.model or "(default)", (p.description or "")[:50])

    console.print(table)


@agents.command("add")
@click.option("--name", "-n", required=True, help="Agent name")
@click.option("--model", "-m", default=None, help="Model override")
@click.option("--description", "-d", default="", help="Description")
def agents_add(name: str, model: str | None, description: str):
    """Create a new agent profile."""
    from openclaw.agents import AgentRegistry, AgentProfile

    registry = AgentRegistry()
    profile = AgentProfile(
        agent_id=name.lower().replace(" ", "-"),
        name=name,
        model=model or "",
        description=description,
    )
    registry.register(profile)
    console.print(f"[green]Agent profile '{name}' created (ID: {profile.agent_id})[/green]")


@agents.command("delete")
@click.argument("agent_id")
def agents_delete(agent_id: str):
    """Delete an agent profile."""
    from openclaw.agents import AgentRegistry

    registry = AgentRegistry()
    if registry.delete(agent_id):
        console.print(f"[green]Agent '{agent_id}' deleted.[/green]")
    else:
        console.print(f"[red]Agent '{agent_id}' not found (or is the default 'main' agent).[/red]")


# -----------------------------------------------------------------------
# Config command group
# -----------------------------------------------------------------------

@main.group()
def config():
    """View and manage configuration."""
    pass


@config.command("show")
def config_show():
    """Show current configuration."""
    from openclaw.config import OpenClawConfig

    cfg = OpenClawConfig.load()
    data = cfg.model_dump()

    # Redact API keys
    if "api_keys" in data:
        for key in data["api_keys"]:
            val = data["api_keys"][key]
            if val:
                data["api_keys"][key] = val[:8] + "..."

    import json
    console.print_json(json.dumps(data, indent=2))


@config.command("set")
@click.argument("key")
@click.argument("value")
def config_set(key: str, value: str):
    """Set a configuration value (dot-notation, e.g. agent.model)."""
    from openclaw.config import OpenClawConfig

    cfg = OpenClawConfig.load()
    parts = key.split(".")

    if len(parts) == 2:
        section = getattr(cfg, parts[0], None)
        if section and hasattr(section, parts[1]):
            field_type = type(getattr(section, parts[1]))
            if field_type == bool:
                setattr(section, parts[1], value.lower() in ("true", "1", "yes"))
            elif field_type == int:
                setattr(section, parts[1], int(value))
            elif field_type == float:
                setattr(section, parts[1], float(value))
            else:
                setattr(section, parts[1], value)
            cfg.save()
            console.print(f"[green]Set {key} = {value}[/green]")
        else:
            console.print(f"[red]Unknown config key: {key}[/red]")
    else:
        console.print("[red]Use dot-notation: e.g. agent.model, gateway.port[/red]")


@config.command("path")
def config_path():
    """Show configuration file location."""
    from openclaw.config import CONFIG_SEARCH_PATHS
    from pathlib import Path

    for p in CONFIG_SEARCH_PATHS:
        resolved = p.expanduser().resolve()
        if resolved.exists():
            console.print(f"[green]Active config:[/green] {resolved}")
            return

    console.print("[dim]No config file found. Run: openclaw setup[/dim]")


if __name__ == "__main__":
    main()

