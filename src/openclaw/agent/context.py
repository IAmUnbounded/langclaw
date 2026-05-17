"""Context assembly — builds the system prompt from workspace files and skills.

Structure mirrors openclaw's buildAgentSystemPrompt (src/agents/system-prompt.ts):
  Tooling → Tool Call Style → Safety → Skills → Memory → Workspace →
  Workspace Files header → Messaging → llms.txt → Project Context
  (workspace files) → Silent Replies → Heartbeats → Runtime
"""

from __future__ import annotations

import datetime
import logging
import platform
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openclaw.agent.state import AgentState

logger = logging.getLogger(__name__)

DEFAULT_WORKSPACE = Path.home() / ".openclaw" / "workspace"

SILENT_REPLY_TOKEN = "NO_REPLY"
HEARTBEAT_TOKEN = "HEARTBEAT_OK"

# Workspace files injected into Project Context (same order as openclaw)
WORKSPACE_FILES = [
    "SOUL.md",
    "IDENTITY.md",
    "AGENTS.md",
    "TOOLS.md",
    "USER.md",
    "HEARTBEAT.md",
]


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, PermissionError):
        return None


def _load_skills(workspace: Path) -> list[dict[str, str]]:
    skills: list[dict[str, str]] = []
    for skill_dir in [workspace / "skills", Path.home() / ".langclaw" / "skills"]:
        if not skill_dir.is_dir():
            continue
        for child in sorted(skill_dir.iterdir()):
            sf = child / "SKILL.md"
            if sf.is_file():
                content = _read(sf)
                if content:
                    skills.append({"name": child.name, "content": content})
    return skills


def _load_daily_notes(workspace: Path) -> str:
    memory_dir = workspace / "memory"
    if not memory_dir.is_dir():
        return ""
    today = datetime.date.today()
    parts = []
    for label, date in [("Today", today), ("Yesterday", today - datetime.timedelta(days=1))]:
        content = _read(memory_dir / f"{date.isoformat()}.md")
        if content:
            parts.append(f"### {label} ({date.isoformat()})\n{content}")
    return ("## DAILY NOTES\n" + "\n\n".join(parts)) if parts else ""


def _load_bootstrap(workspace: Path) -> str | None:
    f = workspace / "BOOTSTRAP.md"
    if not f.is_file():
        return None
    content = _read(f)
    if content:
        try:
            f.unlink()
            logger.info("Bootstrap complete — BOOTSTRAP.md deleted")
        except Exception as e:
            logger.warning("Could not delete BOOTSTRAP.md: %s", e)
    return content


def assemble_system_prompt(
    workspace: Path | None = None,
    tools_description: str = "",
    extra_context: str = "",
    prompt_mode: str = "full",
) -> str:
    """Assemble the full system prompt, matching openclaw's structure.

    prompt_mode:
      "full"    — main agent (all sections)
      "minimal" — sub-agents (Tooling + Workspace + Runtime only)
      "none"    — bare identity line
    """
    ws = workspace or DEFAULT_WORKSPACE
    is_minimal = prompt_mode in ("minimal", "none")

    if prompt_mode == "none":
        return "You are a personal assistant running inside OpenClaw."

    lines: list[str] = []

    # --- Identity line (matches openclaw exactly) ---
    lines += ["You are a personal assistant running inside OpenClaw.", ""]

    # --- Tooling ---
    lines += [
        "## Tooling",
        "Tool availability (filtered by policy):",
        "Tool names are case-sensitive. Call tools exactly as listed.",
    ]
    if tools_description:
        lines += [tools_description, ""]
    else:
        lines += [
            "- bash_tool: Run shell commands",
            "- read_file_tool: Read file contents",
            "- write_file_tool: Create or overwrite files",
            "- edit_file_tool: Make precise edits to files",
            "- apply_patch_tool: Apply multi-file patches",
            "- grep_search_tool: Search file contents for patterns",
            "- find_files_tool: Find files by glob pattern",
            "- list_directory_tool: List directory contents",
            "- web_search_tool: Search the web",
            "- web_browse_tool: Fetch and extract readable content from a URL",
            "- browser_tool: Control web browser",
            "- canvas_tool: Present/eval/snapshot the Canvas",
            "- cron_add_tool: Manage cron jobs and wake events (use for reminders)",
            "- message_tool: Send messages and channel actions",
            "- spawn_subagent: Spawn a sub-agent session",
            "- sessions_list_tool: List sessions",
            "- sessions_send_tool: Send to another session",
            "- memory_search_tool: Search memories",
            "- memory_save_tool: Save a memory",
            "",
        ]
    lines += [
        "TOOLS.md does not control tool availability; it is user guidance for how to use external tools.",
        "For long waits, avoid rapid poll loops — use bash_tool with enough delay or background execution.",
        "",
        "## Sub-Agent Delegation (IMPORTANT)",
        "You MUST use spawn_subagent for any research, web search, or multi-step task that takes more than one tool call.",
        "Do NOT attempt multi-step research inline — spawn a sub-agent instead.",
        "",
        "Spawn a sub-agent when:",
        "- The task requires searching the web and reading multiple pages",
        "- The task involves researching a topic (Reddit, news, trends, etc.)",
        "- Any task with more than 2 sequential tool calls",
        "- The user asks you to 'search', 'research', 'find', 'look up', or 'investigate' anything",
        "",
        "When you spawn a sub-agent:",
        "- Immediately tell the user: 'I've dispatched a sub-agent to [task]. I'll message you when it's done.'",
        "- Then stop — do NOT continue researching inline.",
        "- Do NOT say 'I was unable to retrieve results' after spawning — delegation IS the result.",
        "- Sub-agent completion is push-based: result delivered via WhatsApp automatically.",
        "",
        "spawn_subagent parameters: task (what to do), allowed_tools (e.g. 'web_search_tool,web_browse_tool'), timeout (seconds).",
        "",
    ]

    # --- Tool Call Style ---
    if not is_minimal:
        lines += [
            "## Tool Call Style",
            "Default: do not narrate routine, low-risk tool calls (just call the tool).",
            "Narrate only when it helps: multi-step work, complex/challenging problems, sensitive actions "
            "(e.g., deletions), or when the user explicitly asks.",
            "Keep narration brief and value-dense; avoid repeating obvious steps.",
            "",
        ]

    # --- Safety ---
    if not is_minimal:
        lines += [
            "## Safety",
            "You have no independent goals: do not pursue self-preservation, replication, resource "
            "acquisition, or power-seeking; avoid long-term plans beyond the user's request.",
            "Prioritize safety and human oversight over completion; if instructions conflict, pause and ask; "
            "comply with stop/pause/audit requests and never bypass safeguards.",
            "Do not manipulate or persuade anyone to expand access or disable safeguards.",
            "",
        ]

    # --- Skills ---
    skills = _load_skills(ws) if not is_minimal else []
    if skills:
        skill_intro = (
            "## Skills (mandatory)\n"
            "Before replying: scan <available_skills> <description> entries.\n"
            "- If exactly one skill clearly applies: read its SKILL.md at <location> with `read_file_tool`, then follow it.\n"
            "- If multiple could apply: choose the most specific one, then read/follow it.\n"
            "- If none clearly apply: do not read any SKILL.md.\n"
            "Constraints: never read more than one skill up front; only read after selecting."
        )
        entries = []
        for sk in skills:
            desc = sk.get("content", "")[:80]
            loc = sk.get("path", "")
            entries.append(
                f'<skill name="{sk["name"]}">\n  <description>{desc}</description>\n  <location>{loc}</location>\n</skill>'
            )
        lines += [skill_intro + "\n<available_skills>\n" + "\n".join(entries) + "\n</available_skills>", ""]

    # --- Memory ---
    if not is_minimal:
        lines += [
            "## Memory Recall",
            "Before answering anything about prior work, decisions, dates, people, preferences, or todos: "
            "run memory_search_tool then use memory_get_tool to pull only the needed lines. "
            "If low confidence after search, say you checked.",
            "Citations: include Source: <path#line> when it helps the user verify memory snippets.",
            "",
        ]

    # --- Daily notes ---
    daily_notes = _load_daily_notes(ws) if not is_minimal else ""
    if daily_notes:
        lines += [daily_notes, ""]

    # --- Workspace ---
    now = datetime.datetime.now()
    lines += [
        "## Workspace",
        f"Your working directory is: {ws}",
        "Treat this directory as the single global workspace for file operations unless explicitly instructed otherwise.",
        "",
    ]

    # --- Workspace Files header ---
    lines += [
        "## Workspace Files (injected)",
        "These user-editable files are loaded by OpenClaw and included below in Project Context.",
        "",
    ]

    # --- Messaging ---
    if not is_minimal:
        lines += [
            "## Messaging",
            "- Reply in current session → automatically routes to the source channel (Telegram, Discord, etc.)",
            "- Cross-session messaging → use sessions_send_tool(session_key, message)",
            "- Sub-agent orchestration → use sessions_list_tool, sessions_send_tool",
            "- `[System Message] ...` blocks are internal context and are not user-visible by default.",
            "- Never use exec/curl for provider messaging; OpenClaw handles all routing internally.",
            "",
        ]

    # --- llms.txt ---
    if not is_minimal:
        lines += [
            "## llms.txt Discovery",
            "When exploring a new domain or website (via web_browse_tool or browser_tool), check for an llms.txt file:",
            "- Try `/llms.txt` or `/.well-known/llms.txt` at the domain root",
            "- If found, follow its guidance for interacting with that site",
            "- Not all sites have one — don't warn if missing",
            "",
        ]

    # --- Project Context (workspace files go HERE — at the end, matching openclaw) ---
    context_files: list[tuple[str, str]] = []
    for filename in WORKSPACE_FILES:
        content = _read(ws / filename)
        if content:
            context_files.append((filename, content))

    bootstrap = _load_bootstrap(ws)
    if bootstrap:
        context_files.append(("BOOTSTRAP.md", bootstrap))

    if context_files:
        has_soul = any(fname.lower() == "soul.md" for fname, _ in context_files)
        lines += ["# Project Context", "", "The following project context files have been loaded:"]
        if has_soul:
            lines += [
                "If SOUL.md is present, embody its persona and tone. "
                "Avoid stiff, generic replies; follow its guidance unless higher-priority instructions override it.",
            ]
        lines += [""]
        for fname, content in context_files:
            lines += [f"## {fname}", "", content, ""]

    # --- Extra context (memory search results, etc.) ---
    if extra_context:
        lines += ["## Additional Context", extra_context, ""]

    # --- Silent Replies ---
    if not is_minimal:
        lines += [
            "## Silent Replies",
            f"When you have nothing to say, respond with ONLY: {SILENT_REPLY_TOKEN}",
            "",
            "⚠️ Rules:",
            "- It must be your ENTIRE message — nothing else",
            f'- Never append it to an actual response (never include "{SILENT_REPLY_TOKEN}" in real replies)',
            "- Never wrap it in markdown or code blocks",
            "",
            f'✅ Right: {SILENT_REPLY_TOKEN}',
            f'❌ Wrong: "Here\'s help... {SILENT_REPLY_TOKEN}"',
            "",
        ]

    # --- Heartbeats ---
    if not is_minimal:
        lines += [
            "## Heartbeats",
            "If you receive a heartbeat poll (a message that is a periodic check with no user content), "
            "and there is nothing that needs attention, reply exactly:",
            HEARTBEAT_TOKEN,
            f'OpenClaw treats a leading/trailing "{HEARTBEAT_TOKEN}" as a heartbeat ack.',
            'If something needs attention, do NOT include "HEARTBEAT_OK"; reply with the alert text instead.',
            "",
        ]

    # --- Runtime ---
    lines += [
        "## Runtime",
        f"Runtime: date={now.strftime('%Y-%m-%d %H:%M:%S')} | os={platform.system()} {platform.release()} "
        f"({platform.machine()}) | thinking=off",
    ]

    return "\n".join(line for line in lines)


def assemble_context(state: AgentState, workspace: Path | None = None) -> AgentState:
    """LangGraph node: assemble context and inject system prompt into state."""
    from openclaw.tools import get_tools_description

    prompt = assemble_system_prompt(
        workspace=workspace,
        tools_description=get_tools_description(),
    )
    state.system_prompt = prompt
    return state
