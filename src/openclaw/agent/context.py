"""Context assembly — builds the system prompt from workspace files and skills."""

from __future__ import annotations

import datetime
import platform
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openclaw.agent.state import AgentState

# Default workspace directory
DEFAULT_WORKSPACE = Path.home() / ".openclaw" / "workspace"

# Bootstrap files injected into the system prompt (same as OpenClaw)
BOOTSTRAP_FILES = [
    "IDENTITY.md",
    "SOUL.md",
    "AGENTS.md",
    "TOOLS.md",
    "USER.md",
]


def _read_file_safe(path: Path) -> str | None:
    """Read a file, returning None if it doesn't exist."""
    try:
        return path.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, PermissionError):
        return None


def _load_skills(workspace: Path) -> list[dict[str, str]]:
    """Load skills from workspace/skills/ and ~/.openclaw/skills/."""
    skills: list[dict[str, str]] = []
    skill_dirs = [
        workspace / "skills",
        Path.home() / ".openclaw" / "skills",
    ]
    for skill_dir in skill_dirs:
        if not skill_dir.is_dir():
            continue
        for child in sorted(skill_dir.iterdir()):
            skill_file = child / "SKILL.md"
            if skill_file.is_file():
                content = _read_file_safe(skill_file)
                if content:
                    skills.append({"name": child.name, "content": content})
    return skills


def assemble_system_prompt(
    workspace: Path | None = None,
    tools_description: str = "",
    extra_context: str = "",
) -> str:
    """Assemble the full system prompt from workspace files, skills, and runtime info.

    This mirrors OpenClaw's prompt assembly pipeline:
    1. Base identity + persona from IDENTITY.md / SOUL.md
    2. Operating instructions from AGENTS.md
    3. Tool notes from TOOLS.md
    4. User profile from USER.md
    5. Skills prompts from workspace/skills/
    6. Runtime info (date/time, platform, available tools)
    """
    ws = workspace or DEFAULT_WORKSPACE
    sections: list[str] = []

    # --- Header ---
    sections.append(
        "You are an AI personal assistant powered by OpenClaw-Lang. "
        "You help the user by answering questions, running tools, "
        "executing shell commands, reading/writing files, searching the web, "
        "and automating tasks."
    )

    # --- Bootstrap files ---
    for filename in BOOTSTRAP_FILES:
        content = _read_file_safe(ws / filename)
        if content:
            label = filename.replace(".md", "").upper()
            sections.append(f"## {label}\n{content}")

    # --- Skills ---
    skills = _load_skills(ws)
    if skills:
        skill_parts = []
        for skill in skills:
            skill_parts.append(f"### Skill: {skill['name']}\n{skill['content']}")
        sections.append("## ACTIVE SKILLS\n" + "\n\n".join(skill_parts))

    # --- Available tools ---
    if tools_description:
        sections.append(f"## AVAILABLE TOOLS\n{tools_description}")

    # --- Runtime context ---
    now = datetime.datetime.now()
    runtime_info = (
        f"## RUNTIME CONTEXT\n"
        f"- Current date/time: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
        f"- Platform: {platform.system()} {platform.release()} ({platform.machine()})\n"
        f"- Working directory: {Path.cwd()}\n"
        f"- Workspace: {ws}"
    )
    sections.append(runtime_info)

    # --- Extra context (memory, etc.) ---
    if extra_context:
        sections.append(f"## ADDITIONAL CONTEXT\n{extra_context}")

    return "\n\n".join(sections)


def assemble_context(state: AgentState, workspace: Path | None = None) -> AgentState:
    """LangGraph node: assemble context and inject system prompt into state."""
    from openclaw.tools import get_tools_description

    prompt = assemble_system_prompt(
        workspace=workspace,
        tools_description=get_tools_description(),
    )
    state.system_prompt = prompt
    return state
