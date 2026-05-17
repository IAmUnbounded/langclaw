"""Task planning and artifact tools — structured work management.

Lets the agent create task plans with checklists and produce
named artifacts (markdown documents) for organized output.

Storage: ~/.langclaw/plans/ and ~/.langclaw/artifacts/
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

PLANS_DIR = Path.home() / ".langclaw" / "plans"
ARTIFACTS_DIR = Path.home() / ".langclaw" / "artifacts"


def _ensure_dirs():
    PLANS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


@tool
def create_plan_tool(title: str, phases: str) -> str:
    """Create a task plan with phases and checklist items.

    Use this to organize complex work into trackable phases.

    Args:
        title: Title of the plan (e.g. "Refactor auth module").
        phases: Phases in the format "Phase 1: Description | item1 | item2 ;; Phase 2: ..."
                Separate phases with ";;" and items with "|".

    Returns:
        Confirmation with plan ID and formatted plan.
    """
    _ensure_dirs()
    plan_id = str(uuid.uuid4())[:8]

    parsed_phases = []
    for phase_str in phases.split(";;"):
        phase_str = phase_str.strip()
        if not phase_str:
            continue
        parts = [p.strip() for p in phase_str.split("|")]
        phase_name = parts[0] if parts else "Phase"
        items = [{"text": item, "done": False} for item in parts[1:] if item]
        parsed_phases.append({"name": phase_name, "items": items})

    plan = {
        "plan_id": plan_id,
        "title": title,
        "phases": parsed_phases,
        "created_at": time.time(),
        "updated_at": time.time(),
        "status": "active",
    }

    plan_file = PLANS_DIR / f"{plan_id}.json"
    plan_file.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    # Format for display
    lines = [f"✅ Plan created: **{title}** (`{plan_id}`)\n"]
    for phase in parsed_phases:
        lines.append(f"### {phase['name']}")
        for item in phase["items"]:
            lines.append(f"  - [ ] {item['text']}")
        lines.append("")

    return "\n".join(lines)


@tool
def update_plan_tool(plan_id: str, phase_index: int, item_index: int, done: bool = True) -> str:
    """Update a checklist item in an existing plan.

    Args:
        plan_id: The plan ID to update.
        phase_index: 0-based phase index.
        item_index: 0-based item index within the phase.
        done: Whether the item is completed (default True).

    Returns:
        Updated plan status.
    """
    _ensure_dirs()
    plan_file = PLANS_DIR / f"{plan_id}.json"

    if not plan_file.exists():
        return f"❌ Plan not found: {plan_id}"

    plan = json.loads(plan_file.read_text(encoding="utf-8"))

    try:
        plan["phases"][phase_index]["items"][item_index]["done"] = done
        plan["updated_at"] = time.time()
    except (IndexError, KeyError):
        return f"❌ Invalid phase/item index: phase={phase_index}, item={item_index}"

    plan_file.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    # Check completion
    total = sum(len(p["items"]) for p in plan["phases"])
    completed = sum(1 for p in plan["phases"] for i in p["items"] if i["done"])
    item_text = plan["phases"][phase_index]["items"][item_index]["text"]

    status = "done" if done else "undone"
    return (
        f"✅ Marked as {status}: {item_text}\n"
        f"Progress: {completed}/{total} items complete"
    )


@tool
def create_artifact_tool(name: str, content: str) -> str:
    """Create a named markdown artifact document.

    Use this to produce structured output like reports, analyses,
    documentation, or implementation plans.

    Args:
        name: Name for the artifact (e.g. "analysis_results", "api_design").
              Will be saved as {name}.md
        content: Markdown content for the artifact.

    Returns:
        Confirmation with file path.
    """
    _ensure_dirs()
    # Sanitize name
    safe_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in name)
    if not safe_name.endswith(".md"):
        safe_name += ".md"

    artifact_path = ARTIFACTS_DIR / safe_name
    artifact_path.write_text(content, encoding="utf-8")

    return f"✅ Artifact created: {artifact_path}\n({len(content)} chars)"


@tool
def read_artifact_tool(name: str) -> str:
    """Read an existing artifact document.

    Args:
        name: Name of the artifact (with or without .md extension).

    Returns:
        Artifact content or error message.
    """
    _ensure_dirs()
    safe_name = name if name.endswith(".md") else f"{name}.md"
    artifact_path = ARTIFACTS_DIR / safe_name

    if not artifact_path.exists():
        # List available artifacts
        available = [f.stem for f in ARTIFACTS_DIR.glob("*.md")]
        if available:
            return f"❌ Artifact not found: {name}\nAvailable: {', '.join(available)}"
        return f"❌ Artifact not found: {name}. No artifacts exist yet."

    content = artifact_path.read_text(encoding="utf-8")
    return f"📄 **{name}**\n\n{content}"
