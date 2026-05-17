"""Skills loader — discovers and loads skill definitions from the workspace."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_skills(workspace: Path) -> list[dict[str, Any]]:
    """Load skills from workspace/skills/ and ~/.langclaw/skills/.

    Each skill is a directory containing a SKILL.md file.
    The SKILL.md can have YAML frontmatter with name/description.

    Returns list of dicts with: name, content, path
    """
    skills: list[dict[str, Any]] = []
    search_dirs = [
        workspace / "skills",
        Path.home() / ".langclaw" / "skills",
    ]

    for skill_dir in search_dirs:
        if not skill_dir.is_dir():
            continue
        for child in sorted(skill_dir.iterdir()):
            if not child.is_dir():
                continue
            skill_file = child / "SKILL.md"
            if not skill_file.is_file():
                continue

            try:
                content = skill_file.read_text(encoding="utf-8").strip()
            except Exception:
                continue

            # Parse optional YAML frontmatter
            name = child.name
            description = ""

            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    frontmatter = parts[1].strip()
                    body = parts[2].strip()
                    for line in frontmatter.split("\n"):
                        line = line.strip()
                        if line.startswith("name:"):
                            name = line.split(":", 1)[1].strip().strip("\"'")
                        elif line.startswith("description:"):
                            description = line.split(":", 1)[1].strip().strip("\"'")
                    content = body

            skills.append({
                "name": name,
                "description": description,
                "content": content,
                "path": str(skill_file),
            })

    return skills


def get_skills_prompt(workspace: Path) -> str:
    """Build a prompt section from all loaded skills."""
    skills = load_skills(workspace)
    if not skills:
        return ""

    parts = ["## Active Skills\n"]
    for skill in skills:
        desc = f" — {skill['description']}" if skill['description'] else ""
        parts.append(f"### {skill['name']}{desc}\n{skill['content']}\n")

    return "\n".join(parts)
