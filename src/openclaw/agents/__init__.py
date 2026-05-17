"""Agents registry — manage multiple agent profiles.

Each agent profile defines a specific persona, model, tool set,
and behavior for different use cases (e.g., "researcher", "coder", "writer").

Agent profiles are stored in ~/.langclaw/agents/ as JSON files.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


AGENTS_DIR = Path.home() / ".langclaw" / "agents"


@dataclass
class AgentProfile:
    """A single agent profile configuration."""

    agent_id: str
    name: str
    description: str
    model: str = ""  # override, empty = use default
    temperature: float = 0.7
    tools: list[str] = field(default_factory=list)  # empty = all tools
    system_prompt_extra: str = ""  # additional system prompt content
    max_iterations: int = 25
    thinking_level: str = "off"  # off, low, mid, high, max
    created_at: float = 0.0
    updated_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize profile to a dictionary."""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "model": self.model,
            "temperature": self.temperature,
            "tools": self.tools,
            "system_prompt_extra": self.system_prompt_extra,
            "max_iterations": self.max_iterations,
            "thinking_level": self.thinking_level,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentProfile:
        """Deserialize profile from a dictionary."""
        return cls(
            agent_id=data["agent_id"],
            name=data["name"],
            description=data["description"],
            model=data.get("model", ""),
            temperature=data.get("temperature", 0.7),
            tools=data.get("tools", []),
            system_prompt_extra=data.get("system_prompt_extra", ""),
            max_iterations=data.get("max_iterations", 25),
            thinking_level=data.get("thinking_level", "off"),
            created_at=data.get("created_at", 0.0),
            updated_at=data.get("updated_at", 0.0),
        )


# Built-in default agent profile
_DEFAULT_AGENT = AgentProfile(
    agent_id="main",
    name="Main Agent",
    description="Default general-purpose agent with all tools enabled.",
    model="",
    temperature=0.7,
    tools=[],
    system_prompt_extra="",
    max_iterations=25,
    thinking_level="off",
    created_at=0.0,
    updated_at=0.0,
)


class AgentRegistry:
    """Registry for managing agent profiles on disk.

    Profiles are persisted as individual JSON files in ``~/.langclaw/agents/``.
    A built-in "main" profile is always available as the default.
    """

    def __init__(self, agents_dir: Path | None = None) -> None:
        self._dir = agents_dir or AGENTS_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        # Ensure the built-in default agent exists on disk
        if not (self._dir / "main.json").exists():
            self._save(_DEFAULT_AGENT)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self, profile: AgentProfile) -> AgentProfile:
        """Register a new agent profile.

        Timestamps are set automatically. If an agent with the same ID
        already exists the existing profile is overwritten.
        """
        now = time.time()
        profile.created_at = now
        profile.updated_at = now
        self._save(profile)
        return profile

    def get(self, agent_id: str) -> AgentProfile | None:
        """Return a single profile by ID, or ``None`` if not found."""
        path = self._dir / f"{agent_id}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return AgentProfile.from_dict(data)
        except (json.JSONDecodeError, KeyError):
            return None

    def list_all(self) -> list[AgentProfile]:
        """Return every registered agent profile."""
        return self._load_all()

    def update(self, agent_id: str, **kwargs: Any) -> AgentProfile | None:
        """Update fields of an existing agent profile.

        Returns the updated profile, or ``None`` if the agent ID does not
        exist.
        """
        profile = self.get(agent_id)
        if profile is None:
            return None
        for key, value in kwargs.items():
            if hasattr(profile, key) and key not in ("agent_id", "created_at"):
                setattr(profile, key, value)
        profile.updated_at = time.time()
        self._save(profile)
        return profile

    def delete(self, agent_id: str) -> bool:
        """Delete an agent profile. Returns ``True`` if it existed."""
        if agent_id == "main":
            return False  # cannot delete the built-in default
        path = self._dir / f"{agent_id}.json"
        if path.exists():
            path.unlink()
            return True
        return False

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _save(self, profile: AgentProfile) -> None:
        """Write a profile to its JSON file."""
        path = self._dir / f"{profile.agent_id}.json"
        path.write_text(
            json.dumps(profile.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )

    def _load_all(self) -> list[AgentProfile]:
        """Load every JSON profile from the agents directory."""
        profiles: list[AgentProfile] = []
        for path in sorted(self._dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                profiles.append(AgentProfile.from_dict(data))
            except (json.JSONDecodeError, KeyError):
                continue
        return profiles
