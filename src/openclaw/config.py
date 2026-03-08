"""Configuration module — Pydantic models for openclaw.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

CONFIG_SEARCH_PATHS = [
    Path("./openclaw.json"),
    Path.home() / ".openclaw" / "openclaw.json",
]


class AgentConfig(BaseModel):
    """Agent configuration."""

    model: str = Field(
        default="openai/gpt-4o",
        description="Model to use in provider/model format (e.g. openai/gpt-4o, anthropic/claude-sonnet-4-20250514, google/gemini-2.0-flash).",
    )
    workspace: str = Field(
        default="~/.openclaw/workspace",
        description="Path to the agent workspace directory.",
    )
    timeout_seconds: int = Field(
        default=600,
        description="Maximum seconds for an agent run.",
    )
    max_iterations: int = Field(
        default=25,
        description="Maximum ReAct loop iterations before stopping.",
    )
    temperature: float = Field(
        default=0.7,
        description="LLM temperature.",
    )
    max_context_tokens: int = Field(
        default=100000,
        description="Maximum context window tokens before compression.",
    )
    skip_bootstrap: bool = Field(
        default=False,
        description="Skip the bootstrap ritual.",
    )


class GatewayConfig(BaseModel):
    """Gateway server configuration."""

    port: int = Field(default=18789, description="Gateway port.")
    host: str = Field(default="127.0.0.1", description="Bind host.")
    cors_origins: list[str] = Field(
        default=["*"],
        description="Allowed CORS origins.",
    )


class TelegramChannelConfig(BaseModel):
    """Telegram channel configuration."""

    enabled: bool = False
    token: str = ""
    allow_from: list[str] = Field(default_factory=list)
    dm_policy: str = "pairing"


class DiscordChannelConfig(BaseModel):
    """Discord channel configuration."""

    enabled: bool = False
    token: str = ""
    allow_from: list[str] = Field(default_factory=list)
    dm_policy: str = "pairing"


class WebChatChannelConfig(BaseModel):
    """WebChat channel configuration."""

    enabled: bool = True


class ChannelsConfig(BaseModel):
    """Channels configuration."""

    telegram: TelegramChannelConfig = Field(default_factory=TelegramChannelConfig)
    discord: DiscordChannelConfig = Field(default_factory=DiscordChannelConfig)
    webchat: WebChatChannelConfig = Field(default_factory=WebChatChannelConfig)


class ApiKeysConfig(BaseModel):
    """API keys for LLM providers."""

    openai: str = Field(default="", description="OpenAI API key (or set OPENAI_API_KEY env var).")
    anthropic: str = Field(default="", description="Anthropic API key (or set ANTHROPIC_API_KEY env var).")
    google: str = Field(default="", description="Google AI API key (or set GOOGLE_API_KEY env var).")


class MCPServerConfig(BaseModel):
    """Configuration for an MCP (Model Context Protocol) server."""

    command: str = Field(description="Command to start the MCP server.")
    args: list[str] = Field(default_factory=list, description="Arguments for the command.")
    env: dict[str, str] = Field(default_factory=dict, description="Environment variables.")


class OpenClawConfig(BaseModel):
    """Root configuration model for openclaw.json."""

    agent: AgentConfig = Field(default_factory=AgentConfig)
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    channels: ChannelsConfig = Field(default_factory=ChannelsConfig)
    api_keys: ApiKeysConfig = Field(default_factory=ApiKeysConfig)
    mcp_servers: dict[str, MCPServerConfig] = Field(
        default_factory=dict,
        description="MCP server configurations keyed by name.",
    )

    @classmethod
    def load(cls, path: Path | None = None) -> "OpenClawConfig":
        """Load configuration from a JSON file.

        Search order:
        1. Explicit path argument
        2. ./openclaw.json
        3. ~/.openclaw/openclaw.json
        """
        if path and path.exists():
            return cls._from_file(path)

        for search_path in CONFIG_SEARCH_PATHS:
            resolved = search_path.expanduser().resolve()
            if resolved.exists():
                return cls._from_file(resolved)

        # Return defaults if no config file found
        return cls()

    @classmethod
    def _from_file(cls, path: Path) -> "OpenClawConfig":
        """Load config from a specific file."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls(**data)
        except (json.JSONDecodeError, TypeError) as e:
            raise ValueError(f"Invalid config file {path}: {e}") from e

    def save(self, path: Path | None = None) -> None:
        """Save configuration to a JSON file."""
        target = path or Path.home() / ".openclaw" / "openclaw.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.model_dump(), indent=2),
            encoding="utf-8",
        )

    def get_workspace_path(self) -> Path:
        """Resolve the workspace path."""
        return Path(self.agent.workspace).expanduser().resolve()

    def parse_model(self) -> tuple[str, str]:
        """Parse provider/model string into (provider, model_name)."""
        model_str = self.agent.model
        if "/" in model_str:
            parts = model_str.split("/", 1)
            return parts[0], parts[1]
        # Default to openai if no provider specified
        return "openai", model_str
