"""Configuration module — Pydantic models for langclaw.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

CONFIG_SEARCH_PATHS = [
    Path("./langclaw.json"),
    Path.home() / ".langclaw" / "langclaw.json",
]


class AgentConfig(BaseModel):
    """Agent configuration."""

    model: str = Field(
        default="openai/gpt-4o",
        description="Model to use in provider/model format (e.g. openai/gpt-4o, anthropic/claude-sonnet-4-20250514, google/gemini-2.0-flash).",
    )
    workspace: str = Field(
        default="~/.langclaw/workspace",
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
    thinking_level: str = Field(
        default="off",
        description="Extended thinking level: off, low, mid, high, max.",
    )
    fallback_models: list[str] = Field(
        default_factory=list,
        description="Fallback model chain (e.g. ['openai/gpt-4o', 'google/gemini-2.0-flash']).",
    )
    sandbox_mode: bool = Field(
        default=False,
        description="Enable sandbox mode (restricts filesystem/shell tools).",
    )
    max_concurrent_agents: int = Field(
        default=3,
        description="Maximum concurrent sub-agent runs.",
    )
    compaction_threshold: int = Field(
        default=200,
        description="Number of messages before auto-compaction.",
    )
    image_model: str = Field(
        default="",
        description="Model to use for image generation/analysis (empty = same as agent model).",
    )


class GatewayConfig(BaseModel):
    """Gateway server configuration."""

    port: int = Field(default=18789, description="Gateway port.")
    host: str = Field(default="127.0.0.1", description="Bind host.")
    cors_origins: list[str] = Field(
        default=["*"],
        description="Allowed CORS origins.",
    )
    require_auth: bool = Field(
        default=False,
        description="Require authentication for API access.",
    )
    allow_loopback: bool = Field(
        default=True,
        description="Allow unauthenticated access from localhost.",
    )
    rate_limit_per_minute: int = Field(
        default=60,
        description="Maximum requests per minute per IP.",
    )
    rate_limit_per_hour: int = Field(
        default=1000,
        description="Maximum requests per hour per IP.",
    )


class SecurityConfig(BaseModel):
    """Security and audit configuration."""

    audit_enabled: bool = Field(
        default=True,
        description="Enable security audit logging.",
    )
    audit_retention_days: int = Field(
        default=90,
        description="Days to retain audit logs.",
    )
    scan_workspace: bool = Field(
        default=True,
        description="Scan workspace for secrets on startup.",
    )
    block_dangerous_urls: bool = Field(
        default=True,
        description="Block SSRF-prone URLs (localhost, metadata endpoints).",
    )
    max_tool_output_chars: int = Field(
        default=15000,
        description="Maximum characters in tool output before truncation.",
    )


class DaemonConfig(BaseModel):
    """Daemon/service configuration."""

    auto_start: bool = Field(
        default=False,
        description="Auto-start gateway on system boot.",
    )
    log_dir: str = Field(
        default="~/.langclaw/logs",
        description="Directory for daemon log files.",
    )
    restart_on_failure: bool = Field(
        default=True,
        description="Restart the service on crash.",
    )


class RoutingConfig(BaseModel):
    """Message routing configuration."""

    default_agent: str = Field(
        default="main",
        description="Default agent ID for unrouted messages.",
    )
    sticky_sessions: bool = Field(
        default=True,
        description="Keep sender-to-session bindings across restarts.",
    )
    session_ttl_days: int = Field(
        default=30,
        description="Days before inactive sessions expire.",
    )


class ToolPolicyConfig(BaseModel):
    """Tool policy configuration."""

    profile: str = Field(
        default="full",
        description="Tool profile: minimal, coding, messaging, full.",
    )
    groups: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Group-specific tool policies.",
    )
    users: list[dict[str, Any]] = Field(
        default_factory=list,
        description="User-specific tool policies.",
    )


class ModelAliasConfig(BaseModel):
    """Custom model aliases."""

    aliases: dict[str, str] = Field(
        default_factory=dict,
        description="Custom model aliases (e.g. {'fast': 'openai/gpt-4o-mini'}).",
    )


class TelegramChannelConfig(BaseModel):
    """Telegram channel configuration."""

    enabled: bool = False
    token: str = ""
    allow_from: list[str] = Field(default_factory=list)
    dm_policy: str = "pairing"
    group_policy: str = "mention"
    parse_mode: str = "Markdown"


class DiscordChannelConfig(BaseModel):
    """Discord channel configuration."""

    enabled: bool = False
    token: str = ""
    allow_from: list[str] = Field(default_factory=list)
    dm_policy: str = "pairing"
    presence_status: str = ""
    presence_activity: str = ""


class SlackChannelConfig(BaseModel):
    """Slack channel configuration."""

    enabled: bool = False
    bot_token: str = ""
    app_token: str = ""
    allow_from: list[str] = Field(default_factory=list)
    threading: str = "first"
    dm_policy: str = "open"


class SignalChannelConfig(BaseModel):
    """Signal channel configuration."""

    enabled: bool = False
    phone_number: str = ""
    signal_cli_path: str = ""
    allow_from: list[str] = Field(default_factory=list)
    dm_policy: str = "pairing"
    group_policy: str = "mention"


class LineChannelConfig(BaseModel):
    """LINE channel configuration."""

    enabled: bool = False
    channel_access_token: str = ""
    channel_secret: str = ""
    allow_from: list[str] = Field(default_factory=list)


class WhatsAppChannelConfig(BaseModel):
    """WhatsApp channel configuration."""

    enabled: bool = False
    session_name: str = "openclaw-wa"
    allow_from: list[str] = Field(default_factory=list)
    dm_policy: str = "open"
    group_policy: str = "mention"
    group_history_count: int = 50
    max_message_length: int = 4000


class WebChatChannelConfig(BaseModel):
    """WebChat channel configuration."""

    enabled: bool = True


class ChannelsConfig(BaseModel):
    """Channels configuration."""

    telegram: TelegramChannelConfig = Field(default_factory=TelegramChannelConfig)
    discord: DiscordChannelConfig = Field(default_factory=DiscordChannelConfig)
    whatsapp: WhatsAppChannelConfig = Field(default_factory=WhatsAppChannelConfig)
    webchat: WebChatChannelConfig = Field(default_factory=WebChatChannelConfig)
    slack: SlackChannelConfig = Field(default_factory=SlackChannelConfig)
    signal: SignalChannelConfig = Field(default_factory=SignalChannelConfig)
    line: LineChannelConfig = Field(default_factory=LineChannelConfig)


class ApiKeysConfig(BaseModel):
    """API keys for LLM providers."""

    openai: str = Field(default="", description="OpenAI API key (or set OPENAI_API_KEY env var).")
    anthropic: str = Field(default="", description="Anthropic API key (or set ANTHROPIC_API_KEY env var).")
    google: str = Field(default="", description="Google AI API key (or set GOOGLE_API_KEY env var).")
    elevenlabs: str = Field(default="", description="ElevenLabs API key for TTS.")
    voyage: str = Field(default="", description="Voyage AI API key for embeddings.")
    tinyfish: str = Field(default="", description="Tinyfish API key for web search and fetch (or set TINYFISH_API_KEY env var).")


class MCPServerConfig(BaseModel):
    """Configuration for an MCP (Model Context Protocol) server."""

    command: str = Field(description="Command to start the MCP server.")
    args: list[str] = Field(default_factory=list, description="Arguments for the command.")
    env: dict[str, str] = Field(default_factory=dict, description="Environment variables.")


class OpenClawConfig(BaseModel):
    """Root configuration model for langclaw.json."""

    agent: AgentConfig = Field(default_factory=AgentConfig)
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    channels: ChannelsConfig = Field(default_factory=ChannelsConfig)
    api_keys: ApiKeysConfig = Field(default_factory=ApiKeysConfig)
    mcp_servers: dict[str, MCPServerConfig] = Field(
        default_factory=dict,
        description="MCP server configurations keyed by name.",
    )
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    daemon: DaemonConfig = Field(default_factory=DaemonConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    tool_policies: ToolPolicyConfig = Field(default_factory=ToolPolicyConfig)
    model_aliases: ModelAliasConfig = Field(default_factory=ModelAliasConfig)

    @classmethod
    def load(cls, path: Path | None = None) -> "OpenClawConfig":
        """Load configuration from a JSON file.

        Search order:
        1. Explicit path argument
        2. ./langclaw.json
        3. ~/.langclaw/langclaw.json
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
        target = path or Path.home() / ".langclaw" / "langclaw.json"
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
