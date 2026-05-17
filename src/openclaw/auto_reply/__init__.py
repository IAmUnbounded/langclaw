"""Auto-reply system — command registry, directive parsing, and message handling.

Mirrors OpenClaw's auto-reply module:
- Command registry: built-in slash commands with authentication
- Directive parsing: model overrides, thinking levels, verbose modes
- Message envelope: normalized message format with metadata
- Group activation: mention-based activation in group chats
- Status reporting: agent status and capability reporting
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Message envelope
# ---------------------------------------------------------------------------

@dataclass
class MessageEnvelope:
    """Normalized inbound message with metadata and parsed directives."""

    content: str  # Raw message content
    sender_id: str = ""
    channel: str = ""
    group_id: str = ""
    reply_to: str = ""  # Message being replied to
    media: list[dict[str, Any]] = field(default_factory=list)  # Attached media
    timestamp: float = field(default_factory=time.time)

    # Parsed directives (populated by parse_directives)
    model_override: str = ""
    thinking_level: str = ""
    verbose: bool = False
    raw_content: str = ""  # Content after directive stripping
    is_command: bool = False
    command_name: str = ""
    command_args: str = ""

    # Group activation
    is_mention: bool = False  # Was the bot mentioned?
    is_reply_to_bot: bool = False  # Is this a reply to a bot message?
    is_dm: bool = False  # Is this a DM?


class GroupActivation(str, Enum):
    """How the bot should be activated in group chats."""

    ALWAYS = "always"      # Respond to all messages
    MENTION = "mention"    # Only respond when mentioned
    REPLY = "reply"        # Only respond to replies to bot messages
    OFF = "off"            # Don't respond in groups


# ---------------------------------------------------------------------------
# Directive parsing
# ---------------------------------------------------------------------------

# Directive patterns at the start of a message:
#   @model:gpt-4o  or  @model:sonnet
#   @think:high    or  @think:max
#   @verbose
#   /command args

DIRECTIVE_PATTERNS = {
    "model": re.compile(r"^@model:(\S+)\s*", re.IGNORECASE),
    "think": re.compile(r"^@think:(\S+)\s*", re.IGNORECASE),
    "verbose": re.compile(r"^@verbose\s*", re.IGNORECASE),
    "agent": re.compile(r"^@agent:(\S+)\s*", re.IGNORECASE),
}

COMMAND_PATTERN = re.compile(r"^/(\w+)\s*(.*)?$", re.DOTALL)


def parse_directives(envelope: MessageEnvelope) -> MessageEnvelope:
    """Parse directives from the message content.

    Directives are prefix tokens that modify agent behavior:
    - @model:NAME — override the model for this message
    - @think:LEVEL — set thinking level (off, low, mid, high, max)
    - @verbose — enable verbose output
    - @agent:NAME — route to a specific agent
    - /command — invoke a slash command

    Directives are stripped from the content after parsing.
    """
    content = envelope.content.strip()
    envelope.raw_content = content

    # Parse directives (can stack multiple)
    while True:
        matched = False

        model_match = DIRECTIVE_PATTERNS["model"].match(content)
        if model_match:
            envelope.model_override = model_match.group(1)
            content = content[model_match.end():]
            matched = True

        think_match = DIRECTIVE_PATTERNS["think"].match(content)
        if think_match:
            envelope.thinking_level = think_match.group(1).lower()
            content = content[think_match.end():]
            matched = True

        verbose_match = DIRECTIVE_PATTERNS["verbose"].match(content)
        if verbose_match:
            envelope.verbose = True
            content = content[verbose_match.end():]
            matched = True

        if not matched:
            break

    # Check for slash commands
    cmd_match = COMMAND_PATTERN.match(content)
    if cmd_match:
        envelope.is_command = True
        envelope.command_name = cmd_match.group(1).lower()
        envelope.command_args = (cmd_match.group(2) or "").strip()
        content = ""  # Commands consume the entire content

    envelope.raw_content = content.strip()
    return envelope


# ---------------------------------------------------------------------------
# Command registry
# ---------------------------------------------------------------------------

@dataclass
class Command:
    """A registered slash command."""

    name: str
    description: str
    handler: Callable[[MessageEnvelope, str], Awaitable[str]]
    requires_auth: bool = False  # Requires owner/admin
    hidden: bool = False  # Hidden from /help listing
    category: str = "general"


class CommandRegistry:
    """Registry of available slash commands.

    Commands are invoked via /name in the message content.
    Built-in commands include status, help, model, clear, etc.
    """

    def __init__(self) -> None:
        self._commands: dict[str, Command] = {}
        self._register_builtins()

    def register(
        self,
        name: str,
        description: str,
        handler: Callable[[MessageEnvelope, str], Awaitable[str]],
        requires_auth: bool = False,
        hidden: bool = False,
        category: str = "general",
    ) -> None:
        """Register a slash command."""
        self._commands[name.lower()] = Command(
            name=name.lower(),
            description=description,
            handler=handler,
            requires_auth=requires_auth,
            hidden=hidden,
            category=category,
        )

    def get(self, name: str) -> Command | None:
        """Get a command by name."""
        return self._commands.get(name.lower())

    def list_commands(self, include_hidden: bool = False) -> list[Command]:
        """List all registered commands."""
        cmds = list(self._commands.values())
        if not include_hidden:
            cmds = [c for c in cmds if not c.hidden]
        return sorted(cmds, key=lambda c: (c.category, c.name))

    async def execute(self, envelope: MessageEnvelope) -> str | None:
        """Execute a slash command from the envelope.

        Returns the command response, or None if not a command.
        """
        if not envelope.is_command:
            return None

        cmd = self.get(envelope.command_name)
        if not cmd:
            return f"Unknown command: /{envelope.command_name}. Type /help for available commands."

        try:
            return await cmd.handler(envelope, envelope.command_args)
        except Exception as e:
            logger.exception("Command /%s failed", envelope.command_name)
            return f"Command error: {e}"

    def _register_builtins(self) -> None:
        """Register built-in commands."""

        async def cmd_help(env: MessageEnvelope, args: str) -> str:
            lines = ["**Available Commands:**\n"]
            by_category: dict[str, list[Command]] = {}
            for cmd in self.list_commands():
                by_category.setdefault(cmd.category, []).append(cmd)
            for cat, cmds in sorted(by_category.items()):
                lines.append(f"\n**{cat.title()}**")
                for c in cmds:
                    lines.append(f"  `/{c.name}` — {c.description}")
            lines.append("\n**Directives:** `@model:NAME`, `@think:LEVEL`, `@verbose`")
            return "\n".join(lines)

        async def cmd_status(env: MessageEnvelope, args: str) -> str:
            from openclaw.config import OpenClawConfig
            config = OpenClawConfig.load()
            providers = []
            import os
            if os.environ.get("OPENAI_API_KEY") or config.api_keys.openai:
                providers.append("OpenAI")
            if os.environ.get("ANTHROPIC_API_KEY") or config.api_keys.anthropic:
                providers.append("Anthropic")
            if os.environ.get("GOOGLE_API_KEY") or config.api_keys.google:
                providers.append("Google")
            return (
                f"**Status:** Online\n"
                f"**Model:** {config.agent.model}\n"
                f"**Providers:** {', '.join(providers) or 'none configured'}\n"
                f"**Channel:** {env.channel}\n"
                f"**Thinking:** {config.agent.thinking_level}"
            )

        async def cmd_model(env: MessageEnvelope, args: str) -> str:
            if not args:
                from openclaw.config import OpenClawConfig
                config = OpenClawConfig.load()
                return f"Current model: `{config.agent.model}`\nUsage: `/model <name>` to switch."
            from openclaw.models import resolve_model, get_model_info
            resolved = resolve_model(args.strip())
            info = get_model_info(resolved)
            if info:
                return (
                    f"Switching to **{info.display_name}** (`{resolved}`)\n"
                    f"Context: {info.context_window:,} tokens | "
                    f"Vision: {'yes' if info.supports_vision else 'no'} | "
                    f"Thinking: {'yes' if info.supports_thinking else 'no'}"
                )
            return f"Model set to: `{resolved}` (not in catalog — will pass through)"

        async def cmd_models(env: MessageEnvelope, args: str) -> str:
            from openclaw.models import get_available_models
            models = get_available_models()
            if not models:
                return "No models available. Configure API keys in langclaw.json."
            lines = ["**Available Models:**\n"]
            for m in models:
                price = f"${m.input_price_per_1m:.2f}/${m.output_price_per_1m:.2f} per 1M tokens"
                flags = []
                if m.supports_vision:
                    flags.append("vision")
                if m.supports_thinking:
                    flags.append("thinking")
                flag_str = f" [{', '.join(flags)}]" if flags else ""
                lines.append(f"  `{m.provider}/{m.model_id}` — {m.display_name}{flag_str} ({price})")
            return "\n".join(lines)

        async def cmd_clear(env: MessageEnvelope, args: str) -> str:
            return "Session cleared. Starting fresh conversation."

        async def cmd_ping(env: MessageEnvelope, args: str) -> str:
            return "Pong!"

        async def cmd_whoami(env: MessageEnvelope, args: str) -> str:
            return (
                f"**Sender:** `{env.sender_id}`\n"
                f"**Channel:** `{env.channel}`\n"
                f"**Group:** `{env.group_id or 'N/A'}`"
            )

        async def cmd_think(env: MessageEnvelope, args: str) -> str:
            valid_levels = ["off", "low", "mid", "high", "max"]
            if args.strip().lower() not in valid_levels:
                return f"Usage: `/think <level>` where level is one of: {', '.join(valid_levels)}"
            return f"Thinking level set to: **{args.strip().lower()}**"

        async def cmd_verbose(env: MessageEnvelope, args: str) -> str:
            return "Verbose mode enabled for this conversation."

        async def cmd_version(env: MessageEnvelope, args: str) -> str:
            return "**OpenClaw-Lang** v0.1.0 (LangGraph Edition)"

        self.register("help", "Show available commands", cmd_help, category="general")
        self.register("status", "Show agent status", cmd_status, category="general")
        self.register("model", "Show or switch the current model", cmd_model, category="model")
        self.register("models", "List available models", cmd_models, category="model")
        self.register("clear", "Clear conversation history", cmd_clear, category="session")
        self.register("ping", "Check if the agent is alive", cmd_ping, category="general")
        self.register("whoami", "Show sender and channel info", cmd_whoami, category="general")
        self.register("think", "Set thinking level", cmd_think, category="model")
        self.register("verbose", "Enable verbose output", cmd_verbose, category="general")
        self.register("version", "Show version info", cmd_version, category="general")


# ---------------------------------------------------------------------------
# Group activation logic
# ---------------------------------------------------------------------------

def should_respond_in_group(
    envelope: MessageEnvelope,
    policy: GroupActivation = GroupActivation.MENTION,
    bot_names: list[str] | None = None,
) -> bool:
    """Determine if the bot should respond to a group message.

    Checks activation policy against the message envelope.
    """
    if envelope.is_dm:
        return True

    if policy == GroupActivation.ALWAYS:
        return True

    if policy == GroupActivation.OFF:
        return False

    if policy == GroupActivation.REPLY:
        return envelope.is_reply_to_bot

    # MENTION policy
    if envelope.is_mention:
        return True

    # Check if bot name appears in content
    if bot_names:
        content_lower = envelope.content.lower()
        for name in bot_names:
            if name.lower() in content_lower:
                return True

    return False


# ---------------------------------------------------------------------------
# Auto-reply processor
# ---------------------------------------------------------------------------

class AutoReplyProcessor:
    """Processes inbound messages through the auto-reply pipeline.

    Pipeline:
    1. Parse directives (@model, @think, @verbose)
    2. Check for slash commands (/help, /status, etc.)
    3. Apply group activation policy
    4. Return processed envelope or command response
    """

    def __init__(
        self,
        group_policy: GroupActivation = GroupActivation.MENTION,
        bot_names: list[str] | None = None,
    ) -> None:
        self.command_registry = CommandRegistry()
        self.group_policy = group_policy
        self.bot_names = bot_names or ["openclaw"]

    async def process(self, envelope: MessageEnvelope) -> tuple[MessageEnvelope, str | None]:
        """Process a message through the auto-reply pipeline.

        Returns:
            (processed_envelope, command_response)
            If command_response is not None, it should be sent back directly
            without passing to the agent.
        """
        # Parse directives
        envelope = parse_directives(envelope)

        # Check commands first
        if envelope.is_command:
            response = await self.command_registry.execute(envelope)
            return envelope, response

        # Check group activation
        if envelope.group_id and not should_respond_in_group(
            envelope, self.group_policy, self.bot_names,
        ):
            return envelope, None  # Skip — don't respond

        return envelope, None  # Continue to agent

    def register_command(
        self,
        name: str,
        description: str,
        handler: Callable[[MessageEnvelope, str], Awaitable[str]],
        **kwargs: Any,
    ) -> None:
        """Register a custom command."""
        self.command_registry.register(name, description, handler, **kwargs)


# Module-level singleton
_processor: AutoReplyProcessor | None = None


def get_auto_reply_processor() -> AutoReplyProcessor:
    """Return the global auto-reply processor singleton."""
    global _processor
    if _processor is None:
        _processor = AutoReplyProcessor()
    return _processor
