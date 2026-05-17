"""Tool Policy Pipeline — channel-based tool permissions, profiles, and sandbox gating.

Mirrors OpenClaw's tool-policy-pipeline.ts:
- Channel default policies (CLI=full, messaging=restricted)
- Sandbox mode (blocks filesystem/shell access)
- Tool profiles (minimal, coding, messaging, full)
- Tool groups (categorized tool collections)
- Group-specific overrides
- Per-user allowlists/blocklists
- Owner-only tool restrictions
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool groups — categorized tool collections
# ---------------------------------------------------------------------------

TOOL_GROUPS: dict[str, list[str]] = {
    "file_ops": [
        "bash_tool", "read_file_tool", "write_file_tool",
        "edit_file_tool", "list_directory_tool",
    ],
    "web": [
        "web_search_tool", "web_browse_tool", "browser_tool",
    ],
    "git": [
        "git_status_tool", "git_diff_tool", "git_log_tool",
        "git_commit_tool", "git_branch_tool",
    ],
    "code": [
        "grep_search_tool", "find_files_tool", "code_outline_tool",
    ],
    "memory": [
        "memory_save_tool", "memory_search_tool", "memory_list_tool",
        "daily_note_read_tool", "daily_note_write_tool", "daily_note_list_tool",
    ],
    "terminal": [
        "terminal_run_tool", "terminal_read_tool",
        "terminal_send_tool", "terminal_kill_tool",
    ],
    "scheduling": [
        "cron_add_tool", "cron_list_tool", "cron_remove_tool", "cron_run_tool",
    ],
    "artifacts": [
        "create_plan_tool", "update_plan_tool",
        "create_artifact_tool", "read_artifact_tool",
    ],
    "media": [
        "image_tool", "canvas_tool", "tts_tool",
    ],
    "agents": [
        "spawn_subagent", "agents_list_tool", "agents_create_tool",
        "agents_get_tool", "agents_delete_tool",
    ],
    "sessions": [
        "sessions_list_tool", "sessions_history_tool", "sessions_send_tool",
        "sessions_spawn_tool", "sessions_status_tool",
    ],
    "messaging": [
        "message_tool", "slack_actions_tool",
        "discord_actions_tool", "telegram_actions_tool",
        "whatsapp_actions_tool",
    ],
}

# Flatten for reverse lookup
_TOOL_TO_GROUP: dict[str, str] = {}
for _group_name, _tools in TOOL_GROUPS.items():
    for _tool_name in _tools:
        _TOOL_TO_GROUP[_tool_name] = _group_name


def get_tool_group(tool_name: str) -> str:
    """Get the group a tool belongs to."""
    return _TOOL_TO_GROUP.get(tool_name, "other")


# ---------------------------------------------------------------------------
# Tool profiles — pre-defined tool sets
# ---------------------------------------------------------------------------

TOOL_PROFILES: dict[str, list[str]] = {
    "minimal": [
        # Read-only tools only
        "read_file_tool", "list_directory_tool",
        "web_search_tool", "web_browse_tool",
        "memory_search_tool", "memory_list_tool",
        "daily_note_read_tool", "daily_note_list_tool",
        "git_status_tool", "git_diff_tool", "git_log_tool",
        "grep_search_tool", "find_files_tool", "code_outline_tool",
        "cron_list_tool",
        "read_artifact_tool",
        "sessions_list_tool", "sessions_history_tool", "sessions_status_tool",
        "agents_list_tool", "agents_get_tool",
    ],
    "coding": [
        # All code-related tools
        *TOOL_GROUPS["file_ops"],
        *TOOL_GROUPS["git"],
        *TOOL_GROUPS["code"],
        *TOOL_GROUPS["terminal"],
        *TOOL_GROUPS["artifacts"],
        "web_search_tool", "web_browse_tool",
        "memory_save_tool", "memory_search_tool", "memory_list_tool",
        "spawn_subagent",
    ],
    "messaging": [
        # Communication and read-only tools
        *TOOL_GROUPS["messaging"],
        *TOOL_GROUPS["memory"],
        *TOOL_GROUPS["sessions"],
        *TOOL_GROUPS["scheduling"],
        "web_search_tool", "web_browse_tool",
        "read_file_tool", "list_directory_tool",
        "image_tool", "tts_tool",
    ],
    "full": [],  # Empty = all tools allowed (no filtering)
}


def get_profile_tools(profile: str) -> list[str] | None:
    """Get the tool list for a profile. Returns None for 'full' (all tools)."""
    if profile not in TOOL_PROFILES:
        logger.warning("Unknown tool profile: %s, defaulting to 'full'", profile)
        return None
    tools = TOOL_PROFILES[profile]
    return tools if tools else None  # Empty list means all tools


# ---------------------------------------------------------------------------
# Owner-only tools
# ---------------------------------------------------------------------------

OWNER_ONLY_TOOLS: set[str] = {
    "agents_create_tool",
    "agents_delete_tool",
    "cron_add_tool",
    "cron_remove_tool",
    "cron_run_tool",
}


@dataclass
class ToolPolicy:
    """Defines which tools are allowed in a given context."""

    allowed_tools: list[str] | None = None  # None = all tools allowed
    blocked_tools: list[str] | None = None  # Explicit blocklist
    sandbox_mode: bool = False
    max_tool_output_chars: int = 15000
    profile: str = "full"  # Tool profile to apply


# Default tools blocked in messaging channels (security-sensitive)
_MESSAGING_BLOCKED = [
    "bash_tool",
    "terminal_run_tool",
    "terminal_send_tool",
    "terminal_kill_tool",
]

CHANNEL_POLICIES: dict[str, ToolPolicy] = {
    "cli": ToolPolicy(),
    "webchat": ToolPolicy(),
    "telegram": ToolPolicy(blocked_tools=list(_MESSAGING_BLOCKED)),
    "discord": ToolPolicy(blocked_tools=list(_MESSAGING_BLOCKED)),
    "whatsapp": ToolPolicy(
        # web_search_tool and web_browse_tool are blocked for the main agent on WhatsApp —
        # the agent must delegate these to a spawn_subagent instead.
        # Sub-agents receive all tools and can call web_search_tool / web_browse_tool freely.
        blocked_tools=[*_MESSAGING_BLOCKED, "browser_tool", "web_search_tool", "web_browse_tool"],
    ),
    "slack": ToolPolicy(blocked_tools=list(_MESSAGING_BLOCKED)),
    "signal": ToolPolicy(blocked_tools=list(_MESSAGING_BLOCKED)),
    "line": ToolPolicy(blocked_tools=list(_MESSAGING_BLOCKED)),
}

SANDBOX_BLOCKED_TOOLS = [
    "bash_tool",
    "terminal_run_tool",
    "terminal_send_tool",
    "terminal_kill_tool",
    "write_file_tool",
    "edit_file_tool",
    "git_commit_tool",
    "git_branch_tool",
    "browser_tool",
]


@dataclass
class GroupPolicy:
    """Per-group tool restrictions."""

    group_id: str
    channel: str
    allowed_tools: list[str] | None = None
    blocked_tools: list[str] | None = None
    label: str = ""


@dataclass
class UserToolPolicy:
    """Per-sender tool restrictions."""

    sender_id: str
    allowed_tools: list[str] | None = None
    blocked_tools: list[str] | None = None


class ToolPolicyPipeline:
    """Evaluates whether a tool call is allowed given the current context.

    Merges policies in priority order (most restrictive wins):
    1. Tool profile restrictions
    2. Sandbox restrictions (if sandbox mode)
    3. Channel default policy
    4. Owner-only restrictions (non-owner blocked)
    5. Group-specific policy (if in a group)
    6. User-specific policy (if configured)
    7. Custom overrides
    """

    def __init__(
        self,
        channel: str = "cli",
        sender_id: str = "",
        group_id: str = "",
        sandbox: bool = False,
        custom_policy: ToolPolicy | None = None,
        group_policies: list[GroupPolicy] | None = None,
        user_policies: list[UserToolPolicy] | None = None,
        profile: str = "full",
        is_owner: bool = True,
    ):
        self.channel = channel
        self.sender_id = sender_id
        self.group_id = group_id
        self.sandbox = sandbox
        self.custom_policy = custom_policy
        self._group_policies = {gp.group_id: gp for gp in (group_policies or [])}
        self._user_policies = {up.sender_id: up for up in (user_policies or [])}
        self.profile = profile
        self.is_owner = is_owner

    def get_policy(self) -> ToolPolicy:
        """Get the effective merged policy for the current context."""
        blocked: set[str] = set()
        allowed: set[str] | None = None  # None = all

        # 0. Profile restrictions
        profile_tools = get_profile_tools(self.profile)
        if profile_tools is not None:
            allowed = set(profile_tools)

        # 1. Sandbox restrictions
        if self.sandbox:
            blocked.update(SANDBOX_BLOCKED_TOOLS)

        # 2. Channel policy
        channel_policy = CHANNEL_POLICIES.get(self.channel, ToolPolicy())
        if channel_policy.blocked_tools:
            blocked.update(channel_policy.blocked_tools)
        if channel_policy.allowed_tools is not None:
            if allowed is not None:
                allowed &= set(channel_policy.allowed_tools)
            else:
                allowed = set(channel_policy.allowed_tools)

        # 3. Owner-only restrictions
        if not self.is_owner:
            blocked.update(OWNER_ONLY_TOOLS)

        # 4. Group policy
        if self.group_id and self.group_id in self._group_policies:
            gp = self._group_policies[self.group_id]
            if gp.blocked_tools:
                blocked.update(gp.blocked_tools)
            if gp.allowed_tools is not None:
                if allowed is not None:
                    allowed &= set(gp.allowed_tools)
                else:
                    allowed = set(gp.allowed_tools)

        # 5. User policy
        if self.sender_id and self.sender_id in self._user_policies:
            up = self._user_policies[self.sender_id]
            if up.blocked_tools:
                blocked.update(up.blocked_tools)
            if up.allowed_tools is not None:
                if allowed is not None:
                    allowed &= set(up.allowed_tools)
                else:
                    allowed = set(up.allowed_tools)

        # 6. Custom overrides
        if self.custom_policy:
            if self.custom_policy.blocked_tools:
                blocked.update(self.custom_policy.blocked_tools)
            if self.custom_policy.allowed_tools is not None:
                if allowed is not None:
                    allowed &= set(self.custom_policy.allowed_tools)
                else:
                    allowed = set(self.custom_policy.allowed_tools)

        return ToolPolicy(
            allowed_tools=sorted(allowed) if allowed is not None else None,
            blocked_tools=sorted(blocked) if blocked else None,
            sandbox_mode=self.sandbox,
        )

    def is_tool_allowed(self, tool_name: str) -> bool:
        """Check if a specific tool is allowed."""
        policy = self.get_policy()
        if policy.blocked_tools and tool_name in policy.blocked_tools:
            return False
        if policy.allowed_tools is not None and tool_name not in policy.allowed_tools:
            return False
        return True

    def filter_tools(self, tools: list) -> list:
        """Filter a list of LangChain tools based on policy."""
        policy = self.get_policy()
        result = []
        for t in tools:
            name = t.name if hasattr(t, "name") else str(t)
            if policy.blocked_tools and name in policy.blocked_tools:
                continue
            if policy.allowed_tools is not None and name not in policy.allowed_tools:
                continue
            result.append(t)

        filtered_count = len(tools) - len(result)
        if filtered_count > 0:
            logger.info(
                f"Tool policy: {filtered_count} tools blocked "
                f"(channel={self.channel}, sandbox={self.sandbox})"
            )
        return result

    def explain_blocked(self, tool_name: str) -> str:
        """Explain why a tool is blocked."""
        reasons: list[str] = []

        if self.sandbox and tool_name in SANDBOX_BLOCKED_TOOLS:
            reasons.append("sandbox mode restricts this tool")

        channel_policy = CHANNEL_POLICIES.get(self.channel, ToolPolicy())
        if channel_policy.blocked_tools and tool_name in channel_policy.blocked_tools:
            reasons.append(f"blocked by default for channel '{self.channel}'")

        if self.group_id and self.group_id in self._group_policies:
            gp = self._group_policies[self.group_id]
            if gp.blocked_tools and tool_name in gp.blocked_tools:
                reasons.append(f"blocked by group policy '{gp.label or gp.group_id}'")

        if self.sender_id and self.sender_id in self._user_policies:
            up = self._user_policies[self.sender_id]
            if up.blocked_tools and tool_name in up.blocked_tools:
                reasons.append(f"blocked by user policy for '{self.sender_id}'")

        if not reasons:
            return f"Tool '{tool_name}' is allowed"
        return f"Tool '{tool_name}' is blocked: {'; '.join(reasons)}"


def load_policies_from_config(config_data: dict[str, Any]) -> tuple[list[GroupPolicy], list[UserToolPolicy]]:
    """Load tool policies from langclaw.json config.

    Expected config structure:
    {
        "tool_policies": {
            "groups": [
                {"group_id": "g1", "channel": "discord", "blocked_tools": ["bash_tool"]}
            ],
            "users": [
                {"sender_id": "u1", "allowed_tools": ["web_search_tool"]}
            ]
        }
    }
    """
    policies = config_data.get("tool_policies", {})

    groups = [
        GroupPolicy(**gp)
        for gp in policies.get("groups", [])
    ]
    users = [
        UserToolPolicy(**up)
        for up in policies.get("users", [])
    ]
    return groups, users
