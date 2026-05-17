"""Built-in tools for the OpenClaw agent.

Tool registry that mirrors OpenClaw's built-in tool system:
- bash: shell command execution
- read_file / write_file / edit_file / list_directory: file operations
- apply_patch: diff-based file editing with Begin/End Patch format
- web_search: DuckDuckGo web search
- web_browse: fetch and extract web page content
- cron_add / cron_list / cron_remove / cron_run: scheduled jobs & proactive nudges
- spawn_subagent: delegate tasks to child agents (inline, same process)
- subagents_tool: list/kill/steer background sub-agent sessions
- memory_save / memory_search / memory_get / memory_list: persistent RAG memory
- daily_note_read / daily_note_write / daily_note_list: date-based memory notes
- git_status / git_diff / git_log / git_commit / git_branch: git operations
- grep_search / find_files / code_outline: code intelligence
- create_plan / update_plan / create_artifact / read_artifact: task planning
- terminal_run / terminal_read / terminal_send / terminal_kill: terminal management
- image: image processing and analysis (describe, resize, convert, info, thumbnail, crop)
- canvas: server-side UI canvas management with A2UI support
- browser: Chrome DevTools Protocol browser automation (navigate, screenshot, click, type, evaluate JS)
- agents_list / agents_create / agents_get / agents_delete: agent profile management
- sessions_list / sessions_history / sessions_send / sessions_spawn / sessions_status: session management
- message: cross-channel message delivery (send to any channel)
- tts: text-to-speech conversion (OpenAI, ElevenLabs, Edge TTS)
- slack_actions: Slack-specific operations (send, edit, delete, react, pin)
- discord_actions: Discord-specific operations (send, edit, delete, react, threads, moderation)
- telegram_actions: Telegram-specific operations (send, edit, delete, react, buttons)
- whatsapp_actions: WhatsApp-specific operations (send, react, groups)
"""

from __future__ import annotations

from langchain_core.tools import BaseTool

from openclaw.tools.bash import bash_tool
from openclaw.tools.cron import cron_add_tool, cron_list_tool, cron_remove_tool, cron_run_tool
from openclaw.tools.file_ops import edit_file_tool, list_directory_tool, read_file_tool, write_file_tool
from openclaw.tools.apply_patch import apply_patch_tool
from openclaw.tools.web_browse import web_browse_tool
from openclaw.tools.web_search import web_search_tool
from openclaw.tools.subagent import spawn_subagent
from openclaw.tools.subagents_tool import subagents_tool
from openclaw.tools.memory import memory_save_tool, memory_search_tool, memory_get_tool, memory_list_tool
from openclaw.tools.daily_notes import daily_note_read_tool, daily_note_write_tool, daily_note_list_tool
from openclaw.tools.git import git_status_tool, git_diff_tool, git_log_tool, git_commit_tool, git_branch_tool
from openclaw.tools.code_search import grep_search_tool, find_files_tool
from openclaw.tools.code_analysis import code_outline_tool
from openclaw.tools.artifacts import create_plan_tool, update_plan_tool, create_artifact_tool, read_artifact_tool
from openclaw.tools.terminal import terminal_run_tool, terminal_read_tool, terminal_send_tool, terminal_kill_tool
from openclaw.tools.image import image_tool
from openclaw.tools.canvas import canvas_tool
from openclaw.tools.browser import browser_tool
from openclaw.tools.agents import agents_list_tool, agents_create_tool, agents_get_tool, agents_delete_tool
from openclaw.tools.sessions import (
    sessions_list_tool, sessions_history_tool, sessions_send_tool,
    sessions_spawn_tool, sessions_status_tool,
)
from openclaw.tools.message import message_tool
from openclaw.tools.tts import tts_tool
from openclaw.tools.slack_actions import slack_actions_tool
from openclaw.tools.discord_actions import discord_actions_tool
from openclaw.tools.telegram_actions import telegram_actions_tool
from openclaw.tools.whatsapp_actions import whatsapp_actions_tool

ALL_TOOLS: list[BaseTool] = [
    bash_tool,
    read_file_tool,
    write_file_tool,
    edit_file_tool,
    list_directory_tool,
    apply_patch_tool,
    web_search_tool,
    web_browse_tool,
    cron_add_tool,
    cron_list_tool,
    cron_remove_tool,
    cron_run_tool,
    spawn_subagent,
    subagents_tool,
    memory_save_tool,
    memory_search_tool,
    memory_get_tool,
    memory_list_tool,
    daily_note_read_tool,
    daily_note_write_tool,
    daily_note_list_tool,
    git_status_tool,
    git_diff_tool,
    git_log_tool,
    git_commit_tool,
    git_branch_tool,
    grep_search_tool,
    find_files_tool,
    code_outline_tool,
    create_plan_tool,
    update_plan_tool,
    create_artifact_tool,
    read_artifact_tool,
    terminal_run_tool,
    terminal_read_tool,
    terminal_send_tool,
    terminal_kill_tool,
    image_tool,
    canvas_tool,
    browser_tool,
    agents_list_tool,
    agents_create_tool,
    agents_get_tool,
    agents_delete_tool,
    sessions_list_tool,
    sessions_history_tool,
    sessions_send_tool,
    sessions_spawn_tool,
    sessions_status_tool,
    message_tool,
    tts_tool,
    slack_actions_tool,
    discord_actions_tool,
    telegram_actions_tool,
    whatsapp_actions_tool,
]


def get_all_tools() -> list[BaseTool]:
    """Return all built-in tools."""
    return list(ALL_TOOLS)


def get_tools_by_name(*names: str) -> list[BaseTool]:
    """Return tools matching the given names."""
    name_set = set(names)
    return [t for t in ALL_TOOLS if t.name in name_set]


def get_tools_description() -> str:
    """Return a human-readable description of all available tools."""
    lines = []
    for tool in ALL_TOOLS:
        lines.append(f"- **{tool.name}**: {tool.description}")
    return "\n".join(lines)

