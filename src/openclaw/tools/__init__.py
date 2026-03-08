"""Built-in tools for the OpenClaw agent.

Tool registry that mirrors OpenClaw's built-in tool system:
- bash: shell command execution
- read_file / write_file / edit_file / list_directory: file operations
- web_search: DuckDuckGo web search
- web_browse: fetch and extract web page content
- cron_add / cron_list / cron_remove / cron_run: scheduled jobs & proactive nudges
- spawn_subagent: delegate tasks to child agents
- memory_save / memory_search / memory_list: persistent RAG memory
- git_status / git_diff / git_log / git_commit / git_branch: git operations
- grep_search / find_files / code_outline: code intelligence
- create_plan / update_plan / create_artifact / read_artifact: task planning
- terminal_run / terminal_read / terminal_send / terminal_kill: terminal management
"""

from __future__ import annotations

from langchain_core.tools import BaseTool

from openclaw.tools.bash import bash_tool
from openclaw.tools.cron import cron_add_tool, cron_list_tool, cron_remove_tool, cron_run_tool
from openclaw.tools.file_ops import edit_file_tool, list_directory_tool, read_file_tool, write_file_tool
from openclaw.tools.web_browse import web_browse_tool
from openclaw.tools.web_search import web_search_tool
from openclaw.tools.subagent import spawn_subagent
from openclaw.tools.memory import memory_save_tool, memory_search_tool, memory_list_tool
from openclaw.tools.git import git_status_tool, git_diff_tool, git_log_tool, git_commit_tool, git_branch_tool
from openclaw.tools.code_search import grep_search_tool, find_files_tool
from openclaw.tools.code_analysis import code_outline_tool
from openclaw.tools.artifacts import create_plan_tool, update_plan_tool, create_artifact_tool, read_artifact_tool
from openclaw.tools.terminal import terminal_run_tool, terminal_read_tool, terminal_send_tool, terminal_kill_tool

ALL_TOOLS: list[BaseTool] = [
    bash_tool,
    read_file_tool,
    write_file_tool,
    edit_file_tool,
    list_directory_tool,
    web_search_tool,
    web_browse_tool,
    cron_add_tool,
    cron_list_tool,
    cron_remove_tool,
    cron_run_tool,
    spawn_subagent,
    memory_save_tool,
    memory_search_tool,
    memory_list_tool,
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

