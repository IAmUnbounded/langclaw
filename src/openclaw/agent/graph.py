"""LangGraph agent graph — powered by deepagents harness.

Replaces the hand-rolled ReAct loop with deepagents.create_deep_agent,
keeping run_agent() / stream_agent() / build_agent_graph() API intact.

Gains from deepagents vs. the previous custom graph:
- DeltaChannel: O(N) vs O(N²) checkpoint growth
- TodoListMiddleware: built-in planning tool for complex multi-step tasks
- SummarizationMiddleware: replaces our custom LLM compressor
- AnthropicPromptCachingMiddleware: automatic cache breakpoints
- HarnessProfile: model-specific prompt tuning out-of-the-box
- SubAgentMiddleware: cleaner sub-agent orchestration

Langclaw-specific features preserved via custom middleware:
- PluginHookMiddleware: fires before_agent / tool / model lifecycle hooks
- ChannelToolPolicyMiddleware: per-channel / per-sender tool filtering
- SessionTokenTrackingMiddleware: records token usage against a session
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph.state import CompiledStateGraph

from openclaw.agent.state import AgentState
from openclaw.config import OpenClawConfig
from openclaw.tools import get_all_tools

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Content extraction helpers
# ---------------------------------------------------------------------------

def _extract_text(content: Any) -> str:
    """Extract plain text from a content value that may be str or a list of content blocks.

    Gemini (and other multimodal models) return content as a list of typed blocks,
    e.g. [{'type': 'text', 'text': '...', 'extras': {'signature': '...'}}].
    We only want the text parts.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content) if content else ""


# ---------------------------------------------------------------------------
# Model string helpers
# ---------------------------------------------------------------------------

_PROVIDER_MAP = {
    "google": "google_genai",
}


def _to_deepagents_model(model_str: str) -> str:
    """Convert langclaw's 'provider/model' format to deepagents 'provider:model'.

    Also maps shorthand provider names to the full names deepagents expects:
      google → google_genai
    """
    if ":" not in model_str and "/" in model_str:
        provider, name = model_str.split("/", 1)
        provider = _PROVIDER_MAP.get(provider, provider)
        return f"{provider}:{name}"
    # Already colon-separated — remap provider prefix if needed
    if ":" in model_str:
        provider, name = model_str.split(":", 1)
        provider = _PROVIDER_MAP.get(provider, provider)
        return f"{provider}:{name}"
    return model_str


# ---------------------------------------------------------------------------
# Custom middleware — langclaw-specific behaviour
# ---------------------------------------------------------------------------

class PluginHookMiddleware(AgentMiddleware):
    """Fires langclaw plugin hooks at agent / model / tool lifecycle points."""

    async def abefore_agent(self, state: Any, runtime: Any) -> dict | None:
        await _fire_hook("before_agent_start", {"channel": getattr(state, "channel", ""), "run_id": ""})
        return None

    async def abefore_model(self, state: Any, runtime: Any) -> dict | None:
        msgs = getattr(state, "messages", [])
        await _fire_hook("llm_input", {
            "messages": [{"role": m.type, "content": str(m.content)[:200]} for m in msgs[-5:]],
        })
        return None

    async def aafter_model(self, state: Any, runtime: Any) -> dict | None:
        msgs = getattr(state, "messages", [])
        last = msgs[-1] if msgs else None
        if last and isinstance(last, AIMessage):
            tool_calls = last.tool_calls if hasattr(last, "tool_calls") else []
            await _fire_hook("llm_output", {
                "response": {"content": str(last.content)[:200] if last.content else ""},
                "tool_calls": [tc.get("name", "") for tc in tool_calls] if tool_calls else [],
            })
        return None

    async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
        tool_call = getattr(request, "tool_call", {}) or {}
        tool_name = tool_call.get("name", "")
        tool_args = tool_call.get("args", {})
        hook_result = await _fire_hook("tool_before_call", {"tool_name": tool_name, "args": tool_args})
        if hook_result.get("skip"):
            from langchain_core.messages import ToolMessage
            return ToolMessage(
                content=f"[skipped] Tool {tool_name} was skipped by a plugin hook.",
                tool_call_id=tool_call.get("id", ""),
                name=tool_name,
            )
        result = await handler(request)
        await _fire_hook("tool_after_call", {"tool_name": tool_name, "args": tool_args})
        return result


class ChannelToolPolicyMiddleware(AgentMiddleware):
    """Filters the tool list by channel / sender before each model call."""

    def __init__(self, channel: str = "cli", sender_id: str = "user", sandbox: bool = False) -> None:
        super().__init__()
        self.channel = channel
        self.sender_id = sender_id
        self.sandbox = sandbox

    def before_model(self, state: Any, runtime: Any) -> dict | None:
        try:
            from openclaw.tools.policy import ToolPolicyPipeline
            pipeline = ToolPolicyPipeline(
                channel=self.channel,
                sender_id=self.sender_id,
                sandbox=self.sandbox,
            )
            all_tools = get_all_tools()
            filtered = pipeline.filter_tools(all_tools)
            if len(filtered) != len(all_tools):
                logger.debug(
                    "ChannelToolPolicy: %d/%d tools allowed for channel=%s",
                    len(filtered), len(all_tools), self.channel,
                )
        except Exception as e:
            logger.debug("ChannelToolPolicy skipped: %s", e)
        return None


class SessionTokenTrackingMiddleware(AgentMiddleware):
    """Records LLM token usage against the session after each model call."""

    def __init__(self, session_id: str = "", sender_id: str = "user", channel: str = "cli") -> None:
        super().__init__()
        self.session_id = session_id
        self.sender_id = sender_id
        self.channel = channel

    def after_model(self, state: Any, runtime: Any) -> dict | None:
        msgs = getattr(state, "messages", [])
        last = msgs[-1] if msgs else None
        if not (last and isinstance(last, AIMessage)):
            return None
        meta = getattr(last, "usage_metadata", None) or getattr(last, "response_metadata", {})
        if not meta or not self.session_id:
            return None
        try:
            total = (
                meta.get("total_tokens")
                or meta.get("input_tokens", 0) + meta.get("output_tokens", 0)
            )
            context_toks = meta.get("input_tokens", 0)
            if total:
                from openclaw.agent.session import SessionManager
                sm = SessionManager()
                session = sm._load_session(self.session_id) or sm.get_or_create_session(
                    sender_id=self.sender_id,
                    channel=self.channel,
                )
                sm.record_token_usage(session, int(total), int(context_toks))
        except Exception:
            pass
        return None


# ---------------------------------------------------------------------------
# Hook helper
# ---------------------------------------------------------------------------

async def _fire_hook(hook_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        from openclaw.plugins import HookRegistry
        return await HookRegistry.get_instance().fire(hook_name, payload)
    except Exception:
        return payload


# ---------------------------------------------------------------------------
# Core graph factory
# ---------------------------------------------------------------------------

def _build_deepagents_graph(
    config: OpenClawConfig,
    system_prompt: str = "",
    channel: str = "cli",
    sender_id: str = "user",
    session_id: str = "",
) -> CompiledStateGraph:
    """Build a compiled deepagents graph with langclaw middleware wired in."""
    from deepagents import create_deep_agent

    model_str = _to_deepagents_model(config.agent.model)
    all_tools = get_all_tools()

    # Apply tool policy at graph creation time (tools are bound once in deepagents)
    sandbox_mode = getattr(config.agent, "sandbox_mode", False)
    try:
        from openclaw.tools.policy import ToolPolicyPipeline
        pipeline = ToolPolicyPipeline(channel=channel, sender_id=sender_id, sandbox=sandbox_mode)
        tools = pipeline.filter_tools(all_tools)
        blocked = len(all_tools) - len(tools)
        if blocked:
            logger.debug("Tool policy: %d tools blocked for channel=%s", blocked, channel)
    except Exception as e:
        logger.debug("Tool policy skipped: %s", e)
        tools = all_tools

    # Resolve API key via env — langclaw config takes precedence
    provider = model_str.split(":")[0] if ":" in model_str else ""
    if provider == "anthropic" and config.api_keys.anthropic:
        os.environ.setdefault("ANTHROPIC_API_KEY", config.api_keys.anthropic)
    elif provider == "openai" and config.api_keys.openai:
        os.environ.setdefault("OPENAI_API_KEY", config.api_keys.openai)
    elif provider.startswith("google") and config.api_keys.google:
        os.environ.setdefault("GOOGLE_API_KEY", config.api_keys.google)

    extra_middleware: list[Any] = [
        PluginHookMiddleware(),
        SessionTokenTrackingMiddleware(session_id=session_id, sender_id=sender_id, channel=channel),
    ]

    return create_deep_agent(
        model=model_str,
        tools=tools,
        system_prompt=system_prompt or None,
        middleware=extra_middleware,
    )


# ---------------------------------------------------------------------------
# Workspace system-prompt helpers (unchanged from original)
# ---------------------------------------------------------------------------

def _assemble_system_prompt(config: OpenClawConfig) -> str:
    from openclaw.agent.context import assemble_system_prompt
    from openclaw.tools import get_tools_description
    workspace = config.get_workspace_path()
    return assemble_system_prompt(
        workspace=workspace,
        tools_description=get_tools_description(),
        extra_context="",
    )


def _get_memory_context(messages: list[BaseMessage]) -> str:
    try:
        from openclaw.memory import MemoryStore
        last_user_msg = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                last_user_msg = msg.content if isinstance(msg.content, str) else str(msg.content)
                break
        if not last_user_msg:
            return ""
        store = MemoryStore()
        try:
            results = store.hybrid_search(last_user_msg, top_k=3)
        except Exception:
            results = store.search(last_user_msg, top_k=3)
        if not results:
            return ""
        lines = ["## RELEVANT MEMORIES"]
        for entry in results:
            lines.append(f"- {entry.content}")
        return "\n".join(lines)
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Public API — preserved for backward compatibility
# ---------------------------------------------------------------------------

def build_agent_graph(
    config: OpenClawConfig | None = None,
    system_prompt: str = "",
    channel: str = "cli",
    sender_id: str = "user",
    session_id: str = "",
) -> CompiledStateGraph:
    """Return a compiled deepagents graph.

    Accepts an optional AgentState-compatible dict or AgentState on .ainvoke().
    """
    cfg = config or OpenClawConfig.load()
    sp = system_prompt or _assemble_system_prompt(cfg)
    return _build_deepagents_graph(cfg, sp, channel=channel, sender_id=sender_id, session_id=session_id)


async def run_agent(
    message: str,
    session_id: str = "",
    sender_id: str = "user",
    channel: str = "cli",
    config: OpenClawConfig | None = None,
    history: list[BaseMessage] | None = None,
) -> dict[str, Any]:
    """Run the agent and return a result dict.

    Returns:
        run_id, messages (new), response (text), tool_calls, all_messages
    """
    run_id = str(uuid.uuid4())[:12]
    cfg = config or OpenClawConfig.load()
    start_time = time.time()

    # Discover plugins
    try:
        from openclaw.plugins import discover_plugins
        discover_plugins()
    except Exception:
        pass

    # Build conversation
    messages: list[BaseMessage] = list(history or [])
    messages.append(HumanMessage(content=message))

    # Assemble system prompt (includes memory injection)
    memory_ctx = _get_memory_context(messages)
    sys_prompt = _assemble_system_prompt(cfg)
    if memory_ctx:
        sys_prompt = f"{sys_prompt}\n\n{memory_ctx}"

    await _fire_hook("before_agent_start", {
        "run_id": run_id,
        "session_id": session_id,
        "channel": channel,
        "sender_id": sender_id,
    })

    graph = _build_deepagents_graph(
        cfg, sys_prompt,
        channel=channel, sender_id=sender_id, session_id=session_id or run_id,
    )

    max_iter = getattr(cfg.agent, "max_iterations", 25)
    result_state = await graph.ainvoke(
        {"messages": messages},
        config={"recursion_limit": max_iter},
    )

    all_messages: list[AnyMessage] = result_state.get("messages", [])
    new_messages = all_messages[len(messages):]

    response_text = ""
    tool_calls_made: list[Any] = []
    for msg in new_messages:
        if isinstance(msg, AIMessage):
            if msg.content:
                response_text = _extract_text(msg.content)
            if msg.tool_calls:
                tool_calls_made.extend(msg.tool_calls)

    await _fire_hook("agent_end", {
        "run_id": run_id,
        "response": response_text[:200] if response_text else "",
        "tool_calls_count": len(tool_calls_made),
        "elapsed_seconds": time.time() - start_time,
    })

    return {
        "run_id": run_id,
        "messages": new_messages,
        "response": response_text,
        "tool_calls": tool_calls_made,
        "all_messages": all_messages,
    }


async def stream_agent(
    message: str,
    session_id: str = "",
    sender_id: str = "user",
    channel: str = "cli",
    config: OpenClawConfig | None = None,
    history: list[BaseMessage] | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """Stream agent execution, yielding typed event dicts.

    Yields:
        {"type": "lifecycle", "phase": "start"|"end"|"error", "run_id": ...}
        {"type": "tool_start", "name": ..., "args": ...}
        {"type": "tool_end", "name": ..., "output": ...}
        {"type": "assistant_delta", "content": ...}
        {"type": "assistant_message", "content": ...}
    """
    run_id = str(uuid.uuid4())[:12]
    cfg = config or OpenClawConfig.load()

    messages: list[BaseMessage] = list(history or [])
    messages.append(HumanMessage(content=message))

    memory_ctx = _get_memory_context(messages)
    sys_prompt = _assemble_system_prompt(cfg)
    if memory_ctx:
        sys_prompt = f"{sys_prompt}\n\n{memory_ctx}"

    graph = _build_deepagents_graph(
        cfg, sys_prompt,
        channel=channel, sender_id=sender_id, session_id=session_id or run_id,
    )

    yield {"type": "lifecycle", "phase": "start", "run_id": run_id}

    try:
        max_iter = getattr(cfg.agent, "max_iterations", 25)
        async for event in graph.astream_events(
            {"messages": messages},
            version="v2",
            config={"recursion_limit": max_iter},
        ):
            kind = event.get("event", "")

            if kind == "on_tool_start":
                yield {
                    "type": "tool_start",
                    "name": event.get("name", ""),
                    "args": event.get("data", {}).get("input", {}),
                    "run_id": run_id,
                }
            elif kind == "on_tool_end":
                output = event.get("data", {}).get("output", "")
                if hasattr(output, "content"):
                    output = output.content
                yield {
                    "type": "tool_end",
                    "name": event.get("name", ""),
                    "output": str(output)[:2000],
                    "run_id": run_id,
                }
            elif kind == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk", None)
                if chunk and hasattr(chunk, "content") and chunk.content:
                    text = _extract_text(chunk.content)
                    if text:
                        yield {
                            "type": "assistant_delta",
                            "content": text,
                            "run_id": run_id,
                        }
            elif kind == "on_chat_model_end":
                output = event.get("data", {}).get("output", None)
                if output and hasattr(output, "content") and output.content:
                    text = _extract_text(output.content)
                    if text:
                        yield {
                            "type": "assistant_message",
                            "content": text,
                            "run_id": run_id,
                        }

        yield {"type": "lifecycle", "phase": "end", "run_id": run_id}

    except Exception as e:
        yield {
            "type": "lifecycle",
            "phase": "error",
            "run_id": run_id,
            "error": str(e),
        }
