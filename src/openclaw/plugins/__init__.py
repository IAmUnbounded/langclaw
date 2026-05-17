"""Plugin/Hooks system — lifecycle hooks for the openclaw agent pipeline.

Mirrors OpenClaw's plugin lifecycle hooks, allowing external code to
observe and modify agent behaviour at well-defined points:

    before_agent_start, agent_start, llm_input, llm_output,
    tool_before_call, tool_after_call, agent_end,
    auto_compaction_start, auto_compaction_end

Plugins are discovered from ~/.langclaw/plugins/ (and any additional
directories passed to ``discover_plugins``).  Each plugin is a Python
file or package that exposes a ``register(registry)`` function.
"""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hook types
# ---------------------------------------------------------------------------

class Hook(str, Enum):
    """Known lifecycle hook names.

    Each value documents the payload keys it receives and what (if
    anything) a callback may return to modify behaviour.
    """

    # Called before the agent graph runs.
    # Payload: {"state": AgentState dict, "config": config dict}
    BEFORE_AGENT_START = "before_agent_start"

    # Called when agent starts processing.
    # Payload: {"run_id": str, "session_id": str, "channel": str}
    AGENT_START = "agent_start"

    # Called before LLM invocation.
    # Payload: {"messages": list[dict], "model": str}
    # Callback may return modified messages.
    LLM_INPUT = "llm_input"

    # Called after LLM response.
    # Payload: {"response": dict, "model": str, "tool_calls": list}
    LLM_OUTPUT = "llm_output"

    # Called before tool execution.
    # Payload: {"tool_name": str, "args": dict}
    # Callback may return modified args or {"skip": True} to skip.
    TOOL_BEFORE_CALL = "tool_before_call"

    # Called after tool execution.
    # Payload: {"tool_name": str, "args": dict, "result": str, "elapsed_seconds": float}
    TOOL_AFTER_CALL = "tool_after_call"

    # Called when agent finishes.
    # Payload: {"run_id": str, "response": str, "tool_calls_count": int, "elapsed_seconds": float}
    AGENT_END = "agent_end"

    # Called before context compaction.
    # Payload: {"message_count": int, "estimated_tokens": int}
    AUTO_COMPACTION_START = "auto_compaction_start"

    # Called after compaction.
    # Payload: {"original_count": int, "compressed_count": int}
    AUTO_COMPACTION_END = "auto_compaction_end"

    # Called before sending a message to a channel.
    # Payload: {"content": str, "channel": str, "sender_id": str}
    # Callback may return modified content.
    MESSAGE_PRE_SEND = "message_pre_send"

    # Called after sending a message to a channel.
    # Payload: {"content": str, "channel": str, "sender_id": str, "success": bool}
    MESSAGE_POST_SEND = "message_post_send"

    # Called when a new session starts.
    # Payload: {"session_id": str, "sender_id": str, "channel": str}
    SESSION_START = "session_start"

    # Called when a session ends/expires.
    # Payload: {"session_id": str, "message_count": int}
    SESSION_END = "session_end"

    # Called to allow plugins to override the model selection.
    # Payload: {"model": str, "channel": str, "sender_id": str}
    # Callback may return {"model": "new_model"} to override.
    MODEL_OVERRIDE = "model_override"

    # Called on incoming gateway HTTP requests.
    # Payload: {"method": str, "path": str, "client_ip": str}
    GATEWAY_REQUEST = "gateway_request"

    # Called when a message is received from a channel (before agent processes it).
    # Payload: {"content": str, "channel": str, "sender_id": str, "group_id": str}
    # Callback may return {"content": str} to modify or {"skip": True} to drop.
    MESSAGE_RECEIVED = "message_received"

    # Called when an error occurs during agent execution.
    # Payload: {"error": str, "error_type": str, "run_id": str}
    AGENT_ERROR = "agent_error"

    # Called when memory is saved.
    # Payload: {"content": str, "tags": list, "sender_id": str}
    MEMORY_SAVE = "memory_save"

    # Called when a cron job fires.
    # Payload: {"job_id": str, "job_name": str, "payload": str}
    CRON_FIRE = "cron_fire"


# Convenience set for fast membership tests.
HOOK_NAMES: frozenset[str] = frozenset(h.value for h in Hook)

# ---------------------------------------------------------------------------
# PluginInfo
# ---------------------------------------------------------------------------

@dataclass
class PluginInfo:
    """Metadata about a loaded plugin."""

    name: str
    description: str = ""
    version: str = "0.0.0"
    hooks: list[str] = field(default_factory=list)
    path: str = ""
    enabled: bool = True

# ---------------------------------------------------------------------------
# HookRegistry (singleton)
# ---------------------------------------------------------------------------

class HookRegistry:
    """Central registry for lifecycle hook callbacks.

    Use ``HookRegistry.get_instance()`` to obtain the singleton.  Plugins
    call ``register()`` to subscribe to hooks; the agent pipeline calls
    ``fire()`` to broadcast events.
    """

    _instance: HookRegistry | None = None

    def __init__(self) -> None:
        self._hooks: dict[str, list[Callable[..., Any]]] = defaultdict(list)
        self._plugins: dict[str, PluginInfo] = {}

    # -- singleton ----------------------------------------------------------

    @classmethod
    def get_instance(cls) -> HookRegistry:
        """Return the global singleton, creating it on first access."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def _reset(cls) -> None:
        """Reset the singleton (useful for tests)."""
        cls._instance = None

    # -- registration -------------------------------------------------------

    def register(
        self,
        hook_name: str,
        callback: Callable[..., Any],
        plugin_name: str = "anonymous",
    ) -> None:
        """Register *callback* for *hook_name*.

        Parameters
        ----------
        hook_name:
            One of the ``Hook`` enum values (as a string).
        callback:
            Sync or async callable.  Receives a single ``dict`` payload.
        plugin_name:
            Human-readable name of the plugin registering the callback.
        """
        if hook_name not in HOOK_NAMES:
            logger.warning(
                "Plugin '%s' registered for unknown hook '%s' — "
                "this may be a typo or a custom hook",
                plugin_name,
                hook_name,
            )
        self._hooks[hook_name].append(callback)

        # Track which hooks this plugin registered.
        if plugin_name in self._plugins:
            info = self._plugins[plugin_name]
            if hook_name not in info.hooks:
                info.hooks.append(hook_name)

        logger.debug(
            "Hook '%s' registered by '%s' (total: %d)",
            hook_name,
            plugin_name,
            len(self._hooks[hook_name]),
        )

    def unregister(self, hook_name: str, callback: Callable[..., Any]) -> None:
        """Remove *callback* from *hook_name*."""
        try:
            self._hooks[hook_name].remove(callback)
            logger.debug(
                "Hook '%s' callback removed (remaining: %d)",
                hook_name,
                len(self._hooks[hook_name]),
            )
        except ValueError:
            logger.warning(
                "Attempted to unregister a callback from '%s' that was not registered",
                hook_name,
            )

    # -- firing -------------------------------------------------------------

    async def fire(self, hook_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Fire *hook_name*, passing *payload* through all callbacks.

        Callbacks are invoked in registration order.  If a callback
        returns a ``dict``, its entries are merged into the payload
        before the next callback sees it.  Both sync and async callbacks
        are supported.

        Errors in individual callbacks are logged and swallowed so that
        a misbehaving plugin cannot crash the agent.
        """
        callbacks = self._hooks.get(hook_name)
        if not callbacks:
            return payload

        for cb in callbacks:
            try:
                if asyncio.iscoroutinefunction(cb):
                    result = await cb(payload)
                else:
                    result = cb(payload)

                if isinstance(result, dict):
                    payload.update(result)
            except Exception:
                cb_name = getattr(cb, "__qualname__", repr(cb))
                logger.exception(
                    "Error in hook '%s' callback '%s' — skipping",
                    hook_name,
                    cb_name,
                )

        return payload

    # -- introspection ------------------------------------------------------

    def list_hooks(self) -> dict[str, int]:
        """Return a mapping of hook_name -> registered callback count."""
        return {name: len(cbs) for name, cbs in self._hooks.items() if cbs}

    def list_plugins(self) -> list[PluginInfo]:
        """Return metadata for all loaded plugins."""
        return list(self._plugins.values())

# ---------------------------------------------------------------------------
# Convenience decorator
# ---------------------------------------------------------------------------

def hook(hook_name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that registers a function as a hook callback.

    Usage::

        from openclaw.plugins import hook

        @hook("llm_input")
        def redact_secrets(payload: dict) -> dict:
            # ... modify messages ...
            return {"messages": sanitised}
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        HookRegistry.get_instance().register(hook_name, func)
        return func

    return decorator

# ---------------------------------------------------------------------------
# Plugin discovery
# ---------------------------------------------------------------------------

_DEFAULT_PLUGIN_DIR = Path.home() / ".langclaw" / "plugins"


def discover_plugins(
    plugin_dirs: list[Path] | None = None,
) -> list[PluginInfo]:
    """Discover and load plugins from the filesystem.

    Search order:

    1. ``~/.langclaw/plugins/``
    2. Any additional directories passed via *plugin_dirs*.

    Each plugin is a ``.py`` file (or a package directory with
    ``__init__.py``) that exposes:

    * A ``register(registry: HookRegistry)`` function (required).
    * An optional ``PLUGIN_INFO`` dict with ``name``, ``description``,
      and ``version`` keys.
    """
    registry = HookRegistry.get_instance()

    dirs: list[Path] = [_DEFAULT_PLUGIN_DIR]
    if plugin_dirs:
        dirs.extend(plugin_dirs)

    loaded: list[PluginInfo] = []

    for plugin_dir in dirs:
        plugin_dir = plugin_dir.expanduser().resolve()
        if not plugin_dir.is_dir():
            logger.debug("Plugin directory does not exist, skipping: %s", plugin_dir)
            continue

        for candidate in sorted(plugin_dir.iterdir()):
            info = _load_plugin(candidate, registry)
            if info is not None:
                loaded.append(info)

    logger.info(
        "Plugin discovery complete: %d plugin(s) loaded from %d director%s",
        len(loaded),
        len(dirs),
        "y" if len(dirs) == 1 else "ies",
    )
    return loaded


def _load_plugin(path: Path, registry: HookRegistry) -> PluginInfo | None:
    """Attempt to load a single plugin from *path*.

    *path* may be:
    * A ``.py`` file
    * A directory containing ``__init__.py``

    Returns ``PluginInfo`` on success, ``None`` if the path is not a
    valid plugin.
    """
    # Determine the module file to import.
    if path.is_file() and path.suffix == ".py" and not path.name.startswith("_"):
        module_file = path
        module_name = f"openclaw_plugin_{path.stem}"
    elif path.is_dir() and (path / "__init__.py").is_file():
        module_file = path / "__init__.py"
        module_name = f"openclaw_plugin_{path.name}"
    else:
        return None

    logger.debug("Loading plugin: %s", module_file)

    try:
        spec = importlib.util.spec_from_file_location(module_name, module_file)
        if spec is None or spec.loader is None:
            logger.warning("Could not create import spec for %s", module_file)
            return None

        module = importlib.util.module_from_spec(spec)
        # Make the module importable by name for any internal imports.
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    except Exception:
        logger.exception("Failed to import plugin %s", module_file)
        return None

    # --- Extract PLUGIN_INFO (optional) -----------------------------------
    raw_info: dict[str, Any] = getattr(module, "PLUGIN_INFO", {})
    info = PluginInfo(
        name=raw_info.get("name", path.stem),
        description=raw_info.get("description", ""),
        version=raw_info.get("version", "0.0.0"),
        path=str(module_file),
    )

    # --- Call register(registry) ------------------------------------------
    register_fn = getattr(module, "register", None)
    if register_fn is None or not callable(register_fn):
        logger.warning(
            "Plugin '%s' at %s has no register() function — skipping",
            info.name,
            module_file,
        )
        # Clean up the module we just imported.
        sys.modules.pop(module_name, None)
        return None

    # Pre-register the PluginInfo so that register() calls to
    # registry.register(..., plugin_name=...) can record hooks.
    registry._plugins[info.name] = info

    try:
        sig = inspect.signature(register_fn)
        if len(sig.parameters) >= 1:
            register_fn(registry)
        else:
            register_fn()
    except Exception:
        logger.exception(
            "Plugin '%s' register() raised an exception", info.name
        )
        registry._plugins.pop(info.name, None)
        sys.modules.pop(module_name, None)
        return None

    logger.info(
        "Plugin loaded: %s v%s (%d hook(s): %s)",
        info.name,
        info.version,
        len(info.hooks),
        ", ".join(info.hooks) or "none",
    )
    return info
