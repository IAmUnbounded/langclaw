"""LangGraph agent graph — ReAct-style agentic loop with advanced capabilities.

This is the heart of the system, mirroring OpenClaw's Pi agent runtime.

Architecture:
    context_assembly → compress → llm_call ←→ parallel_tool_execution
                                     ↓
                                  respond

Capabilities:
- Parallel tool execution (multiple tool calls run concurrently)
- Context compression (smart truncation when context exceeds budget)
- MCP tool discovery (dynamically loads tools from MCP servers)
- Sub-agent spawning (tools can spawn child agents)
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, StateGraph

from openclaw.agent.context import assemble_system_prompt
from openclaw.agent.state import AgentState
from openclaw.config import OpenClawConfig
from openclaw.tools import get_all_tools

logger = logging.getLogger(__name__)


def _get_llm(config: OpenClawConfig) -> BaseChatModel:
    """Get the appropriate LLM based on configuration.

    Supports: openai, anthropic, google providers.
    API keys are read from config or environment variables.
    """
    provider, model_name = config.parse_model()

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        api_key = config.api_keys.openai or os.environ.get("OPENAI_API_KEY", "")
        return ChatOpenAI(
            model=model_name,
            temperature=config.agent.temperature,
            api_key=api_key or None,
        )
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        api_key = config.api_keys.anthropic or os.environ.get("ANTHROPIC_API_KEY", "")
        return ChatAnthropic(
            model=model_name,
            temperature=config.agent.temperature,
            api_key=api_key or None,
        )
    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = config.api_keys.google or os.environ.get("GOOGLE_API_KEY", "")
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=config.agent.temperature,
            google_api_key=api_key or None,
        )
    else:
        raise ValueError(
            f"Unsupported provider: {provider}. "
            f"Use openai/model, anthropic/model, or google/model."
        )


def _context_assembly_node(state: AgentState) -> dict[str, Any]:
    """Node: Assemble system prompt from workspace files, skills, and memory."""
    from openclaw.tools import get_tools_description

    config = OpenClawConfig.load()
    workspace = config.get_workspace_path()

    # Inject relevant memories into context
    memory_context = _get_memory_context(state)

    system_prompt = assemble_system_prompt(
        workspace=workspace,
        tools_description=get_tools_description(),
        extra_context=memory_context,
    )

    return {"system_prompt": system_prompt}


def _get_memory_context(state: AgentState) -> str:
    """Retrieve relevant memories based on the current conversation."""
    try:
        from openclaw.memory import MemoryStore

        # Get the last user message to search memories
        last_user_msg = ""
        for msg in reversed(state.messages):
            if isinstance(msg, HumanMessage):
                last_user_msg = msg.content if isinstance(msg.content, str) else str(msg.content)
                break

        if not last_user_msg:
            return ""

        store = MemoryStore()
        results = store.search(last_user_msg, top_k=3)

        if not results:
            return ""

        memory_lines = ["## RELEVANT MEMORIES"]
        for entry in results:
            memory_lines.append(f"- {entry.content}")

        return "\n".join(memory_lines)

    except Exception:
        return ""


def _compress_context_node(state: AgentState) -> dict[str, Any]:
    """Node: Compress context if it exceeds the token budget."""
    from openclaw.agent.compressor import compress_messages

    config = OpenClawConfig.load()
    max_tokens = config.agent.max_context_tokens

    compressed = compress_messages(
        list(state.messages),
        max_tokens=max_tokens,
    )

    if len(compressed) != len(state.messages):
        logger.info(
            f"Context compressed: {len(state.messages)} → {len(compressed)} messages"
        )
        return {"messages": compressed}

    return {}


def _llm_call_node(state: AgentState) -> dict[str, Any]:
    """Node: Call the LLM with the conversation + system prompt.

    This node:
    1. Loads the LLM from config
    2. Binds all tools to the LLM
    3. Prepends the system prompt
    4. Calls the LLM
    5. Returns the AI message (which may contain tool calls)
    """
    config = OpenClawConfig.load()
    llm = _get_llm(config)
    tools = get_all_tools()

    # Bind tools to the LLM
    llm_with_tools = llm.bind_tools(tools)

    # Build messages: system prompt + conversation history
    messages: list[BaseMessage] = []

    # Add system prompt
    if state.system_prompt:
        messages.append(SystemMessage(content=state.system_prompt))

    # Add conversation history
    messages.extend(state.messages)

    # Call the LLM
    response = llm_with_tools.invoke(messages)

    return {"messages": [response]}


async def _parallel_tool_node(state: AgentState) -> dict[str, Any]:
    """Node: Execute tool calls in parallel using asyncio.gather.

    When the LLM returns multiple tool_calls in one response,
    this node runs them concurrently for maximum throughput.
    """
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return {}

    tools = get_all_tools()
    tool_map = {t.name: t for t in tools}

    async def _execute_tool_call(tool_call: dict) -> ToolMessage:
        """Execute a single tool call."""
        tool_name = tool_call.get("name", "")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id", str(uuid.uuid4())[:8])

        tool_instance = tool_map.get(tool_name)
        if not tool_instance:
            return ToolMessage(
                content=f"[error] Tool not found: {tool_name}",
                tool_call_id=tool_call_id,
            )

        try:
            # Check if tool is async
            if hasattr(tool_instance, 'ainvoke'):
                result = await tool_instance.ainvoke(tool_args)
            else:
                # Run sync tool in thread pool
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, lambda: tool_instance.invoke(tool_args)
                )

            content = result if isinstance(result, str) else str(result)

            # Truncate very large outputs
            if len(content) > 15000:
                content = content[:15000] + "\n\n... (output truncated)"

            return ToolMessage(
                content=content,
                tool_call_id=tool_call_id,
                name=tool_name,
            )

        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}")
            return ToolMessage(
                content=f"[error] Tool execution failed: {e}",
                tool_call_id=tool_call_id,
                name=tool_name,
            )

    # Execute all tool calls in parallel
    tool_calls = last_message.tool_calls
    if len(tool_calls) > 1:
        logger.info(f"⚡ Executing {len(tool_calls)} tool calls in parallel")

    tool_messages = await asyncio.gather(
        *[_execute_tool_call(tc) for tc in tool_calls]
    )

    return {"messages": list(tool_messages)}


def _should_continue(state: AgentState) -> Literal["tools", "end"]:
    """Edge: Decide whether to call tools or finish.

    If the last message has tool_calls → route to tools node.
    Otherwise → end the loop.
    """
    last_message = state.messages[-1]

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return "end"


def build_agent_graph(config: OpenClawConfig | None = None) -> StateGraph:
    """Build the LangGraph agent graph.

    Returns a compiled StateGraph that implements the ReAct loop:
    context_assembly → compress → llm_call ←→ parallel_tools → llm_call → ... → end

    The graph supports streaming, so callers can observe tool calls
    and assistant responses in real-time.
    """
    # Build the graph
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("context_assembly", _context_assembly_node)
    graph.add_node("compress", _compress_context_node)
    graph.add_node("llm_call", _llm_call_node)
    graph.add_node("tools", _parallel_tool_node)

    # Set entry point
    graph.set_entry_point("context_assembly")

    # Add edges
    graph.add_edge("context_assembly", "compress")
    graph.add_edge("compress", "llm_call")

    # Conditional edge from llm_call: tools or end
    graph.add_conditional_edges(
        "llm_call",
        _should_continue,
        {
            "tools": "tools",
            "end": END,
        },
    )

    # After tools, go back to llm_call (ReAct loop)
    graph.add_edge("tools", "llm_call")

    return graph.compile()


async def run_agent(
    message: str,
    session_id: str = "",
    sender_id: str = "user",
    channel: str = "cli",
    config: OpenClawConfig | None = None,
    history: list[BaseMessage] | None = None,
) -> dict[str, Any]:
    """Run the agent with a message and return the result.

    Args:
        message: User message text.
        session_id: Session identifier.
        sender_id: Sender identifier.
        channel: Channel name.
        config: Optional config override.
        history: Optional conversation history to prepend.

    Returns:
        Dict with:
        - run_id: Unique run identifier
        - messages: List of new messages (AI + tool messages)
        - response: Final text response
        - tool_calls: List of tool calls made
    """
    run_id = str(uuid.uuid4())[:12]
    cfg = config or OpenClawConfig.load()

    # Build initial state
    messages: list[BaseMessage] = []
    if history:
        messages.extend(history)
    messages.append(HumanMessage(content=message))

    initial_state = AgentState(
        messages=messages,
        session_id=session_id or run_id,
        sender_id=sender_id,
        channel=channel,
        run_id=run_id,
    )

    # Run the graph
    graph = build_agent_graph(cfg)
    result = await graph.ainvoke(initial_state)

    # Extract results
    new_messages = result["messages"][len(messages):]  # Only new messages
    response_text = ""
    tool_calls_made = []

    for msg in new_messages:
        if isinstance(msg, AIMessage):
            if msg.content:
                response_text = msg.content
            if msg.tool_calls:
                tool_calls_made.extend(msg.tool_calls)

    return {
        "run_id": run_id,
        "messages": new_messages,
        "response": response_text,
        "tool_calls": tool_calls_made,
        "all_messages": result["messages"],
    }


async def stream_agent(
    message: str,
    session_id: str = "",
    sender_id: str = "user",
    channel: str = "cli",
    config: OpenClawConfig | None = None,
    history: list[BaseMessage] | None = None,
):
    """Stream agent execution, yielding events as they happen.

    Yields dicts with event type and data:
    - {"type": "lifecycle", "phase": "start", "run_id": ...}
    - {"type": "tool_start", "name": ..., "args": ...}
    - {"type": "tool_end", "name": ..., "output": ...}
    - {"type": "assistant_delta", "content": ...}
    - {"type": "assistant_message", "content": ...}
    - {"type": "lifecycle", "phase": "end", "run_id": ...}
    """
    run_id = str(uuid.uuid4())[:12]
    cfg = config or OpenClawConfig.load()

    # Build initial state
    messages: list[BaseMessage] = []
    if history:
        messages.extend(history)
    messages.append(HumanMessage(content=message))

    initial_state = AgentState(
        messages=messages,
        session_id=session_id or run_id,
        sender_id=sender_id,
        channel=channel,
        run_id=run_id,
    )

    yield {"type": "lifecycle", "phase": "start", "run_id": run_id}

    try:
        graph = build_agent_graph(cfg)

        async for event in graph.astream_events(initial_state, version="v2"):
            kind = event.get("event", "")

            # Tool start
            if kind == "on_tool_start":
                yield {
                    "type": "tool_start",
                    "name": event.get("name", ""),
                    "args": event.get("data", {}).get("input", {}),
                    "run_id": run_id,
                }

            # Tool end
            elif kind == "on_tool_end":
                output = event.get("data", {}).get("output", "")
                if hasattr(output, "content"):
                    output = output.content
                yield {
                    "type": "tool_end",
                    "name": event.get("name", ""),
                    "output": str(output)[:2000],  # Truncate for streaming
                    "run_id": run_id,
                }

            # LLM streaming tokens
            elif kind == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk", None)
                if chunk and hasattr(chunk, "content") and chunk.content:
                    yield {
                        "type": "assistant_delta",
                        "content": chunk.content,
                        "run_id": run_id,
                    }

            # LLM final message
            elif kind == "on_chat_model_end":
                output = event.get("data", {}).get("output", None)
                if output and hasattr(output, "content") and output.content:
                    yield {
                        "type": "assistant_message",
                        "content": output.content,
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

