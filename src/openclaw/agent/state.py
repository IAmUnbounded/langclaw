"""Agent state schema for LangGraph."""

from __future__ import annotations

from typing import Annotated, Any, Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class AgentState(BaseModel):
    """State flowing through the agent LangGraph graph.

    Mirrors OpenClaw's session state model:
    - messages: conversation history (LangChain message objects)
    - session_id: unique session identifier
    - sender_id: who is talking (channel-specific user ID)
    - channel: which channel the message came from
    - run_id: unique run identifier for this invocation
    - metadata: arbitrary key-value pairs for context
    """

    messages: Annotated[Sequence[BaseMessage], add_messages] = Field(
        default_factory=list,
        description="Conversation history (appended via add_messages reducer).",
    )
    session_id: str = Field(
        default="",
        description="Unique session identifier.",
    )
    sender_id: str = Field(
        default="",
        description="Sender / user identifier.",
    )
    channel: str = Field(
        default="cli",
        description="Channel the message originated from (cli, webchat, telegram, discord).",
    )
    run_id: str = Field(
        default="",
        description="Unique run ID for this agent invocation.",
    )
    system_prompt: str = Field(
        default="",
        description="Assembled system prompt (injected by context_assembly).",
    )
    parent_run_id: str = Field(
        default="",
        description="Parent agent run ID (empty if top-level agent).",
    )
    depth: int = Field(
        default=0,
        description="Sub-agent nesting depth (0 = top-level). Max 3.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata.",
    )

    class Config:
        arbitrary_types_allowed = True
