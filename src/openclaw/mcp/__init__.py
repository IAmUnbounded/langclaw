"""MCP (Model Context Protocol) client — connect to external tool servers.

Supports connecting to MCP servers via stdio transport, discovering
their tools, and calling them dynamically. This allows OpenClaw-Lang
to integrate with any MCP-compatible tool server.

Configuration example in openclaw.json:
{
    "mcp_servers": {
        "filesystem": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path"],
            "env": {}
        }
    }
}
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server."""

    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class MCPToolDef:
    """A tool definition from an MCP server."""

    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)
    server_name: str = ""


class MCPClient:
    """Client for communicating with MCP servers via stdio.

    Uses a simple JSON-RPC-over-stdio protocol to communicate with
    MCP servers. Handles server lifecycle, tool discovery, and tool calling.
    """

    def __init__(self, name: str, config: MCPServerConfig):
        self.name = name
        self.config = config
        self._process: subprocess.Popen | None = None
        self._request_id = 0
        self._initialized = False

    async def connect(self) -> bool:
        """Start the MCP server process and initialize the connection."""
        try:
            env = dict(**self.config.env) if self.config.env else None

            self._process = subprocess.Popen(
                [self.config.command] + self.config.args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                bufsize=1,
            )

            # Send initialize request
            init_result = await self._send_request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "openclaw-lang", "version": "0.1.0"},
            })

            if init_result is not None:
                # Send initialized notification
                await self._send_notification("notifications/initialized", {})
                self._initialized = True
                logger.info(f"MCP server '{self.name}' connected")
                return True
            else:
                logger.warning(f"MCP server '{self.name}' initialization failed")
                return False

        except FileNotFoundError:
            logger.error(f"MCP server command not found: {self.config.command}")
            return False
        except Exception as e:
            logger.error(f"MCP server '{self.name}' connection failed: {e}")
            return False

    async def disconnect(self):
        """Stop the MCP server process."""
        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except Exception:
                if self._process:
                    self._process.kill()
            self._process = None
            self._initialized = False

    async def list_tools(self) -> list[MCPToolDef]:
        """Discover available tools from the MCP server."""
        if not self._initialized:
            return []

        result = await self._send_request("tools/list", {})
        if result is None:
            return []

        tools = []
        for tool_data in result.get("tools", []):
            tools.append(MCPToolDef(
                name=tool_data.get("name", ""),
                description=tool_data.get("description", ""),
                input_schema=tool_data.get("inputSchema", {}),
                server_name=self.name,
            ))

        return tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Call a tool on the MCP server."""
        if not self._initialized:
            return "[error] MCP server not connected"

        result = await self._send_request("tools/call", {
            "name": name,
            "arguments": arguments,
        })

        if result is None:
            return "[error] MCP tool call failed"

        # Extract content from result
        content_parts = []
        for item in result.get("content", []):
            if item.get("type") == "text":
                content_parts.append(item.get("text", ""))

        return "\n".join(content_parts) if content_parts else str(result)

    async def _send_request(self, method: str, params: dict) -> dict | None:
        """Send a JSON-RPC request and wait for response."""
        if not self._process or not self._process.stdin or not self._process.stdout:
            return None

        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params,
        }

        try:
            request_str = json.dumps(request) + "\n"
            self._process.stdin.write(request_str)
            self._process.stdin.flush()

            # Read response (with timeout via thread)
            loop = asyncio.get_event_loop()
            response_str = await asyncio.wait_for(
                loop.run_in_executor(None, self._process.stdout.readline),
                timeout=30,
            )

            if not response_str:
                return None

            response = json.loads(response_str)
            if "error" in response:
                logger.warning(f"MCP error: {response['error']}")
                return None

            return response.get("result", {})

        except asyncio.TimeoutError:
            logger.warning(f"MCP request timed out: {method}")
            return None
        except Exception as e:
            logger.warning(f"MCP request failed: {e}")
            return None

    async def _send_notification(self, method: str, params: dict):
        """Send a JSON-RPC notification (no response expected)."""
        if not self._process or not self._process.stdin:
            return

        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }

        try:
            self._process.stdin.write(json.dumps(notification) + "\n")
            self._process.stdin.flush()
        except Exception:
            pass


def mcp_tool_to_langchain(tool_def: MCPToolDef, client: MCPClient) -> BaseTool:
    """Convert an MCP tool definition to a LangChain tool."""

    async def _call_mcp(**kwargs) -> str:
        return await client.call_tool(tool_def.name, kwargs)

    description = tool_def.description or f"MCP tool: {tool_def.name}"
    if len(description) > 200:
        description = description[:197] + "..."

    return StructuredTool.from_function(
        func=lambda **kwargs: asyncio.run(_call_mcp(**kwargs)),
        coroutine=_call_mcp,
        name=f"mcp_{tool_def.server_name}_{tool_def.name}",
        description=f"[MCP:{tool_def.server_name}] {description}",
    )


async def discover_mcp_tools(servers: dict[str, MCPServerConfig]) -> list[BaseTool]:
    """Connect to all configured MCP servers and discover their tools.

    Returns LangChain tools that proxy to the MCP servers.
    """
    tools = []

    for name, config in servers.items():
        client = MCPClient(name, config)
        if await client.connect():
            mcp_tools = await client.list_tools()
            for td in mcp_tools:
                lc_tool = mcp_tool_to_langchain(td, client)
                tools.append(lc_tool)
            logger.info(f"Discovered {len(mcp_tools)} tools from MCP server '{name}'")

    return tools
