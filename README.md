# 🦞 OpenClaw-Lang

> **OpenClaw-equivalent personal AI assistant — built with LangGraph.**

A full-featured, local-first AI assistant inspired by [OpenClaw](https://github.com/openclaw/openclaw), rebuilt from scratch in Python using **LangGraph** for the agent runtime.

## ✨ Features

| Feature | Description |
|---------|-------------|
| **🧠 LangGraph Agent** | ReAct-style agentic loop with tool use, streaming, and multi-turn conversations |
| **🔧 Built-in Tools** | Shell commands, file operations, web search (DuckDuckGo), web browsing |
| **🌐 Gateway Server** | FastAPI control plane with REST API + WebSocket event streaming |
| **💬 WebChat UI** | Premium dark-mode chat interface with real-time tool visualization |
| **📱 Channel Adapters** | Telegram, Discord, and WebChat support |
| **📂 Workspace & Skills** | Customizable agent personality via markdown files + extensible skills |
| **🤖 Multi-Provider** | OpenAI, Anthropic, and Google Gemini support |
| **💾 Session Persistence** | JSONL-based conversation history with per-sender isolation |

## 🚀 Quick Start

### 1. Install

```bash
cd openclaw-lang
pip install -e ".[dev]"
```

### 2. Configure

```bash
# Set your API key (choose one)
export OPENAI_API_KEY="sk-..."
# or
export ANTHROPIC_API_KEY="sk-ant-..."
# or
export GOOGLE_API_KEY="AI..."

# Create workspace with default files
openclaw setup
```

Or create `~/.openclaw/openclaw.json`:

```json
{
  "agent": {
    "model": "openai/gpt-4o"
  },
  "api_keys": {
    "openai": "sk-..."
  }
}
```

### 3. Run

```bash
# Start the Gateway (REST API + WebChat)
openclaw gateway

# Interactive chat (CLI)
openclaw chat

# Single message
openclaw agent --message "What is LangGraph?"

# Check setup
openclaw doctor
```

Then open **http://localhost:18789/webchat** for the WebChat UI.

## 📁 Architecture

```
Channels                    Gateway (FastAPI :18789)           Agent (LangGraph)
┌──────────┐              ┌─────────────────────┐           ┌────────────────────┐
│ Telegram │──┐           │  POST /api/agent    │           │  context_assembly  │
│ Discord  │──┼──────────>│  GET  /api/sessions │──invoke──>│        │           │
│ WebChat  │──┤           │  GET  /api/status   │           │        ▼           │
│ CLI      │──┘           │  WS   /ws           │<─stream──│    llm_call        │
└──────────┘              │  GET  /webchat      │           │    ┌───┴───┐      │
                          └─────────────────────┘           │    │tools? │      │
                                                            │    yes   no       │
Workspace (~/.openclaw/)   Sessions                         │     ▼     ▼       │
├── AGENTS.md              ~/.openclaw/sessions/            │  tools  respond   │
├── SOUL.md                ├── {id}.jsonl                   │     │             │
├── TOOLS.md               └── {id}.meta.json               │     └──> llm_call │
├── IDENTITY.md                                             └────────────────────┘
├── USER.md
└── skills/
```

## 🔧 Tools

| Tool | Description |
|------|-------------|
| `bash_tool` | Execute shell commands |
| `read_file_tool` | Read file contents (with line ranges) |
| `write_file_tool` | Create/overwrite files |
| `edit_file_tool` | Search-and-replace file editing |
| `list_directory_tool` | List directory contents |
| `web_search_tool` | Search the web (DuckDuckGo) |
| `web_browse_tool` | Fetch and extract web page content |

## 🧩 Skills

Add custom skills to extend the agent:

```
~/.openclaw/workspace/skills/
  my-skill/
    SKILL.md    # with optional YAML frontmatter
```

## 📡 Channel Adapters

| Channel | Status | Config Key |
|---------|--------|------------|
| WebChat | ✅ Built-in | `channels.webchat` |
| Telegram | ✅ Ready | `channels.telegram.token` |
| Discord | ✅ Ready | `channels.discord.token` |

## 🔑 Supported Models

```json
{
  "agent": {
    "model": "openai/gpt-4o"
  }
}
```

- `openai/gpt-4o`, `openai/gpt-4o-mini`, `openai/o1`
- `anthropic/claude-sonnet-4-20250514`, `anthropic/claude-opus-4-6`
- `google/gemini-2.0-flash`, `google/gemini-2.5-pro`

## 📄 License

MIT
