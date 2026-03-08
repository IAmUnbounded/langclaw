"""FastAPI Gateway server — the control plane for OpenClaw-Lang.

Mirrors OpenClaw's Gateway:
- REST API for agent invocations, sessions, status
- WebSocket for real-time event streaming
- Static file serving for WebChat UI
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from openclaw.agent.graph import run_agent, stream_agent
from openclaw.agent.session import SessionManager
from openclaw.automation import CronJob, CronScheduler
from openclaw.config import OpenClawConfig
from openclaw.gateway.events import event_to_dict

# Global state
_config: OpenClawConfig | None = None
_session_manager: SessionManager | None = None
_connected_clients: set[WebSocket] = set()
_start_time: float = 0.0
_cron_scheduler: CronScheduler | None = None

STATIC_DIR = Path(__file__).parent / "static"


# --- Request / Response models ---

class AgentRequest(BaseModel):
    """Request body for POST /api/agent."""
    message: str
    sender_id: str = "webchat-user"
    channel: str = "webchat"
    session_id: str = ""


class AgentResponse(BaseModel):
    """Response for POST /api/agent."""
    run_id: str
    response: str
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    session_id: str = ""


# --- Lifespan ---

async def _cron_job_callback(job: CronJob) -> str | None:
    """Callback fired by the cron scheduler when a job is due.

    Runs the agent with the job's payload and broadcasts the
    result as a 'nudge' to all connected WebSocket clients.
    """
    import logging
    logger = logging.getLogger(__name__)

    logger.info(f"🔔 Cron nudge firing: {job.name} — {job.payload}")

    try:
        result = await run_agent(
            message=f"[Scheduled job: {job.name}] {job.payload}",
            sender_id="cron-scheduler",
            channel=job.delivery.channel,
            config=_config,
        )

        response = result.get("response", "")

        # Broadcast nudge to all connected WebSocket clients
        nudge_event = {
            "type": "nudge",
            "job_id": job.job_id,
            "job_name": job.name,
            "content": response,
            "timestamp": time.time(),
        }
        for client in list(_connected_clients):
            try:
                await client.send_json(nudge_event)
            except Exception:
                _connected_clients.discard(client)

        return response

    except Exception as e:
        logger.error(f"Cron job {job.name} failed: {e}")
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle — starts the cron scheduler."""
    global _config, _session_manager, _start_time, _cron_scheduler
    _config = OpenClawConfig.load()
    _session_manager = SessionManager()
    _start_time = time.time()

    # Start cron scheduler for proactive nudges
    _cron_scheduler = CronScheduler(callback=_cron_job_callback, check_interval=30.0)
    await _cron_scheduler.start()

    yield

    # Shutdown cron scheduler
    if _cron_scheduler:
        await _cron_scheduler.stop()


# --- App ---

def create_app(config: OpenClawConfig | None = None) -> FastAPI:
    """Create the FastAPI gateway application."""
    global _config
    if config:
        _config = config

    app = FastAPI(
        title="🦞 OpenClaw-Lang Gateway",
        description="Personal AI Assistant — Gateway Control Plane",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS
    cfg = config or OpenClawConfig()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.gateway.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- REST API Routes ---

    @app.get("/api/status")
    async def get_status():
        """Gateway health and status."""
        return {
            "status": "ok",
            "version": "0.1.0",
            "uptime_seconds": round(time.time() - _start_time, 1),
            "connected_clients": len(_connected_clients),
            "model": (_config or OpenClawConfig()).agent.model,
            "port": (_config or OpenClawConfig()).gateway.port,
        }

    @app.post("/api/agent", response_model=AgentResponse)
    async def invoke_agent(request: AgentRequest):
        """Submit a message to the agent and get a response."""
        sm = _session_manager or SessionManager()
        session = sm.get_or_create_session(request.sender_id, request.channel)

        from langchain_core.messages import HumanMessage, AIMessage

        # Run agent
        result = await run_agent(
            message=request.message,
            session_id=session.metadata.session_id,
            sender_id=request.sender_id,
            channel=request.channel,
            config=_config,
            history=session.messages,
        )

        # Persist messages
        sm.append_message(session, HumanMessage(content=request.message))
        for msg in result.get("messages", []):
            sm.append_message(session, msg)

        return AgentResponse(
            run_id=result["run_id"],
            response=result["response"],
            tool_calls=[
                {"name": tc.get("name", ""), "args": tc.get("args", {})}
                for tc in result.get("tool_calls", [])
            ],
            session_id=session.metadata.session_id,
        )

    @app.get("/api/sessions")
    async def list_sessions():
        """List all sessions."""
        sm = _session_manager or SessionManager()
        sessions = sm.list_sessions()
        return [
            {
                "session_id": s.session_id,
                "sender_id": s.sender_id,
                "channel": s.channel,
                "created_at": s.created_at,
                "last_active": s.last_active,
                "message_count": s.message_count,
                "title": s.title,
            }
            for s in sessions
        ]

    @app.get("/api/sessions/{session_id}/history")
    async def get_session_history(session_id: str):
        """Get session conversation history."""
        sm = _session_manager or SessionManager()
        history = sm.get_session_history(session_id)
        return {"session_id": session_id, "messages": history}

    @app.get("/api/config")
    async def get_config():
        """Get current configuration (API keys redacted)."""
        cfg = _config or OpenClawConfig()
        data = cfg.model_dump()
        # Redact API keys
        if "api_keys" in data:
            for key in data["api_keys"]:
                val = data["api_keys"][key]
                if val:
                    data["api_keys"][key] = val[:8] + "..." if len(val) > 8 else "***"
        return data

    # --- Cron API ---

    @app.get("/api/cron/jobs")
    async def list_cron_jobs():
        """List all scheduled cron jobs."""
        if _cron_scheduler:
            return [j.to_dict() for j in _cron_scheduler.store.list_all()]
        return []

    @app.post("/api/cron/jobs/{job_id}/run")
    async def run_cron_job(job_id: str):
        """Manually trigger a cron job."""
        if _cron_scheduler:
            job = _cron_scheduler.store.jobs.get(job_id)
            if job:
                result = await _cron_job_callback(job)
                _cron_scheduler.store.mark_ran(job_id)
                return {"status": "ok", "result": result}
        return {"status": "error", "message": "Job not found"}

    @app.delete("/api/cron/jobs/{job_id}")
    async def delete_cron_job(job_id: str):
        """Delete a cron job."""
        if _cron_scheduler and _cron_scheduler.store.remove(job_id):
            return {"status": "ok"}
        return {"status": "error", "message": "Job not found"}

    # --- WebSocket ---

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket):
        """WebSocket endpoint for real-time event streaming.

        Protocol:
        - Client sends: {"type": "message", "content": "...", "sender_id": "...", "channel": "webchat"}
        - Server streams: lifecycle, tool_start, tool_end, assistant_delta, assistant_message events
        """
        await ws.accept()
        _connected_clients.add(ws)

        try:
            while True:
                # Receive message from client
                raw = await ws.receive_text()
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    await ws.send_json({"type": "error", "message": "Invalid JSON"})
                    continue

                msg_type = data.get("type", "")

                if msg_type == "message":
                    content = data.get("content", "").strip()
                    sender_id = data.get("sender_id", "webchat-user")
                    channel = data.get("channel", "webchat")

                    if not content:
                        await ws.send_json({"type": "error", "message": "Empty message"})
                        continue

                    # Get or create session
                    sm = _session_manager or SessionManager()
                    session = sm.get_or_create_session(sender_id, channel)

                    from langchain_core.messages import HumanMessage

                    # Persist user message
                    sm.append_message(session, HumanMessage(content=content))

                    # Stream agent response
                    async for event in stream_agent(
                        message=content,
                        session_id=session.metadata.session_id,
                        sender_id=sender_id,
                        channel=channel,
                        config=_config,
                        history=session.messages[:-1],  # Exclude the just-added message
                    ):
                        gateway_event = event_to_dict(event)
                        await ws.send_json(gateway_event)

                        # Broadcast to other connected clients
                        for client in _connected_clients:
                            if client != ws:
                                try:
                                    await client.send_json(gateway_event)
                                except Exception:
                                    pass

                    # Persist AI response to session
                    # The last assistant_message event has the final content
                    # (already handled in the stream)

                elif msg_type == "ping":
                    await ws.send_json({"type": "pong", "timestamp": time.time()})

        except WebSocketDisconnect:
            pass
        finally:
            _connected_clients.discard(ws)

    # --- WebChat static files ---

    @app.get("/webchat")
    async def webchat():
        """Serve the WebChat UI."""
        index_path = STATIC_DIR / "index.html"
        if index_path.exists():
            return FileResponse(index_path, media_type="text/html")
        return HTMLResponse("<h1>WebChat UI not found</h1>", status_code=404)

    # Mount static files
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    return app


def run_gateway(config: OpenClawConfig | None = None):
    """Start the gateway server."""
    import uvicorn

    cfg = config or OpenClawConfig.load()
    app = create_app(cfg)

    print(f"\n🦞 OpenClaw-Lang Gateway starting...")
    print(f"   http://{cfg.gateway.host}:{cfg.gateway.port}")
    print(f"   WebChat: http://{cfg.gateway.host}:{cfg.gateway.port}/webchat")
    print(f"   WebSocket: ws://{cfg.gateway.host}:{cfg.gateway.port}/ws")
    print(f"   Cron API: http://{cfg.gateway.host}:{cfg.gateway.port}/api/cron/jobs")
    print(f"   Model: {cfg.agent.model}")
    print(f"   ⏰ Cron scheduler: active (checks every 30s)")
    print()

    uvicorn.run(
        app,
        host=cfg.gateway.host,
        port=cfg.gateway.port,
        log_level="info",
    )
