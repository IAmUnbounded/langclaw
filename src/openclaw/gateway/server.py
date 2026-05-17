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
from openclaw.channels.whatsapp import WhatsAppAdapter
from openclaw.config import OpenClawConfig
from openclaw.gateway.auth import GatewayAuth
from openclaw.gateway.events import event_to_dict
from openclaw.gateway.rate_limit import RateLimitConfig, SlidingWindowRateLimiter

# Global state
_config: OpenClawConfig | None = None
_session_manager: SessionManager | None = None
_connected_clients: set[WebSocket] = set()
_start_time: float = 0.0
_cron_scheduler: CronScheduler | None = None
_whatsapp_adapter: WhatsAppAdapter | None = None
_whatsapp_ws_clients: set[WebSocket] = set()
_gateway_auth: GatewayAuth | None = None
_rate_limiter: SlidingWindowRateLimiter | None = None

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


async def _whatsapp_message_handler(message) -> str:
    """Handle inbound WhatsApp messages by running the agent."""
    import logging
    logger = logging.getLogger(__name__)

    logger.info(f"WhatsApp message from {message.sender_id}: {message.content[:80]}...")

    try:
        sm = _session_manager or SessionManager()
        session = sm.get_or_create_session(message.sender_id, "whatsapp")

        from langchain_core.messages import HumanMessage

        sm.append_message(session, HumanMessage(content=message.content))

        result = await run_agent(
            message=message.content,
            session_id=session.metadata.session_id,
            sender_id=message.sender_id,
            channel="whatsapp",
            config=_config,
            history=session.messages[:-1],
        )

        response = result.get("response", "I couldn't process that message.")

        # The response may be a list of content blocks — extract plain text
        if isinstance(response, list):
            text_parts = []
            for block in response:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block["text"])
                elif isinstance(block, str):
                    text_parts.append(block)
            response = "\n".join(text_parts) if text_parts else str(response)

        for msg in result.get("messages", []):
            sm.append_message(session, msg)

        return response
    except Exception as e:
        logger.error(f"WhatsApp handler error: {e}")
        return "Sorry, I encountered an error processing your message."


async def _broadcast_wa_status(status: dict) -> None:
    """Broadcast WhatsApp status to all connected WS clients."""
    event = {"type": "status", **status}
    for client in list(_whatsapp_ws_clients):
        try:
            await client.send_json(event)
        except Exception:
            _whatsapp_ws_clients.discard(client)


async def _broadcast_wa_qr(qr_data: str) -> None:
    """Broadcast QR code data to all connected WS clients."""
    event = {"type": "qr", "data": qr_data}
    for client in list(_whatsapp_ws_clients):
        try:
            await client.send_json(event)
        except Exception:
            _whatsapp_ws_clients.discard(client)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle — starts the cron scheduler and WhatsApp adapter."""
    global _config, _session_manager, _start_time, _cron_scheduler, _whatsapp_adapter
    global _gateway_auth, _rate_limiter
    _config = OpenClawConfig.load()
    _session_manager = SessionManager()
    _start_time = time.time()

    # Initialize auth and rate limiting
    _gateway_auth = GatewayAuth(
        require_auth=_config.gateway.require_auth,
        allow_loopback=_config.gateway.allow_loopback,
    )
    _rate_limiter = SlidingWindowRateLimiter(RateLimitConfig(
        requests_per_minute=_config.gateway.rate_limit_per_minute,
        requests_per_hour=_config.gateway.rate_limit_per_hour,
    ))

    # Start cron scheduler for proactive nudges
    _cron_scheduler = CronScheduler(callback=_cron_job_callback, check_interval=30.0)
    await _cron_scheduler.start()

    # Initialize WhatsApp adapter (always available for linking)
    wa_cfg = _config.channels.whatsapp
    _whatsapp_adapter = WhatsAppAdapter(
        session_name=wa_cfg.session_name,
        allow_from=wa_cfg.allow_from,
        dm_policy=wa_cfg.dm_policy,
        group_policy=wa_cfg.group_policy,
        group_history_count=wa_cfg.group_history_count,
        max_message_length=wa_cfg.max_message_length,
        handler=_whatsapp_message_handler,
    )
    _whatsapp_adapter.add_qr_listener(_broadcast_wa_qr)
    _whatsapp_adapter.add_status_listener(_broadcast_wa_status)

    # Auto-start if previously linked (enabled in config)
    if wa_cfg.enabled:
        await _whatsapp_adapter.start()

    # Initialize heartbeat from HEARTBEAT.md
    try:
        from openclaw.automation import CronStore as _CronStore
        from openclaw.automation.heartbeat import ensure_heartbeat_job
        workspace = _config.get_workspace_path()
        heartbeat = ensure_heartbeat_job(workspace)
        if heartbeat:
            # Reload cron store so scheduler picks up the new job
            _cron_scheduler.store = _CronStore()
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Heartbeat init skipped: {e}")

    # Cleanup expired sessions periodically
    try:
        sm = _session_manager
        cleaned = sm.cleanup_expired_sessions() if sm else 0
        if cleaned:
            import logging
            logging.getLogger(__name__).info(f"Cleaned up {cleaned} expired sessions")
    except Exception:
        pass

    yield

    # Shutdown
    if _whatsapp_adapter:
        await _whatsapp_adapter.stop()
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

    # --- Auth & Rate Limiting Middleware ---

    @app.middleware("http")
    async def auth_and_rate_limit_middleware(request, call_next):
        """Apply authentication and rate limiting to API requests."""
        # Skip for non-API routes (static files, WebChat, etc.)
        if not request.url.path.startswith("/api/"):
            return await call_next(request)

        # Rate limiting
        if _rate_limiter:
            client_ip = request.client.host if request.client else "unknown"
            rl_result = _rate_limiter.check_request(client_ip)
            if not rl_result.allowed:
                return JSONResponse(
                    status_code=429,
                    content={"error": rl_result.reason},
                    headers={"Retry-After": str(int(rl_result.retry_after_seconds))},
                )

        # Authentication
        if _gateway_auth and _gateway_auth.require_auth:
            auth_result = _gateway_auth.authenticate(request)
            if not auth_result.authenticated:
                if _rate_limiter:
                    client_ip = request.client.host if request.client else "unknown"
                    _rate_limiter.record_auth_failure(client_ip)
                return JSONResponse(
                    status_code=401,
                    content={"error": auth_result.reason},
                )

        return await call_next(request)

    # --- Models API ---

    @app.get("/api/models")
    async def list_models(provider: str | None = None):
        """List available models with metadata."""
        from openclaw.models import get_available_models
        models = get_available_models(provider)
        return [
            {
                "provider": m.provider,
                "model_id": m.model_id,
                "display_name": m.display_name,
                "context_window": m.context_window,
                "supports_vision": m.supports_vision,
                "supports_tools": m.supports_tools,
                "supports_thinking": m.supports_thinking,
                "input_price_per_1m": m.input_price_per_1m,
                "output_price_per_1m": m.output_price_per_1m,
            }
            for m in models
        ]

    @app.get("/api/models/aliases")
    async def list_model_aliases():
        """List model aliases."""
        from openclaw.models import list_aliases
        return list_aliases()

    @app.get("/api/models/providers")
    async def list_providers():
        """List available LLM providers and their auth status."""
        from openclaw.models import get_available_providers
        return get_available_providers()

    # --- Plugins API ---

    @app.get("/api/plugins")
    async def list_plugins():
        """List loaded plugins."""
        from openclaw.plugins import HookRegistry
        registry = HookRegistry.get_instance()
        return {
            "plugins": [
                {
                    "name": p.name,
                    "version": p.version,
                    "description": p.description,
                    "hooks": p.hooks,
                    "enabled": p.enabled,
                }
                for p in registry.list_plugins()
            ],
            "hooks": registry.list_hooks(),
        }

    # --- Security API ---

    @app.get("/api/security/audit")
    async def security_audit():
        """Run a security audit and return findings."""
        from openclaw.security import run_security_audit
        cfg = _config or OpenClawConfig()
        return run_security_audit(cfg.get_workspace_path())

    @app.get("/api/security/audit/log")
    async def audit_log(days: int = 7, event_type: str | None = None, limit: int = 100):
        """Query the audit log."""
        from openclaw.security import get_audit_log
        log = get_audit_log()
        return log.query(days=days, event_type=event_type, limit=limit)

    # --- Routing API ---

    @app.get("/api/routing/bindings")
    async def list_route_bindings(channel: str | None = None):
        """List active route bindings."""
        from openclaw.routing import get_resolver
        resolver = get_resolver()
        bindings = resolver.list_bindings(channel)
        return [
            {
                "channel": b.channel,
                "sender_id": b.sender_id,
                "session_id": b.session_id,
                "agent_id": b.agent_id,
                "group_id": b.group_id,
                "sticky": b.sticky,
                "last_used": b.last_used,
            }
            for b in bindings
        ]

    @app.get("/api/routing/stats")
    async def routing_stats():
        """Get routing statistics."""
        from openclaw.routing import get_resolver
        return get_resolver().get_stats()

    # --- Rate Limit API ---

    @app.get("/api/rate-limit/stats")
    async def rate_limit_stats():
        """Get rate limiter statistics."""
        if _rate_limiter:
            return _rate_limiter.get_stats()
        return {"error": "Rate limiter not initialized"}

    # --- Auth API ---

    @app.post("/api/auth/token")
    async def generate_auth_token(label: str = "", ttl: int = 0):
        """Generate a new auth token."""
        if _gateway_auth:
            token = _gateway_auth.generate_token(label=label, ttl_seconds=ttl)
            return {"token": token, "label": label}
        return {"error": "Auth not initialized"}

    @app.get("/api/auth/tokens")
    async def list_auth_tokens():
        """List active auth tokens (values redacted)."""
        if _gateway_auth:
            return _gateway_auth.list_tokens()
        return []

    # --- Daemon API ---

    @app.get("/api/daemon/status")
    async def daemon_status():
        """Get daemon/service status."""
        try:
            from openclaw.daemon import get_service_manager
            mgr = get_service_manager()
            return mgr.status().to_dict()
        except NotImplementedError as e:
            return {"error": str(e)}

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

    # --- WhatsApp WebSocket ---

    @app.websocket("/ws/whatsapp")
    async def whatsapp_ws(ws: WebSocket):
        """WebSocket for WhatsApp linking — streams QR codes and status updates."""
        await ws.accept()
        _whatsapp_ws_clients.add(ws)

        try:
            # Send current status on connect
            if _whatsapp_adapter:
                if _whatsapp_adapter.is_connected:
                    await ws.send_json({
                        "type": "status",
                        "state": "connected",
                        "message": f"Connected as {_whatsapp_adapter.phone_number or 'unknown'}",
                        "phone": _whatsapp_adapter.phone_number,
                    })
                elif _whatsapp_adapter.current_qr:
                    await ws.send_json({
                        "type": "qr",
                        "data": _whatsapp_adapter.current_qr,
                    })
                else:
                    await ws.send_json({
                        "type": "status",
                        "state": "idle",
                        "message": "Ready to link. Click Start Linking to begin.",
                    })

            while True:
                raw = await ws.receive_text()
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    await ws.send_json({"type": "error", "message": "Invalid JSON"})
                    continue

                msg_type = data.get("type", "")

                if msg_type == "status":
                    # Re-send current status
                    if _whatsapp_adapter and _whatsapp_adapter.is_connected:
                        await ws.send_json({
                            "type": "status",
                            "state": "connected",
                            "phone": _whatsapp_adapter.phone_number,
                            "message": f"Connected as {_whatsapp_adapter.phone_number or 'unknown'}",
                        })
                    elif _whatsapp_adapter and _whatsapp_adapter.current_qr:
                        await ws.send_json({
                            "type": "qr",
                            "data": _whatsapp_adapter.current_qr,
                        })
                    else:
                        # Not yet started — start the adapter to get a QR code
                        if _whatsapp_adapter and not _whatsapp_adapter.is_connected and not _whatsapp_adapter._client:
                            await _whatsapp_adapter.start()

                elif msg_type == "refresh_qr":
                    if _whatsapp_adapter:
                        await _whatsapp_adapter.restart_linking()

                elif msg_type == "retry":
                    if _whatsapp_adapter:
                        await _whatsapp_adapter.start()

                elif msg_type == "unlink":
                    if _whatsapp_adapter:
                        await _whatsapp_adapter.unlink()
                        await ws.send_json({
                            "type": "unlinked",
                            "message": "WhatsApp account unlinked",
                        })

                elif msg_type == "ping":
                    await ws.send_json({"type": "pong", "timestamp": time.time()})

        except WebSocketDisconnect:
            pass
        finally:
            _whatsapp_ws_clients.discard(ws)

    # --- WhatsApp linking page ---

    @app.get("/whatsapp/link")
    async def whatsapp_link_page():
        """Serve the WhatsApp linking UI."""
        wa_path = STATIC_DIR / "whatsapp.html"
        if wa_path.exists():
            return FileResponse(wa_path, media_type="text/html")
        return HTMLResponse("<h1>WhatsApp link page not found</h1>", status_code=404)

    @app.get("/api/whatsapp/status")
    async def whatsapp_status():
        """Get WhatsApp connection status."""
        if not _whatsapp_adapter:
            return {"state": "not_initialized"}
        return {
            "state": "connected" if _whatsapp_adapter.is_connected else "disconnected",
            "phone": _whatsapp_adapter.phone_number,
            "has_qr": _whatsapp_adapter.current_qr is not None,
        }

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
    print(f"   WhatsApp: http://{cfg.gateway.host}:{cfg.gateway.port}/whatsapp/link")
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
