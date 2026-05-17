"""Canvas tool -- server-side UI canvas management with A2UI support.

Mirrors the original OpenClaw canvas tool, providing actions to create,
present, update, navigate, evaluate, snapshot, list, destroy canvases
and push/reset A2UI JSONL definitions.
"""

from __future__ import annotations

import json

from langchain_core.tools import tool

from openclaw.canvas import CanvasManager


@tool
def canvas_tool(
    action: str,
    canvas_id: str = "",
    url: str = "",
    content: str = "",
    javascript: str = "",
    title: str = "",
    width: int = 800,
    height: int = 600,
) -> str:
    """Manage server-side UI canvases that can be presented to users.

    Canvases support HTML, markdown, and A2UI JSONL rendering.

    Args:
        action: One of "create", "present", "hide", "update", "navigate",
                "evaluate", "snapshot", "list", "destroy", "a2ui_push",
                "a2ui_reset".
        canvas_id: Canvas identifier. Auto-generated for "create"; required
                   for all other actions except "list".
        url: URL for the "navigate" action.
        content: HTML/markdown content for "create", "update", or raw JSONL
                 for "a2ui_push".
        javascript: JavaScript to evaluate in the canvas context (simulated).
        title: Canvas title (used by "create" and "update").
        width: Canvas width in pixels (used by "create").
        height: Canvas height in pixels (used by "create").

    Returns:
        A human-readable status message or JSON payload depending on the
        action.
    """
    mgr = CanvasManager.get_instance()

    # ---- create --------------------------------------------------------
    if action == "create":
        canvas = mgr.create(
            title=title,
            content=content,
            url=url,
            width=width,
            height=height,
        )
        return (
            f"Canvas created: {canvas.canvas_id} "
            f"(title={canvas.title!r}, {canvas.width}x{canvas.height})"
        )

    # ---- list ----------------------------------------------------------
    if action == "list":
        canvases = mgr.list_all()
        if not canvases:
            return "No canvases exist."
        lines = [f"Canvases ({len(canvases)}):"]
        for c in canvases:
            vis = "visible" if c.visible else "hidden"
            lines.append(
                f"  - {c.canvas_id}: {c.title!r} [{vis}] "
                f"{c.width}x{c.height}"
            )
        return "\n".join(lines)

    # ---- actions that require an existing canvas -----------------------
    if not canvas_id:
        return f"Error: canvas_id is required for action {action!r}."

    if action == "present":
        canvas = mgr.present(canvas_id)
        if canvas is None:
            return f"Error: canvas {canvas_id!r} not found."
        return f"Canvas {canvas_id} is now visible."

    if action == "hide":
        canvas = mgr.hide(canvas_id)
        if canvas is None:
            return f"Error: canvas {canvas_id!r} not found."
        return f"Canvas {canvas_id} is now hidden."

    if action == "update":
        kwargs: dict = {}
        if content:
            kwargs["content"] = content
        if title:
            kwargs["title"] = title
        if not kwargs:
            return "Error: nothing to update (provide content or title)."
        canvas = mgr.update(canvas_id, **kwargs)
        if canvas is None:
            return f"Error: canvas {canvas_id!r} not found."
        return f"Canvas {canvas_id} updated."

    if action == "navigate":
        if not url:
            return "Error: url is required for navigate action."
        canvas = mgr.update(canvas_id, url=url)
        if canvas is None:
            return f"Error: canvas {canvas_id!r} not found."
        return f"Canvas {canvas_id} navigated to {url}"

    if action == "evaluate":
        if not javascript:
            return "Error: javascript is required for evaluate action."
        canvas = mgr.get(canvas_id)
        if canvas is None:
            return f"Error: canvas {canvas_id!r} not found."
        # Simulated evaluation -- store the script in state for the UI
        # layer to pick up and execute.
        eval_entry = {
            "script": javascript,
            "queued_at": __import__("time").time(),
        }
        state = dict(canvas.state)
        evals = state.get("pending_evals", [])
        evals.append(eval_entry)
        state["pending_evals"] = evals
        mgr.update(canvas_id, state=state)
        return (
            f"JavaScript queued for canvas {canvas_id} "
            f"({len(evals)} pending eval(s))."
        )

    if action == "snapshot":
        snap = mgr.snapshot(canvas_id)
        if snap is None:
            return f"Error: canvas {canvas_id!r} not found."
        return json.dumps(snap, indent=2)

    if action == "destroy":
        removed = mgr.destroy(canvas_id)
        if not removed:
            return f"Error: canvas {canvas_id!r} not found."
        return f"Canvas {canvas_id} destroyed."

    if action == "a2ui_push":
        if not content:
            return "Error: content (JSONL) is required for a2ui_push."
        canvas = mgr.a2ui_push(canvas_id, content)
        if canvas is None:
            return f"Error: canvas {canvas_id!r} not found."
        lines_count = len(canvas.a2ui_definition.strip().splitlines())
        return (
            f"A2UI definition pushed to canvas {canvas_id} "
            f"({lines_count} JSONL line(s) total)."
        )

    if action == "a2ui_reset":
        canvas = mgr.a2ui_reset(canvas_id)
        if canvas is None:
            return f"Error: canvas {canvas_id!r} not found."
        return f"Canvas {canvas_id} reset to blank."

    return f"Error: unknown action {action!r}."
