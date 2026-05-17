"""Canvas/A2UI rendering system for OpenClaw-Lang.

Provides server-side UI canvas management with A2UI support.
Canvases can render HTML, markdown, or A2UI JSONL definitions
and be presented to users via the gateway.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CANVAS_DIR = Path.home() / ".langclaw" / "canvas"


@dataclass
class Canvas:
    """A single server-side UI canvas."""

    canvas_id: str
    title: str = "Untitled"
    content: str = ""
    url: str = ""
    width: int = 800
    height: int = 600
    visible: bool = False
    created_at: float = 0.0
    updated_at: float = 0.0
    state: dict[str, Any] = field(default_factory=dict)
    a2ui_definition: str = ""  # Raw A2UI JSONL

    def to_dict(self) -> dict[str, Any]:
        """Serialize canvas to a plain dictionary."""
        return {
            "canvas_id": self.canvas_id,
            "title": self.title,
            "content": self.content,
            "url": self.url,
            "width": self.width,
            "height": self.height,
            "visible": self.visible,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "state": self.state,
            "a2ui_definition": self.a2ui_definition,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Canvas":
        """Deserialize a canvas from a plain dictionary."""
        return cls(
            canvas_id=data["canvas_id"],
            title=data.get("title", "Untitled"),
            content=data.get("content", ""),
            url=data.get("url", ""),
            width=data.get("width", 800),
            height=data.get("height", 600),
            visible=data.get("visible", False),
            created_at=data.get("created_at", 0.0),
            updated_at=data.get("updated_at", 0.0),
            state=data.get("state", {}),
            a2ui_definition=data.get("a2ui_definition", ""),
        )


class CanvasManager:
    """Manages canvas lifecycle and persistence.

    Uses the singleton pattern so all parts of the application share
    the same set of canvases.
    """

    _instance: CanvasManager | None = None

    def __init__(self) -> None:
        self._canvases: dict[str, Canvas] = {}
        self._dir = CANVAS_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        self._load_from_disk()

    @classmethod
    def get_instance(cls) -> "CanvasManager":
        """Return the singleton CanvasManager, creating it if needed."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Canvas CRUD
    # ------------------------------------------------------------------

    def create(
        self,
        title: str = "",
        content: str = "",
        url: str = "",
        width: int = 800,
        height: int = 600,
    ) -> Canvas:
        """Create a new canvas and persist it."""
        now = time.time()
        canvas = Canvas(
            canvas_id=str(uuid.uuid4())[:8],
            title=title or "Untitled",
            content=content,
            url=url,
            width=width,
            height=height,
            visible=False,
            created_at=now,
            updated_at=now,
        )
        self._canvases[canvas.canvas_id] = canvas
        self._save_to_disk()
        return canvas

    def get(self, canvas_id: str) -> Canvas | None:
        """Return a canvas by ID, or None if not found."""
        return self._canvases.get(canvas_id)

    def update(self, canvas_id: str, **kwargs: Any) -> Canvas | None:
        """Update arbitrary fields on a canvas.

        Only recognised Canvas fields are applied; unknown keys are
        silently ignored.
        """
        canvas = self._canvases.get(canvas_id)
        if canvas is None:
            return None

        allowed = {
            "title", "content", "url", "width", "height",
            "visible", "state", "a2ui_definition",
        }
        for key, value in kwargs.items():
            if key in allowed:
                setattr(canvas, key, value)

        canvas.updated_at = time.time()
        self._save_to_disk()
        return canvas

    def present(self, canvas_id: str) -> Canvas | None:
        """Mark a canvas as visible (for UI rendering)."""
        return self.update(canvas_id, visible=True)

    def hide(self, canvas_id: str) -> Canvas | None:
        """Mark a canvas as hidden."""
        return self.update(canvas_id, visible=False)

    def destroy(self, canvas_id: str) -> bool:
        """Remove a canvas entirely. Returns True if it existed."""
        if canvas_id not in self._canvases:
            return False
        del self._canvases[canvas_id]
        self._save_to_disk()
        return True

    def list_all(self) -> list[Canvas]:
        """Return all canvases, sorted by creation time (newest first)."""
        return sorted(
            self._canvases.values(),
            key=lambda c: c.created_at,
            reverse=True,
        )

    def snapshot(self, canvas_id: str) -> dict[str, Any] | None:
        """Return the full state of a canvas as a dictionary."""
        canvas = self._canvases.get(canvas_id)
        if canvas is None:
            return None
        return canvas.to_dict()

    # ------------------------------------------------------------------
    # A2UI helpers
    # ------------------------------------------------------------------

    def a2ui_push(self, canvas_id: str, jsonl: str) -> Canvas | None:
        """Push an A2UI JSONL definition to a canvas.

        The JSONL payload is appended to any existing definition so
        incremental updates work naturally.
        """
        canvas = self._canvases.get(canvas_id)
        if canvas is None:
            return None

        if canvas.a2ui_definition:
            canvas.a2ui_definition += "\n" + jsonl
        else:
            canvas.a2ui_definition = jsonl

        canvas.updated_at = time.time()
        self._save_to_disk()
        return canvas

    def a2ui_reset(self, canvas_id: str) -> Canvas | None:
        """Reset a canvas to blank (clear content and A2UI definition)."""
        canvas = self._canvases.get(canvas_id)
        if canvas is None:
            return None

        canvas.content = ""
        canvas.a2ui_definition = ""
        canvas.state = {}
        canvas.updated_at = time.time()
        self._save_to_disk()
        return canvas

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_to_disk(self) -> None:
        """Persist all canvases to ~/.langclaw/canvas/canvases.json."""
        data = {cid: c.to_dict() for cid, c in self._canvases.items()}
        path = self._dir / "canvases.json"
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _load_from_disk(self) -> None:
        """Load canvases from disk if the persistence file exists."""
        path = self._dir / "canvases.json"
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for cid, cdata in data.items():
                self._canvases[cid] = Canvas.from_dict(cdata)
        except (json.JSONDecodeError, KeyError):
            # Corrupted file -- start fresh
            self._canvases = {}
