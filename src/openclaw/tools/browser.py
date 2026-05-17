"""Browser control tool — Chrome DevTools Protocol (CDP) browser automation.

Provides full browser automation via CDP, including navigation, screenshots,
clicking, typing, JavaScript evaluation, and content extraction. Uses
asyncio.subprocess to launch Chrome/Chromium and communicates via the
WebSocket-based CDP protocol.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# Chrome / Chromium path discovery
# ---------------------------------------------------------------------------

_CHROME_CANDIDATES: dict[str, list[str]] = {
    "darwin": [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    ],
    "linux": [
        "google-chrome",
        "google-chrome-stable",
        "chromium",
        "chromium-browser",
        "/usr/bin/google-chrome",
        "/usr/bin/google-chrome-stable",
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
        "/snap/bin/chromium",
    ],
    "win32": [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
        r"C:\Program Files\Chromium\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    ],
}


def _find_chrome_binary() -> str | None:
    """Discover a Chrome/Chromium binary on the current platform."""
    plat = sys.platform
    candidates = _CHROME_CANDIDATES.get(plat, _CHROME_CANDIDATES.get("linux", []))

    for candidate in candidates:
        # Absolute path — check exists
        if os.path.isabs(candidate):
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate
        else:
            # Bare command name — look up via PATH
            found = shutil.which(candidate)
            if found:
                return found

    return None


# ---------------------------------------------------------------------------
# Screenshot storage
# ---------------------------------------------------------------------------

SCREENSHOTS_DIR = Path.home() / ".langclaw" / "browser" / "screenshots"


def _ensure_screenshots_dir() -> Path:
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    return SCREENSHOTS_DIR


# ---------------------------------------------------------------------------
# CDP helpers
# ---------------------------------------------------------------------------


def _run_async(coro):
    """Run an async coroutine from synchronous code.

    Handles the case where an event loop is already running (e.g. inside
    FastAPI/uvicorn) by creating a new thread with its own loop.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        # We are inside an already-running event loop — run in a new thread.
        result: list[Any] = []
        exc: list[BaseException] = []

        def _thread_target():
            try:
                result.append(asyncio.run(coro))
            except BaseException as e:
                exc.append(e)

        t = threading.Thread(target=_thread_target)
        t.start()
        t.join()
        if exc:
            raise exc[0]
        return result[0]
    else:
        return asyncio.run(coro)


async def _cdp_send(ws, method: str, params: dict | None = None, timeout: float = 30) -> dict:
    """Send a CDP command over websocket and wait for the corresponding result."""
    import websockets

    msg_id = uuid.uuid4().int & 0xFFFFFFFF
    payload = {"id": msg_id, "method": method}
    if params:
        payload["params"] = params

    await ws.send(json.dumps(payload))

    deadline = asyncio.get_event_loop().time() + timeout
    while True:
        remaining = deadline - asyncio.get_event_loop().time()
        if remaining <= 0:
            raise TimeoutError(f"CDP command {method} timed out after {timeout}s")
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
        except asyncio.TimeoutError:
            raise TimeoutError(f"CDP command {method} timed out after {timeout}s")
        data = json.loads(raw)
        if data.get("id") == msg_id:
            if "error" in data:
                err = data["error"]
                raise RuntimeError(f"CDP error ({err.get('code')}): {err.get('message')}")
            return data.get("result", {})
        # Otherwise it's an event — discard and keep waiting.


# ---------------------------------------------------------------------------
# Tab tracking
# ---------------------------------------------------------------------------

@dataclass
class BrowserTab:
    """Represents one browser tab / target."""

    target_id: str
    title: str
    url: str
    ws_url: str
    ws: Any = None  # websockets connection


# ---------------------------------------------------------------------------
# BrowserManager
# ---------------------------------------------------------------------------


class BrowserManager:
    """Manages a Chrome/Chromium browser instance via the Chrome DevTools Protocol.

    This is a singleton — only one browser session can be active at a time.
    """

    _instance: "BrowserManager | None" = None
    _lock = threading.Lock()

    def __new__(cls) -> "BrowserManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialised = False
            return cls._instance

    def __init__(self) -> None:
        if self._initialised:
            return
        self._process: subprocess.Popen | None = None
        self._debug_port: int = 0
        self._tabs: list[BrowserTab] = []
        self._active_tab_index: int = 0
        self._headless: bool = True
        self._initialised = True

    # -- Properties ----------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @property
    def active_tab(self) -> BrowserTab | None:
        if not self._tabs:
            return None
        if self._active_tab_index >= len(self._tabs):
            self._active_tab_index = 0
        return self._tabs[self._active_tab_index]

    # -- Lifecycle -----------------------------------------------------------

    async def start(self, headless: bool = True, debug_port: int = 0) -> str:
        """Launch the browser and connect via CDP."""
        if self.is_running:
            return "Browser is already running."

        chrome_path = _find_chrome_binary()
        if not chrome_path:
            return (
                "[error] Could not find Chrome or Chromium. "
                "Please install Google Chrome or set the CHROME_BIN environment variable."
            )

        # Allow override via env
        chrome_path = os.environ.get("CHROME_BIN", chrome_path)

        # Pick a debug port
        if debug_port == 0:
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", 0))
                debug_port = s.getsockname()[1]

        self._debug_port = debug_port
        self._headless = headless

        user_data_dir = Path.home() / ".langclaw" / "browser" / "chrome-profile"
        user_data_dir.mkdir(parents=True, exist_ok=True)

        args = [
            chrome_path,
            f"--remote-debugging-port={debug_port}",
            f"--user-data-dir={user_data_dir}",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-extensions",
            "--disable-popup-blocking",
            "--disable-translate",
            "--disable-background-networking",
            "--disable-sync",
            "--metrics-recording-only",
            "--safebrowsing-disable-auto-update",
            f"--window-size=1280,720",
        ]

        if headless:
            args.append("--headless=new")

        args.append("about:blank")

        try:
            self._process = subprocess.Popen(
                args,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
        except Exception as e:
            return f"[error] Failed to launch browser: {e}"

        # Wait for the CDP endpoint to become available
        import httpx

        for attempt in range(40):  # up to ~10 seconds
            await asyncio.sleep(0.25)
            if self._process.poll() is not None:
                stderr_out = self._process.stderr.read() if self._process.stderr else ""
                return f"[error] Browser exited immediately.\n{stderr_out[:500]}"
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(
                        f"http://127.0.0.1:{debug_port}/json/version",
                        timeout=2,
                    )
                    if resp.status_code == 200:
                        break
            except Exception:
                continue
        else:
            self.stop_sync()
            return "[error] Browser started but CDP endpoint did not become available."

        # Discover initial tabs
        await self._refresh_tabs()

        # Connect to the first page target
        if self._tabs:
            await self._connect_tab(0)

        mode = "headless" if headless else "visible"
        return (
            f"Browser started ({mode}) on CDP port {debug_port}. "
            f"Tabs open: {len(self._tabs)}"
        )

    async def stop(self) -> str:
        """Stop the browser and close all connections."""
        if not self.is_running:
            return "Browser is not running."

        # Close websocket connections
        for tab in self._tabs:
            if tab.ws is not None:
                try:
                    await tab.ws.close()
                except Exception:
                    pass

        self._tabs.clear()
        self._active_tab_index = 0

        try:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=3)
        except Exception:
            pass

        self._process = None
        return "Browser stopped."

    def stop_sync(self) -> str:
        """Synchronous stop for cleanup paths."""
        if self._process is None:
            return "Browser is not running."
        try:
            self._process.terminate()
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.kill()
        except Exception:
            pass
        self._tabs.clear()
        self._active_tab_index = 0
        self._process = None
        return "Browser stopped."

    # -- Tab management ------------------------------------------------------

    async def _refresh_tabs(self) -> None:
        """Refresh the list of open tabs from the CDP endpoint."""
        import httpx

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"http://127.0.0.1:{self._debug_port}/json",
                    timeout=5,
                )
                targets = resp.json()
        except Exception:
            return

        self._tabs = [
            BrowserTab(
                target_id=t["id"],
                title=t.get("title", ""),
                url=t.get("url", ""),
                ws_url=t.get("webSocketDebuggerUrl", ""),
            )
            for t in targets
            if t.get("type") == "page"
        ]

    async def _connect_tab(self, index: int) -> None:
        """Ensure the websocket connection to the given tab is open."""
        import websockets

        if index < 0 or index >= len(self._tabs):
            return

        tab = self._tabs[index]
        if tab.ws is not None:
            # Check if still open
            try:
                await tab.ws.ping()
                self._active_tab_index = index
                return
            except Exception:
                tab.ws = None

        if not tab.ws_url:
            return

        try:
            tab.ws = await websockets.connect(
                tab.ws_url,
                max_size=50 * 1024 * 1024,  # 50 MB for large screenshots
                close_timeout=5,
            )
            self._active_tab_index = index
            # Enable useful CDP domains
            await _cdp_send(tab.ws, "Page.enable")
            await _cdp_send(tab.ws, "Network.enable")
            await _cdp_send(tab.ws, "DOM.enable")
        except Exception:
            tab.ws = None

    async def _ensure_connected(self) -> BrowserTab | None:
        """Return the active tab with a live websocket, or None."""
        tab = self.active_tab
        if tab is None:
            return None
        if tab.ws is None:
            await self._connect_tab(self._active_tab_index)
            tab = self.active_tab
        return tab if tab and tab.ws else None

    # -- Navigation ----------------------------------------------------------

    async def navigate(self, url: str, timeout: int = 30) -> str:
        """Navigate the active tab to the given URL."""
        tab = await self._ensure_connected()
        if not tab:
            return "[error] No active browser tab. Start the browser first."

        try:
            result = await _cdp_send(tab.ws, "Page.navigate", {"url": url}, timeout=timeout)
            frame_id = result.get("frameId", "")
            error_text = result.get("errorText")
            if error_text:
                return f"[error] Navigation failed: {error_text}"

            # Wait for load event
            try:
                deadline = asyncio.get_event_loop().time() + timeout
                while True:
                    remaining = deadline - asyncio.get_event_loop().time()
                    if remaining <= 0:
                        break
                    raw = await asyncio.wait_for(tab.ws.recv(), timeout=remaining)
                    event = json.loads(raw)
                    if event.get("method") in (
                        "Page.loadEventFired",
                        "Page.frameStoppedLoading",
                    ):
                        break
            except (asyncio.TimeoutError, TimeoutError):
                pass  # Page may still be usable even if load event didn't fire

            # Refresh tab metadata
            await self._refresh_tabs()

            return f"Navigated to {url}"

        except Exception as e:
            return f"[error] Navigation failed: {e}"

    # -- Screenshot ----------------------------------------------------------

    async def screenshot(self, width: int = 1280, height: int = 720) -> str:
        """Capture a screenshot of the active tab and save to disk.

        Returns the file path to the saved PNG.
        """
        tab = await self._ensure_connected()
        if not tab:
            return "[error] No active browser tab. Start the browser first."

        try:
            # Set viewport size
            await _cdp_send(
                tab.ws,
                "Emulation.setDeviceMetricsOverride",
                {
                    "width": width,
                    "height": height,
                    "deviceScaleFactor": 1,
                    "mobile": False,
                },
            )

            result = await _cdp_send(
                tab.ws,
                "Page.captureScreenshot",
                {"format": "png", "quality": 85},
            )

            image_data = base64.b64decode(result["data"])

            screenshots_dir = _ensure_screenshots_dir()
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}_{uuid.uuid4().hex[:6]}.png"
            filepath = screenshots_dir / filename
            filepath.write_bytes(image_data)

            size_kb = len(image_data) / 1024
            return (
                f"Screenshot saved: {filepath}\n"
                f"Size: {size_kb:.1f} KB ({width}x{height})"
            )

        except Exception as e:
            return f"[error] Screenshot failed: {e}"

    # -- Click ---------------------------------------------------------------

    async def click(self, selector: str, timeout: int = 30) -> str:
        """Click on an element matching the given CSS selector."""
        tab = await self._ensure_connected()
        if not tab:
            return "[error] No active browser tab. Start the browser first."

        try:
            # Get the document root
            doc_result = await _cdp_send(tab.ws, "DOM.getDocument", {}, timeout=timeout)
            root_node_id = doc_result["root"]["nodeId"]

            # Find the element
            query_result = await _cdp_send(
                tab.ws,
                "DOM.querySelector",
                {"nodeId": root_node_id, "selector": selector},
                timeout=timeout,
            )
            node_id = query_result.get("nodeId", 0)
            if node_id == 0:
                return f"[error] Element not found: {selector}"

            # Get the box model so we know where to click
            box_result = await _cdp_send(
                tab.ws, "DOM.getBoxModel", {"nodeId": node_id}, timeout=timeout
            )
            content_quad = box_result["model"]["content"]
            # content_quad is [x1,y1, x2,y2, x3,y3, x4,y4]
            cx = (content_quad[0] + content_quad[2] + content_quad[4] + content_quad[6]) / 4
            cy = (content_quad[1] + content_quad[3] + content_quad[5] + content_quad[7]) / 4

            # Dispatch mouse events
            for event_type in ("mousePressed", "mouseReleased"):
                await _cdp_send(
                    tab.ws,
                    "Input.dispatchMouseEvent",
                    {
                        "type": event_type,
                        "x": cx,
                        "y": cy,
                        "button": "left",
                        "clickCount": 1,
                    },
                    timeout=timeout,
                )

            return f"Clicked element: {selector} (at {cx:.0f}, {cy:.0f})"

        except Exception as e:
            return f"[error] Click failed: {e}"

    # -- Type text -----------------------------------------------------------

    async def type_text(self, selector: str, text: str, timeout: int = 30) -> str:
        """Click on an element and type text into it."""
        tab = await self._ensure_connected()
        if not tab:
            return "[error] No active browser tab. Start the browser first."

        try:
            # Focus the element first via click
            click_result = await self.click(selector, timeout=timeout)
            if click_result.startswith("[error]"):
                return click_result

            # Small delay to let the element focus
            await asyncio.sleep(0.1)

            # Send each character via Input.dispatchKeyEvent
            for char in text:
                await _cdp_send(
                    tab.ws,
                    "Input.dispatchKeyEvent",
                    {
                        "type": "keyDown",
                        "text": char,
                        "key": char,
                        "unmodifiedText": char,
                    },
                    timeout=timeout,
                )
                await _cdp_send(
                    tab.ws,
                    "Input.dispatchKeyEvent",
                    {
                        "type": "keyUp",
                        "key": char,
                    },
                    timeout=timeout,
                )

            return f"Typed {len(text)} characters into {selector}"

        except Exception as e:
            return f"[error] Type failed: {e}"

    # -- JavaScript evaluation -----------------------------------------------

    async def evaluate_js(self, script: str, timeout: int = 30) -> str:
        """Evaluate JavaScript in the active tab and return the result."""
        tab = await self._ensure_connected()
        if not tab:
            return "[error] No active browser tab. Start the browser first."

        try:
            result = await _cdp_send(
                tab.ws,
                "Runtime.evaluate",
                {
                    "expression": script,
                    "returnByValue": True,
                    "awaitPromise": True,
                    "timeout": timeout * 1000,
                },
                timeout=timeout,
            )

            exception_details = result.get("exceptionDetails")
            if exception_details:
                exc_text = exception_details.get("text", "Unknown error")
                exc_obj = exception_details.get("exception", {})
                desc = exc_obj.get("description", exc_text)
                return f"[JS error] {desc}"

            remote_obj = result.get("result", {})
            value = remote_obj.get("value")
            obj_type = remote_obj.get("type", "undefined")

            if value is None and obj_type == "undefined":
                return "(undefined)"
            if isinstance(value, (dict, list)):
                return json.dumps(value, indent=2, default=str)
            return str(value)

        except Exception as e:
            return f"[error] JS evaluation failed: {e}"

    # -- Page content --------------------------------------------------------

    async def get_page_content(self, timeout: int = 30) -> str:
        """Get the text content of the current page."""
        tab = await self._ensure_connected()
        if not tab:
            return "[error] No active browser tab. Start the browser first."

        try:
            # Use JavaScript to extract text content
            script = """
            (() => {
                const body = document.body;
                if (!body) return '(empty page)';

                // Remove script and style elements from clone
                const clone = body.cloneNode(true);
                const remove = clone.querySelectorAll('script, style, noscript, svg, iframe');
                remove.forEach(el => el.remove());

                // Get text, clean up whitespace
                let text = clone.innerText || clone.textContent || '';
                text = text.replace(/\\n{3,}/g, '\\n\\n').trim();
                return text;
            })()
            """
            result = await self.evaluate_js(script, timeout=timeout)

            # Get current URL and title
            url = await self.evaluate_js("document.location.href", timeout=5)
            title = await self.evaluate_js("document.title", timeout=5)

            # Truncate if very long
            max_len = 15000
            if len(result) > max_len:
                result = result[:max_len] + "\n\n... (content truncated)"

            return f"Page: {title}\nURL: {url}\n\n{result}"

        except Exception as e:
            return f"[error] Failed to get page content: {e}"

    # -- Scroll --------------------------------------------------------------

    async def scroll(self, direction: str = "down", amount: int = 500, timeout: int = 10) -> str:
        """Scroll the page in the given direction."""
        tab = await self._ensure_connected()
        if not tab:
            return "[error] No active browser tab. Start the browser first."

        try:
            dx = 0
            dy = 0
            if direction == "down":
                dy = amount
            elif direction == "up":
                dy = -amount
            elif direction == "right":
                dx = amount
            elif direction == "left":
                dx = -amount
            else:
                return f"[error] Invalid scroll direction: {direction}. Use up/down/left/right."

            # Use Input.dispatchMouseEvent with type=mouseWheel
            await _cdp_send(
                tab.ws,
                "Input.dispatchMouseEvent",
                {
                    "type": "mouseWheel",
                    "x": 640,
                    "y": 360,
                    "deltaX": dx,
                    "deltaY": dy,
                },
                timeout=timeout,
            )

            # Get resulting scroll position
            pos = await self.evaluate_js(
                "JSON.stringify({x: window.scrollX, y: window.scrollY})",
                timeout=5,
            )
            return f"Scrolled {direction} by {amount}px. Position: {pos}"

        except Exception as e:
            return f"[error] Scroll failed: {e}"

    # -- Wait for selector ---------------------------------------------------

    async def wait_for_selector(self, selector: str, timeout: int = 30) -> str:
        """Wait until an element matching the selector appears in the DOM."""
        tab = await self._ensure_connected()
        if not tab:
            return "[error] No active browser tab. Start the browser first."

        try:
            poll_interval = 0.5
            elapsed = 0.0

            while elapsed < timeout:
                # Check via JS for more reliable results
                found = await self.evaluate_js(
                    f"document.querySelector({json.dumps(selector)}) !== null",
                    timeout=5,
                )
                if found == "True" or found == "true":
                    return f"Element found: {selector} (after {elapsed:.1f}s)"

                await asyncio.sleep(poll_interval)
                elapsed += poll_interval

            return f"[error] Timeout waiting for selector: {selector} (waited {timeout}s)"

        except Exception as e:
            return f"[error] Wait failed: {e}"

    # -- List tabs -----------------------------------------------------------

    async def list_tabs(self) -> str:
        """Return a description of all open tabs."""
        await self._refresh_tabs()
        if not self._tabs:
            return "No tabs open."

        lines = []
        for i, tab in enumerate(self._tabs):
            marker = " *" if i == self._active_tab_index else ""
            lines.append(f"  [{i}]{marker} {tab.title or '(untitled)'} — {tab.url}")

        return f"Open tabs ({len(self._tabs)}):\n" + "\n".join(lines)

    # -- Close tab -----------------------------------------------------------

    async def close_tab(self, index: int, timeout: int = 10) -> str:
        """Close the tab at the given index."""
        await self._refresh_tabs()
        if index < 0 or index >= len(self._tabs):
            return f"[error] Invalid tab index: {index}. Open tabs: {len(self._tabs)}"

        tab = self._tabs[index]

        # Close websocket if connected
        if tab.ws:
            try:
                await tab.ws.close()
            except Exception:
                pass

        # Close via CDP HTTP endpoint
        import httpx

        try:
            async with httpx.AsyncClient() as client:
                await client.get(
                    f"http://127.0.0.1:{self._debug_port}/json/close/{tab.target_id}",
                    timeout=timeout,
                )
        except Exception:
            pass

        await self._refresh_tabs()

        # Reconnect to a remaining tab
        if self._tabs:
            self._active_tab_index = min(self._active_tab_index, len(self._tabs) - 1)
            await self._connect_tab(self._active_tab_index)

        return f"Closed tab [{index}]. Remaining tabs: {len(self._tabs)}"


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_browser_manager = BrowserManager()


# ---------------------------------------------------------------------------
# The @tool function
# ---------------------------------------------------------------------------


@tool
def browser_tool(
    action: str,
    url: str = "",
    selector: str = "",
    text: str = "",
    script: str = "",
    direction: str = "down",
    amount: int = 500,
    tab_index: int = 0,
    headless: bool = True,
    timeout: int = 30,
) -> str:
    """Control a browser via Chrome DevTools Protocol.

    Supported actions:
      start     — Launch browser (headless by default).
      stop      — Close the browser.
      navigate  — Go to a URL (pass url=...).
      screenshot — Capture the visible page and save as PNG.
      click     — Click an element (pass selector=...).
      type      — Type text into an element (pass selector=... and text=...).
      evaluate  — Run JavaScript (pass script=...).
      content   — Get the text content of the current page.
      scroll    — Scroll the page (direction=up/down/left/right, amount=pixels).
      wait      — Wait for a CSS selector to appear (pass selector=...).
      tabs      — List all open tabs.
      close_tab — Close a tab (pass tab_index=...).

    Args:
        action: The browser action to perform.
        url: Target URL for navigate action.
        selector: CSS selector for click, type, and wait actions.
        text: Text to enter for the type action.
        script: JavaScript code for the evaluate action.
        direction: Scroll direction — up, down, left, right (default: down).
        amount: Scroll distance in pixels (default: 500).
        tab_index: Tab index for close_tab action (default: 0).
        headless: Whether to run headless when starting (default: True).
        timeout: Timeout in seconds for the action (default: 30).

    Returns:
        Result message describing what happened.
    """
    mgr = _browser_manager

    # Validate that browser is running for actions that need it
    needs_browser = action not in ("start",)
    if needs_browser and not mgr.is_running:
        if action == "stop":
            return "Browser is not running."
        return "[error] Browser is not running. Use action='start' first."

    try:
        if action == "start":
            return _run_async(mgr.start(headless=headless, debug_port=0))

        elif action == "stop":
            return _run_async(mgr.stop())

        elif action == "navigate":
            if not url:
                return "[error] 'url' parameter is required for navigate action."
            return _run_async(mgr.navigate(url, timeout=timeout))

        elif action == "screenshot":
            return _run_async(mgr.screenshot())

        elif action == "click":
            if not selector:
                return "[error] 'selector' parameter is required for click action."
            return _run_async(mgr.click(selector, timeout=timeout))

        elif action == "type":
            if not selector:
                return "[error] 'selector' parameter is required for type action."
            if not text:
                return "[error] 'text' parameter is required for type action."
            return _run_async(mgr.type_text(selector, text, timeout=timeout))

        elif action == "evaluate":
            if not script:
                return "[error] 'script' parameter is required for evaluate action."
            return _run_async(mgr.evaluate_js(script, timeout=timeout))

        elif action == "content":
            return _run_async(mgr.get_page_content(timeout=timeout))

        elif action == "scroll":
            return _run_async(mgr.scroll(direction=direction, amount=amount, timeout=timeout))

        elif action == "wait":
            if not selector:
                return "[error] 'selector' parameter is required for wait action."
            return _run_async(mgr.wait_for_selector(selector, timeout=timeout))

        elif action == "tabs":
            return _run_async(mgr.list_tabs())

        elif action == "close_tab":
            return _run_async(mgr.close_tab(tab_index, timeout=timeout))

        else:
            valid = "start, stop, navigate, screenshot, click, type, evaluate, content, scroll, wait, tabs, close_tab"
            return f"[error] Unknown action: {action}. Valid actions: {valid}"

    except Exception as e:
        return f"[error] Browser action '{action}' failed: {e}"
