"""Daemon/service management — run OpenClaw as a background service.

Mirrors OpenClaw's daemon module:
- LaunchD support (macOS)
- Systemd support (Linux)
- Service lifecycle management (install, start, stop, status, uninstall)
- Service audit logging
- Runtime environment configuration
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
import sys
import textwrap
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SERVICE_NAME = "com.langclaw.gateway"
SERVICE_LABEL = "OpenClaw Gateway"


@dataclass
class ServiceStatus:
    """Current status of the daemon service."""

    installed: bool = False
    running: bool = False
    pid: int | None = None
    uptime_seconds: float = 0
    service_type: str = ""  # launchd, systemd, manual
    log_path: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "installed": self.installed,
            "running": self.running,
            "pid": self.pid,
            "uptime_seconds": round(self.uptime_seconds, 1),
            "service_type": self.service_type,
            "log_path": self.log_path,
            "error": self.error,
        }


class ServiceManager(ABC):
    """Abstract base for platform-specific service management."""

    @abstractmethod
    def install(self, port: int = 18789, host: str = "127.0.0.1") -> bool:
        """Install the service."""
        ...

    @abstractmethod
    def uninstall(self) -> bool:
        """Uninstall the service."""
        ...

    @abstractmethod
    def start(self) -> bool:
        """Start the service."""
        ...

    @abstractmethod
    def stop(self) -> bool:
        """Stop the service."""
        ...

    @abstractmethod
    def status(self) -> ServiceStatus:
        """Get current service status."""
        ...

    @abstractmethod
    def logs(self, lines: int = 50) -> str:
        """Get recent log output."""
        ...


class LaunchDManager(ServiceManager):
    """macOS LaunchD service manager.

    Installs a launchd plist in ~/Library/LaunchAgents/ that runs
    the openclaw gateway as a user-level daemon.
    """

    def __init__(self) -> None:
        self._plist_path = (
            Path.home() / "Library" / "LaunchAgents" / f"{SERVICE_NAME}.plist"
        )
        self._log_dir = Path.home() / ".langclaw" / "logs"
        self._log_dir.mkdir(parents=True, exist_ok=True)

    def _plist_content(self, port: int, host: str) -> str:
        """Generate the launchd plist XML."""
        python = sys.executable
        stdout_log = self._log_dir / "gateway.stdout.log"
        stderr_log = self._log_dir / "gateway.stderr.log"

        return textwrap.dedent(f"""\
        <?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
          "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
        <plist version="1.0">
        <dict>
            <key>Label</key>
            <string>{SERVICE_NAME}</string>
            <key>ProgramArguments</key>
            <array>
                <string>{python}</string>
                <string>-m</string>
                <string>openclaw</string>
                <string>gateway</string>
                <string>--port</string>
                <string>{port}</string>
                <string>--host</string>
                <string>{host}</string>
            </array>
            <key>RunAtLoad</key>
            <true/>
            <key>KeepAlive</key>
            <true/>
            <key>StandardOutPath</key>
            <string>{stdout_log}</string>
            <key>StandardErrorPath</key>
            <string>{stderr_log}</string>
            <key>EnvironmentVariables</key>
            <dict>
                <key>PATH</key>
                <string>{os.environ.get('PATH', '/usr/bin:/usr/local/bin')}</string>
                <key>HOME</key>
                <string>{Path.home()}</string>
            </dict>
            <key>WorkingDirectory</key>
            <string>{Path.home()}</string>
            <key>ProcessType</key>
            <string>Background</string>
        </dict>
        </plist>
        """)

    def install(self, port: int = 18789, host: str = "127.0.0.1") -> bool:
        self._plist_path.parent.mkdir(parents=True, exist_ok=True)
        self._plist_path.write_text(self._plist_content(port, host))
        logger.info("LaunchD plist installed: %s", self._plist_path)
        return True

    def uninstall(self) -> bool:
        self.stop()
        if self._plist_path.exists():
            self._plist_path.unlink()
            logger.info("LaunchD plist removed: %s", self._plist_path)
            return True
        return False

    def start(self) -> bool:
        if not self._plist_path.exists():
            logger.error("Service not installed. Run install first.")
            return False
        try:
            subprocess.run(
                ["launchctl", "load", str(self._plist_path)],
                check=True, capture_output=True, text=True,
            )
            logger.info("Service started via launchctl")
            return True
        except subprocess.CalledProcessError as e:
            logger.error("Failed to start service: %s", e.stderr)
            return False

    def stop(self) -> bool:
        try:
            subprocess.run(
                ["launchctl", "unload", str(self._plist_path)],
                check=True, capture_output=True, text=True,
            )
            logger.info("Service stopped via launchctl")
            return True
        except subprocess.CalledProcessError as e:
            logger.error("Failed to stop service: %s", e.stderr)
            return False

    def status(self) -> ServiceStatus:
        status = ServiceStatus(service_type="launchd")
        status.installed = self._plist_path.exists()
        status.log_path = str(self._log_dir / "gateway.stderr.log")

        if not status.installed:
            return status

        try:
            result = subprocess.run(
                ["launchctl", "list", SERVICE_NAME],
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                status.running = True
                # Parse PID from output
                for line in result.stdout.splitlines():
                    if '"PID"' in line or "PID" in line:
                        parts = line.split()
                        for p in parts:
                            if p.isdigit():
                                status.pid = int(p)
                                break
        except Exception as e:
            status.error = str(e)

        return status

    def logs(self, lines: int = 50) -> str:
        log_file = self._log_dir / "gateway.stderr.log"
        if not log_file.exists():
            return "No log file found."
        try:
            content = log_file.read_text(encoding="utf-8", errors="replace")
            log_lines = content.strip().splitlines()
            return "\n".join(log_lines[-lines:])
        except OSError as e:
            return f"Error reading logs: {e}"


class SystemdManager(ServiceManager):
    """Linux Systemd service manager.

    Installs a systemd user service file in ~/.config/systemd/user/.
    """

    def __init__(self) -> None:
        self._unit_dir = Path.home() / ".config" / "systemd" / "user"
        self._unit_path = self._unit_dir / f"{SERVICE_NAME}.service"
        self._log_dir = Path.home() / ".langclaw" / "logs"
        self._log_dir.mkdir(parents=True, exist_ok=True)

    def _unit_content(self, port: int, host: str) -> str:
        """Generate the systemd unit file."""
        python = sys.executable
        return textwrap.dedent(f"""\
        [Unit]
        Description={SERVICE_LABEL}
        After=network.target

        [Service]
        Type=simple
        ExecStart={python} -m openclaw gateway --port {port} --host {host}
        Restart=on-failure
        RestartSec=5
        WorkingDirectory={Path.home()}
        Environment=PATH={os.environ.get('PATH', '/usr/bin:/usr/local/bin')}
        Environment=HOME={Path.home()}
        StandardOutput=append:{self._log_dir}/gateway.stdout.log
        StandardError=append:{self._log_dir}/gateway.stderr.log

        [Install]
        WantedBy=default.target
        """)

    def install(self, port: int = 18789, host: str = "127.0.0.1") -> bool:
        self._unit_dir.mkdir(parents=True, exist_ok=True)
        self._unit_path.write_text(self._unit_content(port, host))
        subprocess.run(
            ["systemctl", "--user", "daemon-reload"],
            capture_output=True, text=True,
        )
        subprocess.run(
            ["systemctl", "--user", "enable", SERVICE_NAME],
            capture_output=True, text=True,
        )
        logger.info("Systemd unit installed: %s", self._unit_path)
        return True

    def uninstall(self) -> bool:
        self.stop()
        subprocess.run(
            ["systemctl", "--user", "disable", SERVICE_NAME],
            capture_output=True, text=True,
        )
        if self._unit_path.exists():
            self._unit_path.unlink()
            subprocess.run(
                ["systemctl", "--user", "daemon-reload"],
                capture_output=True, text=True,
            )
            logger.info("Systemd unit removed")
            return True
        return False

    def start(self) -> bool:
        if not self._unit_path.exists():
            logger.error("Service not installed")
            return False
        result = subprocess.run(
            ["systemctl", "--user", "start", SERVICE_NAME],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            logger.info("Service started via systemctl")
            return True
        logger.error("Failed to start: %s", result.stderr)
        return False

    def stop(self) -> bool:
        result = subprocess.run(
            ["systemctl", "--user", "stop", SERVICE_NAME],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            logger.info("Service stopped")
            return True
        logger.error("Failed to stop: %s", result.stderr)
        return False

    def status(self) -> ServiceStatus:
        status = ServiceStatus(service_type="systemd")
        status.installed = self._unit_path.exists()
        status.log_path = str(self._log_dir / "gateway.stderr.log")

        if not status.installed:
            return status

        try:
            result = subprocess.run(
                ["systemctl", "--user", "is-active", SERVICE_NAME],
                capture_output=True, text=True,
            )
            status.running = result.stdout.strip() == "active"

            if status.running:
                pid_result = subprocess.run(
                    ["systemctl", "--user", "show", SERVICE_NAME, "--property=MainPID"],
                    capture_output=True, text=True,
                )
                pid_line = pid_result.stdout.strip()
                if "=" in pid_line:
                    pid_str = pid_line.split("=", 1)[1]
                    if pid_str.isdigit() and pid_str != "0":
                        status.pid = int(pid_str)
        except Exception as e:
            status.error = str(e)

        return status

    def logs(self, lines: int = 50) -> str:
        try:
            result = subprocess.run(
                ["journalctl", "--user", "-u", SERVICE_NAME, "-n", str(lines), "--no-pager"],
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                return result.stdout
        except FileNotFoundError:
            pass

        # Fall back to log file
        log_file = self._log_dir / "gateway.stderr.log"
        if log_file.exists():
            content = log_file.read_text(encoding="utf-8", errors="replace")
            return "\n".join(content.strip().splitlines()[-lines:])
        return "No logs available."


def get_service_manager() -> ServiceManager:
    """Get the appropriate service manager for the current platform."""
    system = platform.system()
    if system == "Darwin":
        return LaunchDManager()
    elif system == "Linux":
        return SystemdManager()
    else:
        raise NotImplementedError(
            f"Daemon mode not supported on {system}. "
            f"Run 'openclaw gateway' manually instead."
        )
