"""Security module — audit, dangerous tool detection, and content scanning.

Mirrors OpenClaw's security system:
- Audit system: tracks tool usage and security-relevant events
- Dangerous tools detection: identifies risky tool operations
- File system scanning: detects secrets and sensitive files
- External content security: validates URLs and external content
- Channel metadata security: validates message sources
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Audit system
# ---------------------------------------------------------------------------

AUDIT_DIR = Path.home() / ".langclaw" / "audit"


@dataclass
class AuditEntry:
    """A single audit log entry."""

    timestamp: float = field(default_factory=time.time)
    event_type: str = ""  # tool_call, security_warning, policy_violation, etc.
    channel: str = ""
    sender_id: str = ""
    session_id: str = ""
    tool_name: str = ""
    tool_args: dict[str, Any] = field(default_factory=dict)
    result_summary: str = ""
    risk_level: str = "low"  # low, medium, high, critical
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AuditLog:
    """Persistent audit log for security-relevant events.

    Stores entries as JSONL files in ~/.langclaw/audit/, rotated daily.
    """

    def __init__(self, audit_dir: Path | None = None) -> None:
        self._dir = audit_dir or AUDIT_DIR
        self._dir.mkdir(parents=True, exist_ok=True)

    def _log_path(self) -> Path:
        """Current day's log file."""
        date_str = time.strftime("%Y-%m-%d", time.gmtime())
        return self._dir / f"audit-{date_str}.jsonl"

    def record(self, entry: AuditEntry) -> None:
        """Append an audit entry to the log."""
        try:
            with open(self._log_path(), "a", encoding="utf-8") as f:
                f.write(json.dumps(entry.to_dict()) + "\n")
        except Exception:
            logger.exception("Failed to write audit entry")

    def log_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        channel: str = "",
        sender_id: str = "",
        session_id: str = "",
        risk_level: str = "low",
        result_summary: str = "",
    ) -> None:
        """Convenience method for logging tool calls."""
        self.record(AuditEntry(
            event_type="tool_call",
            channel=channel,
            sender_id=sender_id,
            session_id=session_id,
            tool_name=tool_name,
            tool_args=_sanitize_args(tool_args),
            result_summary=result_summary[:500],
            risk_level=risk_level,
        ))

    def log_security_event(
        self,
        event_type: str,
        details: dict[str, Any],
        risk_level: str = "medium",
        channel: str = "",
        sender_id: str = "",
    ) -> None:
        """Log a security-relevant event."""
        self.record(AuditEntry(
            event_type=event_type,
            channel=channel,
            sender_id=sender_id,
            risk_level=risk_level,
            details=details,
        ))

    def query(
        self,
        days: int = 7,
        event_type: str | None = None,
        risk_level: str | None = None,
        tool_name: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query audit entries with optional filters."""
        entries: list[dict[str, Any]] = []

        for i in range(days):
            ts = time.time() - (i * 86400)
            date_str = time.strftime("%Y-%m-%d", time.gmtime(ts))
            path = self._dir / f"audit-{date_str}.jsonl"
            if not path.exists():
                continue
            try:
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        entry = json.loads(line)
                        if event_type and entry.get("event_type") != event_type:
                            continue
                        if risk_level and entry.get("risk_level") != risk_level:
                            continue
                        if tool_name and entry.get("tool_name") != tool_name:
                            continue
                        entries.append(entry)
            except Exception:
                logger.exception("Error reading audit file %s", path)

        # Most recent first
        entries.sort(key=lambda e: e.get("timestamp", 0), reverse=True)
        return entries[:limit]

    def cleanup(self, max_days: int = 90) -> int:
        """Remove audit logs older than max_days."""
        cutoff = time.time() - (max_days * 86400)
        removed = 0
        for path in self._dir.glob("audit-*.jsonl"):
            try:
                date_str = path.stem.replace("audit-", "")
                file_ts = time.mktime(time.strptime(date_str, "%Y-%m-%d"))
                if file_ts < cutoff:
                    path.unlink()
                    removed += 1
            except (ValueError, OSError):
                continue
        return removed


def _sanitize_args(args: dict[str, Any]) -> dict[str, Any]:
    """Remove sensitive values from tool args before logging."""
    sanitized = {}
    sensitive_keys = {"password", "token", "secret", "api_key", "key", "auth"}
    for k, v in args.items():
        if any(s in k.lower() for s in sensitive_keys):
            sanitized[k] = "***REDACTED***"
        elif isinstance(v, str) and len(v) > 1000:
            sanitized[k] = v[:500] + f"... ({len(v)} chars)"
        else:
            sanitized[k] = v
    return sanitized


# ---------------------------------------------------------------------------
# Dangerous tools detection
# ---------------------------------------------------------------------------

# Tools that can modify system state or access sensitive data
DANGEROUS_TOOLS: dict[str, str] = {
    "bash_tool": "Can execute arbitrary shell commands",
    "write_file_tool": "Can write to arbitrary file paths",
    "edit_file_tool": "Can modify arbitrary files",
    "terminal_run_tool": "Can run persistent terminal processes",
    "terminal_send_tool": "Can send input to running processes",
    "git_commit_tool": "Can create git commits",
    "git_branch_tool": "Can create/delete git branches",
    "browser_tool": "Can navigate to arbitrary URLs and execute JavaScript",
}

# Patterns that indicate potentially dangerous operations
DANGEROUS_PATTERNS: list[tuple[str, str, str]] = [
    # (tool_name, arg_pattern_regex, reason)
    ("bash_tool", r"rm\s+-rf", "Recursive file deletion"),
    ("bash_tool", r"sudo\s+", "Privilege escalation"),
    ("bash_tool", r"chmod\s+777", "Overly permissive file permissions"),
    ("bash_tool", r"curl.*\|\s*(ba)?sh", "Piping remote script to shell"),
    ("bash_tool", r"wget.*\|\s*(ba)?sh", "Piping remote script to shell"),
    ("bash_tool", r"dd\s+if=", "Low-level disk operations"),
    ("bash_tool", r"mkfs\.", "Filesystem formatting"),
    ("bash_tool", r">\s*/dev/", "Writing to device files"),
    ("bash_tool", r"iptables", "Firewall modification"),
    ("bash_tool", r"systemctl\s+(stop|disable|mask)", "Service disruption"),
    ("write_file_tool", r"/etc/", "Writing to system configuration"),
    ("write_file_tool", r"\.ssh/", "Writing to SSH configuration"),
    ("write_file_tool", r"\.env", "Writing to environment files"),
    ("write_file_tool", r"crontab", "Modifying cron jobs"),
    ("edit_file_tool", r"/etc/", "Editing system configuration"),
]


def classify_tool_risk(tool_name: str, args: dict[str, Any]) -> str:
    """Classify the risk level of a tool call.

    Returns: 'low', 'medium', 'high', or 'critical'
    """
    if tool_name not in DANGEROUS_TOOLS:
        return "low"

    # Check for dangerous patterns
    args_str = json.dumps(args, default=str)
    for pattern_tool, pattern, reason in DANGEROUS_PATTERNS:
        if tool_name == pattern_tool and re.search(pattern, args_str, re.IGNORECASE):
            logger.warning(
                "High-risk tool call detected: %s — %s (args: %s)",
                tool_name, reason, args_str[:200],
            )
            return "high"

    return "medium"


def is_dangerous_tool(tool_name: str) -> bool:
    """Check if a tool is in the dangerous tools list."""
    return tool_name in DANGEROUS_TOOLS


def get_tool_risk_explanation(tool_name: str) -> str:
    """Get a human-readable explanation of why a tool is considered dangerous."""
    return DANGEROUS_TOOLS.get(tool_name, "No known risks")


# ---------------------------------------------------------------------------
# File system security scanning
# ---------------------------------------------------------------------------

# Patterns for detecting secrets in files
SECRET_PATTERNS: list[tuple[str, str]] = [
    (r"(?i)(api[_-]?key|apikey)\s*[:=]\s*['\"]?[\w-]{20,}", "API key"),
    (r"(?i)(secret|token)\s*[:=]\s*['\"]?[\w-]{20,}", "Secret/token"),
    (r"(?i)(password|passwd|pwd)\s*[:=]\s*['\"]?.+", "Password"),
    (r"(?i)bearer\s+[\w.-]{20,}", "Bearer token"),
    (r"sk-[a-zA-Z0-9]{20,}", "OpenAI API key"),
    (r"sk-ant-[a-zA-Z0-9-]{20,}", "Anthropic API key"),
    (r"ghp_[a-zA-Z0-9]{36}", "GitHub personal access token"),
    (r"-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----", "Private key"),
    (r"(?i)aws[_-]?(access[_-]?key|secret)", "AWS credentials"),
    (r"(?i)(mysql|postgres|mongodb)://\S+:\S+@", "Database connection string"),
]

# Files that commonly contain secrets
SENSITIVE_FILES: list[str] = [
    ".env",
    ".env.local",
    ".env.production",
    "credentials.json",
    "service-account.json",
    "id_rsa",
    "id_ed25519",
    ".netrc",
    ".pgpass",
    "docker-compose.yml",  # May contain passwords
]


@dataclass
class SecurityFinding:
    """A security issue found during scanning."""

    file_path: str
    finding_type: str  # secret_detected, sensitive_file, permission_issue
    description: str
    severity: str  # info, warning, error, critical
    line_number: int = 0
    recommendation: str = ""


def scan_directory(
    directory: str | Path,
    max_depth: int = 5,
    max_files: int = 1000,
) -> list[SecurityFinding]:
    """Scan a directory for security issues.

    Checks for:
    - Files containing secrets (API keys, tokens, passwords)
    - Sensitive files that shouldn't be in a workspace
    - Overly permissive file permissions
    """
    findings: list[SecurityFinding] = []
    directory = Path(directory)
    scanned = 0

    for root, dirs, files in os.walk(directory):
        # Respect max depth
        depth = len(Path(root).relative_to(directory).parts)
        if depth > max_depth:
            dirs.clear()
            continue

        # Skip common non-relevant directories
        dirs[:] = [
            d for d in dirs
            if d not in {".git", "node_modules", "__pycache__", ".venv", "venv", ".tox"}
        ]

        for fname in files:
            if scanned >= max_files:
                break
            scanned += 1

            fpath = Path(root) / fname

            # Check for sensitive files
            if fname in SENSITIVE_FILES:
                findings.append(SecurityFinding(
                    file_path=str(fpath),
                    finding_type="sensitive_file",
                    description=f"Sensitive file detected: {fname}",
                    severity="warning",
                    recommendation=f"Ensure {fname} is in .gitignore and not committed",
                ))

            # Check file permissions (Unix only)
            try:
                stat = fpath.stat()
                mode = oct(stat.st_mode)[-3:]
                if mode in ("777", "776", "766"):
                    findings.append(SecurityFinding(
                        file_path=str(fpath),
                        finding_type="permission_issue",
                        description=f"Overly permissive file: mode {mode}",
                        severity="warning",
                        recommendation=f"Run: chmod 644 {fpath}",
                    ))
            except OSError:
                continue

            # Scan file content for secrets (text files only)
            if fpath.suffix in {
                ".py", ".js", ".ts", ".json", ".yaml", ".yml",
                ".toml", ".ini", ".cfg", ".conf", ".sh", ".env",
                ".txt", ".md", ".xml", ".properties",
            }:
                try:
                    content = fpath.read_text(encoding="utf-8", errors="ignore")
                    if len(content) > 500_000:  # Skip very large files
                        continue
                    for pattern, secret_type in SECRET_PATTERNS:
                        for match in re.finditer(pattern, content):
                            # Find line number
                            line_no = content[:match.start()].count("\n") + 1
                            findings.append(SecurityFinding(
                                file_path=str(fpath),
                                finding_type="secret_detected",
                                description=f"Possible {secret_type} detected",
                                severity="error",
                                line_number=line_no,
                                recommendation="Remove or rotate this credential",
                            ))
                except (OSError, UnicodeDecodeError):
                    continue

    logger.info(
        "Security scan complete: %d files scanned, %d findings",
        scanned, len(findings),
    )
    return findings


# ---------------------------------------------------------------------------
# External content security
# ---------------------------------------------------------------------------

# URL patterns that should be blocked or flagged
BLOCKED_URL_PATTERNS: list[tuple[str, str]] = [
    (r"^file://", "Local file access via URL"),
    (r"^data:", "Data URI (potential XSS)"),
    (r"localhost|127\.0\.0\.1|0\.0\.0\.0", "Localhost/loopback access"),
    (r"169\.254\.169\.254", "AWS metadata endpoint (SSRF)"),
    (r"metadata\.google\.internal", "GCP metadata endpoint (SSRF)"),
    (r"10\.\d+\.\d+\.\d+|172\.(1[6-9]|2\d|3[01])\.\d+\.\d+|192\.168\.\d+\.\d+",
     "Private network access (SSRF)"),
]


def validate_url(url: str) -> tuple[bool, str]:
    """Validate a URL for security issues.

    Returns (is_safe, reason).
    """
    for pattern, reason in BLOCKED_URL_PATTERNS:
        if re.search(pattern, url, re.IGNORECASE):
            return False, reason
    return True, "URL is safe"


def validate_external_content(content: str, max_length: int = 100_000) -> tuple[bool, str]:
    """Validate external content for injection attempts.

    Returns (is_safe, reason).
    """
    if len(content) > max_length:
        return False, f"Content too large ({len(content)} chars, max {max_length})"

    # Check for common injection patterns
    injection_patterns = [
        (r"<script[^>]*>", "Script injection detected"),
        (r"javascript:", "JavaScript URI detected"),
        (r"on\w+\s*=", "Event handler injection detected"),
    ]
    for pattern, reason in injection_patterns:
        if re.search(pattern, content, re.IGNORECASE):
            return False, reason

    return True, "Content is safe"


# ---------------------------------------------------------------------------
# Security summary
# ---------------------------------------------------------------------------

def run_security_audit(
    workspace_path: str | Path | None = None,
    include_file_scan: bool = True,
) -> dict[str, Any]:
    """Run a comprehensive security audit.

    Returns a summary with findings by category and severity.
    """
    workspace = Path(workspace_path) if workspace_path else Path.home() / ".langclaw"
    findings: list[SecurityFinding] = []

    # Check workspace directory permissions
    if workspace.exists():
        try:
            mode = oct(workspace.stat().st_mode)[-3:]
            if mode not in ("700", "750", "755"):
                findings.append(SecurityFinding(
                    file_path=str(workspace),
                    finding_type="permission_issue",
                    description=f"Workspace directory permissions: {mode}",
                    severity="info",
                    recommendation="Consider: chmod 750 ~/.langclaw",
                ))
        except OSError:
            pass

    # Check for config file security
    config_path = workspace / "langclaw.json"
    if config_path.exists():
        try:
            config_data = json.loads(config_path.read_text())
            api_keys = config_data.get("api_keys", {})
            if any(v for v in api_keys.values() if v):
                findings.append(SecurityFinding(
                    file_path=str(config_path),
                    finding_type="sensitive_file",
                    description="Config file contains API keys in plaintext",
                    severity="warning",
                    recommendation="Use environment variables instead of storing keys in config",
                ))
        except (json.JSONDecodeError, OSError):
            pass

    # File system scan
    if include_file_scan and workspace.exists():
        fs_findings = scan_directory(workspace)
        findings.extend(fs_findings)

    # Categorize results
    by_severity: dict[str, int] = {}
    by_type: dict[str, int] = {}
    for f in findings:
        by_severity[f.severity] = by_severity.get(f.severity, 0) + 1
        by_type[f.finding_type] = by_type.get(f.finding_type, 0) + 1

    return {
        "total_findings": len(findings),
        "by_severity": by_severity,
        "by_type": by_type,
        "findings": [
            {
                "file": f.file_path,
                "type": f.finding_type,
                "description": f.description,
                "severity": f.severity,
                "line": f.line_number,
                "recommendation": f.recommendation,
            }
            for f in findings
        ],
        "timestamp": time.time(),
    }


# Module-level singleton for audit log
_audit_log: AuditLog | None = None


def get_audit_log() -> AuditLog:
    """Return the global audit log singleton."""
    global _audit_log
    if _audit_log is None:
        _audit_log = AuditLog()
    return _audit_log
