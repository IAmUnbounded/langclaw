"""Gateway authentication — token-based auth and device pairing.

Mirrors OpenClaw's gateway auth system:
- Token-based API authentication
- Bearer token validation
- Device pairing with token exchange
- Loopback bypass (localhost access)
- Auth event logging
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import Request

logger = logging.getLogger(__name__)

AUTH_TOKEN_FILE = Path.home() / ".langclaw" / "auth_token"


@dataclass
class AuthToken:
    """An authentication token with metadata."""

    token_hash: str  # SHA-256 hash of the token
    label: str = ""  # Human-readable label
    created_at: float = field(default_factory=time.time)
    last_used: float = 0
    scopes: list[str] = field(default_factory=lambda: ["*"])  # * = all
    expires_at: float = 0  # 0 = never
    device_id: str = ""

    @property
    def is_expired(self) -> bool:
        return self.expires_at > 0 and time.time() > self.expires_at


@dataclass
class AuthResult:
    """Result of an authentication attempt."""

    authenticated: bool = False
    sender_id: str = ""
    device_id: str = ""
    scopes: list[str] = field(default_factory=list)
    reason: str = ""
    is_loopback: bool = False


class GatewayAuth:
    """Authentication manager for the gateway server.

    Supports:
    - Bearer token authentication
    - Loopback bypass (requests from 127.0.0.1 are auto-authenticated)
    - Token generation and management
    - Device pairing tokens
    """

    def __init__(
        self,
        require_auth: bool = False,
        allow_loopback: bool = True,
    ) -> None:
        self.require_auth = require_auth
        self.allow_loopback = allow_loopback
        self._tokens: dict[str, AuthToken] = {}
        self._load_stored_token()

    def _load_stored_token(self) -> None:
        """Load the stored auth token from disk."""
        if AUTH_TOKEN_FILE.exists():
            try:
                token = AUTH_TOKEN_FILE.read_text().strip()
                if token:
                    token_hash = self._hash_token(token)
                    self._tokens[token_hash] = AuthToken(
                        token_hash=token_hash,
                        label="primary",
                        scopes=["*"],
                    )
                    logger.debug("Loaded stored auth token")
            except Exception:
                logger.warning("Failed to load auth token")

    @staticmethod
    def _hash_token(token: str) -> str:
        """Hash a token for storage."""
        return hashlib.sha256(token.encode()).hexdigest()

    def generate_token(
        self,
        label: str = "",
        scopes: list[str] | None = None,
        ttl_seconds: float = 0,
        device_id: str = "",
    ) -> str:
        """Generate a new auth token.

        Returns the plaintext token (store securely — it won't be shown again).
        """
        token = secrets.token_urlsafe(32)
        token_hash = self._hash_token(token)

        auth_token = AuthToken(
            token_hash=token_hash,
            label=label,
            scopes=scopes or ["*"],
            expires_at=time.time() + ttl_seconds if ttl_seconds > 0 else 0,
            device_id=device_id,
        )
        self._tokens[token_hash] = auth_token

        # Store as primary token if first one
        if len(self._tokens) == 1:
            AUTH_TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
            AUTH_TOKEN_FILE.write_text(token)
            AUTH_TOKEN_FILE.chmod(0o600)

        logger.info("Auth token generated: label=%s scopes=%s", label, scopes)
        return token

    def revoke_token(self, token: str) -> bool:
        """Revoke a token by its plaintext value."""
        token_hash = self._hash_token(token)
        if token_hash in self._tokens:
            del self._tokens[token_hash]
            logger.info("Auth token revoked")
            return True
        return False

    def authenticate(self, request: Request) -> AuthResult:
        """Authenticate a request.

        Checks (in order):
        1. Loopback bypass (if enabled)
        2. Bearer token
        3. Query parameter token
        """
        # Check loopback
        client_host = request.client.host if request.client else ""
        is_loopback = client_host in ("127.0.0.1", "::1", "localhost")

        if self.allow_loopback and is_loopback:
            return AuthResult(
                authenticated=True,
                sender_id="local",
                is_loopback=True,
                scopes=["*"],
            )

        # Check Bearer token
        auth_header = request.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            return self._validate_token(token)

        # Check query parameter
        token = request.query_params.get("token", "")
        if token:
            return self._validate_token(token)

        # No auth provided
        if self.require_auth:
            return AuthResult(authenticated=False, reason="Authentication required")

        # Auth not required — allow
        return AuthResult(authenticated=True, sender_id="anonymous", scopes=["*"])

    def _validate_token(self, token: str) -> AuthResult:
        """Validate a token and return auth result."""
        token_hash = self._hash_token(token)
        auth_token = self._tokens.get(token_hash)

        if not auth_token:
            return AuthResult(authenticated=False, reason="Invalid token")

        if auth_token.is_expired:
            del self._tokens[token_hash]
            return AuthResult(authenticated=False, reason="Token expired")

        auth_token.last_used = time.time()
        return AuthResult(
            authenticated=True,
            device_id=auth_token.device_id,
            scopes=auth_token.scopes,
        )

    def list_tokens(self) -> list[dict[str, Any]]:
        """List all active tokens (without the actual token values)."""
        return [
            {
                "label": t.label,
                "created_at": t.created_at,
                "last_used": t.last_used,
                "scopes": t.scopes,
                "device_id": t.device_id,
                "is_expired": t.is_expired,
            }
            for t in self._tokens.values()
        ]
