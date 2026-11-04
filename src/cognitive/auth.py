"""API key authentication."""
from __future__ import annotations

import hmac
import logging
from typing import Optional

log = logging.getLogger("cognitive.auth")


class AuthError(Exception):
    pass


class Authenticator:
    def __init__(self, api_key: str, require_auth: bool = True):
        self.api_key = api_key or ""
        self.require_auth = require_auth
        if require_auth and not self.api_key:
            raise ValueError("Auth required but no API key configured")

    def check(self, headers: dict) -> None:
        if not self.require_auth:
            return
        # Accept Authorization: Bearer <key> or X-API-Key: <key>
        auth = headers.get("Authorization") or headers.get("authorization") or ""
        xkey = headers.get("X-API-Key") or headers.get("x-api-key") or ""
        token = ""
        if auth.lower().startswith("bearer "):
            token = auth[7:].strip()
        elif xkey:
            token = xkey.strip()
        if not token or not hmac.compare_digest(token, self.api_key):
            raise AuthError("unauthorized")
