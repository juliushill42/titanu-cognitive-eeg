#!/usr/bin/env python3
"""
cognitive.sh v2 — Cognitive Bandwidth Multiplier
Hardened: auth, validation, thread-safe engine, persistent memory,
input/model adapters, XSS-safe dashboard, bind-localhost by default.
"""
from __future__ import annotations

import html
import json
import logging
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from typing import Any, Optional
from urllib.parse import urlparse

# Ensure package import works when run as python -m src.cognitive.main
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.cognitive.config import load_settings
from src.cognitive.auth import Authenticator, AuthError
from src.cognitive.memory_store import PersistentMemory
from src.cognitive.adapters import build_neural_adapter, build_model_client
from src.cognitive.engine import CognitiveEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("cognitive")

SETTINGS = load_settings()
START = time.time()

# Resolve API key (may be generated)
API_KEY = SETTINGS.api_key or getattr(SETTINGS, "_generated_key", "") or ""
if SETTINGS.require_auth and not API_KEY:
    import secrets
    API_KEY = secrets.token_urlsafe(32)

AUTH = Authenticator(API_KEY, require_auth=SETTINGS.require_auth)

os.makedirs(SETTINGS.data_dir, exist_ok=True)
MEMORY = PersistentMemory(SETTINGS.resolved_db_path())
NEURAL = build_neural_adapter(SETTINGS.neural_backend, SETTINGS.neural_file)
MODEL = build_model_client(SETTINGS.model_backend, SETTINGS.model_url, SETTINGS.model_api_key)
ENGINE = CognitiveEngine(
    channels=SETTINGS.channels,
    neural=NEURAL,
    model=MODEL,
    memory=MEMORY,
)

DASHBOARD_HTML = ""


def load_dashboard() -> None:
    global DASHBOARD_HTML
    path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "templates", "dashboard.html")
    )
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            DASHBOARD_HTML = f.read()
    else:
        DASHBOARD_HTML = "<!DOCTYPE html><html><body><h1>dashboard missing</h1></body></html>"


def simulation_loop() -> None:
    while True:
        try:
            ENGINE.tick()
        except Exception:
            log.exception("tick failed")
        time.sleep(SETTINGS.tick_sec)


class Handler(BaseHTTPRequestHandler):
    server_version = "cognitive.sh/2.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        log.info("%s - %s", self.address_string(), fmt % args)

    def _headers_dict(self) -> dict:
        return {k: v for k, v in self.headers.items()}

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        if length < 0 or length > SETTINGS.max_body_bytes:
            raise ValueError(f"body too large (max {SETTINGS.max_body_bytes})")
        if length == 0:
            return b""
        return self.rfile.read(length)

    def _json(self, code: int, obj: Any) -> None:
        body = json.dumps(obj, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        if SETTINGS.cors_origin:
            self.send_header("Access-Control-Allow-Origin", SETTINGS.cors_origin)
            self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type, X-API-Key")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def _html(self, code: int, text: str) -> None:
        body = text.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'")
        self.end_headers()
        self.wfile.write(body)

    def _require_auth(self) -> Optional[str]:
        try:
            AUTH.check(self._headers_dict())
            return None
        except AuthError:
            self._json(401, {"error": "unauthorized", "hint": "Send Authorization: Bearer <COG_API_KEY> or X-API-Key header"})
            return "unauthorized"

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        if SETTINGS.cors_origin:
            self.send_header("Access-Control-Allow-Origin", SETTINGS.cors_origin)
            self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type, X-API-Key")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self) -> None:
        path = urlparse(self.path).path

        if path == "/health":
            # Health is unauthenticated for orchestrators
            snap = ENGINE.snapshot()
            return self._json(200, {
                "status": "ok",
                "version": "2.0.0",
                "uptime_sec": round(time.time() - START, 1),
                "mode": snap["mode"],
                "cognitive_load": snap["cognitive_load"],
                "working_memory": snap["working_memory_items"],
                "external_memory": snap["external_memory_items"],
                "ai_calls": snap["ai_calls_total"],
                "auth_required": SETTINGS.require_auth,
            })

        if path == "/":
            return self._json(200, {
                "service": "cognitive.sh",
                "version": "2.0.0",
                "message": "Cognitive Bandwidth Multiplier — Brain-AI Symbiosis Layer",
                "dashboard": "/dashboard",
                "health": "/health",
                "auth": "required" if SETTINGS.require_auth else "disabled",
            })

        if path == "/dashboard":
            # Dashboard page itself is public; API calls from it must carry the key
            # Inject API key only if generated for local demo — prefer env in production
            page = DASHBOARD_HTML
            # Never embed the real key in HTML for production; use meta only when auth disabled
            if not SETTINGS.require_auth:
                page = page.replace("/*INJECT_KEY*/", "window.COG_API_KEY='';")
            else:
                page = page.replace("/*INJECT_KEY*/", "window.COG_API_KEY=''; // set via localStorage or prompt")
            return self._html(200, page)

        if path == "/docs":
            return self._html(200, """<!DOCTYPE html><html><head><meta charset="utf-8"><title>cognitive.sh docs</title></head>
<body style="font-family:monospace;background:#0a0e17;color:#e2e8f0;padding:2rem">
<h1>cognitive.sh v2 API</h1>
<p>All /api/* routes require <code>Authorization: Bearer &lt;COG_API_KEY&gt;</code> when auth is enabled.</p>
<ul>
<li>GET /health — public</li>
<li>GET /api/v1/state</li>
<li>GET /api/v1/attention</li>
<li>GET /api/v1/memory</li>
<li>GET /api/v1/events</li>
<li>POST /api/v1/recall — {"query":"..."}</li>
<li>POST /api/v1/intent — {"intent":"...","call_ai":true}</li>
</ul>
</body></html>""")

        # Authenticated API
        if path.startswith("/api/"):
            if self._require_auth():
                return
            if path == "/api/v1/state":
                return self._json(200, ENGINE.snapshot())
            if path == "/api/v1/attention":
                s = ENGINE.snapshot()
                return self._json(200, s["attention"])
            if path == "/api/v1/memory":
                s = ENGINE.snapshot()
                return self._json(200, {
                    "working": s["working_memory"],
                    "external": s["external_memory"],
                })
            if path == "/api/v1/events":
                s = ENGINE.snapshot()
                return self._json(200, list(reversed(s["recent_events"])))
            return self._json(404, {"error": "not found"})

        return self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if not path.startswith("/api/"):
            return self._json(404, {"error": "not found"})
        if self._require_auth():
            return
        try:
            raw = self._read_body()
        except ValueError as e:
            return self._json(413, {"error": str(e)})
        try:
            body = json.loads(raw.decode("utf-8") or "{}")
        except (UnicodeDecodeError, json.JSONDecodeError):
            return self._json(400, {"error": "invalid json"})
        if not isinstance(body, dict):
            return self._json(400, {"error": "json object required"})

        if path == "/api/v1/recall":
            query = body.get("query", "")
            if not isinstance(query, str):
                return self._json(400, {"error": "query must be string"})
            limit = body.get("limit", 5)
            try:
                limit = int(limit)
            except (TypeError, ValueError):
                limit = 5
            return self._json(200, ENGINE.recall(query, limit=limit))

        if path == "/api/v1/intent":
            intent = body.get("intent", "")
            if not isinstance(intent, str) or not intent.strip():
                return self._json(400, {"error": "intent must be non-empty string"})
            call_ai = bool(body.get("call_ai", False))
            return self._json(200, ENGINE.set_intent(intent.strip(), call_ai=call_ai))

        return self._json(404, {"error": "not found"})


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main() -> None:
    load_dashboard()
    log.info("cognitive.sh v2 starting")
    log.info("bind=%s:%s auth=%s neural=%s model=%s db=%s",
             SETTINGS.host, SETTINGS.port, SETTINGS.require_auth,
             SETTINGS.neural_backend, SETTINGS.model_backend,
             SETTINGS.resolved_db_path())
    if SETTINGS.require_auth:
        log.info("API key (store securely): %s", API_KEY)

    t = threading.Thread(target=simulation_loop, daemon=True, name="cog-tick")
    t.start()

    server = ThreadedHTTPServer((SETTINGS.host, SETTINGS.port), Handler)
    log.info("Symbiosis layer online at http://%s:%s", SETTINGS.host, SETTINGS.port)
    log.info("Dashboard: http://%s:%s/dashboard", SETTINGS.host, SETTINGS.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("shutdown")
        server.shutdown()


if __name__ == "__main__":
    main()
