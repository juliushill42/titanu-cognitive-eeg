"""Runtime configuration from environment."""
from __future__ import annotations

import os
import secrets
from dataclasses import dataclass


def _bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    port: int = int(os.environ.get("COG_PORT", "8100"))
    tick_sec: float = float(os.environ.get("COG_TICK_INTERVAL_SEC", "1.2"))
    channels: int = int(os.environ.get("COG_CHANNELS", "1024"))
    host: str = os.environ.get("COG_HOST", "127.0.0.1")  # default localhost, not 0.0.0.0
    api_key: str = os.environ.get("COG_API_KEY", "")
    require_auth: bool = _bool("COG_REQUIRE_AUTH", True)
    max_body_bytes: int = int(os.environ.get("COG_MAX_BODY_BYTES", "65536"))
    data_dir: str = os.environ.get("COG_DATA_DIR", "./data")
    db_path: str = os.environ.get("COG_DB_PATH", "")
    model_backend: str = os.environ.get("COG_MODEL_BACKEND", "simulator")  # simulator | http
    model_url: str = os.environ.get("COG_MODEL_URL", "")
    model_api_key: str = os.environ.get("COG_MODEL_API_KEY", "")
    neural_backend: str = os.environ.get("COG_NEURAL_BACKEND", "eeg")  # eeg | simulator | file
    neural_file: str = os.environ.get("COG_NEURAL_FILE", "")
    cors_origin: str = os.environ.get("COG_CORS_ORIGIN", "")

    def resolved_db_path(self) -> str:
        if self.db_path:
            return self.db_path
        return os.path.join(self.data_dir, "cognitive.db")

    def ensure_api_key(self) -> str:
        """Return configured key or generate a one-time key for this process."""
        if self.api_key:
            return self.api_key
        # Generated keys are logged once at startup when auth is required
        return getattr(self, "_generated_key", "") or ""


def load_settings() -> Settings:
    s = Settings()
    if s.require_auth and not s.api_key:
        # Generate ephemeral key if none provided
        object.__setattr__(s, "_generated_key", secrets.token_urlsafe(32))  # type: ignore
    return s
