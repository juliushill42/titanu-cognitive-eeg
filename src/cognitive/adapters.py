"""Input and model adapter interfaces with concrete backends."""
from __future__ import annotations

import json
import logging
import random
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

log = logging.getLogger("cognitive.adapters")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class NeuralFrame:
    timestamp: str
    channels: int
    raw_energy: float
    spike_rate: float
    focus_index: float
    load_index: float
    valence: float
    dominant_intent: str

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "channels": self.channels,
            "raw_energy": self.raw_energy,
            "spike_rate": self.spike_rate,
            "focus_index": self.focus_index,
            "load_index": self.load_index,
            "valence": self.valence,
            "dominant_intent": self.dominant_intent,
        }


class NeuralInputAdapter(ABC):
    """Interface for real or simulated BCI input."""

    @abstractmethod
    def read_frame(self, channels: int, mode: str, focus: float, load: float, intent: str) -> NeuralFrame:
        ...


class SimulatorNeuralAdapter(NeuralInputAdapter):
    """Deterministic-ish synthetic neural stream for development and CI."""

    INTENT_POOL = [
        "retrieve_context", "generate_plan", "summarize_stream",
        "expand_idea", "search_memory", "draft_response",
        "compare_options", "focus_narrow", "explore_related",
        "store_insight", "clear_distraction", "deep_work",
    ]

    def read_frame(self, channels: int, mode: str, focus: float, load: float, intent: str) -> NeuralFrame:
        target_load = {
            "idle": 0.2, "focus": 0.55, "explore": 0.45,
            "create": 0.7, "recall": 0.4, "overload": 0.92,
        }.get(mode, 0.3)
        load = max(0.05, min(0.98, load + (target_load - load) * 0.15 + random.uniform(-0.04, 0.04)))
        focus = max(0.1, min(0.95, focus + random.uniform(-0.06, 0.08)))
        if random.random() < 0.3:
            intent = random.choice(self.INTENT_POOL)
        return NeuralFrame(
            timestamp=_now(),
            channels=channels,
            raw_energy=round(random.uniform(0.3, 1.4) * (0.6 + load), 3),
            spike_rate=round(random.uniform(8, 45) * (0.5 + focus), 2),
            focus_index=round(focus, 3),
            load_index=round(load, 3),
            valence=round(random.uniform(-0.2, 0.3), 3),
            dominant_intent=intent,
        )


class FileNeuralAdapter(NeuralInputAdapter):
    """Replay frames from a JSONL file (for recorded traces / offline tests)."""

    def __init__(self, path: str):
        self.path = path
        self._frames: List[dict] = []
        self._idx = 0
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self._frames.append(json.loads(line))
            log.info("Loaded %d neural frames from %s", len(self._frames), path)
        except FileNotFoundError:
            log.warning("Neural file not found: %s — falling back to empty", path)

    def read_frame(self, channels: int, mode: str, focus: float, load: float, intent: str) -> NeuralFrame:
        if not self._frames:
            return SimulatorNeuralAdapter().read_frame(channels, mode, focus, load, intent)
        raw = self._frames[self._idx % len(self._frames)]
        self._idx += 1
        return NeuralFrame(
            timestamp=raw.get("timestamp", _now()),
            channels=int(raw.get("channels", channels)),
            raw_energy=float(raw.get("raw_energy", 0.5)),
            spike_rate=float(raw.get("spike_rate", 20)),
            focus_index=float(raw.get("focus_index", focus)),
            load_index=float(raw.get("load_index", load)),
            valence=float(raw.get("valence", 0.0)),
            dominant_intent=str(raw.get("dominant_intent", intent)),
        )


class ModelClient(ABC):
    """Interface for external / local frontier models."""

    @abstractmethod
    def complete(self, prompt: str, max_tokens: int = 256, metadata: Optional[dict] = None) -> dict:
        ...


class SimulatorModelClient(ModelClient):
    """Local stub that respects policy metadata so A/B eval is measurable."""

    def __init__(self):
        self.calls = 0

    def complete(self, prompt: str, max_tokens: int = 256, metadata: Optional[dict] = None) -> dict:
        self.calls += 1
        time.sleep(0.01)
        meta = metadata or {}
        policy = meta.get("policy") or ""
        temp = float(meta.get("temperature") or 0.7)
        pl = prompt.lower()

        # Policy-conditioned responses (deterministic enough for CI)
        if "high_load_protect" in policy or "under high cognitive load" in pl:
            content = (
                "1. Open the auth middleware file\n"
                "2. Check the JWT verify path\n"
                "3. Add the missing Bearer parse — next step only."
            )
            completion = min(max_tokens, 48)
        elif "fatigue_simplify" in policy or "fatigue markers" in pl:
            content = (
                "OAuth refresh tokens let an app get a new access token without "
                "logging in again. Store them securely. Short version: refresh = spare key."
            )
            completion = min(max_tokens, 60)
        elif "deep_focus_dense" in policy or "deep focus" in pl:
            content = (
                "```python\ndef merge_sorted(a, b):\n"
                "    i = j = 0\n    out = []\n"
                "    while i < len(a) and j < len(b):\n"
                "        if a[i] <= b[j]:\n            out.append(a[i]); i += 1\n"
                "        else:\n            out.append(b[j]); j += 1\n"
                "    out.extend(a[i:]); out.extend(b[j:])\n    return out\n```"
            )
            completion = min(max_tokens, 120)
        elif "create_expand" in policy or "generative mode" in pl:
            content = (
                "1. CortexKeep\n2. SignalPrivate\n3. MindVault Journal\n"
                "Alternatives: Neurona Lite, QuietBand."
            )
            completion = min(max_tokens, 80)
        elif "idle_minimal" in policy or "wait for clearer direction" in pl:
            content = "Ready when you are."
            completion = min(max_tokens, 16)
        elif "sqlite" in pl and "postgres" in pl:
            content = (
                "SQLite: single file, zero ops, great for one-node agent memory. "
                "Postgres: concurrent writers, richer types, network access. "
                "Choose SQLite until you need multi-writer or remote access."
            )
            completion = min(max_tokens, 90)
        else:
            # Baseline-ish generic helpful response (wordier)
            content = (
                "Certainly! I would be happy to help with that. "
                "As a helpful assistant, here is a detailed overview covering multiple angles, "
                "background context, and several considerations you might want to explore further "
                "depending on your specific constraints and environment."
            )
            completion = min(max_tokens, 96)

        return {
            "id": f"sim-{self.calls}",
            "model": "cognitive-simulator",
            "prompt_tokens": len(prompt.split()) + 20,
            "completion_tokens": completion,
            "content": content,
            "finish_reason": "stop",
            "backend": "simulator",
            "temperature": temp,
        }


class HttpModelClient(ModelClient):
    """Minimal HTTP client for OpenAI-compatible or custom endpoints."""

    def __init__(self, url: str, api_key: str = ""):
        if not url:
            raise ValueError("COG_MODEL_URL required for http backend")
        self.url = url.rstrip("/")
        self.api_key = api_key
        self.calls = 0

    def complete(self, prompt: str, max_tokens: int = 256, metadata: Optional[dict] = None) -> dict:
        self.calls += 1
        payload = {
            "model": "default",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            self.url if self.url.endswith("/chat/completions") else f"{self.url}/chat/completions",
            data=data,
            method="POST",
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}),
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read().decode())
            content = ""
            if "choices" in body and body["choices"]:
                content = body["choices"][0].get("message", {}).get("content", "")
            return {
                "id": body.get("id", f"http-{self.calls}"),
                "model": body.get("model", "remote"),
                "prompt_tokens": body.get("usage", {}).get("prompt_tokens", 0),
                "completion_tokens": body.get("usage", {}).get("completion_tokens", 0),
                "content": content or str(body)[:500],
                "finish_reason": "stop",
                "backend": "http",
            }
        except urllib.error.URLError as e:
            log.error("Model HTTP error: %s", e)
            return {
                "id": f"err-{self.calls}",
                "model": "error",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "content": f"model_error: {e}",
                "finish_reason": "error",
                "backend": "http",
            }




class EEGNeuralAdapter(NeuralInputAdapter):
    """
    Real-shaped EEG path: synthesize or accept windows → band powers → CognitiveState → NeuralFrame.
    For hardware: feed EEGSample windows via push_sample() from an LSL/BrainFlow bridge process.
    """

    def __init__(self):
        from src.cognitive.eeg import CognitiveState
        self._prev_state: CognitiveState | None = None
        self._queue: list = []

    def push_sample(self, sample) -> None:
        self._queue.append(sample)

    def read_frame(self, channels: int, mode: str, focus: float, load: float, intent: str) -> NeuralFrame:
        from src.cognitive.eeg import (
            synthesize_eeg, compute_band_powers, detect_cognitive_state, EEGSample,
        )
        if self._queue:
            sample = self._queue.pop(0)
        else:
            # Continuous self-play stream labeled from current mode
            label = mode if mode in ("idle", "focus", "explore", "create", "overload", "fatigue") else "explore"
            sample = synthesize_eeg(label, n_channels=min(8, max(1, channels // 128 or 4)))
        bands = compute_band_powers(sample)
        state = detect_cognitive_state(bands, self._prev_state)
        self._prev_state = state
        return NeuralFrame(
            timestamp=_now(),
            channels=channels,
            raw_energy=round(sum(bands.to_dict().values()) / 1e6 + state.load, 3),
            spike_rate=round(8 + 40 * state.engagement, 2),
            focus_index=state.focus,
            load_index=state.load,
            valence=round(0.2 - 0.4 * state.fatigue, 3),
            dominant_intent=intent if intent and intent != "none" else state.mode,
        )


def build_neural_adapter(backend: str, path: str = "") -> NeuralInputAdapter:
    if backend == "file" and path:
        return FileNeuralAdapter(path)
    if backend == "eeg":
        return EEGNeuralAdapter()
    return SimulatorNeuralAdapter()


def build_model_client(backend: str, url: str = "", api_key: str = "") -> ModelClient:
    if backend == "http":
        return HttpModelClient(url, api_key)
    return SimulatorModelClient()
