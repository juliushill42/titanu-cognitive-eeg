"""Thread-safe Cognitive Bandwidth Multiplier engine."""
from __future__ import annotations

import logging
import random
import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from src.cognitive.adapters import (
    NeuralFrame, NeuralInputAdapter, ModelClient,
    SimulatorNeuralAdapter, SimulatorModelClient,
)
from src.cognitive.memory_store import PersistentMemory, MemoryRecord
from src.cognitive.adaptive import policy_from_state, apply_policy_to_prompt, ModelPolicy
from src.cognitive.eeg import CognitiveState

log = logging.getLogger("cognitive.engine")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class CognitiveMode(str, Enum):
    IDLE = "idle"
    FOCUS = "focus"
    EXPLORE = "explore"
    CREATE = "create"
    RECALL = "recall"
    OVERLOAD = "overload"


class StreamPriority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    BACKGROUND = "background"


@dataclass
class CompressedToken:
    id: str
    content: str
    priority: str
    salience: float
    source: str
    created_at: str
    tokens_est: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AttentionFocus:
    primary: str
    secondary: List[str]
    suppressed: List[str]
    bandwidth_used: float
    mode: str

    def to_dict(self) -> dict:
        return asdict(self)


class CognitiveEngine:
    """
    Brain-AI Symbiosis core.
    Thread-safe: all public methods acquire self._lock.
    """

    def __init__(
        self,
        channels: int = 1024,
        neural: Optional[NeuralInputAdapter] = None,
        model: Optional[ModelClient] = None,
        memory: Optional[PersistentMemory] = None,
    ):
        self._lock = threading.RLock()
        self.channels = channels
        self.neural = neural or SimulatorNeuralAdapter()
        self.model = model or SimulatorModelClient()
        self.memory = memory

        self.mode = CognitiveMode.IDLE.value
        self.cognitive_load = 0.25
        self.focus_index = 0.55
        self.valence = 0.1
        self.last_intent = "none"
        self.compression_ratio = 12.0
        self.tokens_in_flight = 0
        self.ai_calls_total = 0
        self.working_memory: List[CompressedToken] = []
        self.attention = AttentionFocus(
            primary="ambient", secondary=[], suppressed=[],
            bandwidth_used=0.2, mode=CognitiveMode.IDLE.value,
        )
        self.events: List[dict] = []
        self.frame_history: List[NeuralFrame] = []
        self.last_policy: Optional[dict] = None
        self.last_cognitive_state: Optional[dict] = None

        if self.memory:
            self.memory.seed_if_empty([
                ("working", "Current project goals and constraints", 0.9, ["project", "active"]),
                ("episodic", "Yesterday's design review key decisions", 0.7, ["meeting", "decisions"]),
                ("semantic", "Optimus control API surface summary", 0.85, ["robotics", "api"]),
                ("semantic", "Starship propellant transfer constraints", 0.8, ["space", "engineering"]),
                ("procedural", "Preferred code review checklist", 0.75, ["process"]),
                ("working", "Open questions for next Neuralink session", 0.65, ["bci", "open"]),
            ])

    def _event(self, kind: str, message: str, payload: Optional[dict] = None) -> None:
        self.events.append({
            "timestamp": _now(),
            "kind": kind,
            "message": message,
            "payload": payload or {},
        })
        if len(self.events) > 250:
            self.events = self.events[-250:]

    def tick(self) -> None:
        with self._lock:
            frame = self.neural.read_frame(
                self.channels, self.mode, self.focus_index,
                self.cognitive_load, self.last_intent,
            )
            self.cognitive_load = frame.load_index
            self.focus_index = frame.focus_index
            self.valence = frame.valence
            if frame.dominant_intent:
                self.last_intent = frame.dominant_intent

            if self.cognitive_load > 0.85:
                self.mode = CognitiveMode.OVERLOAD.value
            elif self.focus_index > 0.7 and self.cognitive_load < 0.6:
                self.mode = CognitiveMode.FOCUS.value

            self.frame_history.append(frame)
            if len(self.frame_history) > 100:
                self.frame_history = self.frame_history[-100:]

            tokens = self._compress(frame)
            self._update_attention(tokens, frame)
            self._augment_memory(tokens)

            if frame.focus_index > 0.65 and random.random() < 0.25:
                self._call_model_unlocked(
                    f"Intent: {frame.dominant_intent} | Mode: {self.mode} | Load: {frame.load_index:.2f}"
                )

            if len(self.working_memory) > 7 and random.random() < 0.3:
                self.working_memory.pop(0)

    def _compress(self, frame: NeuralFrame) -> List[CompressedToken]:
        base = int(40 + frame.spike_rate * 1.8)
        ratio = 8.0 + frame.load_index * 25.0
        self.compression_ratio = round(ratio, 1)
        out_count = max(3, int(base / ratio))
        tokens: List[CompressedToken] = []
        for i in range(out_count):
            sal = round(random.uniform(0.3, 1.0) * (0.5 + frame.focus_index * 0.5), 3)
            if sal > 0.75:
                pri = StreamPriority.CRITICAL.value
            elif sal > 0.55:
                pri = StreamPriority.HIGH.value
            elif sal > 0.35:
                pri = StreamPriority.NORMAL.value
            else:
                pri = StreamPriority.BACKGROUND.value
            tokens.append(CompressedToken(
                id=str(uuid.uuid4())[:8],
                content=f"[{frame.dominant_intent}] salient-{i+1} (focus={frame.focus_index:.2f})",
                priority=pri, salience=sal, source="bci_stream",
                created_at=_now(), tokens_est=random.randint(12, 48),
            ))
        return tokens

    def _update_attention(self, tokens: List[CompressedToken], frame: NeuralFrame) -> None:
        critical = [t for t in tokens if t.priority == StreamPriority.CRITICAL.value]
        high = [t for t in tokens if t.priority == StreamPriority.HIGH.value]
        normal = [t for t in tokens if t.priority == StreamPriority.NORMAL.value]

        if frame.load_index > 0.8:
            primary = critical[0].content if critical else "load_protection"
            secondary: List[str] = []
            suppressed = [t.content for t in high + normal]
            bandwidth = 0.35
            mode = CognitiveMode.OVERLOAD.value
        elif frame.focus_index > 0.7:
            primary = critical[0].content if critical else (high[0].content if high else "deep_focus")
            secondary = [t.content for t in high[:2]]
            suppressed = [t.content for t in normal]
            bandwidth = 0.55 + frame.focus_index * 0.25
            mode = CognitiveMode.FOCUS.value
        else:
            primary = (critical + high + normal)[0].content if tokens else "ambient"
            secondary = [t.content for t in (critical + high)[:3]]
            suppressed = []
            bandwidth = 0.3 + frame.focus_index * 0.3
            mode = frame.dominant_intent if frame.dominant_intent in (
                CognitiveMode.CREATE.value, CognitiveMode.EXPLORE.value, CognitiveMode.RECALL.value
            ) else CognitiveMode.EXPLORE.value

        self.attention = AttentionFocus(
            primary=primary, secondary=secondary[:4], suppressed=suppressed[:6],
            bandwidth_used=round(min(0.95, bandwidth), 3), mode=mode,
        )
        self.mode = mode

    def _augment_memory(self, tokens: List[CompressedToken]) -> None:
        for t in tokens:
            if t.salience < 0.6:
                continue
            self.working_memory.append(t)
            if len(self.working_memory) > 9:
                self.working_memory.pop(0)
            if self.memory and t.salience > 0.8 and random.random() < 0.35:
                rec = self.memory.write(
                    mem_type="episodic" if t.salience >= 0.9 else "working",
                    content=t.content, strength=t.salience,
                    tags=["auto", t.priority],
                )
                self._event("memory_write", f"Promoted: {t.content[:48]}", {"id": rec.id})

    def _call_model_unlocked(self, prompt: str, max_tokens: int = 256) -> dict:
        self.ai_calls_total += 1
        # Build CognitiveState-like snapshot from current indices for policy
        state = CognitiveState(
            load=self.cognitive_load,
            focus=self.focus_index,
            engagement=min(1.0, 0.5 * self.focus_index + 0.5 * self.cognitive_load),
            fatigue=max(0.0, self.cognitive_load - self.focus_index),
            mode=self.mode if self.mode in ("idle", "focus", "explore", "create", "overload", "fatigue") else "explore",
            confidence=0.7,
            bands={},
            source="engine",
        )
        policy = policy_from_state(state)
        self.last_policy = policy.to_dict()
        self.last_cognitive_state = state.to_dict()
        mem_bits = [t.content for t in self.working_memory[-5:]]
        full = apply_policy_to_prompt(prompt, policy, memory_snippets=mem_bits)
        use_tokens = policy.max_tokens if max_tokens == 256 else max_tokens
        self.tokens_in_flight = use_tokens
        resp = self.model.complete(
            full, max_tokens=use_tokens,
            metadata={
                "intent": self.last_intent,
                "mode": self.mode,
                "load": f"{self.cognitive_load:.2f}",
                "policy": policy.reason,
                "temperature": policy.temperature,
            },
        )
        resp["policy"] = policy.to_dict()
        self.tokens_in_flight = 0
        self._event("ai_call", f"Model call #{self.ai_calls_total} policy={policy.reason}", {"backend": resp.get("backend")})
        return resp

    def set_intent(self, intent: str, call_ai: bool = False) -> dict:
        with self._lock:
            intent = (intent or "").strip()[:200]
            if not intent:
                return {"error": "intent required"}
            self.last_intent = intent
            if "focus" in intent:
                self.mode = CognitiveMode.FOCUS.value
                self.focus_index = min(0.95, self.focus_index + 0.15)
            elif "create" in intent or "expand" in intent:
                self.mode = CognitiveMode.CREATE.value
            elif "recall" in intent or "memory" in intent:
                self.mode = CognitiveMode.RECALL.value
            self._event("intent_set", f"Intent set to '{intent}'")
            out: Dict[str, Any] = {"intent": intent, "mode": self.mode}
            if call_ai:
                out["ai_response"] = self._call_model_unlocked(f"User intent: {intent}")
            return out

    def recall(self, query: str, limit: int = 5) -> dict:
        with self._lock:
            query = (query or "").strip()[:500]
            if not self.memory:
                return {"query": query, "results": []}
            results = self.memory.recall(query, limit=min(20, max(1, limit)))
            self._event("memory_recall", f"Recalled {len(results)} for '{query[:40]}'")
            return {"query": query, "results": [r.to_dict() for r in results]}

    def snapshot(self) -> dict:
        with self._lock:
            ext_count = self.memory.count() if self.memory else 0
            ext_list = [m.to_dict() for m in self.memory.list(12)] if self.memory else []
            return {
                "timestamp": _now(),
                "mode": self.mode,
                "cognitive_load": round(self.cognitive_load, 3),
                "focus_index": round(self.focus_index, 3),
                "valence": round(self.valence, 3),
                "compression_ratio": self.compression_ratio,
                "working_memory_items": len(self.working_memory),
                "external_memory_items": ext_count,
                "tokens_in_flight": self.tokens_in_flight,
                "ai_calls_total": self.ai_calls_total,
                "last_intent": self.last_intent,
                "channels": self.channels,
                "attention": self.attention.to_dict(),
                "working_memory": [t.to_dict() for t in self.working_memory[-9:]],
                "external_memory": ext_list,
                "recent_events": list(self.events[-30:]),
                "latest_frame": self.frame_history[-1].to_dict() if self.frame_history else None,
                "last_policy": self.last_policy,
                "last_cognitive_state": self.last_cognitive_state,
            }
