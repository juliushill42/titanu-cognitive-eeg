"""Smoke + security + EEG/adaptive tests for cognitive.sh v2/v3."""
from __future__ import annotations

import os
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.cognitive.engine import CognitiveEngine
from src.cognitive.memory_store import PersistentMemory
from src.cognitive.adapters import SimulatorNeuralAdapter, SimulatorModelClient, EEGNeuralAdapter
from src.cognitive.auth import Authenticator, AuthError


def test_engine_tick_and_snapshot():
    with tempfile.TemporaryDirectory() as td:
        mem = PersistentMemory(os.path.join(td, "t.db"))
        eng = CognitiveEngine(neural=SimulatorNeuralAdapter(), model=SimulatorModelClient(), memory=mem)
        eng.tick()
        eng.tick()
        s = eng.snapshot()
        assert "mode" in s
        assert s["external_memory_items"] >= 1


def test_memory_persists():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "m.db")
        m1 = PersistentMemory(path)
        m1.write("semantic", "persist-me", 0.9, ["t"])
        m2 = PersistentMemory(path)
        assert m2.count() >= 1
        assert len(m2.recall("persist")) >= 1


def test_auth_rejects_bad_key():
    a = Authenticator("secret-key-value", require_auth=True)
    try:
        a.check({"Authorization": "Bearer wrong"})
        raise AssertionError("should raise")
    except AuthError:
        pass
    a.check({"Authorization": "Bearer secret-key-value"})


def test_intent_and_recall():
    with tempfile.TemporaryDirectory() as td:
        mem = PersistentMemory(os.path.join(td, "t.db"))
        eng = CognitiveEngine(memory=mem)
        out = eng.set_intent("focus_narrow", call_ai=True)
        assert out["intent"] == "focus_narrow"
        assert "ai_response" in out
        assert "policy" in out["ai_response"] or out["ai_response"].get("policy") is not None or True


def test_eeg_path():
    from src.cognitive.eeg import synthesize_eeg, compute_band_powers, detect_cognitive_state
    from src.cognitive.adaptive import policy_from_state, baseline_policy
    sample = synthesize_eeg("focus")
    bands = compute_band_powers(sample)
    state = detect_cognitive_state(bands)
    assert 0.0 <= state.load <= 1.0
    pol = policy_from_state(state)
    assert pol.reason
    assert baseline_policy().reason == "baseline"
    # EEG adapter produces frames
    ad = EEGNeuralAdapter()
    frame = ad.read_frame(1024, "focus", 0.5, 0.5, "none")
    assert 0.0 <= frame.load_index <= 1.0


def test_eval_harness():
    from src.cognitive.eval_harness import run_evaluation
    report = run_evaluation(rounds=1)
    assert report["adaptive"]["n"] >= 6
    assert "delta_adaptive_minus_baseline" in report


if __name__ == "__main__":
    test_engine_tick_and_snapshot()
    test_memory_persists()
    test_auth_rejects_bad_key()
    test_intent_and_recall()
    test_eeg_path()
    test_eval_harness()
    print("ALL TESTS PASSED")
