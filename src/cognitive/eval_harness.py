"""
A/B evaluation: adaptive policy vs conventional baseline.

Metric goals (higher is better unless noted):
- task_success: did the response satisfy a simple rubric for the prompt class
- token_efficiency: useful content / tokens (proxy: inverse of completion length when terse required)
- latency_ms: lower better
- load_alignment: policy matched detected state
- composite: weighted score

Run: PYTHONPATH=. python -m src.cognitive.eval_harness
"""
from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple

from src.cognitive.eeg import synthesize_eeg, compute_band_powers, detect_cognitive_state
from src.cognitive.adaptive import policy_from_state, baseline_policy, apply_policy_to_prompt
from src.cognitive.adapters import SimulatorModelClient


@dataclass
class TrialResult:
    label: str
    arm: str                 # adaptive | baseline
    detected_mode: str
    policy_reason: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    success: float           # 0-1
    composite: float

    def to_dict(self) -> dict:
        return asdict(self)


# Prompt classes with simple success rubrics
TASKS: List[Tuple[str, str, str]] = [
    # (state_label, user_prompt, success_keyword_group)
    ("overload", "I have 30 seconds — what is the single next step to fix a failing auth middleware?", "step"),
    ("fatigue", "Explain OAuth refresh tokens simply.", "token"),
    ("focus", "Write a pure function that merges two sorted integer lists.", "def"),
    ("create", "Propose three product names for a privacy-first EEG journal app.", "1"),
    ("idle", "ok", "ok"),
    ("explore", "Compare SQLite vs Postgres for a single-node agent memory store.", "sqlite"),
]


def _score_success(prompt: str, content: str, state_label: str) -> float:
    c = (content or "").lower()
    if not c or "model_error" in c:
        return 0.0
    words = len(c.split())
    score = 0.0
    # Penalize generic filler
    if "certainly" in c and "happy to help" in c:
        score -= 0.4

    if state_label == "overload":
        if words <= 60:
            score += 0.4
        if any(x in c for x in ("1.", "2.", "3.")):
            score += 0.4
        if "next" in c or "check" in c:
            score += 0.2
    elif state_label == "fatigue":
        if words <= 80:
            score += 0.3
        if "refresh" in c and "token" in c:
            score += 0.5
        if "simply" in c or "short" in c or "spare key" in c:
            score += 0.2
    elif state_label == "focus":
        if "def " in c or "```" in c:
            score += 0.6
        if "merge" in c:
            score += 0.3
        if words > 200:
            score -= 0.2
    elif state_label == "create":
        bullets = sum(c.count(x) for x in ("1.", "2.", "3."))
        score += min(0.7, bullets * 0.25)
        if "alternative" in c:
            score += 0.2
    elif state_label == "idle":
        if words <= 20:
            score += 0.9
        elif words <= 40:
            score += 0.4
    else:  # explore
        if "sqlite" in c and "postgres" in c:
            score += 0.6
        if "single" in c or "multi" in c or "concurrent" in c:
            score += 0.3
    return max(0.0, min(1.0, score))


def run_trial(state_label: str, prompt: str, arm: str, model: SimulatorModelClient) -> TrialResult:
    sample = synthesize_eeg(state_label)
    bands = compute_band_powers(sample)
    state = detect_cognitive_state(bands)
    policy = policy_from_state(state) if arm == "adaptive" else baseline_policy()
    full_prompt = apply_policy_to_prompt(prompt, policy, memory_snippets=["prior context stub"])

    t0 = time.perf_counter()
    resp = model.complete(
        full_prompt,
        max_tokens=policy.max_tokens,
        metadata={
            "arm": arm,
            "label": state_label,
            "policy": policy.reason,
            "temperature": policy.temperature,
        },
    )
    latency = (time.perf_counter() - t0) * 1000.0

    content = resp.get("content") or ""
    success = _score_success(prompt, content, state_label)
    # Token efficiency: reward fewer tokens under overload/fatigue when success holds
    ct = int(resp.get("completion_tokens") or 50)
    efficiency = 1.0
    if state_label in ("overload", "fatigue", "idle") and success >= 0.5:
        efficiency = max(0.3, min(1.0, 120.0 / max(ct, 1)))
    # Alignment: adaptive should pick non-baseline reason under extreme states
    align = 1.0
    if arm == "adaptive" and state_label in ("overload", "fatigue", "focus"):
        align = 1.0 if policy.reason != "baseline" else 0.4
    composite = 0.5 * success + 0.2 * efficiency + 0.15 * align + 0.15 * max(0.0, 1.0 - latency / 500.0)

    return TrialResult(
        label=state_label,
        arm=arm,
        detected_mode=state.mode,
        policy_reason=policy.reason,
        latency_ms=round(latency, 2),
        prompt_tokens=int(resp.get("prompt_tokens") or 0),
        completion_tokens=ct,
        success=round(success, 3),
        composite=round(composite, 3),
    )


def run_evaluation(rounds: int = 2) -> Dict[str, Any]:
    model = SimulatorModelClient()
    results: List[TrialResult] = []
    for _ in range(rounds):
        for label, prompt, _ in TASKS:
            results.append(run_trial(label, prompt, "adaptive", model))
            results.append(run_trial(label, prompt, "baseline", model))

    def _arm(arm: str) -> List[TrialResult]:
        return [r for r in results if r.arm == arm]

    def _summary(arm: str) -> dict:
        rows = _arm(arm)
        return {
            "n": len(rows),
            "mean_success": round(statistics.mean(r.success for r in rows), 3),
            "mean_composite": round(statistics.mean(r.composite for r in rows), 3),
            "mean_latency_ms": round(statistics.mean(r.latency_ms for r in rows), 2),
            "mean_completion_tokens": round(statistics.mean(r.completion_tokens for r in rows), 1),
        }

    adaptive = _summary("adaptive")
    baseline = _summary("baseline")
    delta = {
        "success": round(adaptive["mean_success"] - baseline["mean_success"], 3),
        "composite": round(adaptive["mean_composite"] - baseline["mean_composite"], 3),
        "latency_ms": round(adaptive["mean_latency_ms"] - baseline["mean_latency_ms"], 2),
        "completion_tokens": round(adaptive["mean_completion_tokens"] - baseline["mean_completion_tokens"], 1),
    }
    return {
        "adaptive": adaptive,
        "baseline": baseline,
        "delta_adaptive_minus_baseline": delta,
        "improved": delta["composite"] > 0 and delta["success"] >= 0,
        "trials": [r.to_dict() for r in results],
    }


def main() -> None:
    report = run_evaluation(rounds=2)
    print(json.dumps({k: v for k, v in report.items() if k != "trials"}, indent=2))
    print("---")
    print("improved_over_baseline:", report["improved"])
    # per-label quick view
    from collections import defaultdict
    buckets: dict = defaultdict(lambda: {"adaptive": [], "baseline": []})
    for t in report["trials"]:
        buckets[t["label"]][t["arm"]].append(t["composite"])
    print("per_label_mean_composite:")
    for label, arms in sorted(buckets.items()):
        a = statistics.mean(arms["adaptive"])
        b = statistics.mean(arms["baseline"])
        print(f"  {label:10} adaptive={a:.3f} baseline={b:.3f} delta={a-b:+.3f}")


if __name__ == "__main__":
    main()
