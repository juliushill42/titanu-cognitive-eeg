"""EEG band-power features and cognitive-state detection.

Accepts real multi-channel samples (µV) or precomputed band powers.
Produces a CognitiveState used for adaptive model behavior.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Sequence, Tuple


# Canonical bands (Hz)
BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


@dataclass
class EEGSample:
    """One window of multi-channel EEG."""
    fs: float                          # sample rate Hz
    channels: List[List[float]]        # shape [n_channels][n_samples]
    timestamp: float = 0.0
    label: Optional[str] = None        # optional ground-truth for eval


@dataclass
class BandPowers:
    delta: float = 0.0
    theta: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    gamma: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    def total(self) -> float:
        return max(1e-12, self.delta + self.theta + self.alpha + self.beta + self.gamma)

    def relative(self) -> "BandPowers":
        t = self.total()
        return BandPowers(
            delta=self.delta / t, theta=self.theta / t,
            alpha=self.alpha / t, beta=self.beta / t, gamma=self.gamma / t,
        )


@dataclass
class CognitiveState:
    """Detected cognitive state from EEG features."""
    load: float              # 0-1 estimated cognitive load
    focus: float             # 0-1 attentional focus
    engagement: float        # 0-1
    fatigue: float           # 0-1
    mode: str                # idle|focus|explore|create|overload|fatigue
    confidence: float        # 0-1 detector confidence
    bands: Dict[str, float] = field(default_factory=dict)
    source: str = "eeg"

    def to_dict(self) -> dict:
        return asdict(self)


def _goertzel_power(samples: Sequence[float], fs: float, f_lo: float, f_hi: float) -> float:
    """Approximate band power via summed Goertzel bins (stdlib only)."""
    n = len(samples)
    if n < 8 or fs <= 0:
        return 0.0
    # Sample a few frequencies across the band
    step = max(0.5, (f_hi - f_lo) / 6.0)
    freqs = []
    f = f_lo
    while f <= f_hi + 1e-9:
        freqs.append(f)
        f += step
    power = 0.0
    for freq in freqs:
        if freq <= 0 or freq >= fs / 2:
            continue
        w = 2.0 * math.pi * freq / fs
        coeff = 2.0 * math.cos(w)
        s0 = s1 = s2 = 0.0
        for x in samples:
            s0 = x + coeff * s1 - s2
            s2, s1 = s1, s0
        power += s1 * s1 + s2 * s2 - coeff * s1 * s2
    return max(0.0, power / max(1, len(freqs)))


def compute_band_powers(sample: EEGSample) -> BandPowers:
    """Average band powers across channels for one window."""
    if not sample.channels or not sample.channels[0]:
        return BandPowers()
    acc = {k: 0.0 for k in BANDS}
    n_ch = len(sample.channels)
    for ch in sample.channels:
        for name, (lo, hi) in BANDS.items():
            acc[name] += _goertzel_power(ch, sample.fs, lo, hi)
    n_ch = max(1, n_ch)
    return BandPowers(**{k: v / n_ch for k, v in acc.items()})


def detect_cognitive_state(bands: BandPowers, prev: Optional[CognitiveState] = None) -> CognitiveState:
    """
    Map relative band powers → cognitive state.

    Heuristics grounded in common EEG correlates:
    - High beta/gamma + suppressed alpha → high load / focus
    - High alpha → idle / relaxed
    - Rising theta + falling beta → fatigue
    - Balanced alpha/beta → explore
    """
    r = bands.relative()
    # Load: beta+gamma vs alpha
    load = _clamp((r.beta + r.gamma) / max(1e-6, r.alpha + r.beta + r.gamma) )
    load = _clamp(0.2 + 0.9 * (r.beta + 0.7 * r.gamma - 0.5 * r.alpha))
    focus = _clamp(0.15 + 0.95 * (r.beta - 0.4 * r.theta + 0.2 * r.gamma))
    engagement = _clamp(0.5 * focus + 0.4 * load + 0.1 * r.gamma)
    fatigue = _clamp(0.1 + 0.9 * (r.theta + 0.5 * r.delta - 0.4 * r.beta))

    if fatigue > 0.7 and load < 0.5:
        mode = "fatigue"
    elif load > 0.85:
        mode = "overload"
    elif focus > 0.65 and load > 0.4:
        mode = "focus"
    elif load > 0.55 and focus > 0.45:
        mode = "create"
    elif r.alpha > 0.35 and load < 0.4:
        mode = "idle"
    else:
        mode = "explore"

    # Confidence from band total energy + stability vs prev
    energy = math.log1p(bands.total())
    conf = _clamp(0.35 + 0.15 * min(energy, 4.0))
    if prev and prev.mode == mode:
        conf = _clamp(conf + 0.15)

    return CognitiveState(
        load=round(load, 3),
        focus=round(focus, 3),
        engagement=round(engagement, 3),
        fatigue=round(fatigue, 3),
        mode=mode,
        confidence=round(conf, 3),
        bands=r.to_dict(),
        source="eeg",
    )


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def synthesize_eeg(
    state_label: str,
    fs: float = 256.0,
    n_channels: int = 4,
    n_samples: int = 256,
    noise: float = 0.15,
) -> EEGSample:
    """
    Generate EEG-like multi-channel signals with band profiles matching a label.
    Used for offline eval and CI when no headset is attached.
    """
    # Relative band weights per label
    profiles = {
        "idle":     dict(delta=0.15, theta=0.15, alpha=0.45, beta=0.15, gamma=0.10),
        "focus":    dict(delta=0.08, theta=0.10, alpha=0.15, beta=0.45, gamma=0.22),
        "explore":  dict(delta=0.10, theta=0.18, alpha=0.28, beta=0.28, gamma=0.16),
        "create":   dict(delta=0.08, theta=0.15, alpha=0.20, beta=0.35, gamma=0.22),
        "overload": dict(delta=0.10, theta=0.20, alpha=0.10, beta=0.35, gamma=0.25),
        "fatigue":  dict(delta=0.25, theta=0.35, alpha=0.20, beta=0.12, gamma=0.08),
    }
    prof = profiles.get(state_label, profiles["explore"])
    centers = {"delta": 2.5, "theta": 6.0, "alpha": 10.0, "beta": 20.0, "gamma": 37.0}
    channels: List[List[float]] = []
    for ch in range(n_channels):
        phase = random.random() * 2 * math.pi
        sig: List[float] = []
        for i in range(n_samples):
            t = i / fs
            v = 0.0
            for name, amp in prof.items():
                v += amp * math.sin(2 * math.pi * centers[name] * t + phase + 0.1 * ch)
            v += noise * random.uniform(-1, 1)
            sig.append(v * 20.0)  # ~µV scale
        channels.append(sig)
    return EEGSample(fs=fs, channels=channels, timestamp=0.0, label=state_label)
