"""Adaptive model policy driven by CognitiveState."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

from src.cognitive.eeg import CognitiveState


@dataclass
class ModelPolicy:
    """Parameters applied to the next model call."""
    temperature: float = 0.7
    max_tokens: int = 256
    system_prefix: str = ""
    retrieval_k: int = 3
    style: str = "balanced"          # terse|balanced|elaborate
    refuse_nonessential: bool = False
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def policy_from_state(state: CognitiveState) -> ModelPolicy:
    """
    Map cognitive state → adaptive generation policy.

    Goal: measurable improvement vs fixed baseline under load/fatigue/focus.
    """
    if state.mode == "overload" or state.load > 0.85:
        return ModelPolicy(
            temperature=0.2,
            max_tokens=120,
            system_prefix=(
                "The user is under high cognitive load. Respond in short, numbered steps. "
                "No preamble. Prioritize the single next action."
            ),
            retrieval_k=1,
            style="terse",
            refuse_nonessential=True,
            reason="high_load_protect",
        )
    if state.mode == "fatigue" or state.fatigue > 0.7:
        return ModelPolicy(
            temperature=0.3,
            max_tokens=150,
            system_prefix=(
                "The user shows fatigue markers. Use simple language, short paragraphs, "
                "and confirm understanding. Avoid multi-part digressions."
            ),
            retrieval_k=2,
            style="terse",
            refuse_nonessential=True,
            reason="fatigue_simplify",
        )
    if state.mode == "focus" and state.focus > 0.65:
        return ModelPolicy(
            temperature=0.35,
            max_tokens=400,
            system_prefix=(
                "The user is in deep focus. Be precise, technical, and dense. "
                "Skip motivational filler. Prefer code and concrete constraints."
            ),
            retrieval_k=5,
            style="balanced",
            refuse_nonessential=False,
            reason="deep_focus_dense",
        )
    if state.mode == "create":
        return ModelPolicy(
            temperature=0.85,
            max_tokens=500,
            system_prefix=(
                "The user is in a generative mode. Offer structured options, "
                "analogies, and alternative framings. Still stay grounded."
            ),
            retrieval_k=4,
            style="elaborate",
            refuse_nonessential=False,
            reason="create_expand",
        )
    if state.mode == "idle":
        return ModelPolicy(
            temperature=0.5,
            max_tokens=200,
            system_prefix="Be brief and wait for clearer direction.",
            retrieval_k=2,
            style="terse",
            refuse_nonessential=True,
            reason="idle_minimal",
        )
    # explore / default
    return ModelPolicy(
        temperature=0.7,
        max_tokens=300,
        system_prefix="Be clear and helpful. Match the user's depth.",
        retrieval_k=3,
        style="balanced",
        refuse_nonessential=False,
        reason="explore_default",
    )


def baseline_policy() -> ModelPolicy:
    """Fixed conventional interface policy (control arm)."""
    return ModelPolicy(
        temperature=0.7,
        max_tokens=300,
        system_prefix="You are a helpful assistant.",
        retrieval_k=3,
        style="balanced",
        refuse_nonessential=False,
        reason="baseline",
    )


def apply_policy_to_prompt(user_prompt: str, policy: ModelPolicy, memory_snippets: Optional[list] = None) -> str:
    parts = [policy.system_prefix.strip()]
    if memory_snippets:
        k = max(0, policy.retrieval_k)
        mem = memory_snippets[:k]
        if mem:
            parts.append("Relevant memory:\n- " + "\n- ".join(mem))
    parts.append("User:\n" + user_prompt.strip())
    if policy.style == "terse":
        parts.append("Constraint: answer in under 120 words unless code is required.")
    elif policy.style == "elaborate":
        parts.append("Constraint: provide structure (headings or bullets) and at least one alternative.")
    return "\n\n".join(p for p in parts if p)
