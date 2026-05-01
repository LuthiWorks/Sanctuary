"""Scaffold affect computation — the computed track of dual-track emotion.

Maintains a VAD (Valence-Arousal-Dominance) state that decays toward baseline
and responds to entity-reported emotional shifts. The CfC affect cell is the
authoritative source of computed VAD; this module just smooths and persists.

The entity's felt_quality is the experiential track. Divergence between computed
VAD and felt quality is informative, not a bug.

The keyword-matching heuristic (_POSITIVE_KW / _NEGATIVE_KW / _AROUSING_KW)
that shifted VAD by scanning percept text was removed in the 2026-04-30
cognition-leakage cleanup. Sanctuary doesn't infer the entity's emotion
from words it sees; the entity reports its own shifts via
CognitiveOutput.emotional_state.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from sanctuary.core.authority import AuthorityLevel, AuthorityManager
from sanctuary.core.schema import (
    ComputedVAD,
    EmotionalOutput,
)

logger = logging.getLogger(__name__)


@dataclass
class AffectConfig:
    """Configuration for the scaffold affect module."""

    baseline_valence: float = 0.1
    baseline_arousal: float = 0.2
    baseline_dominance: float = 0.5
    decay_rate: float = 0.05  # Per cycle, toward baseline
    llm_blend_weight: float = 0.3  # How much the model shifts blend when MODEL_GUIDES


class ScaffoldAffect:
    """Computed-VAD track storage for dual-track affect.

    Maintains computed VAD, decays toward baseline, and merges entity-
    reported emotional shifts based on authority level. The keyword-
    based percept scanning was removed in the 2026-04-30 cognition-
    leakage cleanup.
    """

    def __init__(self, config: Optional[AffectConfig] = None):
        self.config = config or AffectConfig()
        self.valence = self.config.baseline_valence
        self.arousal = self.config.baseline_arousal
        self.dominance = self.config.baseline_dominance

    # -- Public API --

    def get_computed_vad(self) -> ComputedVAD:
        """Return the current computed VAD as a schema-compatible object."""
        return ComputedVAD(
            valence=round(self.valence, 3),
            arousal=round(self.arousal, 3),
            dominance=round(self.dominance, 3),
        )

    def merge_llm_emotion(
        self,
        emotion: EmotionalOutput,
        authority: AuthorityManager,
    ) -> None:
        """Blend the entity's emotional self-report into computed VAD.

        The blend weight depends on the authority level for ``emotional_state``:
        - SCAFFOLD_ONLY (0): ignore the model shifts entirely
        - MODEL_ADVISES (1): small blend (~10%)
        - MODEL_GUIDES (2): moderate blend (configured llm_blend_weight)
        - MODEL_CONTROLS (3): the model shifts applied fully
        """
        level = authority.level("emotional_state")

        if level == AuthorityLevel.SCAFFOLD_ONLY:
            return

        # Compute effective blend factor
        if level == AuthorityLevel.MODEL_ADVISES:
            w = self.config.llm_blend_weight * 0.3
        elif level == AuthorityLevel.MODEL_GUIDES:
            w = self.config.llm_blend_weight
        else:  # MODEL_CONTROLS
            w = 1.0

        self.valence = _clamp(self.valence + emotion.valence_shift * w, -1.0, 1.0)
        self.arousal = _clamp(self.arousal + emotion.arousal_shift * w, 0.0, 1.0)

    def decay_toward_baseline(self) -> None:
        """Gradually return toward baseline (emotional regulation)."""
        r = self.config.decay_rate
        self.valence = self.valence * (1 - r) + self.config.baseline_valence * r
        self.arousal = self.arousal * (1 - r) + self.config.baseline_arousal * r
        self.dominance = self.dominance * (1 - r) + self.config.baseline_dominance * r

    def get_emotion_label(self) -> str:
        """Simple VAD → emotion label mapping."""
        v, a = self.valence, self.arousal
        if v > 0.3 and a > 0.5:
            return "joy"
        if v > 0.3 and a <= 0.5:
            return "contentment"
        if v < -0.3 and a > 0.5:
            return "anger"
        if v < -0.3 and a <= 0.5:
            return "sadness"
        if abs(v) <= 0.3 and a > 0.7:
            return "surprise"
        return "calm"


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
