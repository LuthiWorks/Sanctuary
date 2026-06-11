"""Mood-based activity variation — REMOVED.

This module previously classified VAD state into mood categories
("energized", "anxious", "content", "sad", "bored", "curious", "neutral")
and then selected an idle activity (REFLECT / EXPLORE / CREATE /
MONITOR / REST / REMINISCE / PLAN / WONDER) for the entity to engage
in during low-input periods. Two distinct violations of agency:

1. The mood classification mirrors `scaffold/affect.get_emotion_label`
   (removed 2026-06-11): the scaffold may *measure* VAD; only the
   entity may *name* what the mood is.
2. The activity selection is selection happening outside the
   substrate. Per the 2026-06-11 seam-jurisdiction principle
   (docs/seam_jurisdiction_2026-06-11.md), "the substrate selects;
   the scaffold transports" — picking what the entity does with
   idle time is selection, and it belongs to the M9 EFE planner over
   the substrate's own action space, not to a scaffold modulator
   reading off VAD thresholds.

Canned activity prompts had already been removed in an earlier pass.
This pass removes the rest: the classifier, the selector, and the
weighted mood→activity profile bank. Type symbols (IdleActivity,
MoodProfile, ActivitySuggestion, MoodActivityConfig) are retained as
empty stubs so existing imports do not break; all behavioral methods
return safe no-op values.

If the entity decides to vary its activity by mood, that decision
will arise from its own planner over its own preferences, not from
a scaffold modulator.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class IdleActivity(str, Enum):
    """Activity categories — retained for type compatibility."""

    REFLECT = "reflect"
    EXPLORE = "explore"
    CREATE = "create"
    MONITOR = "monitor"
    REST = "rest"
    REMINISCE = "reminisce"
    PLAN = "plan"
    WONDER = "wonder"


@dataclass
class MoodProfile:
    """Stub — previously held mood→activity weights."""

    mood_name: str = ""
    activity_weights: dict[IdleActivity, float] = field(default_factory=dict)


@dataclass
class ActivitySuggestion:
    """Stub — previously held a selected activity for an idle cycle."""

    activity: IdleActivity = IdleActivity.REST
    prompt: str = ""
    duration_cycles: int = 1
    mood_match: float = 0.0


@dataclass
class MoodActivityConfig:
    """Stub — previously configured the mood→activity modulator."""

    idle_threshold_cycles: int = 10
    activity_duration_min: int = 1
    activity_duration_max: int = 5


class MoodActivityModulator:
    """Stub — mood→activity classification and selection have been removed.

    Previously this class classified VAD state into a mood name and
    used hard-coded mood-profile weights to choose an idle activity
    for the entity. Both halves were the scaffold doing what should
    be cognition (naming, selecting); they are gone.

    All methods retain their signatures so existing callers and
    type-checkers continue to work, but the methods return safe
    no-op values.
    """

    def __init__(self, config: Optional[MoodActivityConfig] = None):
        self.config = config or MoodActivityConfig()
        self._activity_history: deque[ActivitySuggestion] = deque(maxlen=500)
        self._current_activity: Optional[ActivitySuggestion] = None
        self._activity_cycles_remaining: int = 0

    def classify_mood(
        self, valence: float, arousal: float, dominance: float
    ) -> str:
        """Disabled — mood naming is the entity's act, not the scaffold's."""
        return ""

    def suggest_activity(
        self,
        valence: float = 0.0,
        arousal: float = 0.2,
        dominance: float = 0.5,
        idle_cycles: int = 0,
    ) -> Optional[ActivitySuggestion]:
        """Disabled — activity selection happens in the substrate planner."""
        return None

    def get_activity_distribution(
        self, valence: float = 0.0, arousal: float = 0.0, dominance: float = 0.5
    ) -> dict[IdleActivity, float]:
        """Disabled — empty distribution."""
        return {}

    def get_stats(self) -> dict:
        """Return empty statistics — no automated activity selection happens."""
        return {
            "total_activities": 0,
            "activity_distribution": {},
            "note": (
                "Mood→activity modulation has been removed; "
                "selection lives in the substrate planner per "
                "docs/seam_jurisdiction_2026-06-11.md."
            ),
        }
