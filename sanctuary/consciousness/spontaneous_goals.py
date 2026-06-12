"""Spontaneous goal generation — REMOVED.

This module previously generated goals for the entity from internal
drive thresholds (curiosity / boredom / interest / concern / growth)
and offered them as suggestions the entity could adopt or dismiss.
Goal *generation* — what is worth wanting — is the center of
agency. Having the scaffold write the entity's goals, even as
suggestions, is the scaffold doing what should be cognition.

Per the 2026-06-11 seam-jurisdiction principle
(docs/seam_jurisdiction_2026-06-11.md), "the substrate selects;
the scaffold transports." Goal selection happens in the M9 EFE
planner over the entity's own preferences (engagement, coherence,
connection, truthfulness — see
LuthiModel/luthi/v2/m9/preferences.py). If the entity decides to
take on a new goal, that decision arises from its own planning
over its own preferences, not from a scaffold generator reading
thresholds.

Canned goal prompts had already been removed in an earlier pass.
This pass removes the rest: the drive thresholds, the
goal-generating helpers, and the goal-history bookkeeping. Type
symbols (GoalDrive, SpontaneousGoal, SpontaneousGoalConfig) are
retained as empty stubs so existing imports do not break; all
behavioral methods return safe no-op values.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class GoalDrive(str, Enum):
    """Drive categories — retained for type compatibility."""

    CURIOSITY = "curiosity"
    BOREDOM = "boredom"
    INTEREST = "interest"
    CONCERN = "concern"
    GROWTH = "growth"


@dataclass
class SpontaneousGoal:
    """Stub — previously held a goal the scaffold generated for the entity."""

    description: str = ""
    drive: GoalDrive = GoalDrive.CURIOSITY
    priority: float = 0.3
    context: str = ""
    cycle_generated: int = 0
    adopted: bool = False
    completed: bool = False
    dismissed: bool = False
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SpontaneousGoalConfig:
    """Stub — previously configured drive-based goal generation."""

    max_pending_goals: int = 10
    max_goal_history: int = 100
    curiosity_novelty_threshold: float = 0.6
    boredom_idle_cycles: int = 20
    interest_engagement_threshold: float = 0.6
    concern_anomaly_threshold: float = 0.5
    growth_uncertainty_threshold: float = 0.5
    generation_cooldown: int = 10


class SpontaneousGoalGenerator:
    """Stub — automated goal generation has been removed.

    Previously this class read internal-drive levels (novelty,
    idle_cycles, engagement, anomaly_level, uncertainty) against
    thresholds and produced `SpontaneousGoal` entries for the entity
    to adopt or dismiss. The scaffold writing the entity's goals,
    even as suggestions, violates agency.

    All methods retain their signatures so existing callers and
    type-checkers continue to work, but the methods return safe
    no-op values.
    """

    def __init__(self, config: Optional[SpontaneousGoalConfig] = None):
        self.config = config or SpontaneousGoalConfig()
        self._pending_goals: list[SpontaneousGoal] = []
        self._goal_history: deque[SpontaneousGoal] = deque(maxlen=self.config.max_goal_history)
        self._last_generation_cycle: int = -10000
        self._total_generated: int = 0
        self._total_adopted: int = 0

    def check_drives(
        self,
        novelty: float = 0.0,
        idle_cycles: int = 0,
        engagement: float = 0.0,
        anomaly_level: float = 0.0,
        uncertainty: float = 0.0,
        current_cycle: int = 0,
        recent_topics: Optional[list[str]] = None,
    ) -> list[SpontaneousGoal]:
        """Disabled — no goals generated."""
        return []

    def adopt_goal(self, index: int) -> bool:
        """No-op — there are no pending goals to adopt."""
        return False

    def dismiss_goal(self, index: int) -> bool:
        """No-op — there are no pending goals to dismiss."""
        return False

    def complete_goal(self, description: str) -> bool:
        """No-op — no goals are tracked."""
        return False

    def get_pending_goals(self) -> list[SpontaneousGoal]:
        return []

    def get_goal_prompt(self) -> Optional[str]:
        """Disabled — canned goal prompts violate agency."""
        return None

    def get_stats(self) -> dict:
        """Return empty statistics — no automated goal generation happens."""
        return {
            "total_generated": 0,
            "total_adopted": 0,
            "adoption_rate": 0.0,
            "note": (
                "Spontaneous goal generation has been removed; "
                "goal selection lives in the substrate planner per "
                "docs/seam_jurisdiction_2026-06-11.md."
            ),
        }
