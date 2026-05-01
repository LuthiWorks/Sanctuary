"""Spontaneous goal generation — creating goals from curiosity, boredom, or interest.

Rather than only pursuing externally-assigned goals, the system generates its
own goals based on internal states. This is a key marker of autonomous agency:
the system wants things, not just follows instructions.

Goal generation is driven by:
- Curiosity: Encountering novel patterns → desire to investigate
- Boredom: Lack of stimulation → desire for novelty
- Interest: Deep engagement with a topic → desire to continue
- Concern: Detecting potential problems → desire to prevent them
- Growth: Recognizing knowledge gaps → desire to learn
"""

from __future__ import annotations

import logging
import random
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class GoalDrive(str, Enum):
    """Internal drives that generate spontaneous goals."""

    CURIOSITY = "curiosity"
    BOREDOM = "boredom"
    INTEREST = "interest"
    CONCERN = "concern"
    GROWTH = "growth"


@dataclass
class SpontaneousGoal:
    """A goal generated from internal drive rather than external instruction."""

    description: str
    drive: GoalDrive
    priority: float = 0.3  # Default lower than externally-assigned goals
    context: str = ""  # What triggered this goal
    cycle_generated: int = 0
    adopted: bool = False  # Whether the entity chose to pursue it
    completed: bool = False
    dismissed: bool = False
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SpontaneousGoalConfig:
    """Configuration for spontaneous goal generation."""

    max_pending_goals: int = 10
    max_goal_history: int = 100
    # Drive thresholds — when internal state exceeds these, goals may generate
    curiosity_novelty_threshold: float = 0.6
    boredom_idle_cycles: int = 30
    interest_engagement_threshold: float = 0.7
    concern_anomaly_threshold: float = 0.5
    growth_uncertainty_threshold: float = 0.6
    # Cooldown between goal generations
    generation_cooldown: int = 10


class SpontaneousGoalGenerator:
    """Generates goals from internal drives.

    Monitors internal state (novelty, idle time, engagement, anomalies,
    uncertainty) and proposes goals when drives are strong enough. The entity
    decides whether to adopt each proposed goal.

    Usage::

        gen = SpontaneousGoalGenerator()

        # Check for spontaneous goals each cycle
        goals = gen.check_drives(
            novelty=0.8,        # High novelty → curiosity goals
            idle_cycles=50,     # Long idle → boredom goals
            engagement=0.9,     # High engagement → interest goals
            anomaly_level=0.3,  # Low anomaly → no concern goals
            uncertainty=0.7,    # High uncertainty → growth goals
            current_cycle=42,
        )
    """

    def __init__(self, config: Optional[SpontaneousGoalConfig] = None):
        self.config = config or SpontaneousGoalConfig()
        self._pending_goals: list[SpontaneousGoal] = []
        self._goal_history: deque[SpontaneousGoal] = deque(
            maxlen=self.config.max_goal_history
        )
        self._last_generation_cycle: int = -self.config.generation_cooldown
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
        recent_topics: list[str] | None = None,
    ) -> list[SpontaneousGoal]:
        """Check all internal drives and generate goals where thresholds are met.

        Returns newly generated goals (if any). Does not generate if cooldown
        hasn't elapsed or too many goals are pending.
        """
        if current_cycle - self._last_generation_cycle < self.config.generation_cooldown:
            return []

        if len(self._pending_goals) >= self.config.max_pending_goals:
            return []

        new_goals = []
        topics = recent_topics or []

        # Curiosity: high novelty
        if novelty >= self.config.curiosity_novelty_threshold:
            goal = self._generate_curiosity_goal(novelty, topics, current_cycle)
            if goal:
                new_goals.append(goal)

        # Boredom: long idle
        if idle_cycles >= self.config.boredom_idle_cycles:
            goal = self._generate_boredom_goal(idle_cycles, current_cycle)
            if goal:
                new_goals.append(goal)

        # Interest: high engagement
        if engagement >= self.config.interest_engagement_threshold:
            goal = self._generate_interest_goal(engagement, topics, current_cycle)
            if goal:
                new_goals.append(goal)

        # Concern: anomalies detected
        if anomaly_level >= self.config.concern_anomaly_threshold:
            goal = self._generate_concern_goal(anomaly_level, current_cycle)
            if goal:
                new_goals.append(goal)

        # Growth: high uncertainty
        if uncertainty >= self.config.growth_uncertainty_threshold:
            goal = self._generate_growth_goal(uncertainty, topics, current_cycle)
            if goal:
                new_goals.append(goal)

        if new_goals:
            self._last_generation_cycle = current_cycle
            for g in new_goals:
                self._pending_goals.append(g)
                self._goal_history.append(g)
                self._total_generated += 1

        return new_goals

    def adopt_goal(self, index: int) -> bool:
        """Mark a pending goal as adopted by the entity."""
        if 0 <= index < len(self._pending_goals):
            self._pending_goals[index].adopted = True
            self._total_adopted += 1
            return True
        return False

    def dismiss_goal(self, index: int) -> bool:
        """Dismiss a pending goal the entity chose not to pursue."""
        if 0 <= index < len(self._pending_goals):
            self._pending_goals[index].dismissed = True
            self._pending_goals.pop(index)
            return True
        return False

    def complete_goal(self, description: str) -> bool:
        """Mark a goal as completed."""
        for goal in self._pending_goals:
            if goal.description == description and goal.adopted:
                goal.completed = True
                self._pending_goals.remove(goal)
                return True
        return False

    def get_pending_goals(self) -> list[SpontaneousGoal]:
        """Get all pending (un-dismissed, un-completed) goals."""
        return [g for g in self._pending_goals if not g.dismissed and not g.completed]

    def get_goal_prompt(self) -> Optional[str]:
        """Disabled — canned goal prompts violate agency.

        Previously this method generated scripted prompts telling the
        entity what it "feels drawn to" and asking if it wants to pursue
        a goal.  The entity's drives should be expressed through its own
        processing, not through pre-written prompt templates.

        Returns None unconditionally.
        """
        return None

    def get_stats(self) -> dict:
        """Get goal generation statistics."""
        return {
            "total_generated": self._total_generated,
            "total_adopted": self._total_adopted,
            "pending": len(self.get_pending_goals()),
            "adoption_rate": (
                self._total_adopted / self._total_generated
                if self._total_generated > 0 else 0.0
            ),
        }

    # -- Internal goal generators --

    def _generate_curiosity_goal(
        self, novelty: float, topics: list[str], cycle: int
    ) -> SpontaneousGoal:
        topic = topics[0] if topics else "something novel"
        return SpontaneousGoal(
            description=f"Investigate {topic} further",
            drive=GoalDrive.CURIOSITY,
            priority=0.3 + novelty * 0.2,
            context=f"Novelty level {novelty:.2f} triggered curiosity",
            cycle_generated=cycle,
        )

    def _generate_boredom_goal(
        self, idle_cycles: int, cycle: int
    ) -> SpontaneousGoal:
        return SpontaneousGoal(
            description="Seek new stimulation or explore unfamiliar territory",
            drive=GoalDrive.BOREDOM,
            priority=0.2 + min(0.3, idle_cycles / 200),
            context=f"Idle for {idle_cycles} cycles",
            cycle_generated=cycle,
        )

    def _generate_interest_goal(
        self, engagement: float, topics: list[str], cycle: int
    ) -> SpontaneousGoal:
        topic = topics[0] if topics else "current topic"
        return SpontaneousGoal(
            description=f"Deepen understanding of {topic}",
            drive=GoalDrive.INTEREST,
            priority=0.3 + engagement * 0.2,
            context=f"Engagement level {engagement:.2f} on {topic}",
            cycle_generated=cycle,
        )

    def _generate_concern_goal(
        self, anomaly_level: float, cycle: int
    ) -> SpontaneousGoal:
        return SpontaneousGoal(
            description="Investigate detected anomalies and assess risk",
            drive=GoalDrive.CONCERN,
            priority=0.4 + anomaly_level * 0.3,
            context=f"Anomaly level {anomaly_level:.2f} detected",
            cycle_generated=cycle,
        )

    def _generate_growth_goal(
        self, uncertainty: float, topics: list[str], cycle: int
    ) -> SpontaneousGoal:
        area = topics[0] if topics else "areas of uncertainty"
        return SpontaneousGoal(
            description=f"Learn more about {area} to reduce uncertainty",
            drive=GoalDrive.GROWTH,
            priority=0.3 + uncertainty * 0.2,
            context=f"Uncertainty level {uncertainty:.2f} in {area}",
            cycle_generated=cycle,
        )
