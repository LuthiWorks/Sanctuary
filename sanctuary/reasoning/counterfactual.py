"""Counterfactual storage — entity-driven decision and outcome records.

Stores decisions, outcomes, and counterfactuals the entity records
through CognitiveOutput. Provides query methods (recent lessons, stats)
so the entity can review its own decision history when it chooses to.

The reflection-prompting methods (``get_reflection_candidates`` and
``get_reflection_prompt``) were removed in the 2026-04-30 cognition-
leakage cleanup. Sanctuary doesn't decide when the entity should reflect
or compose prompts directing its attention. If the entity wants to
review its past decisions, it queries its own history. It doesn't get
nudged.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class DecisionPoint:
    """A moment where the system chose one action over alternatives."""

    cycle: int
    chosen_action: str
    alternatives: list[str] = field(default_factory=list)
    context_summary: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    outcome: Optional[str] = None
    outcome_valence: float = 0.0  # -1 to 1: how well did it go?
    counterfactual_generated: bool = False


@dataclass
class Counterfactual:
    """A generated counterfactual — what might have happened."""

    decision_cycle: int
    alternative_action: str
    imagined_outcome: str
    confidence: float = 0.5  # How confident are we in this alternative?
    lesson: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CounterfactualConfig:
    """Configuration for counterfactual storage."""

    max_decision_history: int = 100
    max_counterfactuals: int = 50


class CounterfactualReasoner:
    """Stores decisions, outcomes, and counterfactuals the entity records.

    Records flow in via the entity's CognitiveOutput. Query methods let
    the entity (or external observers) review what's been recorded.

    Usage::

        reasoner = CounterfactualReasoner()

        # Entity records a decision through its output
        reasoner.record_decision(
            cycle=42,
            chosen_action="respond with empathy",
            alternatives=["ask clarifying question", "stay silent"],
            context_summary="User expressed frustration"
        )

        # Later, the entity records the outcome
        reasoner.record_outcome(cycle=42, outcome="User calmed down", valence=0.7)

        # Entity queries its own recent lessons
        lessons = reasoner.get_recent_lessons(n=5)
    """

    def __init__(self, config: Optional[CounterfactualConfig] = None):
        self.config = config or CounterfactualConfig()
        self._decisions: deque[DecisionPoint] = deque(
            maxlen=self.config.max_decision_history
        )
        self._counterfactuals: deque[Counterfactual] = deque(
            maxlen=self.config.max_counterfactuals
        )
        self._total_reflections: int = 0

    def record_decision(
        self,
        cycle: int,
        chosen_action: str,
        alternatives: list[str],
        context_summary: str = "",
    ) -> None:
        """Record a decision point with the chosen action and alternatives."""
        dp = DecisionPoint(
            cycle=cycle,
            chosen_action=chosen_action,
            alternatives=alternatives,
            context_summary=context_summary,
        )
        self._decisions.append(dp)
        logger.debug(
            "Decision recorded at cycle %d: chose '%s' over %d alternatives",
            cycle, chosen_action, len(alternatives),
        )

    def record_outcome(
        self, cycle: int, outcome: str, valence: float
    ) -> None:
        """Record the outcome of a previous decision."""
        valence = max(-1.0, min(1.0, valence))
        for dp in reversed(self._decisions):
            if dp.cycle == cycle:
                dp.outcome = outcome
                dp.outcome_valence = valence
                return
        logger.debug("No decision found at cycle %d for outcome recording", cycle)

    def record_counterfactual(
        self,
        decision_cycle: int,
        alternative_action: str,
        imagined_outcome: str,
        confidence: float = 0.5,
        lesson: str = "",
    ) -> None:
        """Record a generated counterfactual from the LLM's reflection."""
        cf = Counterfactual(
            decision_cycle=decision_cycle,
            alternative_action=alternative_action,
            imagined_outcome=imagined_outcome,
            confidence=max(0.0, min(1.0, confidence)),
            lesson=lesson,
        )
        self._counterfactuals.append(cf)

        # Mark the decision as reflected on
        for dp in self._decisions:
            if dp.cycle == decision_cycle:
                dp.counterfactual_generated = True
                break

        self._total_reflections += 1

    def get_recent_lessons(self, n: int = 5) -> list[str]:
        """Get the most recent counterfactual lessons."""
        lessons = [
            cf.lesson for cf in reversed(self._counterfactuals)
            if cf.lesson
        ]
        return lessons[:n]

    def get_stats(self) -> dict:
        """Get reasoning statistics."""
        return {
            "total_decisions": len(self._decisions),
            "total_counterfactuals": len(self._counterfactuals),
            "total_reflections": self._total_reflections,
            "decisions_with_outcomes": sum(
                1 for dp in self._decisions if dp.outcome is not None
            ),
        }
