"""Mental simulation — simulating outcomes before taking actions.

Enables the LLM to run "what-if" scenarios before committing to an action.
The scaffold provides the simulation framework (tracking hypothetical branches,
comparing predicted vs. actual outcomes); the LLM generates the actual
simulated content.

Mental simulation is a key component of predictive processing / active inference:
the system imagines futures before selecting actions, reducing free energy
by choosing actions with the best predicted outcomes.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SimulatedScenario:
    """A single hypothetical outcome for an action."""

    action: str
    predicted_outcome: str
    predicted_valence: float = 0.0  # -1 to 1: expected emotional impact
    predicted_confidence: float = 0.5  # How confident in this prediction?
    risks: list[str] = field(default_factory=list)
    benefits: list[str] = field(default_factory=list)


@dataclass
class Simulation:
    """A complete mental simulation — one situation, multiple possible actions."""

    situation: str
    scenarios: list[SimulatedScenario] = field(default_factory=list)
    selected_action: Optional[str] = None
    selection_reasoning: str = ""
    cycle_created: int = 0
    cycle_resolved: Optional[int] = None
    actual_outcome: Optional[str] = None
    actual_valence: Optional[float] = None
    prediction_error: Optional[float] = None  # |predicted - actual| valence
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SimulationConfig:
    """Configuration for mental simulation."""

    max_simulation_history: int = 100
    max_scenarios_per_simulation: int = 5
    min_confidence_to_act: float = 0.3  # Below this, suggest more deliberation
    prediction_error_learning_rate: float = 0.1


class MentalSimulator:
    """Framework for mental simulation — imagining before acting.

    Provides structure for the LLM to:
    1. Enumerate possible actions for a situation
    2. Predict outcomes for each action
    3. Select the best action based on predicted outcomes
    4. Track prediction accuracy to improve future simulations

    Usage::

        sim = MentalSimulator()

        # Start a simulation
        sim_id = sim.begin_simulation(
            situation="User asked a sensitive personal question",
            cycle=42,
        )

        # Add scenarios
        sim.add_scenario(sim_id,
            action="Answer directly",
            predicted_outcome="User feels heard but topic may be uncomfortable",
            predicted_valence=0.3,
            risks=["May overstep boundaries"],
            benefits=["Builds trust"],
        )
        sim.add_scenario(sim_id,
            action="Acknowledge and redirect",
            predicted_outcome="User may feel deflected",
            predicted_valence=-0.1,
            risks=["Seems evasive"],
            benefits=["Respects boundaries"],
        )

        # Select and act
        sim.select_action(sim_id, action="Answer directly", reasoning="Trust building")

        # Later, record what actually happened
        sim.record_outcome(sim_id, outcome="User appreciated honesty", valence=0.6)
    """

    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config or SimulationConfig()
        self._simulations: deque[Simulation] = deque(
            maxlen=self.config.max_simulation_history
        )
        self._sim_counter: int = 0
        self._total_prediction_error: float = 0.0
        self._total_resolved: int = 0

    def begin_simulation(
        self, situation: str, cycle: int = 0
    ) -> int:
        """Start a new mental simulation for a situation. Returns simulation ID."""
        sim = Simulation(
            situation=situation,
            cycle_created=cycle,
        )
        self._simulations.append(sim)
        self._sim_counter += 1
        sim_id = self._sim_counter - 1
        logger.debug("Mental simulation begun: '%s' (id=%d)", situation, sim_id)
        return sim_id

    def add_scenario(
        self,
        sim_id: int,
        action: str,
        predicted_outcome: str,
        predicted_valence: float = 0.0,
        predicted_confidence: float = 0.5,
        risks: list[str] | None = None,
        benefits: list[str] | None = None,
    ) -> bool:
        """Add a hypothetical scenario to a simulation."""
        sim = self._get_simulation(sim_id)
        if sim is None:
            return False

        if len(sim.scenarios) >= self.config.max_scenarios_per_simulation:
            return False

        scenario = SimulatedScenario(
            action=action,
            predicted_outcome=predicted_outcome,
            predicted_valence=max(-1.0, min(1.0, predicted_valence)),
            predicted_confidence=max(0.0, min(1.0, predicted_confidence)),
            risks=risks or [],
            benefits=benefits or [],
        )
        sim.scenarios.append(scenario)
        return True

    def select_action(
        self, sim_id: int, action: str, reasoning: str = ""
    ) -> bool:
        """Record which action was selected from the simulation."""
        sim = self._get_simulation(sim_id)
        if sim is None:
            return False

        sim.selected_action = action
        sim.selection_reasoning = reasoning
        return True

    def record_outcome(
        self,
        sim_id: int,
        outcome: str,
        valence: float,
        cycle: int = 0,
    ) -> Optional[float]:
        """Record the actual outcome and compute prediction error.

        Returns the prediction error (|predicted - actual| valence), or None
        if the simulation wasn't found or had no selected action.
        """
        sim = self._get_simulation(sim_id)
        if sim is None or sim.selected_action is None:
            return None

        sim.actual_outcome = outcome
        sim.actual_valence = max(-1.0, min(1.0, valence))
        sim.cycle_resolved = cycle

        # Find the selected scenario's predicted valence
        predicted_valence = 0.0
        for scenario in sim.scenarios:
            if scenario.action == sim.selected_action:
                predicted_valence = scenario.predicted_valence
                break

        prediction_error = abs(predicted_valence - sim.actual_valence)
        sim.prediction_error = prediction_error
        self._total_prediction_error += prediction_error
        self._total_resolved += 1

        return prediction_error

    def get_simulation_prompt(self, situation: str) -> str:
        """Generate a prompt asking the LLM to simulate outcomes.

        This structures the LLM's deliberation by asking it to enumerate
        actions, predict outcomes, and weigh risks/benefits.
        """
        prompt = (
            f"[Mental simulation] Consider the situation: \"{situation}\". "
            f"Before acting, imagine 2-3 possible responses. For each, predict: "
            f"(1) what would likely happen, (2) emotional impact (-1 to 1), "
            f"(3) risks, (4) benefits. Then choose the best action and explain why."
        )
        return prompt

    def get_recommendation(self, sim_id: int) -> Optional[str]:
        """Get a recommendation based on the simulation's scenarios.

        Returns the action with the best risk-adjusted predicted valence,
        or None if confidence is too low for all scenarios.
        """
        sim = self._get_simulation(sim_id)
        if sim is None or not sim.scenarios:
            return None

        # Score each scenario: valence * confidence, penalized by risk count
        best_score = float("-inf")
        best_action = None
        for scenario in sim.scenarios:
            risk_penalty = len(scenario.risks) * 0.1
            score = (
                scenario.predicted_valence * scenario.predicted_confidence
                - risk_penalty
            )
            if score > best_score:
                best_score = score
                best_action = scenario.action

        return best_action

    def get_average_prediction_error(self) -> float:
        """Get average prediction error across all resolved simulations."""
        if self._total_resolved == 0:
            return 0.0
        return self._total_prediction_error / self._total_resolved

    def get_recent_simulations(
        self, n: int = 5, resolved_only: bool = False
    ) -> list[Simulation]:
        """Get recent simulations."""
        sims = list(self._simulations)
        if resolved_only:
            sims = [s for s in sims if s.actual_outcome is not None]
        return sims[-n:]

    def get_stats(self) -> dict:
        """Get simulation statistics."""
        resolved = [s for s in self._simulations if s.actual_outcome is not None]
        return {
            "total_simulations": len(self._simulations),
            "total_resolved": self._total_resolved,
            "pending": len(self._simulations) - len(resolved),
            "avg_prediction_error": self.get_average_prediction_error(),
            "avg_scenarios_per_sim": (
                sum(len(s.scenarios) for s in self._simulations) / len(self._simulations)
                if self._simulations else 0.0
            ),
        }

    # -- Internal --

    def _get_simulation(self, sim_id: int) -> Optional[Simulation]:
        """Get a simulation by ID.

        sim_id is assigned sequentially at creation (0, 1, 2, ...).
        The deque may have evicted older entries, so we compute the offset
        between the total created count and current deque length to find
        the correct index. Returns None if the simulation was evicted.
        """
        offset = self._sim_counter - len(self._simulations)
        index = sim_id - offset
        if 0 <= index < len(self._simulations):
            return self._simulations[index]
        return None
