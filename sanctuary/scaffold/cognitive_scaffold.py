"""CognitiveScaffold — the ScaffoldProtocol implementation.

Orchestrates all scaffold subsystems to validate, integrate, and broadcast
LLM cognitive output. This is the main entry point for Phase 2.

The scaffold's job:
  1. Detect anomalies in LLM output
  2. Validate actions based on authority levels
  3. Gate external speech through communication system
  4. Integrate goal proposals
  5. Merge LLM emotional output with computed VAD (dual-track)
  6. Report scaffold signals back to the LLM next cycle
  7. Broadcast integrated output (GWT ignition)

The scaffold does NOT do cognition. It validates, persists, and mediates.
"""

from __future__ import annotations

import logging
from typing import Optional

from sanctuary.core.authority import AuthorityManager
from sanctuary.core.schema import (
    CognitiveOutput,
    Percept,
    ScaffoldSignals,
)

from sanctuary.scaffold.affect import AffectConfig, ScaffoldAffect
from sanctuary.scaffold.action_validator import ScaffoldActionValidator
from sanctuary.scaffold.communication import CommunicationConfig, ScaffoldCommunication
from sanctuary.scaffold.goal_integrator import ScaffoldGoalIntegrator

logger = logging.getLogger(__name__)


class CognitiveScaffold:
    """Implements ScaffoldProtocol by orchestrating scaffold subsystems.

    Usage::

        scaffold = CognitiveScaffold()
        cycle = CognitiveCycle(model=placeholder, scaffold=scaffold)
        await cycle.run(max_cycles=5)
    """

    def __init__(
        self,
        affect_config: Optional[AffectConfig] = None,
        communication_config: Optional[CommunicationConfig] = None,
        max_goals: int = 10,
    ):
        self.affect = ScaffoldAffect(affect_config)
        self.communication = ScaffoldCommunication(communication_config)
        self.goals = ScaffoldGoalIntegrator(max_goals=max_goals)
        self.action_validator = ScaffoldActionValidator()

        self._last_validation_issues: list[str] = []
        self._broadcast_handlers: list = []
        self._cycle_count = 0

    # -----------------------------------------------------------------
    # ScaffoldProtocol: integrate
    # -----------------------------------------------------------------

    async def integrate(
        self,
        output: CognitiveOutput,
        authority: AuthorityManager,
    ) -> CognitiveOutput:
        """Validate and integrate LLM output with scaffold subsystems.

        Steps:
        1. Action validation (filters invalid ops based on authority)
        2. Communication gating (controls external speech emission)
        3. Goal integration (records LLM goal proposals)
        4. Affect merge (blends LLM emotional signals with computed VAD)
        5. Affect decay toward baseline

        Returns the integrated (potentially modified) output.
        """
        # 1. Validate actions
        output, self._last_validation_issues = self.action_validator.validate(
            output, authority
        )

        # 2. Communication gating
        has_user_percept = False  # Set by inject_percept_context
        if hasattr(self, "_has_user_percept"):
            has_user_percept = self._has_user_percept

        gated_speech = self.communication.evaluate(
            output.external_speech,
            has_user_percept=has_user_percept,
            authority=authority,
        )
        output.external_speech = gated_speech

        # 3. Goal integration
        self.goals.integrate_proposals(output.goal_proposals, authority)

        # 4. Affect: merge LLM emotional output with computed state
        self.affect.merge_llm_emotion(output.emotional_state, authority)

        # 5. Affect decay
        self.affect.decay_toward_baseline()

        self._cycle_count += 1
        self._has_user_percept = False  # Reset for next cycle

        return output

    # -----------------------------------------------------------------
    # ScaffoldProtocol: get_signals
    # -----------------------------------------------------------------

    def get_signals(self) -> ScaffoldSignals:
        """Return scaffold observations for the LLM's next input.

        These are terse, structured signals — not prose.
        """
        return ScaffoldSignals(
            attention_highlights=self._get_attention_highlights(),
            communication_drives=self.communication.get_signal(),
            goal_status=self.goals.get_status(),
            anomalies=list(self._last_validation_issues),
        )

    # -----------------------------------------------------------------
    # ScaffoldProtocol: broadcast
    # -----------------------------------------------------------------

    async def broadcast(self, output: CognitiveOutput) -> None:
        """Broadcast integrated output to all registered handlers.

        In Phase 2, this is a simple callback mechanism.
        When full GWT broadcast is wired in (Phase 6), this delegates
        to GlobalBroadcaster.
        """
        for handler in self._broadcast_handlers:
            try:
                await handler(output)
            except Exception as e:
                logger.error("Broadcast handler error: %s", e)

    # -----------------------------------------------------------------
    # Additional scaffold API
    # -----------------------------------------------------------------

    def on_broadcast(self, handler) -> None:
        """Register a broadcast listener.

        Handler signature: async def handler(output: CognitiveOutput) -> None
        """
        self._broadcast_handlers.append(handler)

    def notify_percepts(self, percepts: list[Percept]) -> None:
        """Called by the cycle to inform scaffold about incoming percepts.

        This allows the scaffold to:
        - Update affect from percept content
        - Detect user language percepts for communication gating
        """
        self.affect.update_from_percepts(percepts)

        # Check if any percept is from a user (language modality)
        self._has_user_percept = any(
            p.modality == "language" and p.source.startswith("user")
            for p in percepts
        )

    def get_computed_vad(self):
        """Return the scaffold's current computed VAD."""
        return self.affect.get_computed_vad()

    def get_active_goals(self) -> list[str]:
        """Return descriptions of active goals."""
        return self.goals.get_active_goal_descriptions()

    # -----------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------

    def _get_attention_highlights(self) -> list[str]:
        """Generate attention highlights from scaffold state."""
        highlights: list[str] = []

        # Emotional state
        label = self.affect.get_emotion_label()
        if label not in ("calm", "contentment"):
            highlights.append(f"emotional state: {label}")

        return highlights
