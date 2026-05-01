"""CognitiveScaffold — the ScaffoldProtocol implementation.

Orchestrates the remaining scaffold subsystems to validate, integrate,
and broadcast cognitive output.

The scaffold's job:
  1. Validate actions against authority levels
  2. Record entity-driven goal proposals
  3. Track dual-track emotion (computed VAD + entity-reported felt quality)
  4. Report scaffold signals back to the entity next cycle
  5. Broadcast integrated output

The scaffold does NOT do cognition. It validates, persists, and mediates.
The 2026-04-30 cognition-leakage cleanup removed anomaly detection,
communication gating, and auto-staleness inference — Sanctuary doesn't
judge, gate, or infer on the entity's behalf.
"""

from __future__ import annotations

import logging
from typing import Optional

from sanctuary.core.authority import AuthorityManager
from sanctuary.core.schema import (
    CognitiveOutput,
    ScaffoldSignals,
)

from sanctuary.scaffold.affect import AffectConfig, ScaffoldAffect
from sanctuary.scaffold.action_validator import ScaffoldActionValidator
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
        max_goals: int = 10,
    ):
        self.affect = ScaffoldAffect(affect_config)
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
        """Validate and integrate cognitive output with scaffold subsystems.

        Steps:
        1. Action validation (filters invalid ops based on authority)
        2. Goal integration (records entity goal proposals)
        3. Affect merge (blends entity-reported emotional shifts with computed VAD)
        4. Affect decay toward baseline

        external_speech flows through unchanged — the entity speaks when
        it produces speech. Period.

        Returns the integrated (potentially modified) output.
        """
        # 1. Validate actions
        output, self._last_validation_issues = self.action_validator.validate(
            output, authority
        )

        # 2. Goal integration
        self.goals.integrate_proposals(output.goal_proposals, authority)

        # 3. Affect: merge entity-reported emotional shifts with computed state
        self.affect.merge_llm_emotion(output.emotional_state, authority)

        # 4. Affect decay
        self.affect.decay_toward_baseline()

        self._cycle_count += 1

        return output

    # -----------------------------------------------------------------
    # ScaffoldProtocol: get_signals
    # -----------------------------------------------------------------

    def get_signals(self) -> ScaffoldSignals:
        """Return scaffold observations for the entity's next input.

        These are terse, structured signals — not prose.
        """
        return ScaffoldSignals(
            attention_highlights=self._get_attention_highlights(),
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
