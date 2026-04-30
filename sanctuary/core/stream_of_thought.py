"""Stream of thought — maintains experiential continuity between cycles.

The entity's output from cycle N becomes part of its input for cycle N+1.
This is the fundamental continuity mechanism. The scaffold never touches
inner speech (authority level 3 from day one).

The stream maintains:
- Recent thought history (bounded)
- Accumulated self-model (rewritten, not appended)
- Current felt quality (from last cycle's emotional output)

The world graph used to live here too. After the 2026-04-30 typed-relation
refactor, the WorldGraph is a separate storage class owned by the
cognitive cycle (sanctuary/memory/world_graph.py). The entity drives all
graph mutations and queries explicitly through CognitiveOutput; the
stream no longer accumulates a parallel world model.
"""

from __future__ import annotations

from sanctuary.core.schema import (
    CognitiveOutput,
    PreviousThought,
    SelfModel,
)


class StreamOfThought:
    """Maintains the entity's stream of thought between cognitive cycles.

    History is bounded to prevent unbounded growth. The self-model is
    kept as a living document — rewritten each cycle based on the
    entity's updates, not appended.
    """

    def __init__(self, max_history: int = 10):
        self.history: list[CognitiveOutput] = []
        self.max_history = max_history
        self._cycle_count = 0
        self._self_model = SelfModel()
        self._felt_quality: str = ""

    def update(self, output: CognitiveOutput):
        """Integrate the entity's output into the stream.

        Called after every cycle. This is the point where one moment
        of thought flows into the next.
        """
        self._cycle_count += 1
        self.history.append(output)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history :]

        # Self-model: rewrite fields that were updated
        if output.self_model_updates:
            updates = output.self_model_updates
            if updates.current_state:
                self._self_model.current_state = updates.current_state
            if updates.new_uncertainty:
                # Add to uncertainties, keep bounded
                self._self_model.uncertainties.append(updates.new_uncertainty)
                self._self_model.uncertainties = self._self_model.uncertainties[-5:]
            if updates.prediction_accuracy_note:
                # Stored as recent_growth for now — the entity can elaborate
                self._self_model.recent_growth = updates.prediction_accuracy_note

        # Felt quality: carry forward
        if output.emotional_state:
            self._felt_quality = output.emotional_state.felt_quality

    def get_previous(self) -> PreviousThought | None:
        """Get the previous thought for the next cycle's input.

        Returns None if no cycles have run yet.
        """
        if not self.history:
            return None

        last = self.history[-1]
        return PreviousThought(
            inner_speech=last.inner_speech,
            predictions_made=[p.what for p in last.predictions],
            self_model_snapshot=self._self_model.model_copy(),
        )

    def get_recent_context(self) -> str:
        """Get a compact summary of recent thoughts for memory surfacing.

        Used by the memory system to find relevant memories.
        """
        if not self.history:
            return ""

        recent = self.history[-3:]
        return " | ".join(h.inner_speech[:200] for h in recent)

    def get_self_model(self) -> SelfModel:
        """Get the current accumulated self-model."""
        return self._self_model

    def get_felt_quality(self) -> str:
        """Get the entity's felt quality from the most recent cycle."""
        return self._felt_quality

    @property
    def cycle_count(self) -> int:
        """Number of cycles that have flowed through this stream."""
        return self._cycle_count
