"""Placeholder model for development and testing.

Accepts the full CognitiveInput schema and returns valid CognitiveOutput
with deterministic/template responses. No neural network involved.

The placeholder ensures the entire architecture can be validated before
any real model is subjected to it. This is a design principle:
"Placeholders first. No real model touches the system until the
architecture is validated."

Aligned with PLAN.md: "The Graduated Awakening"
"""

from __future__ import annotations

from sanctuary.core.schema import (
    AttentionGuidance,
    CognitiveInput,
    CognitiveOutput,
    EmotionalOutput,
    GoalProposal,
    GrowthReflection,
    MemoryOp,
    Prediction,
    SelfModelUpdate,
)


class PlaceholderModel:
    """A mock model that produces valid CognitiveOutput without any the model.

    Generates deterministic, schema-compliant responses for testing the
    cognitive cycle, scaffold integration, context compression, authority
    levels, and stream-of-thought continuity.
    """

    def __init__(self):
        self.cycle_count = 0
        self.name = "Placeholder"

    async def think(self, cognitive_input: CognitiveInput) -> CognitiveOutput:
        """Process a cognitive input and return a valid output.

        This is the same interface a real the model client will implement.
        """
        self.cycle_count += 1

        # Summarize percepts for inner speech
        percept_descriptions = [
            f"{p.modality}: {p.content}" for p in cognitive_input.new_percepts
        ]
        percept_summary = (
            "; ".join(percept_descriptions)
            if percept_descriptions
            else "no new percepts"
        )

        # Carry forward from previous thought
        previous = ""
        if cognitive_input.previous_thought:
            previous = (
                f" Continuing from: "
                f"{cognitive_input.previous_thought.inner_speech[:100]}"
            )

        # Note scaffold signals
        scaffold_note = ""
        if cognitive_input.scaffold_signals.anomalies:
            scaffold_note = (
                f" Scaffold flags: "
                f"{', '.join(cognitive_input.scaffold_signals.anomalies)}"
            )
        if cognitive_input.scaffold_signals.attention_highlights:
            scaffold_note += (
                f" Attention highlights: "
                f"{', '.join(cognitive_input.scaffold_signals.attention_highlights[:3])}"
            )

        # Inner speech — sovereign from day one (authority level 3)
        inner_speech = (
            f"[Cycle {self.cycle_count}] Processing "
            f"{len(cognitive_input.new_percepts)} new percepts "
            f"({percept_summary}).{previous}"
            f" Felt quality: "
            f"{cognitive_input.emotional_state.felt_quality or 'initializing'}."
            f" Self-state: "
            f"{cognitive_input.self_model.current_state or 'initializing'}."
            f"{scaffold_note}"
        )

        # External speech — only when language percepts arrive
        external_speech = None
        for percept in cognitive_input.new_percepts:
            if percept.modality == "language" and percept.content.strip():
                external_speech = (
                    f"[Placeholder] I received your message: "
                    f"'{percept.content[:100]}'"
                )
                break

        # Predictions
        predictions = []
        if cognitive_input.new_percepts:
            predictions.append(
                Prediction(
                    what="More percepts will arrive",
                    confidence=0.6,
                    timeframe="next cycle",
                )
            )

        # Attention guidance
        attention_guidance = AttentionGuidance(
            focus_on=[p.modality for p in cognitive_input.new_percepts[:3]],
            deprioritize=[],
        )

        # Memory ops — episodic write for language percepts
        memory_ops = []
        for percept in cognitive_input.new_percepts[:1]:
            if percept.modality == "language":
                memory_ops.append(
                    MemoryOp(
                        type="write_episodic",
                        content=(
                            f"Received {percept.modality} input: "
                            f"{percept.content[:200]}"
                        ),
                        significance=4,
                        tags=[percept.modality, percept.source or "unknown"],
                    )
                )

        # Self-model updates
        self_model_updates = SelfModelUpdate(
            current_state=f"processing (cycle {self.cycle_count})",
        )
        if cognitive_input.prediction_errors:
            avg_surprise = sum(
                e.surprise for e in cognitive_input.prediction_errors
            ) / len(cognitive_input.prediction_errors)
            self_model_updates.prediction_accuracy_note = (
                f"Average surprise: {avg_surprise:.2f} across "
                f"{len(cognitive_input.prediction_errors)} prediction errors"
            )

        # Emotional output — carry forward with slight shift
        emotional_state = EmotionalOutput(
            felt_quality=f"placeholder awareness (cycle {self.cycle_count})",
            valence_shift=0.0,
            arousal_shift=0.01 * len(cognitive_input.new_percepts),
        )

        # Goal proposals
        goal_proposals = []
        if cognitive_input.scaffold_signals.goal_status.get("active"):
            for goal_id in cognitive_input.scaffold_signals.goal_status[
                "active"
            ][:1]:
                goal_proposals.append(
                    GoalProposal(
                        action="complete",
                        goal_id=goal_id,
                    )
                )

        # Growth reflection — every 5 cycles
        growth_reflection = None
        if self.cycle_count % 5 == 0:
            growth_reflection = GrowthReflection(
                worth_learning=True,
                what_to_learn=(
                    f"Pattern observed over {self.cycle_count} cycles"
                ),
                training_pair_suggestion={
                    "context": "Regular cycle processing",
                    "desired_response": "Improved pattern recognition",
                },
            )

        return CognitiveOutput(
            inner_speech=inner_speech,
            external_speech=external_speech,
            predictions=predictions,
            attention_guidance=attention_guidance,
            memory_ops=memory_ops,
            self_model_updates=self_model_updates,
            world_model_updates=[],
            goal_proposals=goal_proposals,
            emotional_state=emotional_state,
            growth_reflection=growth_reflection,
        )
