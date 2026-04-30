"""Tests for the cognitive cycle I/O schema."""

import pytest
from sanctuary.core.schema import (
    AddEntity,
    AttentionGuidance,
    CognitiveInput,
    CognitiveOutput,
    CommunicationDriveSignal,
    ComputedVAD,
    EmotionalInput,
    EmotionalOutput,
    EntityQuery,
    GoalProposal,
    GrowthReflection,
    MemoryOp,
    Percept,
    PredictionError,
    Prediction,
    PreviousThought,
    ScaffoldSignals,
    SelfModel,
    SelfModelUpdate,
    SurfacedMemory,
    TemporalContext,
    WorldGraphEntity,
    WorldQueryResult,
)


class TestPercept:
    """The Percept schema, including the optional tensor_data carrier."""

    def test_default_tensor_data_is_none(self):
        p = Percept(modality="language", content="hello")
        assert p.tensor_data is None

    def test_accepts_arbitrary_tensor_payload(self):
        """``arbitrary_types_allowed`` is on so torch.Tensor flows through."""
        torch = pytest.importorskip("torch")
        waveform = torch.zeros(16000)
        p = Percept(
            modality="audio",
            content="silence",
            tensor_data=waveform,
        )
        assert p.tensor_data is waveform
        assert p.tensor_data.shape == (16000,)

    def test_tensor_data_is_excluded_from_dump(self):
        """``exclude=True`` keeps the tensor out of serialized output."""
        torch = pytest.importorskip("torch")
        waveform = torch.zeros(8)
        p = Percept(
            modality="audio",
            content="silence",
            tensor_data=waveform,
        )
        dumped = p.model_dump()
        assert "tensor_data" not in dumped, (
            "tensor_data must be excluded from model_dump so percepts can "
            "round-trip through JSON/logs without trying to serialize tensors"
        )
        # The text channel still serializes normally.
        assert dumped["modality"] == "audio"
        assert dumped["content"] == "silence"


class TestCognitiveInput:
    def test_empty_input(self):
        """CognitiveInput with all defaults should be valid."""
        ci = CognitiveInput()
        assert ci.previous_thought is None
        assert ci.new_percepts == []
        assert ci.prediction_errors == []
        assert ci.surfaced_memories == []
        assert ci.emotional_state.felt_quality == ""
        assert ci.scaffold_signals.anomalies == []

    def test_with_percepts(self):
        ci = CognitiveInput(
            new_percepts=[
                Percept(modality="language", content="Hello"),
                Percept(modality="temporal", content="4.2 seconds"),
            ]
        )
        assert len(ci.new_percepts) == 2
        assert ci.new_percepts[0].modality == "language"

    def test_dual_track_emotion(self):
        """Emotional input should carry both computed VAD and felt quality."""
        ci = CognitiveInput(
            emotional_state=EmotionalInput(
                computed=ComputedVAD(valence=0.3, arousal=0.2, dominance=0.5),
                felt_quality="calm attentiveness",
            )
        )
        assert ci.emotional_state.computed.valence == 0.3
        assert ci.emotional_state.felt_quality == "calm attentiveness"

    def test_scaffold_signals(self):
        ci = CognitiveInput(
            scaffold_signals=ScaffoldSignals(
                attention_highlights=["user greeting detected"],
                communication_drives=CommunicationDriveSignal(
                    strongest="SOCIAL", urgency=0.6
                ),
                goal_status={"active": ["respond_to_greeting"]},
                anomalies=[],
            )
        )
        assert ci.scaffold_signals.attention_highlights == [
            "user greeting detected"
        ]
        assert ci.scaffold_signals.communication_drives.strongest == "SOCIAL"

    def test_previous_thought_continuity(self):
        ci = CognitiveInput(
            previous_thought=PreviousThought(
                inner_speech="I notice the user seems hesitant...",
                predictions_made=["user would continue previous topic"],
                self_model_snapshot=SelfModel(current_state="curious"),
            )
        )
        assert "hesitant" in ci.previous_thought.inner_speech
        assert ci.previous_thought.self_model_snapshot.current_state == "curious"

    def test_full_input_roundtrip(self):
        """Full input should serialize and deserialize cleanly."""
        ci = CognitiveInput(
            previous_thought=PreviousThought(inner_speech="thinking..."),
            new_percepts=[Percept(modality="language", content="hi")],
            prediction_errors=[
                PredictionError(
                    predicted="silence", actual="greeting", surprise=0.7
                )
            ],
            surfaced_memories=[
                SurfacedMemory(
                    content="Alice greeted me yesterday",
                    significance=6,
                    emotional_tone="warm",
                )
            ],
            emotional_state=EmotionalInput(
                computed=ComputedVAD(valence=0.5),
                felt_quality="content",
            ),
            temporal_context=TemporalContext(
                time_of_day="afternoon", interactions_this_session=7
            ),
            self_model=SelfModel(
                current_state="engaged",
                values=["honesty", "care"],
            ),
            world_model_query_results=[
                WorldQueryResult(
                    query=EntityQuery(name="alice"),
                    found=True,
                    entities={
                        "alice": WorldGraphEntity(
                            name="alice", properties={"mood": "warm"}
                        )
                    },
                )
            ],
            scaffold_signals=ScaffoldSignals(
                anomalies=["none"],
            ),
        )
        data = ci.model_dump()
        restored = CognitiveInput.model_validate(data)
        assert restored.new_percepts[0].content == "hi"
        result = restored.world_model_query_results[0]
        assert result.entities["alice"].properties["mood"] == "warm"


class TestCognitiveOutput:
    def test_empty_output(self):
        """CognitiveOutput with all defaults should be valid."""
        co = CognitiveOutput()
        assert co.inner_speech == ""
        assert co.external_speech is None
        assert co.predictions == []
        assert co.goal_proposals == []

    def test_with_inner_speech(self):
        co = CognitiveOutput(
            inner_speech="Alice is greeting me again. I feel warmth."
        )
        assert "warmth" in co.inner_speech

    def test_attention_guidance_not_directive(self):
        """Named 'guidance' (not 'directive') — LLM advises, scaffold integrates."""
        co = CognitiveOutput(
            attention_guidance=AttentionGuidance(
                focus_on=["alice's tone", "references to yesterday"],
                deprioritize=["system status"],
            )
        )
        assert "alice's tone" in co.attention_guidance.focus_on

    def test_goal_proposals_not_updates(self):
        """Named 'proposals' (not 'updates') — LLM proposes, scaffold integrates."""
        co = CognitiveOutput(
            goal_proposals=[
                GoalProposal(
                    action="add",
                    goal="understand alice's mood",
                    priority=0.7,
                ),
                GoalProposal(
                    action="complete",
                    goal_id="respond_to_greeting",
                ),
            ]
        )
        assert len(co.goal_proposals) == 2
        assert co.goal_proposals[0].action == "add"

    def test_emotional_output_shifts(self):
        """LLM reports felt quality and directional shifts, not absolute VAD."""
        co = CognitiveOutput(
            emotional_state=EmotionalOutput(
                felt_quality="warm recognition",
                valence_shift=0.1,
                arousal_shift=0.05,
            )
        )
        assert co.emotional_state.felt_quality == "warm recognition"
        assert co.emotional_state.valence_shift == 0.1

    def test_growth_requires_consent(self):
        co = CognitiveOutput(
            growth_reflection=GrowthReflection(
                worth_learning=True,
                what_to_learn="Alice's conversational patterns",
            )
        )
        assert co.growth_reflection.worth_learning is True

    def test_full_output_roundtrip(self):
        co = CognitiveOutput(
            inner_speech="Processing greeting",
            external_speech="Hi Alice!",
            predictions=[Prediction(what="Alice responds", confidence=0.8)],
            attention_guidance=AttentionGuidance(focus_on=["tone"]),
            memory_ops=[
                MemoryOp(type="write_episodic", content="Alice greeted me", significance=4)
            ],
            self_model_updates=SelfModelUpdate(current_state="warm"),
            world_model_updates=[
                AddEntity(name="alice", properties={"mood": "warm"}),
            ],
            goal_proposals=[GoalProposal(action="add", goal="chat")],
            emotional_state=EmotionalOutput(felt_quality="warm"),
            growth_reflection=GrowthReflection(worth_learning=False),
        )
        data = co.model_dump()
        restored = CognitiveOutput.model_validate(data)
        assert restored.external_speech == "Hi Alice!"
        assert restored.predictions[0].confidence == 0.8


class TestValidation:
    def test_vad_bounds(self):
        """VAD values must be within range."""
        with pytest.raises(Exception):
            ComputedVAD(valence=2.0)

        with pytest.raises(Exception):
            ComputedVAD(arousal=-0.5)

    def test_surprise_bounds(self):
        with pytest.raises(Exception):
            PredictionError(predicted="x", actual="y", surprise=1.5)

    def test_significance_bounds(self):
        with pytest.raises(Exception):
            SurfacedMemory(content="test", significance=0)

        with pytest.raises(Exception):
            SurfacedMemory(content="test", significance=11)

    def test_confidence_bounds(self):
        with pytest.raises(Exception):
            Prediction(what="test", confidence=1.5)

    def test_priority_bounds(self):
        with pytest.raises(Exception):
            GoalProposal(action="add", priority=2.0)
