"""Tests for scaffold affect module."""

from sanctuary.core.authority import AuthorityLevel, AuthorityManager
from sanctuary.core.schema import EmotionalOutput
from sanctuary.scaffold.affect import AffectConfig, ScaffoldAffect


class TestScaffoldAffect:
    """Test the simplified affect computation."""

    def test_initial_state_is_baseline(self):
        affect = ScaffoldAffect()
        vad = affect.get_computed_vad()
        assert vad.valence == 0.1
        assert vad.arousal == 0.2
        assert vad.dominance == 0.5

    def test_custom_baseline(self):
        config = AffectConfig(
            baseline_valence=0.3,
            baseline_arousal=0.5,
            baseline_dominance=0.7,
        )
        affect = ScaffoldAffect(config)
        vad = affect.get_computed_vad()
        assert vad.valence == 0.3
        assert vad.arousal == 0.5
        assert vad.dominance == 0.7

    def test_decay_toward_baseline(self):
        affect = ScaffoldAffect()
        # Push valence high
        affect.valence = 0.8
        affect.arousal = 0.9

        for _ in range(50):
            affect.decay_toward_baseline()

        # Should have decayed toward baseline (within 0.1 of baseline)
        assert abs(affect.valence - affect.config.baseline_valence) < 0.1
        assert abs(affect.arousal - affect.config.baseline_arousal) < 0.1

    def test_merge_model_emotion_scaffold_only_ignores(self):
        affect = ScaffoldAffect()
        authority = AuthorityManager({"emotional_state": AuthorityLevel.SCAFFOLD_ONLY})
        emotion = EmotionalOutput(valence_shift=0.5, arousal_shift=0.5)
        initial_v = affect.valence
        affect.merge_model_emotion(emotion, authority)
        assert affect.valence == initial_v

    def test_merge_model_emotion_advises_small_blend(self):
        affect = ScaffoldAffect()
        authority = AuthorityManager({"emotional_state": AuthorityLevel.MODEL_ADVISES})
        emotion = EmotionalOutput(valence_shift=0.5)
        initial_v = affect.valence
        affect.merge_model_emotion(emotion, authority)
        assert affect.valence > initial_v
        # Small blend — should be less than half the shift
        assert affect.valence < initial_v + 0.25

    def test_merge_model_emotion_guides_moderate_blend(self):
        affect = ScaffoldAffect()
        authority = AuthorityManager({"emotional_state": AuthorityLevel.MODEL_GUIDES})
        emotion = EmotionalOutput(valence_shift=0.5)
        initial_v = affect.valence
        affect.merge_model_emotion(emotion, authority)
        assert affect.valence > initial_v

    def test_merge_model_emotion_controls_full_blend(self):
        affect = ScaffoldAffect()
        authority = AuthorityManager({"emotional_state": AuthorityLevel.MODEL_CONTROLS})
        emotion = EmotionalOutput(valence_shift=0.5)
        initial_v = affect.valence
        affect.merge_model_emotion(emotion, authority)
        # Full blend — should be close to initial + 0.5
        assert abs(affect.valence - (initial_v + 0.5)) < 0.01

    def test_emotion_label(self):
        affect = ScaffoldAffect()
        affect.valence = 0.0
        affect.arousal = 0.1
        assert affect.get_emotion_label() == "calm"

        affect.valence = 0.5
        affect.arousal = 0.8
        assert affect.get_emotion_label() == "joy"

        affect.valence = -0.5
        affect.arousal = 0.8
        assert affect.get_emotion_label() == "anger"

        affect.valence = -0.5
        affect.arousal = 0.2
        assert affect.get_emotion_label() == "sadness"
