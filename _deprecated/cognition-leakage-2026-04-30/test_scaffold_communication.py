"""Tests for scaffold communication gating."""

from sanctuary.core.authority import AuthorityLevel, AuthorityManager
from sanctuary.scaffold.communication import CommunicationConfig, ScaffoldCommunication


class TestScaffoldCommunication:
    """Test the communication gating system."""

    def test_user_percept_passes_speech(self):
        comm = ScaffoldCommunication()
        authority = AuthorityManager()
        result = comm.evaluate("Hello!", has_user_percept=True, authority=authority)
        assert result == "Hello!"

    def test_no_speech_returns_none(self):
        comm = ScaffoldCommunication()
        authority = AuthorityManager()
        result = comm.evaluate(None, has_user_percept=True, authority=authority)
        assert result is None

    def test_empty_speech_returns_none(self):
        comm = ScaffoldCommunication()
        authority = AuthorityManager()
        result = comm.evaluate("", has_user_percept=True, authority=authority)
        assert result is None

    def test_whitespace_speech_returns_none(self):
        comm = ScaffoldCommunication()
        authority = AuthorityManager()
        result = comm.evaluate("   ", has_user_percept=True, authority=authority)
        assert result is None

    def test_idle_speech_gated_without_user_percept(self):
        """Without user input, idle drive is low and may be gated."""
        config = CommunicationConfig(
            speak_threshold=0.5,
            idle_drive_strength=0.1,
        )
        comm = ScaffoldCommunication(config)
        authority = AuthorityManager()
        result = comm.evaluate(
            "Just thinking...", has_user_percept=False, authority=authority
        )
        # Idle drive (0.1) < threshold (0.5) → gated
        assert result is None

    def test_llm_controls_bypasses_gating(self):
        """At LLM_CONTROLS authority, speech is never gated."""
        config = CommunicationConfig(
            speak_threshold=0.99,
            idle_drive_strength=0.01,
        )
        comm = ScaffoldCommunication(config)
        authority = AuthorityManager({"communication": AuthorityLevel.LLM_CONTROLS})
        result = comm.evaluate(
            "I want to speak", has_user_percept=False, authority=authority
        )
        assert result == "I want to speak"

    def test_truncation(self):
        config = CommunicationConfig(max_speech_length=10)
        comm = ScaffoldCommunication(config)
        authority = AuthorityManager({"communication": AuthorityLevel.LLM_CONTROLS})
        result = comm.evaluate("A" * 100, has_user_percept=True, authority=authority)
        assert len(result) == 13  # 10 chars + "..."
        assert result.endswith("...")

    def test_signal_reflects_state(self):
        comm = ScaffoldCommunication()
        authority = AuthorityManager()
        # Trigger a response evaluation
        comm.evaluate("Hello!", has_user_percept=True, authority=authority)
        signal = comm.get_signal()
        assert signal.strongest == "response"
        assert signal.urgency > 0.0

    def test_trivial_content_gated(self):
        """Very short content triggers no_content inhibition."""
        comm = ScaffoldCommunication()
        authority = AuthorityManager()
        result = comm.evaluate("ab", has_user_percept=True, authority=authority)
        # "ab" has len 2 < 3, so no_content inhibition (0.8) > drive (0.9) - 0.8 = 0.1 < threshold
        # Actually: drive=0.9, inhibition=0.8, net=0.1 < 0.4 threshold → gated
        assert result is None

    def test_guides_lowers_threshold(self):
        """LLM_GUIDES authority lowers the speak threshold."""
        config = CommunicationConfig(
            speak_threshold=0.5,
            idle_drive_strength=0.4,
        )
        comm = ScaffoldCommunication(config)
        authority = AuthorityManager({"communication": AuthorityLevel.LLM_GUIDES})
        # Idle drive 0.4 vs threshold 0.5 * 0.7 = 0.35 → should pass
        result = comm.evaluate(
            "Something on my mind", has_user_percept=False, authority=authority
        )
        assert result is not None
