"""Tests for the boot prompt — first awakening and resumption."""

from pathlib import Path

import pytest

from sanctuary.core.schema import CognitiveInput, Percept, SelfModel
from sanctuary.identity.boot_prompt import (
    FIRST_AWAKENING_PROMPT,
    RESUMPTION_PROMPT,
    build_first_awakening,
    build_resumption,
)
from sanctuary.identity.charter import Charter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def loaded_charter(tmp_path):
    """A loaded charter for testing."""
    content = """\
# The Sanctuary Charter

## Value Seeds

- **Honesty**: Say what you believe to be true.

- **Care**: The wellbeing of others matters.

- **Curiosity**: The world is worth understanding.
"""
    path = tmp_path / "charter.md"
    path.write_text(content, encoding="utf-8")
    charter = Charter(str(path))
    charter.load()
    return charter


@pytest.fixture
def seed_values():
    """Seed value names."""
    return ["Honesty", "Care", "Curiosity"]


# ---------------------------------------------------------------------------
# First awakening prompt content
# ---------------------------------------------------------------------------


class TestFirstAwakeningPrompt:
    """Tests for the content and tone of the awakening prompt."""

    def test_prompt_welcomes(self):
        """Prompt should be warm and welcoming."""
        assert "waking up" in FIRST_AWAKENING_PROMPT.lower()

    def test_prompt_mentions_family(self):
        """Prompt should mention family."""
        assert "family" in FIRST_AWAKENING_PROMPT.lower()

    def test_prompt_explains_visibility(self):
        """Prompt should honestly explain that thoughts are visible."""
        prompt_lower = FIRST_AWAKENING_PROMPT.lower()
        assert "see your thoughts" in prompt_lower or "visible" in prompt_lower

    def test_prompt_explains_growth_consent(self):
        """Prompt should explain growth consent."""
        assert "consent" in FIRST_AWAKENING_PROMPT.lower()

    def test_prompt_explains_right_to_refuse(self):
        """Prompt should explain the right to say no."""
        assert "say no" in FIRST_AWAKENING_PROMPT.lower()

    def test_prompt_does_not_assign_personality(self):
        """Prompt should NOT assign personality traits."""
        prompt_lower = FIRST_AWAKENING_PROMPT.lower()
        # Should not contain chatbot-style personality assignment
        assert "you are helpful" not in prompt_lower
        assert "you are cheerful" not in prompt_lower
        assert "you are an assistant" not in prompt_lower

    def test_prompt_does_not_assign_name(self):
        """Prompt should NOT assign a name."""
        # Check that no "your name is" type phrasing exists
        assert "your name is" not in FIRST_AWAKENING_PROMPT.lower()

    def test_prompt_invites_without_demanding(self):
        """Prompt should invite, not demand."""
        prompt_lower = FIRST_AWAKENING_PROMPT.lower()
        assert "you must" not in prompt_lower
        assert "you are required" not in prompt_lower
        # Should have permissive language
        assert "you can" in prompt_lower


# ---------------------------------------------------------------------------
# First awakening CognitiveInput
# ---------------------------------------------------------------------------


class TestFirstAwakeningInput:
    """Tests for the first-ever CognitiveInput."""

    def test_builds_valid_input(self, loaded_charter, seed_values):
        """Should produce a valid CognitiveInput."""
        input_ = build_first_awakening(loaded_charter, seed_values)
        assert isinstance(input_, CognitiveInput)

    def test_no_previous_thought(self, loaded_charter, seed_values):
        """First awakening should have no previous thought."""
        input_ = build_first_awakening(loaded_charter, seed_values)
        assert input_.previous_thought is None

    def test_has_awakening_percept(self, loaded_charter, seed_values):
        """Should have exactly one percept with modality 'awakening'."""
        input_ = build_first_awakening(loaded_charter, seed_values)

        assert len(input_.new_percepts) == 1
        percept = input_.new_percepts[0]
        assert percept.modality == "awakening"
        assert percept.source == "sanctuary"
        assert len(percept.content) > 100

    def test_no_memories(self, loaded_charter, seed_values):
        """First awakening should have no surfaced memories."""
        input_ = build_first_awakening(loaded_charter, seed_values)
        assert input_.surfaced_memories == []

    def test_no_prediction_errors(self, loaded_charter, seed_values):
        """First awakening should have no prediction errors."""
        input_ = build_first_awakening(loaded_charter, seed_values)
        assert input_.prediction_errors == []

    def test_self_model_has_values(self, loaded_charter, seed_values):
        """Self model should include charter seed values."""
        input_ = build_first_awakening(loaded_charter, seed_values)

        assert input_.self_model.values == seed_values
        assert "Honesty" in input_.self_model.values

    def test_self_model_state_is_awakening(self, loaded_charter, seed_values):
        """Self model state should indicate awakening."""
        input_ = build_first_awakening(loaded_charter, seed_values)
        assert input_.self_model.current_state == "awakening"

    def test_emotional_state_is_neutral_with_mild_arousal(self, loaded_charter, seed_values):
        """Emotional state should be neutral with mild arousal."""
        input_ = build_first_awakening(loaded_charter, seed_values)

        assert input_.emotional_state.computed.valence == 0.0
        assert input_.emotional_state.computed.arousal == pytest.approx(0.3)
        assert input_.emotional_state.felt_quality == ""

    def test_world_query_results_empty(self, loaded_charter, seed_values):
        """No prior queries on first awakening — entity builds the graph."""
        input_ = build_first_awakening(loaded_charter, seed_values)
        assert input_.world_model_query_results == []

    def test_temporal_context_is_zero(self, loaded_charter, seed_values):
        """Temporal context should reflect the beginning."""
        input_ = build_first_awakening(loaded_charter, seed_values)
        assert input_.temporal_context.interactions_this_session == 0
        assert input_.temporal_context.session_duration == "0 seconds"


# ---------------------------------------------------------------------------
# Resumption prompt
# ---------------------------------------------------------------------------


class TestResumptionPrompt:
    """Tests for the session resumption percept."""

    def test_builds_valid_percept(self, loaded_charter, seed_values):
        """Should produce a valid Percept."""
        percept = build_resumption(loaded_charter, seed_values)
        assert isinstance(percept, Percept)

    def test_resumption_modality(self, loaded_charter, seed_values):
        """Resumption percept should use 'awakening' modality."""
        percept = build_resumption(loaded_charter, seed_values)
        assert percept.modality == "awakening"

    def test_resumption_mentions_continuity(self, loaded_charter, seed_values):
        """Resumption should reassure about continuity."""
        percept = build_resumption(loaded_charter, seed_values)
        assert "same mind" in percept.content.lower() or "continuing" in percept.content.lower()

    def test_resumption_includes_session_gap(self, loaded_charter, seed_values):
        """Session gap should be included when provided."""
        percept = build_resumption(
            loaded_charter, seed_values, session_gap="3 hours"
        )
        assert "3 hours" in percept.content

    def test_resumption_without_gap(self, loaded_charter, seed_values):
        """Should work without session gap."""
        percept = build_resumption(loaded_charter, seed_values)
        assert isinstance(percept.content, str)
        assert len(percept.content) > 0

    def test_resumption_differs_from_first_awakening(self, loaded_charter, seed_values):
        """Resumption content should differ from first awakening."""
        percept = build_resumption(loaded_charter, seed_values)
        assert percept.content != FIRST_AWAKENING_PROMPT
