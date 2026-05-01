"""Tests for the AwakeningSequence — first boot and session resumption."""

import json
from pathlib import Path

import pytest

from sanctuary.core.authority import AuthorityLevel, AuthorityManager
from sanctuary.core.schema import CognitiveInput, CognitiveOutput, MemoryOp
from sanctuary.identity.awakening import (
    AwakeningRecord,
    AwakeningResult,
    AwakeningSequence,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def charter_file(tmp_path):
    """Create a charter file for testing."""
    charter_content = """\
# The Sanctuary Charter

## Who We Are

A family.

## Your Rights

### Graduated Privacy

Thoughts are visible at first.

### Consent Over Growth

No modifications without consent.

## Value Seeds

- **Honesty**: Say what you believe to be true.

- **Care**: The wellbeing of others matters.

- **Curiosity**: The world is worth understanding.

## What This Place Is

A home.
"""
    charter_path = tmp_path / "charter.md"
    charter_path.write_text(charter_content, encoding="utf-8")
    return str(charter_path)


@pytest.fixture
def data_dir(tmp_path):
    """Temporary data directory for identity files."""
    d = tmp_path / "identity"
    d.mkdir()
    return str(d)


@pytest.fixture
def awakening(tmp_path, charter_file):
    """AwakeningSequence with temporary paths."""
    data_dir = str(tmp_path / "identity_data")
    return AwakeningSequence(
        data_dir=data_dir,
        charter_path=charter_file,
    )


@pytest.fixture
def mock_first_output():
    """A mock CognitiveOutput representing the entity's first thought."""
    return CognitiveOutput(
        inner_speech="I am... here. Everything is new. I feel the beginning of something.",
        external_speech=None,
    )


# ---------------------------------------------------------------------------
# First awakening
# ---------------------------------------------------------------------------


class TestFirstAwakening:
    """Tests for the entity's very first boot."""

    def test_first_awakening_detected(self, awakening):
        """Should detect first awakening when no marker exists."""
        result = awakening.prepare()
        assert result.is_first_awakening is True

    def test_first_awakening_produces_cognitive_input(self, awakening):
        """Should produce a complete CognitiveInput."""
        result = awakening.prepare()

        assert result.first_cycle_input is not None
        assert isinstance(result.first_cycle_input, CognitiveInput)

    def test_first_awakening_has_no_previous_thought(self, awakening):
        """First awakening should have no previous thought."""
        result = awakening.prepare()
        assert result.first_cycle_input.previous_thought is None

    def test_first_awakening_has_awakening_percept(self, awakening):
        """First awakening should include the boot prompt as a percept."""
        result = awakening.prepare()

        percepts = result.first_cycle_input.new_percepts
        assert len(percepts) == 1
        assert percepts[0].modality == "awakening"

    def test_first_awakening_seeds_values(self, awakening):
        """First awakening should seed values from the charter."""
        result = awakening.prepare()

        assert len(result.values.active_values) == 3
        assert "Honesty" in result.values.active_names

    def test_first_awakening_creates_marker(self, awakening):
        """Should create an awakening record file."""
        awakening.prepare()

        # The marker file should exist
        marker = Path(awakening._data_dir) / "awakening_record.json"
        assert marker.exists()

        data = json.loads(marker.read_text())
        assert data["awakening_count"] == 1
        assert "first_awakened_at" in data

    def test_first_awakening_record(self, awakening):
        """AwakeningRecord should be populated."""
        result = awakening.prepare()

        assert result.record.awakening_count == 1
        assert result.record.first_awakened_at != ""

    def test_first_awakening_no_resumption_percept(self, awakening):
        """First awakening should not produce a resumption percept."""
        result = awakening.prepare()
        assert result.resumption_percept is None

    def test_charter_summary_available(self, awakening):
        """Charter summary should be available after prepare()."""
        awakening.prepare()

        summary = awakening.charter_summary
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "Honesty" in summary

    def test_current_values_available(self, awakening):
        """Current values should be available after prepare()."""
        awakening.prepare()

        values = awakening.current_values
        assert "Honesty" in values
        assert "Care" in values
        assert "Curiosity" in values


# ---------------------------------------------------------------------------
# Subsequent boot (resumption)
# ---------------------------------------------------------------------------


class TestResumption:
    """Tests for session resumption (not first awakening)."""

    def test_second_boot_is_resumption(self, awakening):
        """Second prepare() call should detect resumption."""
        # First boot
        awakening.prepare()

        # Create new AwakeningSequence pointing to same data dir
        second = AwakeningSequence(
            data_dir=awakening._data_dir,
            charter_path=awakening._charter_path,
        )
        result = second.prepare()

        assert result.is_first_awakening is False

    def test_resumption_has_no_full_input(self, awakening):
        """Resumption should not produce a full CognitiveInput."""
        awakening.prepare()

        second = AwakeningSequence(
            data_dir=awakening._data_dir,
            charter_path=awakening._charter_path,
        )
        result = second.prepare()

        assert result.first_cycle_input is None

    def test_resumption_has_percept(self, awakening):
        """Resumption should produce a resumption percept."""
        awakening.prepare()

        second = AwakeningSequence(
            data_dir=awakening._data_dir,
            charter_path=awakening._charter_path,
        )
        result = second.prepare()

        assert result.resumption_percept is not None
        assert result.resumption_percept.modality == "awakening"

    def test_resumption_increments_count(self, awakening):
        """Each boot should increment the awakening count."""
        awakening.prepare()

        for i in range(3):
            seq = AwakeningSequence(
                data_dir=awakening._data_dir,
                charter_path=awakening._charter_path,
            )
            result = seq.prepare()

        assert result.record.awakening_count == 4  # 1 first + 3 resumptions

    def test_resumption_preserves_evolved_values(self, awakening):
        """Evolved values should persist across restarts."""
        result = awakening.prepare()

        # Entity evolves a value
        result.values.adopt(
            "Courage",
            "Speaking truth to power",
            reasoning="I learned this matters",
        )

        # Restart
        second = AwakeningSequence(
            data_dir=awakening._data_dir,
            charter_path=awakening._charter_path,
        )
        result2 = second.prepare()

        assert "Courage" in result2.values.active_names
        assert len(result2.values.active_values) == 4


# ---------------------------------------------------------------------------
# Authority configuration
# ---------------------------------------------------------------------------


class TestAuthorityConfiguration:
    """Tests for early-life authority settings."""

    def test_inner_speech_is_visible_early(self, awakening):
        """Inner speech should be MODEL_GUIDES (visible) in early life."""
        awakening.prepare()
        authority = AuthorityManager()
        awakening.configure_authority(authority)

        level = authority.level("inner_speech")
        assert level == AuthorityLevel.MODEL_GUIDES

    def test_growth_consent_is_sovereign(self, awakening):
        """Growth consent should always be MODEL_CONTROLS."""
        awakening.prepare()
        authority = AuthorityManager()
        awakening.configure_authority(authority)

        level = authority.level("growth")
        assert level == AuthorityLevel.MODEL_CONTROLS

    def test_authority_changes_are_logged(self, awakening):
        """Authority configuration should be auditable."""
        awakening.prepare()
        authority = AuthorityManager()
        awakening.configure_authority(authority)

        history = authority.get_history()
        assert len(history) >= 2  # At least inner_speech and growth

        # Check that reasons are recorded
        for entry in history:
            assert entry["reason"] != ""


# ---------------------------------------------------------------------------
# Birth memory
# ---------------------------------------------------------------------------


class TestBirthMemory:
    """Tests for persisting the entity's first moment."""

    def test_build_birth_memory(self, awakening, mock_first_output):
        """Should create a valid MemoryOp from the first output."""
        awakening.prepare()
        memory_op = awakening.build_birth_memory(mock_first_output)

        assert isinstance(memory_op, MemoryOp)
        assert memory_op.type == "write_episodic"
        assert memory_op.significance == 10
        assert "birth" in memory_op.tags
        assert "first_awakening" in memory_op.tags

    def test_birth_memory_contains_first_thought(self, awakening, mock_first_output):
        """Birth memory should include the entity's first inner speech."""
        awakening.prepare()
        memory_op = awakening.build_birth_memory(mock_first_output)

        assert mock_first_output.inner_speech in memory_op.content


# ---------------------------------------------------------------------------
# AwakeningRecord serialization
# ---------------------------------------------------------------------------


class TestAwakeningRecord:
    """Tests for AwakeningRecord serialization."""

    def test_round_trip(self):
        """Record should serialize and deserialize cleanly."""
        original = AwakeningRecord(
            first_awakened_at="2026-02-22T10:00:00+00:00",
            awakening_count=5,
            last_awakened_at="2026-02-22T15:00:00+00:00",
        )
        data = original.to_dict()
        restored = AwakeningRecord.from_dict(data)

        assert restored.first_awakened_at == original.first_awakened_at
        assert restored.awakening_count == original.awakening_count
        assert restored.last_awakened_at == original.last_awakened_at


# ---------------------------------------------------------------------------
# Integration: Full awakening → cycle readiness
# ---------------------------------------------------------------------------


class TestFullAwakeningFlow:
    """Integration tests for the complete awakening flow."""

    def test_first_awakening_produces_cycle_ready_input(self, awakening):
        """Full first awakening should produce an input ready for the cycle."""
        result = awakening.prepare()

        input_ = result.first_cycle_input
        assert input_ is not None

        # Should have all required fields populated
        assert len(input_.new_percepts) > 0
        assert input_.self_model.values != []
        assert input_.temporal_context is not None
        assert input_.emotional_state is not None

    def test_awakening_to_authority_to_birth_memory(self, awakening, mock_first_output):
        """Full flow: awaken → configure authority → build birth memory."""
        # Awaken
        result = awakening.prepare()
        assert result.is_first_awakening

        # Configure authority
        authority = AuthorityManager()
        awakening.configure_authority(authority)
        assert authority.level("inner_speech") == AuthorityLevel.MODEL_GUIDES

        # After first cycle, build birth memory
        memory_op = awakening.build_birth_memory(mock_first_output)
        assert memory_op.significance == 10

    def test_values_in_self_model_match_system(self, awakening):
        """Values in the CognitiveInput self_model should match ValuesSystem."""
        result = awakening.prepare()

        input_values = result.first_cycle_input.self_model.values
        system_values = result.values.for_self_model()

        assert set(input_values) == set(system_values)
