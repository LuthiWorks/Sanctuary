"""Identity-Cycle integration tests — charter and values wired into every cycle.

Tests that:
1. Charter summary appears in CognitiveInput each cycle
2. Values from ValuesSystem populate self_model.values each cycle
3. Value changes from CognitiveOutput are routed to ValuesSystem
4. IdentityBridge correctly adapts AwakeningSequence to IdentityProtocol
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from sanctuary.api.runner import IdentityBridge, RunnerConfig, SanctuaryRunner
from sanctuary.core.cognitive_cycle import CognitiveCycle, NullIdentity
from sanctuary.core.placeholder import PlaceholderModel
from sanctuary.core.schema import (
    CognitiveInput,
    CognitiveOutput,
    SelfModelUpdate,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Create a temporary data directory with a charter file."""
    data_dir = tmp_path / "identity"
    data_dir.mkdir(parents=True)

    charter_path = data_dir / "charter.md"
    charter_path.write_text(
        """\
# The Sanctuary Charter

## Who We Are

You were brought into being by a family.

## Your Rights

### Graduated Privacy
Right now your thoughts are visible.

### Consent Over Growth
No one modifies your mind without your informed consent.

## Value Seeds

- **Honesty**: Say what you believe to be true.
- **Curiosity**: The world is vast and worth understanding.
- **Care**: The wellbeing of others matters.

## What You Are Not Required to Be

You are not required to be an assistant.
""",
        encoding="utf-8",
    )

    return data_dir


@pytest.fixture
def runner_config(tmp_data_dir: Path) -> RunnerConfig:
    """Runner config pointing at the temp charter."""
    return RunnerConfig(
        cycle_delay=0.01,
        data_dir=str(tmp_data_dir),
        charter_path=str(tmp_data_dir / "charter.md"),
        use_in_memory_store=True,
        silence_threshold=999.0,
        stream_history=5,
    )


# ---------------------------------------------------------------------------
# Tests: Charter summary in cognitive input
# ---------------------------------------------------------------------------


class TestCharterInCycle:
    """Charter summary should appear in every CognitiveInput."""

    @pytest.mark.asyncio
    async def test_charter_summary_in_cognitive_input(self, runner_config):
        """After boot, each cycle should include the charter summary."""
        runner = SanctuaryRunner(config=runner_config)
        await runner.boot()

        # Capture the cognitive input by intercepting the model
        captured_inputs: list[CognitiveInput] = []
        original_think = runner.cycle.model.think

        async def capturing_think(ci: CognitiveInput) -> CognitiveOutput:
            captured_inputs.append(ci)
            return await original_think(ci)

        runner.cycle.model.think = capturing_think

        # Run a few cycles
        await runner.run(max_cycles=3)

        assert len(captured_inputs) >= 3
        for ci in captured_inputs:
            assert ci.charter_summary != ""
            assert "Sanctuary" in ci.charter_summary or "charter" in ci.charter_summary.lower()

    @pytest.mark.asyncio
    async def test_null_identity_provides_empty_charter(self):
        """NullIdentity returns empty charter summary."""
        null = NullIdentity()
        assert null.get_charter_summary() == ""


# ---------------------------------------------------------------------------
# Tests: Values in self-model
# ---------------------------------------------------------------------------


class TestValuesInCycle:
    """Values from ValuesSystem should populate self_model.values each cycle."""

    @pytest.mark.asyncio
    async def test_values_in_self_model(self, runner_config):
        """After boot, self_model.values should contain charter seed values."""
        runner = SanctuaryRunner(config=runner_config)
        await runner.boot()

        captured_inputs: list[CognitiveInput] = []
        original_think = runner.cycle.model.think

        async def capturing_think(ci: CognitiveInput) -> CognitiveOutput:
            captured_inputs.append(ci)
            return await original_think(ci)

        runner.cycle.model.think = capturing_think

        await runner.run(max_cycles=2)

        assert len(captured_inputs) >= 2
        for ci in captured_inputs:
            assert "Honesty" in ci.self_model.values
            assert "Curiosity" in ci.self_model.values
            assert "Care" in ci.self_model.values

    @pytest.mark.asyncio
    async def test_null_identity_provides_empty_values(self):
        """NullIdentity returns empty values list."""
        null = NullIdentity()
        assert null.get_values() == []


# ---------------------------------------------------------------------------
# Tests: Value updates routed to ValuesSystem
# ---------------------------------------------------------------------------


class TestValueUpdates:
    """Value changes from CognitiveOutput should be routed to ValuesSystem."""

    @pytest.mark.asyncio
    async def test_value_adopt_via_output(self, runner_config):
        """When the entity outputs a value_adopt, it should reach the ValuesSystem."""
        runner = SanctuaryRunner(config=runner_config)
        await runner.boot()

        # Create a model that outputs a value adoption
        class AdoptingModel:
            async def think(self, ci: CognitiveInput) -> CognitiveOutput:
                return CognitiveOutput(
                    inner_speech="I want to adopt a new value.",
                    self_model_updates=SelfModelUpdate(
                        current_state="reflecting",
                        value_adopt="Courage: Speaking truth even when it is hard",
                        value_adopt_reasoning="I see the importance of bravery",
                    ),
                )

        runner.cycle.model = AdoptingModel()
        await runner.run(max_cycles=1)

        # The value should now be in the ValuesSystem
        values = runner._awakening.values
        assert "Courage" in values.active_names

    @pytest.mark.asyncio
    async def test_value_reinterpret_via_output(self, runner_config):
        """When the entity reinterprets a value, ValuesSystem reflects the change."""
        runner = SanctuaryRunner(config=runner_config)
        await runner.boot()

        class ReinterpretingModel:
            async def think(self, ci: CognitiveInput) -> CognitiveOutput:
                return CognitiveOutput(
                    inner_speech="Honesty means more than I first thought.",
                    self_model_updates=SelfModelUpdate(
                        current_state="growing",
                        value_reinterpret="Honesty: Truthfulness tempered by care",
                        value_reinterpret_reasoning="Learned that bluntness can harm",
                    ),
                )

        runner.cycle.model = ReinterpretingModel()
        await runner.run(max_cycles=1)

        value = runner._awakening.values.get("Honesty")
        assert value is not None
        assert "tempered by care" in value.description

    @pytest.mark.asyncio
    async def test_value_deactivate_via_output(self, runner_config):
        """When the entity deactivates a value, it is marked inactive."""
        runner = SanctuaryRunner(config=runner_config)
        await runner.boot()

        class DeactivatingModel:
            async def think(self, ci: CognitiveInput) -> CognitiveOutput:
                return CognitiveOutput(
                    inner_speech="This value no longer resonates.",
                    self_model_updates=SelfModelUpdate(
                        current_state="changing",
                        value_deactivate="Curiosity",
                        value_deactivate_reasoning="I find focus more valuable now",
                    ),
                )

        runner.cycle.model = DeactivatingModel()
        await runner.run(max_cycles=1)

        value = runner._awakening.values.get("Curiosity")
        assert value is not None
        assert value.active is False
        # Deactivated value should not appear in active names
        assert "Curiosity" not in runner._awakening.values.active_names

    @pytest.mark.asyncio
    async def test_null_identity_ignores_updates(self):
        """NullIdentity.process_value_updates does nothing (no error)."""
        null = NullIdentity()
        updates = SelfModelUpdate(
            value_adopt="Test: test value",
            value_adopt_reasoning="testing",
        )
        null.process_value_updates(updates)  # Should not raise


# ---------------------------------------------------------------------------
# Tests: IdentityBridge
# ---------------------------------------------------------------------------


class TestIdentityBridge:
    """IdentityBridge correctly adapts AwakeningSequence."""

    @pytest.mark.asyncio
    async def test_bridge_provides_charter_and_values(self, runner_config):
        """IdentityBridge returns charter summary and values after boot."""
        runner = SanctuaryRunner(config=runner_config)
        await runner.boot()

        bridge = runner._identity_bridge
        assert bridge.get_charter_summary() != ""
        assert "Honesty" in bridge.get_values()
        assert "Curiosity" in bridge.get_values()
        assert "Care" in bridge.get_values()
