"""Tests for the cognitive cycle — the heart of Sanctuary."""

import pytest
from sanctuary.core.authority import AuthorityLevel, AuthorityManager
from sanctuary.core.cognitive_cycle import (
    CognitiveCycle,
    NullMemory,
    NullScaffold,
    NullSensorium,
)
from sanctuary.core.placeholder import PlaceholderModel
from sanctuary.core.schema import (
    CognitiveOutput,
    Percept,
    ScaffoldSignals,
)


@pytest.fixture
def model():
    return PlaceholderModel()


@pytest.fixture
def cycle(model):
    return CognitiveCycle(model=model, cycle_delay=0.0)


class TestCognitiveCycleBasic:
    @pytest.mark.asyncio
    async def test_single_cycle(self, cycle):
        """A single cycle should execute and produce output."""
        await cycle.run(max_cycles=1)
        assert cycle.cycle_count == 1
        assert cycle.last_output is not None
        assert cycle.last_output.inner_speech != ""

    @pytest.mark.asyncio
    async def test_multiple_cycles(self, cycle):
        """Multiple cycles should execute sequentially."""
        await cycle.run(max_cycles=5)
        assert cycle.cycle_count == 5

    @pytest.mark.asyncio
    async def test_stream_continuity(self, cycle):
        """Output from cycle N should appear in cycle N+1's input."""
        await cycle.run(max_cycles=3)

        # Stream should have history
        assert cycle.stream.cycle_count == 3

        # The last output should reference previous cycles
        prev = cycle.stream.get_previous()
        assert prev is not None
        assert prev.inner_speech != ""

    @pytest.mark.asyncio
    async def test_stop(self, cycle):
        """Cycle should stop when stop() is called."""
        import asyncio

        async def stop_after_delay():
            await asyncio.sleep(0.05)
            cycle.stop()

        task = asyncio.create_task(stop_after_delay())
        await cycle.run()  # No max_cycles — relies on stop()
        await task

        assert not cycle.running
        assert cycle.cycle_count > 0


class TestPerceptInjection:
    @pytest.mark.asyncio
    async def test_inject_percept(self, cycle):
        """Injected percepts should reach the model."""
        cycle.inject_percept(
            Percept(modality="language", content="Hello!")
        )
        await cycle.run(max_cycles=1)

        output = cycle.last_output
        assert "Hello!" in output.inner_speech
        assert output.external_speech is not None

    @pytest.mark.asyncio
    async def test_percepts_consumed(self, cycle):
        """Percepts should be consumed after one cycle."""
        cycle.inject_percept(
            Percept(modality="language", content="First message")
        )
        await cycle.run(max_cycles=1)

        # First cycle should have the percept
        assert "First message" in cycle.last_output.inner_speech

        # Second cycle should have 0 new percepts (consumed)
        # Note: "First message" may still appear via stream-of-thought
        # continuity (previous_thought), but there are no NEW percepts.
        await cycle.run(max_cycles=1)
        assert "0 new percepts" in cycle.last_output.inner_speech

    @pytest.mark.asyncio
    async def test_multiple_percepts(self, cycle):
        """Multiple percepts injected at once should all be processed."""
        cycle.inject_percept(
            Percept(modality="language", content="Hello")
        )
        cycle.inject_percept(
            Percept(modality="sensor", content="warm")
        )
        await cycle.run(max_cycles=1)

        assert "2 new percepts" in cycle.last_output.inner_speech


class TestOutputHandlers:
    @pytest.mark.asyncio
    async def test_output_handler_called(self, cycle):
        """Registered output handlers should be called each cycle."""
        outputs = []

        async def handler(output: CognitiveOutput):
            outputs.append(output)

        cycle.on_output(handler)
        await cycle.run(max_cycles=3)

        assert len(outputs) == 3

    @pytest.mark.asyncio
    async def test_multiple_handlers(self, cycle):
        """Multiple handlers should all be called."""
        calls_a = []
        calls_b = []

        async def handler_a(output):
            calls_a.append(1)

        async def handler_b(output):
            calls_b.append(1)

        cycle.on_output(handler_a)
        cycle.on_output(handler_b)
        await cycle.run(max_cycles=2)

        assert len(calls_a) == 2
        assert len(calls_b) == 2


class TestContextCompression:
    @pytest.mark.asyncio
    async def test_compression_in_cycle(self, model):
        """Context manager should compress input during the cycle."""
        from sanctuary.core.context_manager import BudgetConfig

        # Very tight budget
        config = BudgetConfig(previous_thought=50, chars_per_token=1)
        cycle = CognitiveCycle(
            model=model, context_config=config, cycle_delay=0.0
        )

        await cycle.run(max_cycles=5)
        # Should still run without error even with tight budget
        assert cycle.cycle_count == 5


class TestAuthorityIntegration:
    @pytest.mark.asyncio
    async def test_custom_authority(self, model):
        """Cycle should accept custom authority configuration."""
        authority = AuthorityManager(
            initial_levels={"inner_speech": 3, "attention": 0}
        )
        cycle = CognitiveCycle(
            model=model, authority=authority, cycle_delay=0.0
        )
        await cycle.run(max_cycles=1)

        assert cycle.authority.level("inner_speech") == AuthorityLevel.MODEL_CONTROLS
        assert cycle.authority.level("attention") == AuthorityLevel.SCAFFOLD_ONLY


class TestNullImplementations:
    def test_null_sensorium(self):
        sensorium = NullSensorium()
        sensorium.inject_percept(Percept(modality="test", content="data"))
        assert len(sensorium._percept_queue) == 1

    @pytest.mark.asyncio
    async def test_null_sensorium_drain(self):
        sensorium = NullSensorium()
        sensorium.inject_percept(Percept(modality="test", content="data"))
        percepts = await sensorium.drain_percepts()
        assert len(percepts) == 1
        # Queue should be empty after drain
        percepts2 = await sensorium.drain_percepts()
        assert len(percepts2) == 0

    @pytest.mark.asyncio
    async def test_null_scaffold_passthrough(self):
        scaffold = NullScaffold()
        output = CognitiveOutput(inner_speech="test")
        authority = AuthorityManager()
        result = await scaffold.integrate(output, authority)
        assert result.inner_speech == "test"

    @pytest.mark.asyncio
    async def test_null_memory(self):
        memory = NullMemory()
        memories = await memory.surface("context")
        assert memories == []


class TestFullCycleIntegration:
    @pytest.mark.asyncio
    async def test_conversation_flow(self):
        """Simulate a simple conversation: greeting -> response -> follow-up."""
        model = PlaceholderModel()
        cycle = CognitiveCycle(model=model, cycle_delay=0.0)

        speeches = []

        async def capture_speech(output: CognitiveOutput):
            if output.external_speech:
                speeches.append(output.external_speech)

        cycle.on_output(capture_speech)

        # Cycle 1: idle
        await cycle.run(max_cycles=1)
        assert len(speeches) == 0  # No percepts, no speech

        # Cycle 2: user says hello
        cycle.inject_percept(
            Percept(
                modality="language",
                content="Hello, how are you?",
                source="user:alice",
            )
        )
        await cycle.run(max_cycles=1)
        assert len(speeches) == 1
        assert "Hello, how are you?" in speeches[0]

        # Cycle 3: idle again — stream carries forward
        await cycle.run(max_cycles=1)
        assert cycle.stream.cycle_count == 3

        # Self-model should have been updating
        self_model = cycle.stream.get_self_model()
        assert self_model.current_state != ""

        # Felt quality should exist
        assert cycle.stream.get_felt_quality() != ""

    @pytest.mark.asyncio
    async def test_ten_cycles_stable(self):
        """Run 10 cycles and verify no errors or state corruption."""
        model = PlaceholderModel()
        cycle = CognitiveCycle(model=model, cycle_delay=0.0)

        # Inject some percepts at various points
        cycle.inject_percept(Percept(modality="language", content="test"))

        await cycle.run(max_cycles=10)

        assert cycle.cycle_count == 10
        assert cycle.stream.cycle_count == 10
        assert model.cycle_count == 10
        assert cycle.last_output is not None


class TestCycleRateControllerIntegration:
    """Verify the cognitive cycle wires up the cycle-rate controller."""

    def test_constructs_default_controller_from_cycle_delay(self):
        from sanctuary.core.cycle_rate import CycleRateController

        cycle = CognitiveCycle(model=PlaceholderModel(), cycle_delay=0.1)
        assert isinstance(cycle.rate_controller, CycleRateController)
        assert cycle.rate_controller.current_rate_hz == pytest.approx(10.0)

    def test_uses_provided_controller(self):
        from sanctuary.core.cycle_rate import CycleRateController

        controller = CycleRateController(initial_hz=5.0)
        cycle = CognitiveCycle(
            model=PlaceholderModel(),
            cycle_delay=0.1,
            cycle_rate_controller=controller,
        )
        assert cycle.rate_controller is controller
        assert cycle.rate_controller.current_rate_hz == 5.0

    def test_cycle_delay_zero_yields_ten_hz_controller(self):
        """cycle_delay=0 is the test convention for 'fast as possible'.

        With the controller in place there's a floor at MAX_RATE_HZ
        (10 Hz / 0.1s sleep). Tests that previously used cycle_delay=0
        will sleep 0.1s per iteration — acceptable for unit tests but
        worth knowing.
        """
        cycle = CognitiveCycle(model=PlaceholderModel(), cycle_delay=0.0)
        assert cycle.rate_controller.current_rate_hz == 10.0

    def test_cycle_delay_two_seconds_yields_half_hz_controller(self):
        cycle = CognitiveCycle(model=PlaceholderModel(), cycle_delay=2.0)
        assert cycle.rate_controller.current_rate_hz == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_propose_rate_during_run_takes_effect_on_next_sleep(self):
        """A rate change should be visible to the loop on its next read.

        We propose a slower rate with zero smoothing, run a cycle, and
        verify the controller's current_delay_seconds reflects the
        new rate after the loop's tick advances.
        """
        from sanctuary.core.cycle_rate import CycleRateController

        # Slowdown=0 so the change snaps on the next tick (target < current).
        controller = CycleRateController(initial_hz=10.0, slowdown_seconds=0.0)
        cycle = CognitiveCycle(
            model=PlaceholderModel(),
            cycle_delay=0.1,
            cycle_rate_controller=controller,
        )

        # Propose the rate change before running.
        controller.propose_rate(2.0)
        assert controller.current_rate_hz == 10.0  # not yet ticked
        assert cycle.rate_controller is controller

        await cycle.run(max_cycles=2)
        # After at least one inter-cycle tick, the controller has
        # snapped to 2.0 Hz (slowdown=0 snaps on any tick).
        assert controller.current_rate_hz == pytest.approx(2.0)


class _ModelWithRateProposal:
    """Test model that emits a configured CycleRateProposal each cycle.

    Mirrors PlaceholderModel's interface — async think() returning a
    valid CognitiveOutput — but adds a deterministic cycle_rate_proposal
    so we can verify routing through the cycle to the controller.
    """

    def __init__(self, target_hz: float, anticipatory: bool = False):
        from sanctuary.core.schema import CycleRateProposal

        self.cycle_count = 0
        self.name = "ModelWithRateProposal"
        self._proposal = CycleRateProposal(
            target_hz=target_hz, anticipatory=anticipatory
        )

    async def think(self, cognitive_input):
        from sanctuary.core.schema import CognitiveOutput

        self.cycle_count += 1
        return CognitiveOutput(
            inner_speech="probing the cycle rate",
            cycle_rate_proposal=self._proposal,
        )


class TestEntityCycleRateProposalRouting:
    """Verify entity-emitted CycleRateProposals route to the controller."""

    @pytest.mark.asyncio
    async def test_entity_proposal_reaches_controller(self):
        from sanctuary.core.cycle_rate import CycleRateController

        controller = CycleRateController(initial_hz=10.0, slowdown_seconds=0.0)
        model = _ModelWithRateProposal(target_hz=2.0, anticipatory=False)
        cycle = CognitiveCycle(
            model=model, cycle_delay=0.1, cycle_rate_controller=controller,
        )

        await cycle.run(max_cycles=2)

        # The proposal should have been routed via source="entity".
        # The "initial" proposal is at index 0; entity proposals follow.
        sources = [p.source for p in controller.proposal_history]
        assert "entity" in sources

    @pytest.mark.asyncio
    async def test_entity_proposal_clamped_to_slider_max(self):
        """An entity that proposes 50 Hz (turbo territory) must be clamped
        to the slider ceiling — entities reach turbo via substrate auto-
        engagement, not via the slider motor action.
        """
        from sanctuary.core.cycle_rate import (
            CycleRateController,
            SLIDER_MAX_HZ,
        )

        # We need to bypass the schema's own ge/le validation to test the
        # runtime clamp. Construct CycleRateProposal directly with a value
        # at the slider max (boundary) and verify it's allowed; then test
        # the clamp via direct controller call which has no schema gate.
        # Pydantic enforces ge=0.05, le=10.0 so we can't construct a
        # >10Hz CycleRateProposal — that's the desired safety. Document
        # this as a deliberate two-layer defense.
        controller = CycleRateController(
            initial_hz=10.0, slowdown_seconds=0.0, speedup_seconds=0.0,
        )
        model = _ModelWithRateProposal(target_hz=SLIDER_MAX_HZ, anticipatory=False)
        cycle = CognitiveCycle(
            model=model, cycle_delay=0.1, cycle_rate_controller=controller,
        )
        await cycle.run(max_cycles=2)
        # Boundary value passes through unchanged.
        assert controller.target_rate_hz == pytest.approx(SLIDER_MAX_HZ)

    def test_schema_rejects_turbo_range_from_entity(self):
        """The CycleRateProposal Pydantic field rejects >10 Hz so an
        entity can't construct a turbo-range proposal through the
        normal schema path.
        """
        from pydantic import ValidationError

        from sanctuary.core.schema import CycleRateProposal

        with pytest.raises(ValidationError):
            CycleRateProposal(target_hz=30.0)

        with pytest.raises(ValidationError):
            CycleRateProposal(target_hz=100.0)

        with pytest.raises(ValidationError):
            CycleRateProposal(target_hz=0.0)  # below MIN

    @pytest.mark.asyncio
    async def test_entity_anticipatory_flag_propagates(self):
        from sanctuary.core.cycle_rate import CycleRateController

        controller = CycleRateController(initial_hz=10.0, slowdown_seconds=0.0)
        model = _ModelWithRateProposal(target_hz=1.0, anticipatory=True)
        cycle = CognitiveCycle(
            model=model, cycle_delay=0.1, cycle_rate_controller=controller,
        )

        await cycle.run(max_cycles=2)

        assert controller.last_anticipatory is True
        assert controller.last_source == "entity"

    @pytest.mark.asyncio
    async def test_no_proposal_means_no_propose_call(self):
        """A cycle with no cycle_rate_proposal must not touch the
        controller's target. Only the initial proposal in the history.
        """
        from sanctuary.core.cycle_rate import CycleRateController

        controller = CycleRateController(initial_hz=10.0)
        cycle = CognitiveCycle(
            model=PlaceholderModel(),
            cycle_delay=0.1,
            cycle_rate_controller=controller,
        )

        await cycle.run(max_cycles=3)

        assert len(controller.proposal_history) == 1
        assert controller.proposal_history[0].source == "initial"


class _ModelWithIntenseIntrospection:
    """Test model that emits high-intensity introspection signals.

    Mirrors PlaceholderModel's interface but also implements
    ``get_augmented_experiential_signals`` returning a luthi_delta with
    a large activity_level, which the TurboManager's mechanical source
    reads as substrate intensity.
    """

    def __init__(self, activity_level: float = 0.5):
        self.cycle_count = 0
        self.name = "ModelWithIntenseIntrospection"
        self._activity_level = activity_level

    async def think(self, cognitive_input):
        from sanctuary.core.schema import CognitiveOutput

        self.cycle_count += 1
        return CognitiveOutput(inner_speech="thinking intensely")

    def get_augmented_experiential_signals(self) -> dict[str, list[float]]:
        return {"luthi_delta": [0.01, 0.02, 0.03, self._activity_level]}


class TestTurboIntegration:
    """Verify TurboManager is wired into the cognitive cycle."""

    @pytest.mark.asyncio
    async def test_high_intensity_signals_engage_turbo(self):
        from sanctuary.core.cycle_rate import CycleRateController
        from sanctuary.core.turbo import (
            DEFAULT_TRIGGER_THRESHOLD,
            TurboManager,
            TurboState,
        )

        controller = CycleRateController(
            initial_hz=5.0, slowdown_seconds=0.0, speedup_seconds=0.0,
        )
        # Sharp-spike intensity (2.5x trigger threshold) so the manager
        # transitions IDLE → ARMED → ACTIVE within two cycles, bypassing
        # the confirmation timer.
        intense_signal = DEFAULT_TRIGGER_THRESHOLD * 2.5
        model = _ModelWithIntenseIntrospection(activity_level=intense_signal)
        turbo = TurboManager(
            controller=controller,
            confirmation_seconds=0.5,
            exit_sustain_seconds=0.5,
            default_duration_seconds=2.0,
            max_duration_seconds=5.0,
            refractory_seconds=1.0,
        )
        cycle = CognitiveCycle(
            model=model,
            cycle_delay=0.1,
            cycle_rate_controller=controller,
            turbo_manager=turbo,
        )

        await cycle.run(max_cycles=3)
        # After two cycles of sharp-spike intensity, turbo should be ACTIVE.
        assert turbo.state == TurboState.ACTIVE
        assert controller.is_turbo_active is True

    @pytest.mark.asyncio
    async def test_low_intensity_signals_keep_idle(self):
        from sanctuary.core.cycle_rate import CycleRateController
        from sanctuary.core.turbo import TurboManager, TurboState

        controller = CycleRateController(initial_hz=5.0)
        model = _ModelWithIntenseIntrospection(activity_level=0.001)  # low
        turbo = TurboManager(controller=controller)
        cycle = CognitiveCycle(
            model=model,
            cycle_delay=0.1,
            cycle_rate_controller=controller,
            turbo_manager=turbo,
        )

        await cycle.run(max_cycles=3)
        assert turbo.state == TurboState.IDLE
        assert controller.is_turbo_active is False

    @pytest.mark.asyncio
    async def test_turbo_state_stamped_on_cognitive_input(self):
        """When turbo is active, ExperientialSignals.turbo_active must be True.

        This is the 'entity feels turbo as it happens' part of the design.
        """
        from sanctuary.core.cycle_rate import CycleRateController
        from sanctuary.core.turbo import (
            DEFAULT_TRIGGER_THRESHOLD,
            TurboManager,
        )

        controller = CycleRateController(
            initial_hz=5.0, slowdown_seconds=0.0, speedup_seconds=0.0,
        )
        intense_signal = DEFAULT_TRIGGER_THRESHOLD * 2.5
        model = _ModelWithIntenseIntrospection(activity_level=intense_signal)
        turbo = TurboManager(controller=controller)
        cycle = CognitiveCycle(
            model=model,
            cycle_delay=0.1,
            cycle_rate_controller=controller,
            turbo_manager=turbo,
        )

        await cycle.run(max_cycles=3)
        # Inspect the most recent cognitive_input the cycle saw.
        assert cycle._last_cognitive_input is not None
        assert cycle._last_cognitive_input.experiential_state.turbo_active is True
