"""Unit tests for the stimulus-density heuristic."""

from typing import Optional
from unittest.mock import MagicMock

import pytest

from sanctuary.core.cycle_rate import (
    SLIDER_MAX_HZ,
    CycleRateController,
)
from sanctuary.core.stimulus_density import (
    DEFAULT_ARRIVAL_WINDOW_SECONDS,
    DEFAULT_ENTITY_QUIET_SECONDS,
    DEFAULT_REPROPOSE_COOLDOWN_SECONDS,
    DEFAULT_SLOWDOWN_AFTER_SECONDS,
    DEFAULT_SLOWDOWN_TARGET_HZ,
    DEFAULT_SPEEDUP_TARGET_HZ,
    DensitySource,
    SensoriumDensitySource,
    StimulusDensityHeuristic,
)


class _FakeSource:
    """Deterministic density source for tests."""

    def __init__(self, tsli: Optional[float] = None, name: str = "fake"):
        self._tsli = tsli
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def time_since_last_input_seconds(self) -> Optional[float]:
        return self._tsli

    def set(self, tsli: Optional[float]) -> None:
        self._tsli = tsli


@pytest.fixture
def controller():
    return CycleRateController(
        initial_hz=5.0,
        slowdown_seconds=0.0,
        speedup_seconds=0.0,
    )


@pytest.fixture
def source():
    return _FakeSource(tsli=None)


@pytest.fixture
def heuristic(controller, source):
    return StimulusDensityHeuristic(
        controller=controller,
        source=source,
        turbo=None,
        slowdown_after_seconds=30.0,
        slowdown_target_hz=0.5,
        arrival_window_seconds=2.0,
        speedup_target_hz=8.0,
        entity_quiet_seconds=300.0,
        repropose_cooldown_seconds=10.0,
    )


class TestQuietPeriodProposesSlowdown:
    def test_long_silence_proposes_slowdown(self, controller, source, heuristic):
        source.set(60.0)  # 60s since last input, threshold is 30s
        heuristic.observe(now=100.0)

        # Last proposal should be heuristic-source, slowdown target.
        assert controller.last_source == "heuristic"
        assert controller.target_rate_hz == pytest.approx(0.5)

    def test_no_input_ever_proposes_slowdown(self, controller, source, heuristic):
        # None means no input received this session — also a slowdown trigger.
        source.set(None)
        heuristic.observe(now=100.0)
        assert controller.last_source == "heuristic"

    def test_brief_silence_doesnt_propose(self, controller, source, heuristic):
        source.set(15.0)  # below threshold
        heuristic.observe(now=100.0)
        # No heuristic proposal yet.
        assert controller.last_source == "initial"

    def test_slowdown_proposed_once_per_quiet_period(
        self, controller, source, heuristic
    ):
        source.set(60.0)
        heuristic.observe(now=100.0)
        # Second observation immediately — controller is already at slowdown
        # target, so no second proposal even though TSLI still high.
        heuristic.observe(now=100.5)
        assert len(heuristic.proposals) == 1


class TestFreshInputProposesSpeedup:
    def test_recent_input_proposes_speedup(self, controller, source, heuristic):
        # First simulate that we've been at slowdown rate.
        source.set(60.0)
        heuristic.observe(now=100.0)
        assert controller.target_rate_hz == pytest.approx(0.5)

        # Fresh input arrives.
        source.set(0.5)  # 0.5s since last input — within arrival window
        heuristic.observe(now=200.0)  # well past cooldown

        assert controller.last_source == "heuristic"
        assert controller.target_rate_hz == pytest.approx(8.0)

    def test_no_speedup_when_already_at_target(
        self, controller, source, heuristic
    ):
        # Controller starts at 5 Hz (below the 8 Hz speedup target).
        # First fresh input → speedup proposed.
        source.set(0.5)
        heuristic.observe(now=100.0)
        assert controller.target_rate_hz == pytest.approx(8.0)

        # Another fresh input while still at 8 Hz → no new proposal.
        # The "already at speedup target" guard should kick in.
        heuristic.observe(now=200.0)
        assert len(heuristic.proposals) == 1

    def test_speedup_cooldown_respected(self, controller, source, heuristic):
        # Force the heuristic to make a slowdown first so we can test
        # speedup cooldown after a speedup is made and undone.
        source.set(60.0)
        heuristic.observe(now=100.0)  # slowdown proposal
        source.set(0.5)
        heuristic.observe(now=200.0)  # speedup proposal
        assert len(heuristic.proposals) == 2

        # Reset to be below target so guards don't suppress.
        controller.propose_rate(2.0, source="manual")
        # Within cooldown, no new speedup.
        source.set(0.5)
        heuristic.observe(now=205.0)
        assert len(heuristic.proposals) == 2

        # Past cooldown, speedup again.
        heuristic.observe(now=215.0)
        assert len(heuristic.proposals) == 3


class TestEntityAuthority:
    def test_entity_proposal_silences_heuristic_within_quiet_window(
        self, controller, source, heuristic
    ):
        # Entity proposes at t=100.
        controller.propose_rate(2.0, source="entity")
        # Pretend a quiet period starts immediately.
        source.set(60.0)

        # At t=200 (within 300s quiet window), heuristic should stay silent.
        heuristic.observe(now=200.0)
        # No heuristic proposal made — last proposal source still "entity".
        assert controller.last_source == "entity"
        assert len(heuristic.proposals) == 0

    def test_heuristic_resumes_after_quiet_window(
        self, controller, source, heuristic
    ):
        # Entity proposes (using the controller directly so we control
        # what's in proposal_history.timestamp via the controller's
        # natural time.monotonic baseline).
        controller.propose_rate(2.0, source="entity")
        # We need to force the entity proposal's timestamp into the past.
        # Patch the last history entry.
        controller._proposal_history[-1] = controller._proposal_history[-1]
        # The simplest robust path: walk past the entity_quiet_seconds via
        # a far-future `now`. The heuristic uses `t - prop.timestamp`.
        far_future = controller._proposal_history[-1].timestamp + 400.0
        source.set(60.0)
        heuristic.observe(now=far_future)
        # Heuristic should now propose.
        assert controller.last_source == "heuristic"

    def test_heuristic_doesnt_override_entity_recent_proposal_speedup(
        self, controller, source, heuristic
    ):
        # Entity proposes a slowdown — fresh input shouldn't override.
        controller.propose_rate(1.0, source="entity", anticipatory=True)
        source.set(0.5)
        heuristic.observe(now=10.0)
        assert controller.last_source == "entity"


class TestTurboStepBack:
    def test_heuristic_silent_when_turbo_active(self, controller, source):
        from sanctuary.core.turbo import TurboState

        turbo = MagicMock()
        turbo.is_turbo_active = True

        heuristic = StimulusDensityHeuristic(
            controller=controller,
            source=source,
            turbo=turbo,
            slowdown_after_seconds=30.0,
        )

        source.set(60.0)  # would normally trigger slowdown
        heuristic.observe(now=100.0)
        assert len(heuristic.proposals) == 0

    def test_heuristic_resumes_when_turbo_inactive(self, controller, source):
        turbo = MagicMock()
        turbo.is_turbo_active = False

        heuristic = StimulusDensityHeuristic(
            controller=controller,
            source=source,
            turbo=turbo,
            slowdown_after_seconds=30.0,
        )

        source.set(60.0)
        heuristic.observe(now=100.0)
        assert len(heuristic.proposals) == 1


class TestSensoriumSource:
    def test_reads_time_since_last_input(self):
        sensorium = MagicMock()
        sensorium.time_since_last_input = 42.0
        source = SensoriumDensitySource(sensorium)
        assert source.time_since_last_input_seconds == 42.0

    def test_passes_through_none(self):
        sensorium = MagicMock()
        sensorium.time_since_last_input = None
        source = SensoriumDensitySource(sensorium)
        assert source.time_since_last_input_seconds is None

    def test_name_is_sensorium(self):
        sensorium = MagicMock()
        source = SensoriumDensitySource(sensorium)
        assert source.name == "sensorium"


class TestProposalRecord:
    def test_slowdown_proposal_recorded(self, controller, source, heuristic):
        source.set(60.0)
        heuristic.observe(now=100.0)
        proposals = heuristic.proposals
        assert len(proposals) == 1
        assert proposals[0].direction == "slowdown"
        assert proposals[0].target_hz == pytest.approx(0.5)
        assert "no input" in proposals[0].reason

    def test_speedup_proposal_recorded(self, controller, source, heuristic):
        source.set(0.5)
        heuristic.observe(now=100.0)
        proposals = heuristic.proposals
        assert len(proposals) == 1
        assert proposals[0].direction == "speedup"
        assert proposals[0].target_hz == pytest.approx(8.0)
        assert "input arrived" in proposals[0].reason

    def test_proposals_is_a_copy(self, controller, source, heuristic):
        source.set(60.0)
        heuristic.observe(now=100.0)
        proposals = heuristic.proposals
        proposals.clear()
        # Internal list still has the proposal.
        assert len(heuristic.proposals) == 1


class TestClamping:
    def test_targets_clamped_to_slider(self, controller, source):
        # Construct a heuristic with out-of-range targets.
        heuristic = StimulusDensityHeuristic(
            controller=controller,
            source=source,
            slowdown_target_hz=0.001,  # below MIN
            speedup_target_hz=50.0,    # above SLIDER_MAX
        )
        source.set(60.0)
        heuristic.observe(now=100.0)
        # Should land at MIN (0.05), not the requested 0.001.
        assert controller.target_rate_hz == pytest.approx(0.05)

        # Speedup target should clamp to SLIDER_MAX (10).
        source.set(0.5)
        # Skip the cooldown by directly invoking.
        heuristic._last_speedup_proposed_at = None
        heuristic.observe(now=200.0)
        assert controller.target_rate_hz == pytest.approx(SLIDER_MAX_HZ)
