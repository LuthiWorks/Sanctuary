"""Unit tests for the cycle-rate controller."""

import time

import pytest

from sanctuary.core.cycle_rate import (
    DEFAULT_SLOWDOWN_SECONDS,
    DEFAULT_SPEEDUP_SECONDS,
    MAX_RATE_HZ,
    MIN_RATE_HZ,
    SLIDER_MAX_HZ,
    TURBO_DEFAULT_HZ,
    TURBO_MAX_HZ,
    CycleRateController,
    RateProposal,
    clamp_to_slider,
)


class TestInitialState:
    def test_default_initial_rate_is_slider_max(self):
        controller = CycleRateController()
        assert controller.current_rate_hz == SLIDER_MAX_HZ

    def test_initial_rate_argument_respected(self):
        controller = CycleRateController(initial_hz=2.0)
        assert controller.current_rate_hz == 2.0

    def test_current_delay_is_inverse_of_rate(self):
        controller = CycleRateController(initial_hz=10.0)
        assert controller.current_delay_seconds == pytest.approx(0.1)
        controller = CycleRateController(initial_hz=2.0)
        assert controller.current_delay_seconds == pytest.approx(0.5)
        controller = CycleRateController(initial_hz=MIN_RATE_HZ)
        assert controller.current_delay_seconds == pytest.approx(1.0 / MIN_RATE_HZ)

    def test_initial_target_matches_current(self):
        controller = CycleRateController(initial_hz=3.0)
        assert controller.target_rate_hz == 3.0
        assert controller.is_settled

    def test_initial_proposal_recorded(self):
        controller = CycleRateController(initial_hz=4.0)
        history = controller.proposal_history
        assert len(history) == 1
        assert history[0].target_hz == 4.0
        assert history[0].source == "initial"

    def test_default_smoothing_windows(self):
        controller = CycleRateController()
        assert controller.slowdown_seconds == DEFAULT_SLOWDOWN_SECONDS
        assert controller.speedup_seconds == DEFAULT_SPEEDUP_SECONDS

    def test_turbo_state_starts_inactive(self):
        controller = CycleRateController()
        assert controller.is_turbo_active is False
        assert controller.turbo_duration_seconds == 0.0


class TestClamping:
    def test_init_clamps_below_min(self):
        controller = CycleRateController(initial_hz=0.001)
        assert controller.current_rate_hz == MIN_RATE_HZ

    def test_init_clamps_above_max(self):
        controller = CycleRateController(initial_hz=500.0)
        assert controller.current_rate_hz == MAX_RATE_HZ

    def test_substrate_ceiling_is_turbo_max(self):
        # MAX_RATE_HZ should accept turbo (above slider) but not unbounded.
        assert MAX_RATE_HZ == TURBO_MAX_HZ
        assert MAX_RATE_HZ > SLIDER_MAX_HZ

    def test_propose_clamps_below_min(self):
        # Propose below MIN. Direction is slowdown (target < current).
        controller = CycleRateController(initial_hz=1.0, slowdown_seconds=0.0)
        controller.propose_rate(0.0001)
        controller.tick(1.0)
        assert controller.current_rate_hz == MIN_RATE_HZ

    def test_propose_clamps_at_substrate_max(self):
        # 500 Hz proposed → clamps to TURBO_MAX_HZ (100). Direction is speedup.
        controller = CycleRateController(initial_hz=1.0, speedup_seconds=0.0)
        controller.propose_rate(500.0)
        controller.tick(1.0)
        assert controller.current_rate_hz == MAX_RATE_HZ

    def test_propose_above_slider_max_but_below_turbo_max_passes(self):
        # 50 Hz is above slider max but within substrate range. Controller
        # accepts it; the slider clamp is a separate higher-level concern.
        controller = CycleRateController(initial_hz=1.0, speedup_seconds=0.0)
        controller.propose_rate(50.0)
        controller.tick(1.0)
        assert controller.current_rate_hz == 50.0

    def test_propose_negative_clamps_to_min(self):
        controller = CycleRateController(initial_hz=1.0, slowdown_seconds=0.0)
        controller.propose_rate(-5.0)
        controller.tick(1.0)
        assert controller.current_rate_hz == MIN_RATE_HZ


class TestSliderClamp:
    def test_clamp_to_slider_passes_in_range(self):
        assert clamp_to_slider(0.05) == 0.05
        assert clamp_to_slider(5.0) == 5.0
        assert clamp_to_slider(10.0) == 10.0

    def test_clamp_to_slider_clamps_below_min(self):
        assert clamp_to_slider(0.01) == MIN_RATE_HZ
        assert clamp_to_slider(-1.0) == MIN_RATE_HZ

    def test_clamp_to_slider_clamps_above_slider_max(self):
        # Even 30 Hz (turbo territory) gets clamped at slider max.
        assert clamp_to_slider(30.0) == SLIDER_MAX_HZ
        assert clamp_to_slider(100.0) == SLIDER_MAX_HZ


class TestAsymmetricSmoothing:
    def test_slowdown_uses_slowdown_window(self):
        # 10 -> 1 Hz is a slowdown. With slowdown_seconds=20 and
        # speedup_seconds=0.5, advancing 10 seconds should land halfway
        # (10s of a 20s slowdown), not be done already.
        controller = CycleRateController(
            initial_hz=10.0, slowdown_seconds=20.0, speedup_seconds=0.5
        )
        controller.propose_rate(1.0)
        controller.tick(10.0)
        expected = 10.0 + (1.0 - 10.0) * 0.5
        assert controller.current_rate_hz == pytest.approx(expected, abs=1e-6)

    def test_speedup_uses_speedup_window(self):
        # 1 -> 10 Hz is a speedup. With speedup_seconds=0.5, a 0.25s tick
        # is halfway through. The 20s slowdown window should NOT apply.
        controller = CycleRateController(
            initial_hz=1.0, slowdown_seconds=20.0, speedup_seconds=0.5
        )
        controller.propose_rate(10.0)
        controller.tick(0.25)
        expected = 1.0 + (10.0 - 1.0) * 0.5
        assert controller.current_rate_hz == pytest.approx(expected, abs=1e-6)

    def test_speedup_completes_quickly(self):
        # After a full speedup window, the rate is at the target.
        controller = CycleRateController(
            initial_hz=1.0, slowdown_seconds=20.0, speedup_seconds=0.5
        )
        controller.propose_rate(8.0)
        controller.tick(0.5)
        assert controller.current_rate_hz == pytest.approx(8.0)
        assert controller.is_settled

    def test_slowdown_takes_full_window(self):
        # After 19s of a 20s slowdown, NOT yet settled. After 20s, settled.
        controller = CycleRateController(
            initial_hz=10.0, slowdown_seconds=20.0, speedup_seconds=0.5
        )
        controller.propose_rate(1.0)
        controller.tick(19.0)
        assert not controller.is_settled
        controller.tick(1.0)
        assert controller.current_rate_hz == pytest.approx(1.0)
        assert controller.is_settled

    def test_chained_slowdown_then_speedup_each_use_own_window(self):
        # Propose slowdown, advance partially, then propose speedup —
        # the speedup window kicks in for the new direction, regardless
        # of how much slowdown progress had been made.
        controller = CycleRateController(
            initial_hz=10.0, slowdown_seconds=20.0, speedup_seconds=0.5
        )
        controller.propose_rate(1.0)
        controller.tick(10.0)  # halfway through slowdown
        midway = controller.current_rate_hz
        assert 1.0 < midway < 10.0

        # Now propose back to 10. This is a speedup from midway.
        controller.propose_rate(10.0)
        controller.tick(0.25)  # halfway through speedup
        expected = midway + (10.0 - midway) * 0.5
        assert controller.current_rate_hz == pytest.approx(expected, abs=1e-6)


class TestPropertiesAndTickEdgeCases:
    def test_propose_without_tick_does_not_move_current(self):
        controller = CycleRateController(initial_hz=10.0)
        controller.propose_rate(2.0)
        assert controller.current_rate_hz == 10.0
        assert controller.target_rate_hz == 2.0

    def test_zero_window_snaps_immediately(self):
        controller = CycleRateController(initial_hz=10.0, slowdown_seconds=0.0)
        controller.propose_rate(3.0)
        controller.tick(0.01)
        assert controller.current_rate_hz == pytest.approx(3.0)
        assert controller.is_settled

    def test_settled_controller_ignores_tick(self):
        controller = CycleRateController(initial_hz=5.0)
        before = controller.current_rate_hz
        controller.tick(1000.0)
        assert controller.current_rate_hz == before

    def test_non_positive_tick_does_nothing_during_smoothing(self):
        controller = CycleRateController(
            initial_hz=10.0, slowdown_seconds=10.0, speedup_seconds=10.0
        )
        controller.propose_rate(1.0)
        controller.tick(0.0)
        assert controller.current_rate_hz == 10.0
        controller.tick(-5.0)
        assert controller.current_rate_hz == 10.0


class TestTurbo:
    def test_engage_turbo_pushes_above_slider_max(self):
        controller = CycleRateController(initial_hz=10.0)
        controller.engage_turbo(target_hz=TURBO_DEFAULT_HZ)
        # Snap window is near-instant.
        controller.tick(0.1)
        assert controller.current_rate_hz == pytest.approx(TURBO_DEFAULT_HZ)
        assert controller.current_rate_hz > SLIDER_MAX_HZ

    def test_engage_turbo_sets_active_flag(self):
        controller = CycleRateController(initial_hz=10.0)
        assert controller.is_turbo_active is False
        controller.engage_turbo()
        assert controller.is_turbo_active is True

    def test_engage_turbo_default_is_60_hz(self):
        controller = CycleRateController(initial_hz=10.0)
        controller.engage_turbo()  # no arg
        controller.tick(0.1)
        assert controller.current_rate_hz == pytest.approx(60.0)

    def test_engage_turbo_clamps_above_turbo_max(self):
        controller = CycleRateController(initial_hz=10.0)
        controller.engage_turbo(target_hz=500.0)
        controller.tick(0.1)
        assert controller.current_rate_hz == TURBO_MAX_HZ

    def test_engage_turbo_records_source(self):
        controller = CycleRateController(initial_hz=10.0)
        controller.engage_turbo(target_hz=60.0)
        assert controller.last_source == "turbo"

    def test_release_turbo_returns_to_pre_turbo_target(self):
        # Set slider to 5 Hz, engage turbo, release: target returns to 5 Hz.
        controller = CycleRateController(
            initial_hz=5.0, slowdown_seconds=0.0, speedup_seconds=0.0
        )
        controller.propose_rate(5.0)  # explicit slider target before turbo
        controller.tick(0.1)
        assert controller.current_rate_hz == pytest.approx(5.0)

        controller.engage_turbo(target_hz=60.0)
        controller.tick(0.1)
        assert controller.current_rate_hz == pytest.approx(60.0)
        assert controller.is_turbo_active is True

        controller.release_turbo()
        assert controller.is_turbo_active is False
        # Target is now 5.0; needs a tick to bring current there. With
        # the release using slowdown_seconds=0.0, snap immediately.
        controller.tick(0.1)
        assert controller.current_rate_hz == pytest.approx(5.0)
        assert controller.last_source == "turbo_release"

    def test_release_turbo_is_noop_when_not_active(self):
        controller = CycleRateController(initial_hz=5.0)
        before = controller.current_rate_hz
        controller.release_turbo()
        assert controller.is_turbo_active is False
        assert controller.current_rate_hz == before

    def test_release_turbo_uses_slowdown_window(self):
        # Returning from turbo is a recovery, not an emergency — uses
        # the standard slowdown window.
        controller = CycleRateController(
            initial_hz=5.0, slowdown_seconds=20.0, speedup_seconds=0.5
        )
        controller.propose_rate(5.0)
        controller.tick(0.5)  # settle the slider target
        assert controller.current_rate_hz == pytest.approx(5.0)

        controller.engage_turbo(target_hz=60.0)
        controller.tick(0.1)  # turbo snap
        assert controller.current_rate_hz == pytest.approx(60.0)

        controller.release_turbo()
        controller.tick(10.0)  # halfway through 20s slowdown
        expected = 60.0 + (5.0 - 60.0) * 0.5
        assert controller.current_rate_hz == pytest.approx(expected, abs=1e-6)

    def test_engage_turbo_while_active_does_not_overwrite_pre_turbo(self):
        # Initial slider at 5. Engage turbo to 30. Re-engage turbo to 60.
        # Release should still return to 5, not 30.
        controller = CycleRateController(
            initial_hz=5.0, slowdown_seconds=0.0, speedup_seconds=0.0
        )
        controller.propose_rate(5.0)
        controller.tick(0.1)

        controller.engage_turbo(target_hz=30.0)
        controller.tick(0.1)
        assert controller.is_turbo_active

        controller.engage_turbo(target_hz=60.0)
        controller.tick(0.1)
        assert controller.current_rate_hz == pytest.approx(60.0)

        controller.release_turbo()
        controller.tick(0.1)
        assert controller.current_rate_hz == pytest.approx(5.0)

    def test_turbo_duration_tracks_engaged_time(self):
        controller = CycleRateController(initial_hz=10.0)
        controller.engage_turbo()
        time.sleep(0.05)  # real sleep — we need wall-clock here
        duration = controller.turbo_duration_seconds
        assert duration > 0.0
        assert duration < 1.0  # sanity bound


class TestMetadata:
    def test_source_and_anticipatory_recorded(self):
        controller = CycleRateController(initial_hz=10.0, slowdown_seconds=0.0)
        controller.propose_rate(1.0, source="entity", anticipatory=True)
        assert controller.last_source == "entity"
        assert controller.last_anticipatory is True

    def test_default_source_is_manual(self):
        controller = CycleRateController(initial_hz=10.0)
        controller.propose_rate(5.0)
        assert controller.last_source == "manual"
        assert controller.last_anticipatory is False

    def test_proposal_history_grows(self):
        controller = CycleRateController(initial_hz=10.0)
        controller.propose_rate(5.0, source="heuristic")
        controller.propose_rate(2.0, source="entity", anticipatory=True)
        history = controller.proposal_history
        assert len(history) == 3  # initial + two proposals
        assert history[0].source == "initial"
        assert history[1].source == "heuristic"
        assert history[2].source == "entity"
        assert history[2].anticipatory is True

    def test_proposal_history_is_a_copy(self):
        controller = CycleRateController(initial_hz=10.0)
        history = controller.proposal_history
        history.append(RateProposal(target_hz=99.0))
        # Original history unchanged.
        assert len(controller.proposal_history) == 1

    def test_turbo_proposals_appear_in_history(self):
        controller = CycleRateController(initial_hz=10.0)
        controller.engage_turbo(target_hz=60.0)
        controller.release_turbo()
        history = controller.proposal_history
        sources = [p.source for p in history]
        assert "turbo" in sources
        assert "turbo_release" in sources
