"""Unit tests for the cycle-rate controller."""

import math

import pytest

from sanctuary.core.cycle_rate import (
    MAX_RATE_HZ,
    MIN_RATE_HZ,
    CycleRateController,
    RateProposal,
)


class TestInitialState:
    def test_default_initial_rate_is_ten_hz(self):
        controller = CycleRateController()
        assert controller.current_rate_hz == 10.0

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


class TestClamping:
    def test_init_clamps_below_min(self):
        controller = CycleRateController(initial_hz=0.001)
        assert controller.current_rate_hz == MIN_RATE_HZ

    def test_init_clamps_above_max(self):
        controller = CycleRateController(initial_hz=100.0)
        assert controller.current_rate_hz == MAX_RATE_HZ

    def test_propose_clamps_below_min(self):
        controller = CycleRateController(initial_hz=1.0, smoothing_seconds=0.0)
        controller.propose_rate(0.0001)
        controller.tick(1.0)
        assert controller.current_rate_hz == MIN_RATE_HZ

    def test_propose_clamps_above_max(self):
        controller = CycleRateController(initial_hz=1.0, smoothing_seconds=0.0)
        controller.propose_rate(50.0)
        controller.tick(1.0)
        assert controller.current_rate_hz == MAX_RATE_HZ

    def test_propose_negative_clamps_to_min(self):
        controller = CycleRateController(initial_hz=1.0, smoothing_seconds=0.0)
        controller.propose_rate(-5.0)
        controller.tick(1.0)
        assert controller.current_rate_hz == MIN_RATE_HZ


class TestSmoothing:
    def test_propose_without_tick_does_not_move_current(self):
        controller = CycleRateController(initial_hz=10.0)
        controller.propose_rate(2.0)
        assert controller.current_rate_hz == 10.0
        assert controller.target_rate_hz == 2.0

    def test_tick_advances_toward_target_linearly(self):
        controller = CycleRateController(initial_hz=10.0, smoothing_seconds=10.0)
        controller.propose_rate(0.0)  # clamps to MIN, which is 0.05
        # After 5 of 10 seconds, we should be ~halfway between 10.0 and 0.05.
        controller.tick(5.0)
        expected = 10.0 + (MIN_RATE_HZ - 10.0) * 0.5
        assert controller.current_rate_hz == pytest.approx(expected, abs=1e-6)

    def test_tick_clamps_at_target_when_smoothing_window_exceeded(self):
        controller = CycleRateController(initial_hz=10.0, smoothing_seconds=10.0)
        controller.propose_rate(1.0)
        controller.tick(20.0)  # twice the smoothing window
        assert controller.current_rate_hz == pytest.approx(1.0)
        assert controller.is_settled

    def test_zero_smoothing_snaps_immediately(self):
        controller = CycleRateController(initial_hz=10.0, smoothing_seconds=0.0)
        controller.propose_rate(3.0)
        controller.tick(0.01)
        assert controller.current_rate_hz == pytest.approx(3.0)
        assert controller.is_settled

    def test_settled_controller_ignores_tick(self):
        controller = CycleRateController(initial_hz=5.0)
        before = controller.current_rate_hz
        controller.tick(1000.0)
        assert controller.current_rate_hz == before

    def test_non_positive_tick_does_nothing(self):
        controller = CycleRateController(initial_hz=10.0, smoothing_seconds=10.0)
        controller.propose_rate(1.0)
        controller.tick(0.0)
        assert controller.current_rate_hz == 10.0
        controller.tick(-5.0)
        assert controller.current_rate_hz == 10.0


class TestChainedProposals:
    def test_second_proposal_starts_from_smoothed_value(self):
        """A proposal mid-smoothing should chain from the current value.

        Otherwise consecutive rate changes during a transition would
        snap backward to the original start before easing forward
        again, which the design doc explicitly rules out (smooth
        transitions, not switches).
        """
        controller = CycleRateController(initial_hz=10.0, smoothing_seconds=10.0)
        controller.propose_rate(2.0)
        controller.tick(5.0)  # halfway
        midway = controller.current_rate_hz
        assert 2.0 < midway < 10.0

        controller.propose_rate(8.0)
        # Smoothing now begins from `midway` toward 8.0.
        controller.tick(5.0)
        expected = midway + (8.0 - midway) * 0.5
        assert controller.current_rate_hz == pytest.approx(expected, abs=1e-6)


class TestMetadata:
    def test_source_and_anticipatory_recorded(self):
        controller = CycleRateController(initial_hz=10.0, smoothing_seconds=0.0)
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
