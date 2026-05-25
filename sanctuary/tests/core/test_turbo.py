"""Unit tests for the turbo state machine."""

from unittest.mock import MagicMock

import pytest

from sanctuary.core.cycle_rate import (
    SLIDER_MAX_HZ,
    TURBO_DEFAULT_HZ,
    TURBO_MAX_HZ,
    CycleRateController,
)
from sanctuary.core.schema import ExperientialSignals
from sanctuary.core.turbo import (
    DEFAULT_ARM_THRESHOLD,
    DEFAULT_TRIGGER_THRESHOLD,
    IntensitySource,
    MechanicalIntensitySource,
    TurboManager,
    TurboState,
)


class _FakeSource:
    """Deterministic intensity source for tests."""

    def __init__(self, value: float = 0.0, name: str = "fake"):
        self._value = value
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def intensity(self, signals: ExperientialSignals) -> float:
        return self._value

    def set(self, value: float) -> None:
        self._value = value


@pytest.fixture
def controller():
    return CycleRateController(
        initial_hz=5.0,
        slowdown_seconds=0.0,
        speedup_seconds=0.0,
    )


@pytest.fixture
def signals():
    return ExperientialSignals()


@pytest.fixture
def source():
    return _FakeSource(value=0.0)


@pytest.fixture
def manager(controller, source):
    return TurboManager(
        controller=controller,
        sources=[source],
        # Aggressive timings so tests don't wait for real time.
        confirmation_seconds=0.5,
        exit_sustain_seconds=0.5,
        default_duration_seconds=2.0,
        max_duration_seconds=5.0,
        refractory_seconds=1.0,
    )


class TestInitialState:
    def test_starts_idle(self, manager):
        assert manager.state == TurboState.IDLE
        assert not manager.is_turbo_active

    def test_default_target_is_60hz(self, manager):
        # The default turbo_target_hz is TURBO_DEFAULT_HZ.
        assert manager._turbo_target_hz == TURBO_DEFAULT_HZ

    def test_default_source_is_mechanical(self, controller):
        m = TurboManager(controller=controller)
        assert len(m.sources) == 1
        assert isinstance(m.sources[0], MechanicalIntensitySource)


class TestStateTransitions:
    def test_low_intensity_stays_idle(self, manager, signals, source):
        source.set(0.001)
        manager.observe(signals, now=10.0)
        assert manager.state == TurboState.IDLE

    def test_arm_threshold_crosses_idle_to_armed(self, manager, signals, source):
        source.set(DEFAULT_ARM_THRESHOLD * 1.5)
        manager.observe(signals, now=10.0)
        assert manager.state == TurboState.ARMED

    def test_intensity_drops_armed_back_to_idle(self, manager, signals, source):
        source.set(DEFAULT_ARM_THRESHOLD * 1.5)
        manager.observe(signals, now=10.0)
        assert manager.state == TurboState.ARMED

        source.set(0.001)
        manager.observe(signals, now=10.5)
        assert manager.state == TurboState.IDLE

    def test_armed_to_active_after_confirmation(self, manager, signals, source):
        # Drive to ARMED.
        source.set(DEFAULT_TRIGGER_THRESHOLD * 1.1)
        manager.observe(signals, now=10.0)
        assert manager.state == TurboState.ARMED

        # Hold above trigger for confirmation_seconds.
        manager.observe(signals, now=10.6)
        assert manager.state == TurboState.ACTIVE

    def test_sharp_spike_bypasses_confirmation(self, manager, signals, source):
        # 2x trigger threshold → straight to ACTIVE.
        source.set(DEFAULT_TRIGGER_THRESHOLD * 2.5)
        # First tick takes IDLE → ARMED.
        manager.observe(signals, now=10.0)
        # Second tick observes the still-high spike — bypass to ACTIVE.
        manager.observe(signals, now=10.01)
        assert manager.state == TurboState.ACTIVE

    def test_active_to_refractory_on_max_duration(self, manager, signals, source):
        # Get into ACTIVE.
        source.set(DEFAULT_TRIGGER_THRESHOLD * 2.5)
        manager.observe(signals, now=10.0)
        manager.observe(signals, now=10.01)
        assert manager.state == TurboState.ACTIVE

        # Keep intensity high but elapse beyond max_duration.
        manager.observe(signals, now=10.01 + 5.0)
        assert manager.state == TurboState.REFRACTORY

    def test_active_to_refractory_on_sustained_low_signal(self, manager, signals, source):
        # Get into ACTIVE.
        source.set(DEFAULT_TRIGGER_THRESHOLD * 2.5)
        manager.observe(signals, now=10.0)
        manager.observe(signals, now=10.01)

        # Drop signal below exit threshold and sustain.
        source.set(0.001)
        manager.observe(signals, now=10.5)
        # Not yet exited — needs exit_sustain_seconds.
        assert manager.state == TurboState.ACTIVE
        manager.observe(signals, now=11.5)
        assert manager.state == TurboState.REFRACTORY

    def test_refractory_to_idle_after_refractory_seconds(self, manager, signals, source):
        # Force ACTIVE then REFRACTORY.
        source.set(DEFAULT_TRIGGER_THRESHOLD * 2.5)
        manager.observe(signals, now=10.0)
        manager.observe(signals, now=10.01)
        manager.observe(signals, now=10.01 + 5.0)  # max duration
        assert manager.state == TurboState.REFRACTORY

        # Refractory period is 1.0s. Hold low signal.
        source.set(0.001)
        manager.observe(signals, now=10.01 + 5.0 + 0.5)
        assert manager.state == TurboState.REFRACTORY
        manager.observe(signals, now=10.01 + 5.0 + 1.1)
        assert manager.state == TurboState.IDLE


class TestControllerInteraction:
    def test_engage_called_on_active(self, controller, manager, signals, source):
        source.set(DEFAULT_TRIGGER_THRESHOLD * 2.5)
        manager.observe(signals, now=10.0)
        manager.observe(signals, now=10.01)
        assert controller.is_turbo_active is True

    def test_release_called_on_refractory(self, controller, manager, signals, source):
        source.set(DEFAULT_TRIGGER_THRESHOLD * 2.5)
        manager.observe(signals, now=10.0)
        manager.observe(signals, now=10.01)
        assert controller.is_turbo_active is True

        manager.observe(signals, now=10.01 + 5.0)  # exceed max_duration
        assert controller.is_turbo_active is False

    def test_controller_rate_reaches_turbo_target_when_active(
        self, controller, manager, signals, source
    ):
        source.set(DEFAULT_TRIGGER_THRESHOLD * 2.5)
        manager.observe(signals, now=10.0)
        manager.observe(signals, now=10.01)
        # Controller snap window is small; one tick should advance to target.
        controller.tick(0.1)
        assert controller.current_rate_hz == pytest.approx(TURBO_DEFAULT_HZ)


class TestJournalAutoEntry:
    def test_journal_write_on_refractory(self, controller, signals):
        journal = MagicMock()
        source = _FakeSource(value=0.0)
        manager = TurboManager(
            controller=controller,
            journal=journal,
            sources=[source],
            confirmation_seconds=0.5,
            exit_sustain_seconds=0.5,
            default_duration_seconds=2.0,
            max_duration_seconds=5.0,
            refractory_seconds=1.0,
        )

        source.set(DEFAULT_TRIGGER_THRESHOLD * 2.5)
        manager.observe(signals, now=10.0)
        manager.observe(signals, now=10.01)
        # Exceed max duration to force exit.
        manager.observe(signals, now=10.01 + 5.0)

        journal.write.assert_called_once()
        call_kwargs = journal.write.call_args.kwargs
        assert "turbo" in call_kwargs["tags"]
        assert "system-generated" in call_kwargs["tags"]
        assert "max_duration" in call_kwargs["tags"]
        assert call_kwargs["significance"] == 6
        assert "elevated" in call_kwargs["emotional_tone"]

    def test_no_journal_means_no_write_error(self, controller, signals):
        source = _FakeSource(value=0.0)
        manager = TurboManager(
            controller=controller,
            journal=None,
            sources=[source],
            confirmation_seconds=0.5,
            exit_sustain_seconds=0.5,
            default_duration_seconds=2.0,
            max_duration_seconds=5.0,
            refractory_seconds=1.0,
        )
        source.set(DEFAULT_TRIGGER_THRESHOLD * 2.5)
        manager.observe(signals, now=10.0)
        manager.observe(signals, now=10.01)
        # Should not raise even without a journal.
        manager.observe(signals, now=10.01 + 5.0)
        assert manager.state == TurboState.REFRACTORY

    def test_journal_write_failure_is_non_fatal(self, controller, signals):
        journal = MagicMock()
        journal.write.side_effect = RuntimeError("disk full")
        source = _FakeSource(value=0.0)
        manager = TurboManager(
            controller=controller,
            journal=journal,
            sources=[source],
            confirmation_seconds=0.5,
            exit_sustain_seconds=0.5,
            default_duration_seconds=2.0,
            max_duration_seconds=5.0,
            refractory_seconds=1.0,
        )
        source.set(DEFAULT_TRIGGER_THRESHOLD * 2.5)
        manager.observe(signals, now=10.0)
        manager.observe(signals, now=10.01)
        # Journal raises but turbo still transitions cleanly.
        manager.observe(signals, now=10.01 + 5.0)
        assert manager.state == TurboState.REFRACTORY


class TestSignalStamping:
    def test_apply_to_signals_when_idle(self, manager, signals):
        out = manager.apply_to_signals(signals)
        assert out.turbo_active is False
        assert out.turbo_duration_seconds == 0.0

    def test_apply_to_signals_when_active(self, manager, signals, source):
        source.set(DEFAULT_TRIGGER_THRESHOLD * 2.5)
        manager.observe(signals, now=10.0)
        manager.observe(signals, now=10.01)
        # Now ACTIVE. apply_to_signals uses time.monotonic() internally,
        # so duration won't be deterministic; just check the flag.
        out = manager.apply_to_signals(signals)
        assert out.turbo_active is True
        assert out.turbo_duration_seconds >= 0.0


class TestMechanicalIntensitySource:
    def test_reads_activity_index(self):
        source = MechanicalIntensitySource()
        signals = ExperientialSignals(
            knowledge_signals={"luthi_delta": [0.01, 0.02, 0.03, 0.4]}
        )
        assert source.intensity(signals) == pytest.approx(0.4)

    def test_missing_key_returns_zero(self):
        source = MechanicalIntensitySource()
        signals = ExperientialSignals()
        assert source.intensity(signals) == 0.0

    def test_short_list_returns_zero(self):
        source = MechanicalIntensitySource()
        signals = ExperientialSignals(
            knowledge_signals={"luthi_delta": [0.01, 0.02]}
        )
        assert source.intensity(signals) == 0.0

    def test_negative_values_are_abs(self):
        source = MechanicalIntensitySource()
        signals = ExperientialSignals(
            knowledge_signals={"luthi_delta": [0.0, 0.0, 0.0, -0.5]}
        )
        assert source.intensity(signals) == pytest.approx(0.5)

    def test_configurable_key_and_index(self):
        source = MechanicalIntensitySource(
            signal_key="custom", activity_index=1
        )
        signals = ExperientialSignals(
            knowledge_signals={"custom": [0.1, 0.7, 0.3]}
        )
        assert source.intensity(signals) == pytest.approx(0.7)


class TestMultipleSources:
    def test_max_of_sources_used(self, controller, signals):
        a = _FakeSource(value=0.02, name="a")
        b = _FakeSource(value=0.08, name="b")
        manager = TurboManager(
            controller=controller,
            sources=[a, b],
        )
        manager.observe(signals, now=10.0)
        # b's value (0.08) > arm threshold (0.05), so ARMED.
        assert manager.state == TurboState.ARMED

    def test_source_exception_treated_as_zero(self, controller, signals):
        class _Broken:
            @property
            def name(self):
                return "broken"

            def intensity(self, _signals):
                raise RuntimeError("intentional test failure")

        good = _FakeSource(value=0.0, name="good")
        manager = TurboManager(controller=controller, sources=[_Broken(), good])
        # Should not raise; the broken source contributes 0.
        manager.observe(signals, now=10.0)
        assert manager.state == TurboState.IDLE


class TestTraceLogging:
    """Trace logging writes one JSONL line per observe() call."""

    def test_trace_path_writes_jsonl(self, controller, signals, tmp_path):
        import json

        trace_path = tmp_path / "turbo_trace.jsonl"
        source = _FakeSource(value=0.0)
        manager = TurboManager(
            controller=controller,
            sources=[source],
            trace_path=trace_path,
            confirmation_seconds=0.5,
            exit_sustain_seconds=0.5,
            default_duration_seconds=2.0,
            max_duration_seconds=5.0,
            refractory_seconds=1.0,
        )

        # Drive through a state transition: low → arming spike → active.
        source.set(0.001)
        manager.observe(signals, now=10.0)
        source.set(DEFAULT_TRIGGER_THRESHOLD * 2.5)
        manager.observe(signals, now=10.5)
        manager.observe(signals, now=10.6)

        lines = trace_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 3

        # Each line is parseable JSON with the expected schema.
        records = [json.loads(line) for line in lines]
        for r in records:
            assert "t" in r
            assert "per_source" in r
            assert "fake" in r["per_source"]
            assert "intensity" in r
            assert "dominant_source" in r
            assert "state_before" in r
            assert "state_after" in r
            assert "current_rate_hz" in r
            assert "is_turbo_active" in r

        # First observe: state stays IDLE (intensity below arm threshold).
        assert records[0]["state_before"] == "idle"
        assert records[0]["state_after"] == "idle"
        # Second observe: intensity above trigger, IDLE → ARMED.
        assert records[1]["state_before"] == "idle"
        assert records[1]["state_after"] == "armed"
        # Third observe: sharp spike persists; ARMED → ACTIVE.
        assert records[2]["state_after"] == "active"
        assert records[2]["is_turbo_active"] is True

    def test_no_trace_path_means_no_file(self, controller, signals, tmp_path):
        source = _FakeSource(value=DEFAULT_TRIGGER_THRESHOLD * 2.5)
        manager = TurboManager(
            controller=controller,
            sources=[source],
            trace_path=None,
        )
        manager.observe(signals, now=10.0)
        # No files created in tmp_path.
        assert list(tmp_path.iterdir()) == []

    def test_trace_creates_parent_directories(self, controller, signals, tmp_path):
        nested_path = tmp_path / "deep" / "nested" / "trace.jsonl"
        source = _FakeSource(value=0.001)
        manager = TurboManager(
            controller=controller,
            sources=[source],
            trace_path=nested_path,
        )
        manager.observe(signals, now=10.0)
        assert nested_path.exists()


class TestEventHistory:
    def test_event_recorded_on_exit(self, controller, signals):
        source = _FakeSource(value=0.0)
        manager = TurboManager(
            controller=controller,
            sources=[source],
            confirmation_seconds=0.5,
            exit_sustain_seconds=0.5,
            default_duration_seconds=2.0,
            max_duration_seconds=5.0,
            refractory_seconds=1.0,
        )
        source.set(DEFAULT_TRIGGER_THRESHOLD * 2.5)
        manager.observe(signals, now=10.0)
        manager.observe(signals, now=10.01)
        manager.observe(signals, now=10.01 + 5.0)  # exit on max_duration

        history = manager.event_history
        assert len(history) == 1
        event = history[0]
        assert event.exit_reason == "max_duration"
        assert event.trigger_source == "fake"
        assert event.trigger_intensity > 0
        assert event.target_rate_hz == TURBO_DEFAULT_HZ
