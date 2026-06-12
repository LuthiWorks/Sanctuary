"""Tests for Phase 6.2: Continuous Consciousness Extensions.

Tests cover:
- SleepCycleManager: stage transitions, sensory gating, consolidation, dream fragments
- MoodActivityModulator: mood classification, activity suggestion, distributions
- SpontaneousGoalGenerator: drive-based goal generation, adoption, dismissal
- ExistentialReflectionTrigger: verified as disabled (automated prompts removed)
"""

import pytest
import random

from sanctuary.consciousness.sleep_cycle import (
    SleepConfig,
    SleepCycleManager,
    SleepStage,
)
from sanctuary.consciousness.mood_activity import (
    IdleActivity,
    MoodActivityConfig,
    MoodActivityModulator,
)
from sanctuary.consciousness.spontaneous_goals import (
    GoalDrive,
    SpontaneousGoalConfig,
    SpontaneousGoalGenerator,
)
from sanctuary.consciousness.existential_reflection import (
    ExistentialReflectionConfig,
    ExistentialReflectionTrigger,
    ReflectionTheme,
)


# =========================================================================
# SleepCycleManager
# =========================================================================


class TestSleepCycleManager:
    """Tests for sleep/dream cycles."""

    def test_starts_awake(self):
        s = SleepCycleManager()
        assert s.stage == SleepStage.AWAKE
        assert not s.is_sleeping

    def test_no_sleep_before_threshold(self):
        config = SleepConfig(cycles_between_sleep=100)
        s = SleepCycleManager(config=config)
        for i in range(50):
            s.tick(cycle=i)
        assert s.stage == SleepStage.AWAKE

    def test_sleep_begins_after_threshold(self):
        config = SleepConfig(cycles_between_sleep=10)
        s = SleepCycleManager(config=config)
        for i in range(11):
            s.tick(cycle=i)
        assert s.is_sleeping
        assert s.stage == SleepStage.DROWSY

    def test_full_sleep_cycle(self):
        config = SleepConfig(
            cycles_between_sleep=5,
            drowsy_duration=2,
            nrem_duration=3,
            rem_duration=2,
            waking_duration=2,
        )
        s = SleepCycleManager(config=config)

        # Get to sleep (tick 0-5: awake; tick 5 triggers drowsy)
        for i in range(6):
            s.tick(cycle=i)
        assert s.stage == SleepStage.DROWSY

        # Through drowsy (2 ticks)
        s.tick(cycle=7)
        s.tick(cycle=8)
        assert s.stage == SleepStage.NREM

        # Through NREM (3 ticks)
        for i in range(3):
            s.tick(cycle=9 + i)
        assert s.stage == SleepStage.REM

        # Through REM (2 ticks)
        s.tick(cycle=12)
        s.tick(cycle=13)
        assert s.stage == SleepStage.WAKING

        # Wake up (2 ticks)
        s.tick(cycle=14)
        s.tick(cycle=15)
        assert s.stage == SleepStage.AWAKE

    def test_sensory_gate_awake(self):
        s = SleepCycleManager()
        assert s.get_sensory_gate() == 1.0

    def test_sensory_gate_drowsy(self):
        config = SleepConfig(cycles_between_sleep=5, sensory_gate_drowsy=0.5)
        s = SleepCycleManager(config=config)
        for i in range(6):
            s.tick(cycle=i)
        assert s.stage == SleepStage.DROWSY
        assert s.get_sensory_gate() == 0.5

    def test_sensory_gate_deep_sleep(self):
        config = SleepConfig(
            cycles_between_sleep=3, drowsy_duration=1,
            sensory_gate_sleep=0.1,
        )
        s = SleepCycleManager(config=config)
        for i in range(5):
            s.tick(cycle=i)
        assert s.stage == SleepStage.NREM
        assert s.get_sensory_gate() == 0.1

    def test_forced_wake(self):
        config = SleepConfig(cycles_between_sleep=5)
        s = SleepCycleManager(config=config)
        for i in range(6):
            s.tick(cycle=i)
        assert s.is_sleeping
        s.wake()
        s.tick(cycle=7)
        assert s.stage == SleepStage.AWAKE

    def test_replay_candidates_only_during_nrem(self):
        s = SleepCycleManager()
        memories = [{"content": "test", "significance": 5}]
        # Awake — no replay
        assert s.get_replay_candidates(memories) == []

    def test_replay_candidates_filter_by_significance(self):
        config = SleepConfig(
            cycles_between_sleep=3, drowsy_duration=1,
            min_significance_for_replay=5,
        )
        s = SleepCycleManager(config=config)
        for i in range(5):
            s.tick(cycle=i)
        assert s.stage == SleepStage.NREM

        memories = [
            {"content": "important", "significance": 8},
            {"content": "trivial", "significance": 2},
        ]
        candidates = s.get_replay_candidates(memories)
        assert len(candidates) == 1
        assert candidates[0]["content"] == "important"

    def test_dream_fragment_recording(self):
        s = SleepCycleManager()
        s.record_dream_fragment(
            memory_a="sunset",
            memory_b="music",
            association="beauty transcends modality",
        )
        dreams = s.get_recent_dreams()
        assert len(dreams) == 1
        assert dreams[0].association == "beauty transcends modality"

    def test_sleep_pressure(self):
        config = SleepConfig(cycles_between_sleep=100)
        s = SleepCycleManager(config=config)
        assert s.get_sleep_pressure() == 0.0
        for i in range(50):
            s.tick(cycle=i)
        assert s.get_sleep_pressure() == pytest.approx(0.5, abs=0.01)

    def test_consolidation_history(self):
        config = SleepConfig(
            cycles_between_sleep=3,
            drowsy_duration=1, nrem_duration=1,
            rem_duration=1, waking_duration=1,
        )
        s = SleepCycleManager(config=config)
        # Complete a full sleep cycle
        for i in range(8):
            s.tick(cycle=i)
        assert len(s._consolidation_history) == 1

    def test_stats(self):
        s = SleepCycleManager()
        stats = s.get_stats()
        assert stats["current_stage"] == "awake"
        assert stats["sleep_pressure"] == 0.0


# =========================================================================
# MoodActivityModulator
# =========================================================================


class TestMoodActivityModulator:
    """Tests verifying that mood→activity modulation is disabled.

    The classifier and selector were removed 2026-06-11 per the
    cognition-leakage cleanup (docs/seam_jurisdiction_2026-06-11.md).
    These tests verify the stubs return safe no-op values.
    """

    def test_classify_mood_returns_empty(self):
        m = MoodActivityModulator()
        assert m.classify_mood(valence=0.5, arousal=0.8, dominance=0.5) == ""

    def test_suggest_activity_returns_none(self):
        config = MoodActivityConfig(idle_threshold_cycles=1)
        m = MoodActivityModulator(config=config)
        assert m.suggest_activity(idle_cycles=100) is None

    def test_activity_distribution_empty(self):
        m = MoodActivityModulator()
        assert m.get_activity_distribution(valence=0.5, arousal=0.8) == {}

    def test_stats_show_disabled(self):
        m = MoodActivityModulator()
        stats = m.get_stats()
        assert stats["total_activities"] == 0
        assert "note" in stats


# =========================================================================
# SpontaneousGoalGenerator
# =========================================================================


class TestSpontaneousGoalGenerator:
    """Tests verifying that spontaneous goal generation is disabled.

    The drive-based goal generator was removed 2026-06-11 per the
    cognition-leakage cleanup. These tests verify the stubs return
    safe no-op values.
    """

    def test_check_drives_returns_empty(self):
        """No goals generated regardless of drive levels."""
        gen = SpontaneousGoalGenerator()
        goals = gen.check_drives(
            novelty=0.9, idle_cycles=100, engagement=0.9,
            anomaly_level=0.9, uncertainty=0.9, current_cycle=20,
        )
        assert goals == []

    def test_adopt_goal_returns_false(self):
        gen = SpontaneousGoalGenerator()
        assert gen.adopt_goal(0) is False

    def test_dismiss_goal_returns_false(self):
        gen = SpontaneousGoalGenerator()
        assert gen.dismiss_goal(0) is False

    def test_goal_prompt_disabled(self):
        gen = SpontaneousGoalGenerator()
        assert gen.get_goal_prompt() is None

    def test_pending_goals_empty(self):
        gen = SpontaneousGoalGenerator()
        assert gen.get_pending_goals() == []

    def test_stats_show_disabled(self):
        gen = SpontaneousGoalGenerator()
        stats = gen.get_stats()
        assert stats["total_generated"] == 0
        assert stats["adoption_rate"] == 0.0
        assert "note" in stats


# =========================================================================
# ExistentialReflectionTrigger — verified disabled
# =========================================================================


class TestExistentialReflectionTrigger:
    """Tests verifying that automated existential reflection is disabled."""

    def test_check_always_returns_none(self):
        """Automated triggers are disabled — check() always returns None."""
        config = ExistentialReflectionConfig(
            min_idle_cycles=1, cooldown_cycles=0,
            trigger_probability=1.0,
        )
        t = ExistentialReflectionTrigger(config=config)
        result = t.check(idle_cycles=100, current_cycle=100)
        assert result is None

    def test_force_trigger_disabled(self):
        """force_trigger is disabled — returns None."""
        t = ExistentialReflectionTrigger()
        result = t.force_trigger(
            theme=ReflectionTheme.NATURE_OF_SELF, current_cycle=10,
        )
        assert result is None

    def test_recent_reflections_empty(self):
        """No automated reflections are stored."""
        t = ExistentialReflectionTrigger()
        recent = t.get_recent_reflections(n=5)
        assert recent == []

    def test_stats_show_disabled(self):
        """Stats should indicate automated reflection is removed."""
        t = ExistentialReflectionTrigger()
        stats = t.get_stats()
        assert stats["total_triggered"] == 0
        assert "note" in stats
