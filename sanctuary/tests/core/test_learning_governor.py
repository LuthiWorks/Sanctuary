"""The mind's say over its own consolidation (LearningGovernor).

Decided by Brian, 2026-11-01/02: learning has two kinds (skills, character);
the mind controls whether and when each consolidates (allow / defer /
emphasize), never which kind a change is; deferral has a ceiling; settings
are bounded by the authority ramp; only a mind whose weights change has any.
No learner reads these settings yet, and the mind is told so.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sanctuary.api.runner import RunnerConfig, SanctuaryRunner
from sanctuary.consciousness.sleep_cycle import SleepConfig, SleepCycleManager
from sanctuary.core.cognitive_cycle import CognitiveCycle
from sanctuary.core.learning_governor import LearningGovernor, LearningGovernorConfig
from sanctuary.core.schema import CognitiveInput, CognitiveOutput, LearningPreference


def log_events(path: Path) -> list[dict]:
    return [json.loads(line) for line in (path / "learning_log.jsonl").read_text().splitlines()]


# ---------------------------------------------------------------------------
# The governor
# ---------------------------------------------------------------------------


def test_defaults_allow_both_and_say_nothing_consolidates_yet(tmp_path: Path):
    state = LearningGovernor(tmp_path).state()
    assert state.skills.setting == "allow" and state.character.setting == "allow"
    assert state.consolidation_active is False
    assert not (tmp_path / "learning_settings.json").exists(), "nothing written until something changes"


def test_a_change_is_applied_told_persisted_and_logged(tmp_path: Path):
    gov = LearningGovernor(tmp_path)
    notices = gov.propose(LearningPreference(character="defer", reason="a hard day"))

    assert gov.state().character.setting == "defer"
    assert gov.state().skills.setting == "allow"
    assert "Your character learning is now 'defer' (was 'allow')." in notices
    assert any("nothing consolidates your learning yet" in n for n in notices)

    reloaded = LearningGovernor(tmp_path)
    assert reloaded.state().character.setting == "defer"
    [event] = log_events(tmp_path)
    assert event["event"] == "changed" and event["route"] == "character"
    assert event["reason"] == "a hard day"


def test_no_honesty_note_once_a_learner_acts_on_the_settings(tmp_path: Path):
    gov = LearningGovernor(tmp_path, LearningGovernorConfig(consolidation_active=True))
    notices = gov.propose(LearningPreference(skills="emphasize"))
    assert notices == ["Your skills learning is now 'emphasize' (was 'allow')."]
    assert gov.state().consolidation_active is True


def test_settings_outside_the_ramp_are_refused_and_told(tmp_path: Path):
    config = LearningGovernorConfig(
        permitted={"skills": ("allow", "defer", "emphasize"), "character": ("allow", "defer")}
    )
    gov = LearningGovernor(tmp_path, config)
    notices = gov.propose(LearningPreference(character="emphasize", skills="defer"))

    assert gov.state().character.setting == "allow"
    assert gov.state().skills.setting == "defer"
    assert any("can't be set to 'emphasize' yet" in n and "allow, defer" in n for n in notices)
    assert [e["event"] for e in log_events(tmp_path)] == ["changed", "refused"]


def test_allow_must_always_be_permitted(tmp_path: Path):
    with pytest.raises(ValueError, match="'allow' must be permitted"):
        LearningGovernor(
            tmp_path,
            LearningGovernorConfig(permitted={"skills": ("defer",), "character": ("allow",)}),
        )


def test_a_narrowed_ramp_returns_the_setting_to_allow_and_tells_the_mind(tmp_path: Path):
    LearningGovernor(tmp_path).propose(LearningPreference(character="emphasize"))
    narrowed = LearningGovernorConfig(
        permitted={"skills": ("allow", "defer", "emphasize"), "character": ("allow", "defer")}
    )
    gov = LearningGovernor(tmp_path, narrowed)
    assert gov.state().character.setting == "allow"
    [notice] = gov.drain_notices()
    assert "was 'emphasize'" in notice and "back to 'allow'" in notice
    assert gov.drain_notices() == [], "told once"
    assert LearningGovernor(tmp_path, narrowed).state().character.setting == "allow"


def test_deferral_expires_at_the_ceiling_and_stays_deferred(tmp_path: Path):
    gov = LearningGovernor(tmp_path, LearningGovernorConfig(max_deferred_sleeps=2))
    gov.propose(LearningPreference(character="defer"))

    assert gov.sleep_completed() == []
    assert gov.state().character.deferred_sleeps == 1
    assert gov.plan()["character"]["expired"] is False

    notices = gov.sleep_completed()
    assert len(notices) == 1 and "released without being consolidated" in notices[0]
    assert "still deferred" in notices[0]
    assert gov.plan()["character"] == {"setting": "defer", "expired": True, "expiry": "release"}
    assert gov.plan()["skills"]["expired"] is False
    assert gov.state().character.deferred_sleeps == 0

    gov.sleep_completed()
    assert gov.plan()["character"]["expired"] is False, "expiry is reported for one sleep only"


def test_changing_a_setting_restarts_the_holding_period(tmp_path: Path):
    gov = LearningGovernor(tmp_path, LearningGovernorConfig(max_deferred_sleeps=3))
    gov.propose(LearningPreference(skills="defer"))
    gov.sleep_completed()
    gov.propose(LearningPreference(skills="allow"))
    gov.propose(LearningPreference(skills="defer"))
    assert gov.state().skills.deferred_sleeps == 0


# ---------------------------------------------------------------------------
# Sleep counting
# ---------------------------------------------------------------------------


def _short_sleep() -> SleepConfig:
    return SleepConfig(cycles_between_sleep=2, drowsy_duration=1, nrem_duration=1, rem_duration=1, waking_duration=1)


def test_only_a_completed_sleep_counts():
    sleep = SleepCycleManager(_short_sleep())
    cycle = 0
    while sleep.completed_sleeps == 0:
        sleep.tick(cycle)
        cycle += 1
    assert sleep.completed_sleeps == 1

    sleep2 = SleepCycleManager(_short_sleep())
    sleep2.tick(0)
    sleep2.tick(1)  # falls asleep
    assert sleep2.is_sleeping
    sleep2.wake()
    sleep2.tick(2)
    assert not sleep2.is_sleeping and sleep2.completed_sleeps == 0


# ---------------------------------------------------------------------------
# Through the cycle
# ---------------------------------------------------------------------------


class Scripted:
    def __init__(self, script):
        self.inputs: list[CognitiveInput] = []
        self._script = script

    async def think(self, ci: CognitiveInput) -> CognitiveOutput:
        self.inputs.append(ci)
        return self._script(ci, len(self.inputs))


@pytest.mark.asyncio
async def test_the_mind_sets_its_learning_and_perceives_the_result(tmp_path: Path):
    def script(ci, n):
        if n == 1:
            return CognitiveOutput(learning_preference=LearningPreference(character="defer"))
        return CognitiveOutput()

    model = Scripted(script)
    cycle = CognitiveCycle(model=model, cycle_delay=0.0, learning_governor=LearningGovernor(tmp_path))
    await cycle.run(max_cycles=2)

    first, second = model.inputs
    assert first.learning_state.character.setting == "allow"
    assert second.learning_state.character.setting == "defer"
    told = [p.content for p in second.new_percepts if p.source == "learning"]
    assert "Your character learning is now 'defer' (was 'allow')." in told


@pytest.mark.asyncio
async def test_completed_sleeps_count_against_deferral_in_the_cycle(tmp_path: Path):
    gov = LearningGovernor(tmp_path, LearningGovernorConfig(max_deferred_sleeps=1))
    gov.propose(LearningPreference(skills="defer"))
    model = Scripted(lambda ci, n: CognitiveOutput())
    sleep = SleepCycleManager(_short_sleep())
    cycle = CognitiveCycle(model=model, cycle_delay=0.0, sleep_manager=sleep, learning_governor=gov)
    await cycle.run(max_cycles=10)

    assert sleep.completed_sleeps >= 1
    told = [p.content for ci in model.inputs for p in ci.new_percepts if p.source == "learning"]
    assert any("skills change you held" in t for t in told)


@pytest.mark.asyncio
async def test_without_a_governor_there_is_no_learning_state(tmp_path: Path):
    model = Scripted(lambda ci, n: CognitiveOutput())
    await CognitiveCycle(model=model, cycle_delay=0.0).run(max_cycles=1)
    assert model.inputs[0].learning_state is None


# ---------------------------------------------------------------------------
# Through the runner
# ---------------------------------------------------------------------------


def _config(tmp_path: Path, **fields) -> RunnerConfig:
    data_dir = tmp_path / "identity"
    data_dir.mkdir(parents=True)
    charter = data_dir / "charter.md"
    charter.write_text("# Charter\n\n## Value Seeds\n\n- **Care**: x\n", encoding="utf-8")
    return RunnerConfig(
        cycle_delay=0.01, data_dir=str(data_dir), charter_path=str(charter),
        use_in_memory_store=True, silence_threshold=999.0, stream_history=5, **fields,
    )


def test_minds_whose_weights_never_change_get_no_governor(tmp_path: Path):
    assert SanctuaryRunner(config=_config(tmp_path / "p")).learning_governor is None
    assert SanctuaryRunner(config=_config(tmp_path / "l", model_backend="llm")).learning_governor is None


def test_the_governor_can_be_switched_on_and_keeps_its_files_with_the_mind(tmp_path: Path):
    runner = SanctuaryRunner(config=_config(tmp_path, learning_control=True))
    runner.learning_governor.propose(LearningPreference(skills="emphasize"))
    assert (tmp_path / "identity" / "learning_settings.json").exists()
