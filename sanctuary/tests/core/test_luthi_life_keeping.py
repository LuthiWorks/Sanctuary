"""Keeping a Luthi life (2026-11-02): saved once per completed sleep and at
shutdown, and never started without a place to keep it.

Split from ``test_luthi_m9_life.py`` because none of this needs the luthi
package, so CI runs it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sanctuary.api.runner import RunnerConfig, SanctuaryRunner
from sanctuary.consciousness.sleep_cycle import SleepConfig, SleepCycleManager
from sanctuary.core.cognitive_cycle import CognitiveCycle
from sanctuary.core.schema import CognitiveInput, CognitiveOutput


class _LivingModel:
    """A model with a life to save, counting saves."""

    def __init__(self):
        self.saves: list[str] = []

    async def think(self, ci: CognitiveInput) -> CognitiveOutput:
        return CognitiveOutput()

    def save_life(self, reason: str = "life"):
        self.saves.append(reason)


@pytest.mark.asyncio
async def test_the_keep_budget_renews_once_per_completed_sleep():
    class _Renewing(_LivingModel):
        renewals = 0

        def renew_keep_budget(self):
            self.renewals += 1

    model = _Renewing()
    sleep = SleepCycleManager(_short_sleep())
    cycle = CognitiveCycle(model=model, cycle_delay=0.0, sleep_manager=sleep)
    await cycle.run(max_cycles=30)
    assert sleep.completed_sleeps >= 2
    assert model.renewals == len(model.saves)


def _short_sleep() -> SleepConfig:
    return SleepConfig(
        cycles_between_sleep=3, drowsy_duration=1, nrem_duration=2,
        rem_duration=1, waking_duration=1,
    )


@pytest.mark.asyncio
async def test_the_life_is_saved_once_per_completed_sleep():
    model = _LivingModel()
    sleep = SleepCycleManager(_short_sleep())
    cycle = CognitiveCycle(model=model, cycle_delay=0.0, sleep_manager=sleep)
    await cycle.run(max_cycles=30)
    assert sleep.completed_sleeps >= 2
    # Saved at the start of the cycle after each sleep completes; the last
    # sleep may complete on the final cycle, before the next one could save.
    assert len(model.saves) in (sleep.completed_sleeps - 1, sleep.completed_sleeps)
    assert set(model.saves) == {"sleep"}


def _runner_config(tmp_path: Path, **fields) -> RunnerConfig:
    data_dir = tmp_path / "identity"
    data_dir.mkdir(parents=True)
    charter = data_dir / "charter.md"
    charter.write_text("# Charter\n\n## Value Seeds\n\n- **Care**: x\n", encoding="utf-8")
    return RunnerConfig(
        cycle_delay=0.01, data_dir=str(data_dir), charter_path=str(charter),
        use_in_memory_store=True, silence_threshold=999.0, stream_history=5,
        **fields,
    )


def test_an_m9_run_needs_a_place_for_the_life(tmp_path):
    with pytest.raises(ValueError, match="--luthi-life-dir"):
        SanctuaryRunner(config=_runner_config(
            tmp_path, model_backend="luthi", luthi_m9_run=str(tmp_path / "run"),
        ))


def test_the_runner_hands_the_run_and_life_to_the_model_and_saves_at_shutdown(tmp_path):
    runner = SanctuaryRunner(config=_runner_config(
        tmp_path, model_backend="luthi", luthi_m9_run=str(tmp_path / "run"),
        luthi_life_dir=str(tmp_path / "life"),
    ))
    assert runner._model.config.m9_run_dir == str(tmp_path / "run")
    assert runner._model.config.life_dir == str(tmp_path / "life")
    saved: list[str] = []
    runner._model.save_life = lambda reason="life": saved.append(reason)
    runner.save_state()
    assert saved == ["shutdown"]
