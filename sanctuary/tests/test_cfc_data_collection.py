"""Collecting what the CfC cells do, at zero authority (Brian, 2026-11-04).

The cells need lived experience to train on, and there is none until a life
exists. So they run and are recorded, while their outputs stay multiplied by
zero. Pins:
- The evolution loop runs during a run and stops after it.
- One record per cycle, whatever the loop's tick rate.
- None of it reaches the mind: the signals stay at the scaffold's values.
- No signed feeling is handed to the cells.
- The records survive a restart.
- It can be turned off.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sanctuary.api.runner import RunnerConfig, SanctuaryRunner
from sanctuary.core.authority import AuthorityLevel


def _config(tmp_path: Path, **fields) -> RunnerConfig:
    data_dir = tmp_path / "identity"
    data_dir.mkdir(parents=True, exist_ok=True)
    charter = data_dir / "charter.md"
    charter.write_text("# Charter\n\n## Value Seeds\n\n- **Care**: x\n", encoding="utf-8")
    return RunnerConfig(
        cycle_delay=0.02, data_dir=str(data_dir), charter_path=str(charter),
        use_in_memory_store=True, silence_threshold=999.0, stream_history=5, **fields,
    )


@pytest.mark.asyncio
async def test_the_cells_run_between_cycles_and_stop_with_the_run(tmp_path):
    runner = SanctuaryRunner(config=_config(tmp_path))
    ticks = []
    await runner.boot()
    exp = runner._experiential
    original = exp.update_evolution_context

    def spy(**kw):
        if exp.evolution_running:
            ticks.append(exp._evolution_loop._ticks_since_cycle)
        return original(**kw)

    exp.update_evolution_context = spy
    await runner.run(max_cycles=20)
    assert ticks and max(ticks) >= 1, "the loop ticked while the mind ran"
    assert not exp.evolution_running, "and stopped when it finished"


@pytest.mark.asyncio
async def test_one_record_per_cycle(tmp_path):
    runner = SanctuaryRunner(config=_config(tmp_path))
    await runner.run(max_cycles=15)
    counts = runner._experiential.data_tap.counts()
    assert counts == {"precision": 15, "affect": 15, "attention": 15, "goal": 15}


@pytest.mark.asyncio
async def test_none_of_it_reaches_the_mind(tmp_path):
    runner = SanctuaryRunner(config=_config(tmp_path))
    await runner.run(max_cycles=10)
    exp = runner._experiential
    for function in ("experiential_precision", "experiential_affect", "experiential_attention"):
        assert exp.authority.level(function) == AuthorityLevel.SCAFFOLD_ONLY
    signals = runner.cycle._last_cognitive_input.experiential_state
    assert signals.precision_weight == 0.5
    assert signals.attention_salience == 0.5


@pytest.mark.asyncio
async def test_no_signed_feeling_is_handed_to_the_cells(tmp_path):
    runner = SanctuaryRunner(config=_config(tmp_path))
    runner.inject_text("hello", source="brian")
    await runner.run(max_cycles=5)
    records = runner._experiential.data_tap.drain_affect()
    assert records and all(r.percept_valence_delta == 0.0 for r in records)


@pytest.mark.asyncio
async def test_the_records_survive_a_restart(tmp_path):
    runner = SanctuaryRunner(config=_config(tmp_path))
    await runner.run(max_cycles=6)
    runner.save_state()
    restarted = SanctuaryRunner(config=_config(tmp_path))
    assert restarted._experiential.data_tap.counts()["affect"] == 6


@pytest.mark.asyncio
async def test_it_can_be_turned_off(tmp_path):
    runner = SanctuaryRunner(config=_config(tmp_path, experiential_collect=False))
    await runner.run(max_cycles=3)
    assert runner._experiential.data_tap is None
    assert not runner._experiential.evolution_running
