"""Learning to avoid what is unpleasant, the Sanctuary half (2026-11-03;
Brian: "We still need Luthi to learn to avoid dangerous or unpleasant
things"; LuthiModel docs/research/2026-11-03_learning-to-avoid.md).

Pins:
- A percept can say how it felt (valence, -1..1), and nothing else changes
  when it does not.
- The adapter takes the cycle's strongest felt valence and hands it to the
  learner with the realized transition, sync and async.
- When a moment matters (strongly felt, or chosen to be remembered), the
  living layers keep more of it, scaled by the mind's own character
  setting: defer keeps nothing, emphasize doubles. Sleep's home drift
  follows the same setting.
- An M9 life's learning governor reports that it acts on the settings.

The adapter tests need the luthi package (set LUTHI_PATH); the schema and
runner tests do not.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from pydantic import ValidationError

from sanctuary.api.runner import RunnerConfig, SanctuaryRunner
from sanctuary.core.schema import (
    CognitiveInput,
    LearningRouteState,
    LearningState,
    Percept,
)


def test_a_percept_can_say_how_it_felt():
    assert Percept(modality="sensor", content="took damage", valence=-0.8).valence == -0.8
    assert Percept(modality="language", content="hi").valence is None
    with pytest.raises(ValidationError):
        Percept(modality="sensor", content="x", valence=-1.5)


def test_a_felt_event_survives_batching():
    from sanctuary.core.context_manager import BudgetConfig, ContextManager

    percepts = [Percept(modality="sensor", content=f"reading_{i}" * 10) for i in range(20)]
    percepts[3] = percepts[3].model_copy(update={"valence": -0.8})
    percepts[7] = percepts[7].model_copy(update={"valence": 0.4})
    compressed = ContextManager(BudgetConfig(new_percepts=100, chars_per_token=1)).compress(
        CognitiveInput(new_percepts=percepts)
    )
    assert len(compressed.new_percepts) < 20, "the burst was batched"
    assert [p.valence for p in compressed.new_percepts] == [-0.8, None]


def _sleeping_cycle():
    from sanctuary.consciousness.sleep_cycle import SleepConfig, SleepCycleManager, SleepStage
    from sanctuary.core.cognitive_cycle import CognitiveCycle
    from sanctuary.core.placeholder import PlaceholderModel

    sleep = SleepCycleManager(SleepConfig(sensory_gate_sleep=0.1))
    sleep._stage = SleepStage.NREM
    return CognitiveCycle(model=PlaceholderModel(), sleep_manager=sleep), sleep


@pytest.mark.asyncio
async def test_sleep_never_gates_out_a_felt_event():
    cycle, sleep = _sleeping_cycle()
    for i in range(10):
        cycle.inject_percept(Percept(modality="sensor", content=f"hum {i}"))
    cycle.inject_percept(Percept(modality="sensor", content="cold", valence=-0.3))
    ci = await cycle._assemble_input()
    assert [p.content for p in ci.new_percepts] == ["hum 0", "cold"]
    assert sleep.is_sleeping, "mild discomfort does not wake"


@pytest.mark.asyncio
async def test_something_that_hurts_wakes_the_mind():
    cycle, sleep = _sleeping_cycle()
    cycle.inject_percept(Percept(modality="sensor", content="took damage", valence=-0.7))
    await cycle._assemble_input()
    assert sleep._forced_wake


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


def test_an_m9_life_is_told_its_settings_are_acted_on(tmp_path):
    runner = SanctuaryRunner(config=_runner_config(
        tmp_path, model_backend="luthi", luthi_m9_run=str(tmp_path / "run"),
        luthi_life_dir=str(tmp_path / "life"),
    ))
    assert runner.learning_governor.state().consolidation_active is True
    assert runner._model.config.felt_valence is True


# ---------------------------------------------------------------------------
# The adapter (needs luthi)
# ---------------------------------------------------------------------------

_LUTHI_PATH = os.environ.get("LUTHI_PATH", "")
if _LUTHI_PATH and _LUTHI_PATH not in sys.path:
    sys.path.insert(0, _LUTHI_PATH)
try:
    from luthi.seam_types import ActionSelection, GenerationState, PlanSnapshot
    HAVE_LUTHI = True
except ImportError:
    HAVE_LUTHI = False

needs_luthi = pytest.mark.skipif(not HAVE_LUTHI, reason="luthi not importable; set LUTHI_PATH")
D = 8


def _ci(*valences, setting: str | None = None) -> CognitiveInput:
    percepts = [
        Percept(modality="sensor", content=f"event {i}", valence=v)
        for i, v in enumerate(valences)
    ]
    learning = None
    if setting is not None:
        learning = LearningState(
            skills=LearningRouteState(), character=LearningRouteState(setting=setting),
        )
    return CognitiveInput(new_percepts=percepts, learning_state=learning)


@needs_luthi
def test_the_strongest_felt_valence_counts():
    from sanctuary.core.luthi_model import LuthiModel

    assert LuthiModel._felt(_ci()) is None
    assert LuthiModel._felt(_ci(None, 0.3)) == 0.3
    assert LuthiModel._felt(_ci(0.6, -0.9, 0.2)) == -0.9


class _Actor:
    def select_action(self, s_t, **kw) -> "ActionSelection":
        return ActionSelection(
            action=torch.zeros(D), readable_summary="a",
            plan_snapshot=PlanSnapshot(
                visit_distribution=torch.tensor([1.0]),
                candidate_actions=torch.zeros(1, D), r_best=0.0,
            ),
        )


class _Sink:
    def __init__(self):
        self.ctxs: list[dict] = []

    def observe_transition(self, s_t, a_t, s_next, ctx: dict[str, Any]) -> dict:
        self.ctxs.append(ctx)
        return {}


def _adapter(**config):
    from sanctuary.core.luthi_model import LuthiModel, LuthiModelConfig

    a = LuthiModel(LuthiModelConfig(**config))
    a.device = torch.device("cpu")
    a.model_config = {"d_model": D, "n_blocks": 1, "seq_len": 32}
    a._loaded = True
    return a


def _state() -> "GenerationState":
    return GenerationState(s_t=torch.ones(1, D), ctx_latents=torch.ones(1, 4, D))


@needs_luthi
def test_the_learner_hears_how_the_next_state_felt():
    adapter = _adapter()
    sink = _Sink()
    adapter.attach_seam(actor=_Actor(), sink=sink)
    adapter._cycle_valence = None
    adapter._run_training_seam(_state())
    adapter._cycle_valence = -0.7
    adapter._run_training_seam(_state())
    assert sink.ctxs[-1]["valence"] == -0.7


@needs_luthi
def test_the_async_transition_carries_it():
    from sanctuary.core.async_learner import AsyncLearner, Transition

    sink = _Sink()
    import threading

    learner = AsyncLearner(sink, threading.Lock(), mode="drain")
    learner.submit(Transition(
        s_t=torch.zeros(1, D), a_t=torch.zeros(D), s_next=torch.zeros(1, D), valence=-0.5,
    ))
    assert sink.ctxs[-1]["valence"] == -0.5
    learner.submit(Transition(s_t=torch.zeros(1, D), a_t=torch.zeros(D), s_next=torch.zeros(1, D)))
    assert "valence" not in sink.ctxs[-1]


@pytest.fixture
def kept(monkeypatch):
    import luthi.sanctuary_interface as si

    calls: list[float] = []
    monkeypatch.setattr(si, "keep_recent", lambda model, fraction: calls.append(fraction) or 1)
    return calls


@needs_luthi
@pytest.mark.parametrize("setting,factor", [("allow", 1.0), ("emphasize", 2.0), ("defer", 0.0)])
def test_a_felt_moment_is_kept_by_the_minds_own_setting(kept, setting, factor):
    adapter = _adapter(keep_on_significance=0.1)
    adapter._cycle_valence = -0.8
    adapter._character_setting = setting
    adapter._keep_if_significant()
    if factor == 0.0:
        assert kept == []
    else:
        assert kept == [pytest.approx(0.1 * 0.8 * factor)]


@needs_luthi
def test_a_moment_chosen_to_be_remembered_is_kept_too(kept):
    adapter = _adapter(keep_on_significance=0.1)
    adapter._cycle_valence = None
    adapter._last_decoded = {"active": {"memory": True}, "memory_salience": 0.6}
    adapter._keep_if_significant()
    assert kept == [pytest.approx(0.06)]
    kept.clear()
    adapter._last_decoded = {"active": {"memory": False}, "memory_salience": 0.6}
    adapter._keep_if_significant()
    assert kept == [], "an unremarkable moment keeps nothing extra"


@needs_luthi
@pytest.mark.parametrize("setting,factor", [("allow", 1.0), ("defer", 0.0), ("emphasize", 2.0)])
def test_sleeps_home_drift_follows_the_setting(setting, factor):
    from luthi.v2.living_layer_pc import PredictiveCodingLayer

    adapter = _adapter(consolidation_rate=0.01)
    layer = PredictiveCodingLayer(D, D, num_episodes=2, context_dim=4, salience_threshold=1e9)

    class _Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.living_ffn = layer

    adapter.model = torch.nn.Module()
    adapter.model.blocks = torch.nn.ModuleList([_Block()])
    with torch.no_grad():
        layer.weight.add_(1.0)
    gap = (layer.weight - layer.set_point).clone()
    home = layer.set_point.clone()
    adapter._character_setting = setting
    adapter.consolidate()
    assert torch.allclose(layer.set_point, home + gap * 0.01 * factor)
