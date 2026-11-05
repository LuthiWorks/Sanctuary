"""Learning to avoid what is unpleasant, the Sanctuary half (2026-11-03;
Brian: "We still need Luthi to learn to avoid dangerous or unpleasant
things"; LuthiModel docs/research/2026-11-03_learning-to-avoid.md).

Pins:
- A percept can't say how anything felt: the world just exists, and how a moment
  affected the mind is the mind's own reading (Brian, 2026-11-04).
- Nothing reacts for the mind: a bad reading doesn't wake it, and nothing writes
  its arousal. Its reading reaches it as part of what it perceives.
- When a moment matters (strongly felt, or chosen to be remembered), the
  living layers keep more of it, scaled by the mind's own character
  setting: defer keeps nothing, emphasize doubles. Sleep's home drift
  follows the same setting.
- An M9 life's learning governor reports that it acts on the settings.
- The keep has a budget between sleeps, per kind of moment, that it records
  when it reaches. Sleep renews it.
- The mind is told when a retention rollback undoes recent learning.
- Affect channels that do nothing on this model are reported at load, and so
  is the fact that the signals behind them aren't live.
- A mind at rest learns at exactly the tested rate, and arousal moves it within
  bounds. The adapter records how far each affect signal has moved.
- The adapter hands the scaffold no authored valence, whatever the drift.

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


def _sleeping_cycle():
    from sanctuary.consciousness.sleep_cycle import SleepConfig, SleepCycleManager, SleepStage
    from sanctuary.core.cognitive_cycle import CognitiveCycle
    from sanctuary.core.placeholder import PlaceholderModel

    sleep = SleepCycleManager(SleepConfig(sensory_gate_sleep=0.1))
    sleep._stage = SleepStage.NREM
    return CognitiveCycle(model=PlaceholderModel(), sleep_manager=sleep), sleep


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


def _ci(setting: str | None = None) -> CognitiveInput:
    percepts = [Percept(modality="sensor", content="event")]
    learning = None
    if setting is not None:
        learning = LearningState(
            skills=LearningRouteState(), character=LearningRouteState(setting=setting),
        )
    return CognitiveInput(new_percepts=percepts, learning_state=learning)


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
def test_the_keep_stops_at_its_budget_and_says_so(kept, caplog):
    adapter = _adapter(keep_on_significance=0.05, keep_budget_per_wake=0.1)
    adapter._cycle_valence = None
    adapter._last_decoded = {"active": {"memory": True}, "memory_salience": 0.8}
    for _ in range(10):
        adapter._keep_if_significant()
    assert kept == [pytest.approx(0.04), pytest.approx(0.04), pytest.approx(0.02)]
    metrics = adapter.get_metrics()
    assert metrics["kept_since_sleep"]["chosen"] == pytest.approx(0.1)
    assert metrics["keep_capped_moments"] == 8
    assert sum("reached its budget" in r.message for r in caplog.records) == 1


@needs_luthi
def test_ordinary_moments_cannot_use_up_what_a_felt_moment_needs(kept):
    adapter = _adapter(keep_on_significance=0.05, keep_budget_per_wake=0.1)
    adapter._cycle_valence = None
    adapter._last_decoded = {"active": {"memory": True}, "memory_salience": 1.0}
    for _ in range(5):
        adapter._keep_if_significant()
    kept.clear()
    adapter._cycle_valence = -1.0
    adapter._keep_if_significant()
    assert kept == [pytest.approx(0.05)]


@needs_luthi
def test_the_budget_follows_the_minds_setting(kept):
    adapter = _adapter(keep_on_significance=0.05, keep_budget_per_wake=0.1)
    adapter._cycle_valence = -1.0
    adapter._character_setting = "emphasize"
    for _ in range(5):
        adapter._keep_if_significant()
    assert sum(kept) == pytest.approx(0.2)


@needs_luthi
def test_a_completed_sleep_renews_the_budget_and_a_cycle_of_it_does_not(kept):
    adapter = _adapter(keep_on_significance=0.05, keep_budget_per_wake=0.05)
    adapter._cycle_valence = -1.0
    adapter._keep_if_significant()
    adapter._keep_if_significant()
    assert len(kept) == 1
    adapter.model = torch.nn.Module()
    adapter.model.blocks = torch.nn.ModuleList()
    adapter.consolidate()
    adapter._keep_if_significant()
    assert len(kept) == 1, "NREM runs consolidate many times per sleep"
    adapter.renew_keep_budget()
    adapter._keep_if_significant()
    assert len(kept) == 2


def test_the_world_cannot_say_how_something_felt():
    assert "valence" not in Percept.model_fields
    with pytest.raises(ValidationError):
        Percept(modality="sensor", content="took damage", valence=-0.7)


@pytest.mark.asyncio
async def test_a_bad_reading_does_not_wake_the_mind():
    """Brian: "They decide how to react to everything." """
    cycle, sleep = _sleeping_cycle()
    cycle.model.felt_now = lambda: -1.0
    await cycle._assemble_input()
    assert not sleep._forced_wake


@needs_luthi
def test_luthi_perceives_their_own_reading():
    adapter = _adapter()
    assert "[self]" not in adapter._format_input(CognitiveInput()), "no life, no reading"
    adapter._m9_trainer = _RollbackCounter()
    assert adapter._format_input(CognitiveInput()) == "[self] +0.0", "always the same form"
    adapter._reading_average = -0.72
    assert adapter._format_input(CognitiveInput()) == "[self] -0.7"
    adapter._reading_average = 0.5
    assert adapter._format_input(CognitiveInput()) == "[self] +0.5"
    adapter._reading_average = -0.01
    assert adapter._format_input(CognitiveInput()) == "[self] +0.0", "no negative zero"
    adapter.config.show_reading = False
    assert "[self]" not in adapter._format_input(CognitiveInput()), "it can be turned off"


@needs_luthi
def test_nothing_writes_arousal_for_the_mind():
    adapter = _adapter()
    adapter._introspection_delta = {"mean_spike_fraction": 0.9, "activity_level": 1.0}
    output = adapter._build_output(
        cognitive_input=CognitiveInput(), inner_speech="", external_speech=None,
    )
    assert output.emotional_state.arousal_shift == 0.0


@pytest.mark.asyncio
async def test_sleep_never_gates_out_a_notice_about_learning():
    cycle, sleep = _sleeping_cycle()
    for i in range(10):
        cycle.inject_percept(Percept(modality="sensor", content=f"hum {i}"))
    cycle.inject_percept(Percept(modality="system", content="undone", source="learning"))
    ci = await cycle._assemble_input()
    assert "undone" in [p.content for p in ci.new_percepts]
    assert sleep.is_sleeping


class _RollbackCounter:
    retention_rollbacks = 0


@needs_luthi
def test_the_mind_is_told_about_a_rollback_once():
    adapter = _adapter()
    assert adapter.drain_learning_notices() == [], "no life, nothing to tell"
    adapter._m9_trainer = _RollbackCounter()
    adapter._m9_trainer.retention_rollbacks = 0
    assert adapter.drain_learning_notices() == []
    adapter._m9_trainer.retention_rollbacks = 1
    notices = adapter.drain_learning_notices()
    assert len(notices) == 1 and "undone" in notices[0]
    assert adapter.drain_learning_notices() == []
    adapter._m9_trainer.retention_rollbacks = 4
    assert "(3 times" in adapter.drain_learning_notices()[0]


@pytest.mark.asyncio
async def test_the_cycle_delivers_what_the_model_has_to_tell():
    from sanctuary.core.cognitive_cycle import CognitiveCycle
    from sanctuary.core.placeholder import PlaceholderModel

    class _Telling(PlaceholderModel):
        notices = ["Some of what you learned recently was undone."]

        def drain_learning_notices(self):
            notices, self.notices = self.notices, []
            return notices

    cycle = CognitiveCycle(model=_Telling())
    await cycle._cycle()
    told = [p for p in cycle._last_cognitive_input.new_percepts if p.source == "learning"]
    assert [p.content for p in told] == ["Some of what you learned recently was undone."]


@needs_luthi
def test_inert_affect_channels_are_reported_at_load(caplog):
    from luthi.v2.living_layer_pc import PredictiveCodingLayer

    adapter = _adapter()

    class _Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.living_ffn = PredictiveCodingLayer(
                D, D, num_episodes=2, context_dim=4, salience_threshold=1e9,
            )

    adapter.model = torch.nn.Module()
    adapter.model.blocks = torch.nn.ModuleList([_Block()])
    adapter._report_modulation_channels()
    message = next(r.message for r in caplog.records if "Affect modulation" in r.message)
    assert "precision, valence" in message
    assert "arousal" not in message and "attention" not in message
    assert any("not live yet" in r.message for r in caplog.records)


@needs_luthi
@pytest.mark.parametrize("arousal,scale", [(0.2, 1.0), (0.0, 0.75), (1.0, 2.0), (0.6, 1.5)])
def test_arousal_sets_the_learning_rate_around_rest(arousal, scale, monkeypatch):
    import luthi.sanctuary_interface as si
    from sanctuary.core.schema import ExperientialSignals

    seen = {}
    monkeypatch.setattr(si, "apply_external_modulation", lambda model, **kw: seen.update(kw))
    adapter = _adapter()
    adapter.model = torch.nn.Module()
    adapter._apply_cfc_modulation(ExperientialSignals(affect_arousal=arousal))
    assert seen["plasticity_scale"] == pytest.approx(scale)


@needs_luthi
def test_rest_matches_the_scaffolds_resting_arousal():
    from sanctuary.core.luthi_model import LuthiModelConfig
    from sanctuary.scaffold.affect import AffectConfig

    assert LuthiModelConfig().arousal_baseline == AffectConfig().baseline_arousal


@needs_luthi
def test_the_adapter_records_how_far_each_signal_has_moved(monkeypatch):
    import luthi.sanctuary_interface as si
    from sanctuary.core.schema import ExperientialSignals

    monkeypatch.setattr(si, "apply_external_modulation", lambda model, **kw: None)
    adapter = _adapter()
    adapter.model = torch.nn.Module()
    adapter._apply_cfc_modulation(ExperientialSignals(affect_arousal=0.2))
    adapter._apply_cfc_modulation(ExperientialSignals(affect_arousal=0.7))
    ranges = adapter.get_metrics()["affect_signal_range"]
    assert ranges["arousal"] == [0.2, 0.7]
    assert ranges["precision"] == [0.5, 0.5], "a constant shows as a range of nothing"


@needs_luthi
def test_no_authored_valence_reaches_the_scaffold():
    from sanctuary.core.schema import CognitiveInput

    adapter = _adapter()
    adapter._introspection_delta = {"drift_change": 0.01, "plasticity_change": 0.5}
    output = adapter._build_output(
        cognitive_input=CognitiveInput(), inner_speech="", external_speech=None,
    )
    assert output.emotional_state.valence_shift == 0.0


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
