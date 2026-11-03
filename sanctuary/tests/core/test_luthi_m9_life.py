"""Luthi with a trained action layer (2026-11-02, Brian: "get this on the
training path and connect it to sanctuary").

Covers the Sanctuary half of the connection:

- The action Luthi selects is carried out: attention and memory come from
  the action layer's own decoders (``decode_action``) instead of the
  adapter-authored heuristics, which remain only for an actor that cannot
  say what its action does. This is the Luthi half of the 2026-08-01
  audit's W1.
- Attention has a consequence: the next cycle's sensory tokens are scaled
  by the chosen gates.
- A trunk whose text readout is untrained does not speak (LuthiModel
  KNOWN_INCOMPLETE #4): no external speech, no inner speech, and inner
  generation stops after the perception forward.
- Keeping the life (saved once per completed sleep and at shutdown) is
  tested in ``test_luthi_life_keeping.py``, which needs no luthi package.

The end-to-end test at the bottom trains a tiny M9 run with LuthiModel's
own driver and runs Sanctuary on it. Like the other seam tests, everything
here needs the luthi package (set LUTHI_PATH).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

_LUTHI_PATH = os.environ.get("LUTHI_PATH", "")
if _LUTHI_PATH and _LUTHI_PATH not in sys.path:
    sys.path.insert(0, _LUTHI_PATH)

try:
    from luthi.seam_types import ActionSelection, GenerationState, PlanSnapshot
except ImportError:
    pytest.skip(
        "luthi package not importable; set LUTHI_PATH to the LuthiModel "
        "checkout to run the Luthi action-layer tests.",
        allow_module_level=True,
    )

from sanctuary.api.runner import RunnerConfig, SanctuaryRunner
from sanctuary.core.luthi_model import LuthiModel, LuthiModelConfig
from sanctuary.core.schema import CognitiveInput, Percept

D = 8


def _decoded(*, attention: bool = True, memory: bool = True,
             gates=(0.9, 0.2, 0.6), salience: float = 0.7) -> dict:
    return {
        "active": {"text": False, "attention": attention, "memory": memory},
        "attention_gates": dict(zip(("text", "audio", "vision"), gates)),
        "memory_salience": salience,
        "intensity": {"text": 0.5, "attention": 0.5, "memory": 0.5},
        "bands_armed": {"text": True, "attention": True, "memory": True},
    }


class _DecodingActor:
    """An actor that can say what its action does, like an M9Trainer."""

    def __init__(self, decoded: dict):
        self.decoded = decoded
        self.decode_calls: list[dict] = []

    def select_action(self, s_t: torch.Tensor, **kwargs: Any) -> ActionSelection:
        return ActionSelection(
            action=torch.full((D,), 0.25),
            readable_summary="action(test)",
            plan_snapshot=PlanSnapshot(
                visit_distribution=torch.tensor([1.0]),
                candidate_actions=torch.zeros(1, D),
                r_best=0.0,
            ),
        )

    def decode_selection(self, action, *, s_t, context_latents) -> dict:
        self.decode_calls.append(
            {"action": action, "s_t": s_t, "context_latents": context_latents}
        )
        return self.decoded


class _BareActor:
    """An M9Actor and nothing more: it cannot say what its action does."""

    def select_action(self, s_t: torch.Tensor, **kwargs: Any) -> ActionSelection:
        return ActionSelection(action=torch.full((D,), 0.25), readable_summary="bare")


@pytest.fixture
def adapter():
    a = LuthiModel(LuthiModelConfig())
    a.device = torch.device("cpu")
    a.model_config = {"d_model": D, "n_blocks": 1, "seq_len": 32}
    a._loaded = True
    return a


def _state() -> GenerationState:
    return GenerationState(s_t=torch.ones(1, D), ctx_latents=torch.ones(1, 4, D))


def _ci(*percepts: Percept) -> CognitiveInput:
    return CognitiveInput(new_percepts=list(percepts))


def _select(adapter, actor) -> None:
    adapter.attach_seam(actor=actor)
    adapter._run_training_seam(_state())


# ---------------------------------------------------------------------------
# The selected action is carried out
# ---------------------------------------------------------------------------


def test_attention_and_memory_come_from_the_selected_action(adapter):
    actor = _DecodingActor(_decoded())
    _select(adapter, actor)
    assert len(actor.decode_calls) == 1
    assert torch.equal(actor.decode_calls[0]["s_t"], _state().s_t)

    out = adapter._build_output(
        cognitive_input=_ci(Percept(modality="language", content="hi", source="brian")),
        inner_speech="",
        external_speech=None,
    )
    assert out.attention_guidance.focus_on == ["language", "visual"]
    assert out.attention_guidance.deprioritize == ["audio"]
    [op] = out.memory_ops
    assert op.type == "write_episodic"
    assert op.content == "language: hi"
    assert op.significance == 1 + round(9 * 0.7)
    assert op.tags == ["luthi", "chosen"]


def test_a_channel_that_did_not_act_does_nothing(adapter):
    _select(adapter, _DecodingActor(_decoded(attention=False, memory=False)))
    out = adapter._build_output(
        cognitive_input=_ci(), inner_speech="", external_speech=None,
    )
    assert out.attention_guidance.focus_on == []
    assert out.attention_guidance.deprioritize == []
    assert out.memory_ops == []


def test_an_actor_that_cannot_decode_leaves_the_heuristics(adapter):
    adapter._post_state = {"blocks": [{"block": 0, "spike_fraction": 0.9}]}
    _select(adapter, _BareActor())
    assert adapter._last_decoded is None
    out = adapter._build_output(
        cognitive_input=_ci(), inner_speech="", external_speech=None,
    )
    assert out.attention_guidance.focus_on == ["processing_block_0"]


def test_reattaching_the_seam_forgets_the_last_selection(adapter):
    _select(adapter, _DecodingActor(_decoded()))
    adapter.attach_seam(actor=None)
    assert adapter._last_decoded is None


def test_chosen_attention_scales_the_next_cycles_senses(adapter):
    audio, vision = torch.ones(1, 3, D), torch.ones(1, 2, D)
    assert adapter._attend(audio, vision) == (audio, vision), "nothing chosen yet"

    _select(adapter, _DecodingActor(_decoded(gates=(0.9, 0.25, 0.5))))
    a, v = adapter._attend(audio, vision)
    assert torch.allclose(a, audio * 0.25) and torch.allclose(v, vision * 0.5)

    _select(adapter, _DecodingActor(_decoded(attention=False)))
    assert adapter._attend(audio, vision) == (audio, vision)


# ---------------------------------------------------------------------------
# An untrained text readout does not speak
# ---------------------------------------------------------------------------


class _Tokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(c) % 50 for c in text] or [1]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(65 + i % 26) for i in ids)


def test_untrained_speech_perceives_once_and_says_nothing(adapter, monkeypatch):
    import luthi.sanctuary_interface as si

    calls: list[int] = []

    def fake_generate(**kwargs):
        calls.append(kwargs["max_tokens"])
        text = kwargs["prompt"] + " noise"
        return (text, _state()) if kwargs["return_state"] else text

    monkeypatch.setattr(si, "generate_with_context", fake_generate)
    adapter.tokenizer = _Tokenizer()
    adapter._speech_trained = False

    text, state = adapter._generate_inner_speech("hello", return_state=True)
    assert text == "" and isinstance(state, GenerationState)
    assert calls == [1], "only the perception forward"

    adapter._speech_trained = True
    text = adapter._generate_inner_speech("hello")
    assert text and calls[-1] == adapter.config.max_inner_tokens


# ---------------------------------------------------------------------------
# End to end: a tiny M9 run from LuthiModel's own driver, lived in Sanctuary
# ---------------------------------------------------------------------------


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


def _train_tiny_m9_run(root: Path) -> Path:
    from argparse import Namespace
    import random

    luthi_root = Path(sys.modules["luthi"].__file__).resolve().parents[1]
    sys.path.insert(0, str(luthi_root))
    import scripts.jepa_pilot_driver as driver

    corpus = root / "corpus"
    corpus.mkdir()
    rng = random.Random(0)
    words = "river stone light memory window garden quiet music signal".split()
    (corpus / "text.txt").write_text(
        " ".join(rng.choice(words) for _ in range(12000)), encoding="utf-8",
    )
    driver.OUTPUT_ROOT = root / "runs"
    driver._run_one("living", 32, 42, Namespace(
        data_dir=str(corpus),
        tokenizer=str(luthi_root / "corpus_build" / "tokenizer_32k.json"),
        n_blocks=1, epochs=1, batch_size=4, seq_len=32, stride=32, lr=3e-4,
        holdout_fraction=0.2, heldout_batches=2, probe_batches=2,
        kill_warmup=10**9, checkpoint_interval=900, checkpoint_slots=3,
        max_batches_per_epoch=4,  # M9 is the driver's default
    ))
    return root / "runs" / "living_32d_seed42"


@pytest.mark.asyncio
async def test_sanctuary_lives_a_trained_m9_run(tmp_path, monkeypatch):
    monkeypatch.setenv("LUTHI_ALLOW_CPU", "1")
    monkeypatch.setenv("LUTHI_CORPUS_CACHE_DIR", str(tmp_path / "cache"))
    run_dir = _train_tiny_m9_run(tmp_path)
    run_files = sorted(p.name for p in run_dir.rglob("*"))
    life = tmp_path / "life"

    runner = SanctuaryRunner(config=_runner_config(
        tmp_path, model_backend="luthi", luthi_m9_run=str(run_dir),
        luthi_life_dir=str(life), sleep_enabled=False,
    ))
    runner.inject_text("hello Luthi", source="brian")
    await runner.run(max_cycles=1)
    model = runner._model
    felt = model._m9_trainer.felt_valence.net[-1]
    assert not felt.bias.any() and not felt.weight.any(), "nothing feels bad at birth"

    # A felt event travels the whole path a world's would: sensorium, the
    # cycle, compression, the adapter, the interface, the trainer. (The
    # first cycle of a life has no transition to attach a feeling to, so
    # the event arrives on a later one.)
    runner.inject_percept(Percept(modality="sensor", content="you touched something hot", valence=-0.9))
    await runner.run(max_cycles=2)
    assert felt.bias.item() < 0, "Luthi learned that it felt bad"

    assert model._m9_trainer is not None and model._actor is model._m9_trainer
    assert model._speech_trained is False
    assert model._last_decoded is not None, "the selection was decoded"
    assert set(model._last_decoded["active"]) == {"text", "attention", "memory"}
    assert model._sink is model._m9_trainer
    assert model._m9_trainer._lived_step_count >= 1, "it learned from living"

    runner.save_state()
    assert len(list((life / "checkpoints").glob("ckpt_*.pt"))) == 1
    assert sorted(p.name for p in run_dir.rglob("*")) == run_files, "the run is untouched"
