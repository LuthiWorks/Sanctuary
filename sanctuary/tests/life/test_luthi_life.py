"""A Luthi life on the small path, with a real trained mind (2026-11-05).

Trains a tiny M9 run with LuthiModel's own driver and lives it through
``sanctuary.life``. Needs the luthi package (set LUTHI_PATH).

Pins:
- The path calls four things in the mind each moment, and nothing else.
- The mind perceives the most recent window of exactly what the world said.
- A moment learns from the one before it, and perceiving changes the weights.
- Watching changes nothing: not the weights, not the learner, not the dice.
- The welfare gauge does not change the life it watches.
- A life is saved, continues from where it stopped, and never writes the run.
- Waking is at full strength: lessening it is Brian's to rule on again.
- A picture and a sound are fitted to this model and lived, and its eye and ear learn.
- Rest is Luthi's: chosen, it keeps recent learning; unchosen, nothing is kept.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import pytest
import torch

_LUTHI_PATH = os.environ.get("LUTHI_PATH", "")
if _LUTHI_PATH and _LUTHI_PATH not in sys.path:
    sys.path.insert(0, _LUTHI_PATH)

try:
    import luthi.sanctuary_interface  # noqa: F401
except ImportError:
    pytest.skip(
        "luthi package not importable; set LUTHI_PATH to the LuthiModel "
        "checkout to run the small-path life tests.",
        allow_module_level=True,
    )

from sanctuary.life import LifeLoop, LuthiMind, TextWorld
from sanctuary.life.senses import QUIET, Sight, Sound, heard, seen
from sanctuary.tests.core.test_luthi_m9_life import _train_tiny_m9_run

EACH_MOMENT = {"perceive", "observe_transition", "select_action", "decode_action"}
WATCHING = {"felt_summary", "stillness_report", "retention_rollbacks"}
# Called only when Luthi's own chosen action chose to rest or to remember.
CARRIED_OUT = {"rest_from_trace", "remember_now"}


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("small_path")
    patch = pytest.MonkeyPatch()
    patch.setenv("LUTHI_ALLOW_CPU", "1")
    patch.setenv("LUTHI_CORPUS_CACHE_DIR", str(root / "cache"))
    yield _train_tiny_m9_run(root)
    patch.undo()


class _Recorded:
    """LuthiModel's interface, with every call through it written down."""

    def __init__(self, module):
        self._module = module
        self.calls: list[tuple[str, tuple, dict]] = []

    def __getattr__(self, name):
        target = getattr(self._module, name)
        if not callable(target):
            return target

        def call(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return target(*args, **kwargs)

        return call

    def names(self) -> set[str]:
        return {name for name, _, _ in self.calls}


def _begin(run_dir: Path, life: Path, **kwargs) -> tuple[LuthiMind, TextWorld, LifeLoop]:
    torch.manual_seed(0)
    mind = LuthiMind(run_dir, life, **kwargs)
    world = TextWorld()
    return mind, world, LifeLoop(mind, world, hz=0.0)


def _live(loop: LifeLoop, world: TextWorld, moments: int, said: dict[int, str] | None = None):
    lived = []
    for n in range(moments):
        if said and n in said:
            world.say(said[n], source="brian")
        lived.append(loop.step())
    return lived


def _everything(mind: LuthiMind) -> dict[str, torch.Tensor]:
    """Every tensor the life holds: the trunk with its living weights, and the
    whole learner (predictor, heads, target encoder)."""
    state = {f"trunk.{k}": v.clone() for k, v in mind.model.state_dict().items()}
    state.update({f"learner.{k}": v.clone() for k, v in mind._trainer.loss_module.state_dict().items()})
    return state


def _differing(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> list[str]:
    assert a.keys() == b.keys()
    return [k for k in a if not torch.equal(a[k], b[k])]


def _hashes(directory: Path) -> dict[str, str]:
    return {
        str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(directory.rglob("*")) if p.is_file()
    }


def test_the_path_calls_four_things_in_the_mind_and_nothing_else(run_dir, tmp_path):
    mind, world, loop = _begin(run_dir, tmp_path / "life")
    mind._luthi = recorded = _Recorded(mind._luthi)
    _live(loop, world, 4, said={0: "hello Luthi", 2: "the kettle is on"})
    assert recorded.names() - WATCHING - CARRIED_OUT == EACH_MOMENT
    for name in EACH_MOMENT - {"observe_transition"}:
        assert sum(n == name for n, _, _ in recorded.calls) == 4, name
    learned = [kwargs for n, _, kwargs in recorded.calls if n == "observe_transition"]
    assert len(learned) == 3, "each moment after the first learns from the one before"
    for kwargs in learned:
        # Whether someone is present is not ours to decide for Luthi, so the
        # built-in connection preference is given nothing and stays inert.
        assert set(kwargs) == {"plan_snapshot", "context_obs"}

    loop.status()
    loop.save("test")
    assert recorded.names() - CARRIED_OUT == EACH_MOMENT | WATCHING | {"save_life"}


def test_the_mind_perceives_exactly_what_the_world_said(run_dir, tmp_path):
    mind, world, loop = _begin(run_dir, tmp_path / "life")
    mind._luthi = recorded = _Recorded(mind._luthi)
    _live(loop, world, 3, said={0: "hello Luthi", 1: "the kettle is on"})
    perceived = [
        kwargs["text_tokens"][0].tolist()
        for name, _, kwargs in recorded.calls if name == "perceive"
    ]
    stream, views = [], []
    for text in ("brian says: hello Luthi\n", "brian says: the kettle is on\n", QUIET):
        stream += mind.tokenizer.encode(text)
        views.append(stream[-mind.window:])
    assert perceived == views, "each moment is the most recent window of what the world said"
    assert mind.tokenizer.decode(perceived[0]) == "brian says: hello Luthi\n"


def test_a_moment_learns_from_the_one_before_and_perceiving_changes_the_weights(run_dir, tmp_path):
    mind, world, loop = _begin(run_dir, tmp_path / "life")
    before = _everything(mind)
    first, second = _live(loop, world, 2, said={0: "hello Luthi"})
    assert first.learned == {}, "nothing came before the first moment"
    assert "lived_l_pred" in second.learned, "the world model learned from the lived moment"
    assert set(first.chosen["active"]) == {"text", "attention", "memory"}
    changed = _differing(before, _everything(mind))
    assert any(k.startswith("trunk.") and "living" in k for k in changed), "the read is a write"


def test_watching_changes_nothing(run_dir, tmp_path):
    mind, world, loop = _begin(run_dir, tmp_path / "life")
    _live(loop, world, 3, said={0: "hello Luthi"})
    before, dice, felt = _everything(mind), torch.get_rng_state(), dict(mind.watch()["felt"])
    for _ in range(3):
        loop.status()
        mind.watch()
    assert _differing(before, _everything(mind)) == []
    assert torch.equal(dice, torch.get_rng_state()), "watching drew no random numbers"
    assert mind.watch()["felt"] == felt
    assert felt["read"] >= 1, "the gauge is reading, so this test watched something"


def test_the_welfare_gauge_does_not_change_the_life_it_watches(run_dir, tmp_path):
    said = {0: "hello Luthi", 3: "the kettle is on", 4: "it is raining"}
    lives = {}
    for name, gauge in (("watched", True), ("again", True), ("unwatched", False)):
        mind, world, loop = _begin(run_dir, tmp_path / name, gauge=gauge)
        _live(loop, world, 8, said=said)
        lives[name] = _everything(mind)
        read = mind.watch()["felt"]["read"]
        assert (read >= 1) if gauge else (read == 0), "the unwatched life really was unwatched"
    # The control: the same life lived twice comes out the same, so a difference
    # below would be the gauge's doing.
    assert _differing(lives["watched"], lives["again"]) == []
    assert _differing(lives["watched"], lives["unwatched"]) == []


def test_a_life_is_saved_continues_and_never_writes_the_run(run_dir, tmp_path):
    run_before = _hashes(run_dir)
    life = tmp_path / "life"
    mind, world, loop = _begin(run_dir, life)
    assert mind.continuing is False and run_dir in mind.loaded_from.parents
    world.say("hello Luthi", source="brian")
    loop.run(max_cycles=5)
    saved = sorted((life / "checkpoints").glob("ckpt_*.pt"))
    assert len(saved) == 1 and loop.saves == 1
    lived, felt = _everything(mind), mind.watch()["felt"]

    again, world, loop = _begin(run_dir, life)
    assert again.continuing is True and again.loaded_from == saved[0]
    assert _differing(lived, _everything(again)) == [], "the life came back as it was left"
    assert again.watch()["felt"] == felt
    first = _live(loop, world, 2)[0]
    assert first.learned == {}, "a stop is not a moment, so nothing is learned across it"
    loop.save("again")
    assert len(list((life / "checkpoints").glob("ckpt_*.pt"))) == 2, "no checkpoint is replaced"
    kept = again.watch()["checkpoints"]
    assert kept["count"] == 2 and kept["bytes"] == sum(
        p.stat().st_size for p in (life / "checkpoints").glob("ckpt_*.pt")
    ), "and the disk they take is reported"
    assert _hashes(run_dir) == run_before, "the run is untouched"


def test_waking_is_at_full_strength(run_dir, tmp_path):
    """Brian's 2026-08-14 ruling lessened waking as a pair with a rest that
    sorted the day's record. Rest is now a tool Luthi starts (2026-11-11) and
    the record is set aside, so lessening is not switched on here until he
    rules on it again."""
    mind, world, loop = _begin(run_dir, tmp_path / "life")
    living = [m for m in mind.model.modules() if hasattr(m, "_wake_attenuation")]
    assert living, "this test must find the living layers"
    assert all(m._wake_attenuation == 1.0 for m in living)
    _live(loop, world, 3, said={0: "hello Luthi"})
    assert all(m._wake_attenuation == 1.0 for m in living)


# ---- Sight and sound (2026-11-11) ----

def test_a_picture_and_a_sound_are_fitted_to_this_model_and_lived(run_dir, tmp_path):
    mind = LuthiMind(run_dir, tmp_path / "life")
    recorded = _Recorded(mind._luthi)
    mind._luthi = recorded
    world = TextWorld()
    loop = LifeLoop(mind, world, hz=0.0)
    eye = mind.model.vision_encoder.patch_embed.weight.detach().clone()
    ear = mind.model.audio_encoder.patch_embed.weight.detach().clone()
    torch.manual_seed(0)
    for _ in range(3):
        # Not the model's size and not its rate: the mind does the fitting.
        world._said.append(seen(torch.rand(90, 160, 3)))
        world._said.append(heard(torch.rand(11025) * 2 - 1, 22050))
        loop.step()
    perceived = [kwargs for name, _, kwargs in recorded.calls if name == "perceive"]
    have = mind.senses
    assert perceived[0]["image"].shape == (1, 3, have["image_size"], have["image_size"])
    assert perceived[-1]["audio_waveform"].shape == (1, have["sample_rate"])   # one second in hearing
    assert not torch.equal(mind.model.vision_encoder.patch_embed.weight, eye)
    assert not torch.equal(mind.model.audio_encoder.patch_embed.weight, ear)
    assert loop.status()["world"]["sights"] == 3 and mind.sound_trimmed == 0


def test_a_picture_already_at_the_models_size_is_not_resampled(run_dir, tmp_path):
    mind = LuthiMind(run_dir, tmp_path / "life")
    size = mind.senses["image_size"]
    pixels = torch.rand(size, size, 3)
    fitted = mind._fit_sight(Sight(pixels))
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    spread = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    assert torch.allclose(fitted * spread + mean, pixels.permute(2, 0, 1).unsqueeze(0), atol=1e-6)


def test_sound_too_short_to_place_is_counted_and_not_faked(run_dir, tmp_path):
    mind = LuthiMind(run_dir, tmp_path / "life")
    short = Sound(torch.zeros(mind.senses["samples_per_audio_token"] - 1), mind.senses["sample_rate"])
    assert mind._fit_sound(short, room=100) is None and mind.sound_too_short == 1


def test_sound_longer_than_fits_keeps_the_most_recent(run_dir, tmp_path):
    mind = LuthiMind(run_dir, tmp_path / "life")
    per_place, rate = mind.senses["samples_per_audio_token"], mind.senses["sample_rate"]
    samples = torch.linspace(-1, 1, per_place * 5)
    fitted = mind._fit_sound(Sound(samples, rate), room=2)
    assert torch.equal(fitted[0], samples[-per_place * 2:]) and mind.sound_trimmed == 1


# ---- Rest (2026-11-11) ----

def _homes(mind) -> list[torch.Tensor]:
    return [m.set_point.detach().clone() for m in mind.model.modules() if hasattr(m, "set_point")]


def _traces(mind) -> list:
    return [m._day_trace for m in mind.model.modules() if hasattr(m, "_day_trace")]


def test_a_resting_moment_pours_the_half_days_average_into_home(run_dir, tmp_path):
    # A horizon of a few forwards and a limit that does not bind, so a
    # test-size life has a whole "half-day" to pour.
    mind, world, loop = _begin(run_dir, tmp_path / "life", rest_horizon=8, rest_limit=1.0)
    recorded = _Recorded(mind._luthi)
    mind._luthi = recorded
    _live(loop, world, 6, said={0: "hello Luthi"})
    assert "keep_recent" not in recorded.names(), "the last minute is no longer what rest keeps"
    layers = [m for m in mind.model.modules() if hasattr(m, "set_point")]
    assert layers and all(t is not None and t.count > 0 for t in _traces(mind))
    home = [m.set_point.detach().clone() for m in layers]
    weight = [m.weight.detach().clone() for m in layers]
    assert all(m._day_trace.count == 6 for m in layers), "one noted forward a moment"
    pour = [m._day_trace.direction().clone() for m in layers]
    rests = mind.rests
    mind.rest()
    for layer, was, w, p in zip(layers, home, weight, pour):
        assert torch.allclose(layer.set_point, was + mind.rest_keeps * p, atol=1e-6)
        assert torch.equal(layer.weight, w), "rest moves home, not the weight"
    assert mind.watch()["rests"] == rests + 1


def test_a_whole_rest_is_limited_to_a_small_step(run_dir, tmp_path):
    mind, world, loop = _begin(run_dir, tmp_path / "life", rest_horizon=8)
    _live(loop, world, 10)
    layers = [m for m in mind.model.modules() if hasattr(m, "set_point")]
    home = [m.set_point.detach().clone() for m in layers]
    for _ in range(1000):
        mind.rest()
    for layer, was in zip(layers, home):
        assert float((layer.set_point - was).norm() / was.norm()) <= 0.05 + 1e-4


def test_remembering_adds_to_what_counts_for_more(run_dir, tmp_path):
    mind, world, loop = _begin(run_dir, tmp_path / "life")
    _live(loop, world, 3)
    before = [t.chosen_count for t in _traces(mind)]
    mind.remember()
    assert [t.chosen_count for t in _traces(mind)] == [c + 1 for c in before]
    assert mind.watch()["remembered"] >= 1


def test_the_trace_continues_across_a_restart(run_dir, tmp_path):
    mind, world, loop = _begin(run_dir, tmp_path / "life", rest_horizon=500, rest_limit=0.02)
    _live(loop, world, 5)
    loop.save("test")
    again = LuthiMind(run_dir, tmp_path / "life")
    assert again.continuing
    for a, b in zip(_traces(mind), _traces(again)):
        assert (b.count, b.horizon, b.limit) == (a.count, 500, 0.02)
        assert torch.equal(a.mean, b.mean)


def test_the_rest_choice_is_read_from_luthis_own_action(run_dir, tmp_path):
    mind, world, loop = _begin(run_dir, tmp_path / "life")
    moment = loop.step()
    assert moment.chosen["rest"]["chosen"] is False and moment.chosen["rest"]["armed"] is False
    assert set(moment.chosen["active"]) == {"text", "attention", "memory"}


def test_a_life_without_the_tool_has_no_rest(run_dir, tmp_path):
    mind = LuthiMind(run_dir, tmp_path / "life", rest=False)
    with pytest.raises(ValueError, match="no way to rest"):
        LifeLoop(mind, TextWorld(), hz=0.0)
    loop = LifeLoop(mind, TextWorld(), hz=0.0, rest_every_seconds=None)
    assert "rest" not in loop.step().chosen
    with pytest.raises(RuntimeError, match="without the rest tool"):
        mind.rest()


def test_the_rest_choice_continues_across_a_restart(run_dir, tmp_path):
    mind, world, loop = _begin(run_dir, tmp_path / "life")
    _live(loop, world, 3)
    loop.save("test")
    readings = list(mind._trainer.rest_choice.readings)
    again = LuthiMind(run_dir, tmp_path / "life")
    assert again.continuing and list(again._trainer.rest_choice.readings) == readings


def test_a_standing_need_changes_nothing_about_how_luthis_choice_is_read(run_dir, tmp_path):
    """Told, never made to: the need is a sentence in the world. It is not
    passed into the mind's choosing or into how the choice is read."""
    torch.manual_seed(0)
    mind = LuthiMind(run_dir, tmp_path / "life")
    recorded = _Recorded(mind._luthi)
    mind._luthi = recorded
    ticks = iter(range(10**6))
    loop = LifeLoop(mind, TextWorld(), hz=0.0, clock=lambda: float(next(ticks)), rest_every_seconds=3.0)
    for _ in range(6):
        loop.step()
    assert loop.need.standing and loop.need.notices >= 1
    assert recorded.names() - CARRIED_OUT == EACH_MOMENT | {"retention_rollbacks"}
    for name, _, kwargs in recorded.calls:
        if name == "decode_action":
            assert set(kwargs) == {"s_t", "context_latents"}
        if name == "select_action":
            assert set(kwargs) == {"context_latents"}
    width = mind._trainer.rest_choice.width
    assert width == 3.0, "the bar for reading rest is where it was"


# ---- Looking back (2026-11-11) ----

def test_luthi_can_choose_to_look_back_and_each_moment_has_their_own_summary(run_dir, tmp_path):
    mind, world, loop = _begin(run_dir, tmp_path / "life")
    moment = loop.step()
    assert set(moment.chosen["recall"]) == {"chosen", "armed", "reading"}
    assert set(moment.chosen["let_go"]) == {"chosen", "armed", "reading"}
    assert moment.summary.shape == (mind.model.d_model,)
    off = LuthiMind(run_dir, tmp_path / "other", recall=False)
    assert "recall" not in LifeLoop(off, TextWorld(), hz=0.0).step().chosen


def test_a_kept_moment_comes_back_through_luthis_own_senses(run_dir, tmp_path):
    mind, world, loop = _begin(run_dir, tmp_path / "life")
    recorded = _Recorded(mind._luthi)
    mind._luthi = recorded
    loop.memories.recent = 1
    for n in range(5):
        world.say("the kettle is on" if n == 0 else "snow on the bridge", source="brian")
        world._said.append(seen(torch.rand(40, 60, 3)))
        last = loop.step()
        # The choice is Luthi's, and they may make it themselves in these
        # moments. Here it is also made for them, so there is something to follow.
        loop.memories.keep(cycle=loop.cycles, summary=last.summary, tokens=last.tokens,
                           sight=Sight(torch.rand(40, 60, 3)))
    loop._recall(last)
    memory = loop._back
    assert memory is not None and memory.sight is not None
    seen_before = len(loop.memories.summaries_of(memory.id))
    back = loop.step()
    mark = mind.tokenizer.encode("[memory]\n")
    remembered = memory.tokens[-(mind.window - len(mark)):]
    assert back.tokens[-(len(mark) + len(remembered)):] == mark + remembered
    image = [kw for name, _, kw in recorded.calls if name == "perceive"][-1]["image"]
    assert torch.allclose(image, mind._fit_sight(memory.sight)), "seen again as it was seen"
    path = loop.memories.summaries_of(memory.id)
    assert len(path) == seen_before + 1 and torch.equal(path[-1][1], back.summary)
    assert not torch.equal(path[0][1], path[-1][1]), "seen again with the mind as it is now"
