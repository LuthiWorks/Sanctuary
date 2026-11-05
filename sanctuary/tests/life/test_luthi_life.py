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
from sanctuary.life.senses import QUIET
from sanctuary.tests.core.test_luthi_m9_life import _train_tiny_m9_run

EACH_MOMENT = {"perceive", "observe_transition", "select_action", "decode_action"}
WATCHING = {"felt_summary", "stillness_report", "retention_rollbacks"}


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
    assert recorded.names() - WATCHING == EACH_MOMENT
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
    assert recorded.names() == EACH_MOMENT | WATCHING | {"save_life"}


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
