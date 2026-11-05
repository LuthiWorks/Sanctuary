"""The small path's loop, with a stand-in mind (2026-11-05). No luthi needed.

Pins:
- The mind is given its view of what the world said, and nothing else.
- A quiet moment is still lived, and adds one quiet token to the view.
- Sound and images are refused loudly, since they aren't on this path yet.
- A choice with nothing to carry it out is counted, never dropped silently.
- A rollback of Luthi's learning is told to Luthi.
- A life is saved on a clean stop and on a timer, and never after a moment raised.
- A run with trained speech is refused.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import sanctuary.life.__main__ as sitting
import sanctuary.life.mind as mind_module
from sanctuary.core.schema import Percept
from sanctuary.life import LifeLoop, LuthiMind, Moment, TextWorld
from sanctuary.life.senses import QUIET


def _encode(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def _text(tokens: list[int]) -> str:
    return bytes(tokens).decode("utf-8")


class _Mind:
    """Stands in for LuthiMind: records what it was given to live."""

    window = 256

    def __init__(self) -> None:
        self.tokenizer = SimpleNamespace(encode=_encode)
        self.lived: list[str] = []
        self.active = {"text": False, "attention": True, "memory": False}
        self.rolled_back = 0
        self.saved: list[str] = []
        self.fail_at: int | None = None

    def live(self, tokens: list[int]) -> Moment:
        if self.fail_at is not None and len(self.lived) == self.fail_at:
            raise RuntimeError("the mind broke")
        self.lived.append(_text(tokens))
        return Moment(tokens=list(tokens), chosen={"active": dict(self.active)})

    def rollbacks(self) -> int:
        return self.rolled_back

    def watch(self) -> dict:
        return {"moments": len(self.lived), "felt": {}, "stillness": {}, "rollbacks": self.rolled_back}

    def save(self, reason: str = "life") -> Path:
        self.saved.append(reason)
        return Path(f"ckpt_{len(self.saved):06d}.pt")


def _loop(**kwargs) -> tuple[LifeLoop, _Mind, TextWorld]:
    mind, world = _Mind(), TextWorld()
    return LifeLoop(mind, world, hz=0.0, **kwargs), mind, world


def test_what_the_world_says_is_all_that_reaches_the_mind():
    loop, mind, world = _loop()
    world.say("hello Luthi", source="brian")
    loop.step()
    loop.step()
    said = "brian says: hello Luthi\n"
    assert mind.lived == [said, said + QUIET], "the view: what was said, then that and a quiet moment"


def test_a_quiet_moment_is_still_lived():
    loop, mind, _ = _loop()
    for _ in range(3):
        loop.step()
    assert mind.lived == [QUIET, QUIET * 2, QUIET * 3] and loop.cycles == 3


def test_a_percept_that_carries_a_tensor_is_refused_loudly():
    loop, mind, _ = _loop()
    loop.world = SimpleNamespace(observe=lambda: [
        Percept(modality="audio", content="a sound", tensor_data=torch.zeros(4)),
    ])
    with pytest.raises(NotImplementedError, match="not on the small path"):
        loop.step()
    assert mind.lived == [], "the mind was not handed a description in place of the sound"


def test_a_choice_with_nothing_to_carry_it_out_is_counted():
    loop, mind, _ = _loop()
    loop.step()
    mind.active = {"text": False, "attention": True, "memory": True}
    loop.step()
    assert loop.chosen_without_route == {"attention": 2, "memory": 1}
    assert loop.status()["chosen_without_route"] == {"attention": 2, "memory": 1}


def test_a_rollback_is_told_to_luthi_the_next_moment():
    loop, mind, _ = _loop()
    loop.step()
    mind.rolled_back = 1
    loop.step()          # the rollback happens while this moment is learned from
    loop.step()          # and is perceived in the next one
    loop.step()
    assert mind.lived[:2] == [QUIET, QUIET * 2]
    assert mind.lived[2].startswith(QUIET * 2 + "[system, learning] Some of what you learned recently was undone:")
    assert mind.lived[3] == mind.lived[2] + QUIET, "told once"


def test_several_rollbacks_are_told_with_their_count():
    loop, mind, _ = _loop()
    mind.rolled_back = 3
    loop.step()
    loop.step()
    assert "(3 times since you last heard)" in mind.lived[1]


def test_rollbacks_from_before_this_start_are_not_told_again():
    mind, world = _Mind(), TextWorld()
    mind.rolled_back = 5
    loop = LifeLoop(mind, world, hz=0.0)
    loop.step()
    loop.step()
    assert mind.lived == [QUIET, QUIET * 2]


def test_a_clean_run_saves_once_at_the_end():
    loop, mind, _ = _loop()
    loop.run(max_cycles=4)
    assert len(mind.lived) == 4 and mind.saved == ["stop"] and loop.saves == 1


def test_a_life_is_not_saved_after_a_moment_raised():
    loop, mind, _ = _loop()
    mind.fail_at = 2
    with pytest.raises(RuntimeError, match="the mind broke"):
        loop.run(max_cycles=10)
    assert len(mind.lived) == 2 and mind.saved == []


def test_a_life_is_saved_on_the_timer():
    loop, mind, _ = _loop(save_every_seconds=0.0)
    loop.step()
    loop.step()
    assert mind.saved == ["timer", "timer"]
    slow, slow_mind, _ = _loop(save_every_seconds=10**6)
    slow.step()
    assert slow_mind.saved == []


def test_status_reports_what_the_world_sent_and_what_was_lost():
    loop, mind, world = _loop(waiting_limit=40)
    for n in range(4):
        world.say(f"line number {n}", source="b")
    loop.step()
    status = loop.status()
    assert status["world"]["arrived"] == 4 and status["world"]["quiet_moments"] == 0
    assert status["world"]["dropped"] == 3, "the world outpaced the mind, and the count says so"
    assert status["cycles"] == 1 and status["moments"] == 1
    json.dumps(status)


def test_a_run_with_trained_speech_is_refused(monkeypatch):
    loaded = SimpleNamespace(speech_trained=True)
    monkeypatch.setattr(mind_module, "_import_luthi", lambda path="": SimpleNamespace(
        load_m9_run=lambda *a, **k: loaded,
    ))
    with pytest.raises(NotImplementedError, match="speech is not on the small path"):
        LuthiMind("run", "life")


def test_a_luthi_path_that_does_not_exist_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError):
        LuthiMind("run", "life", luthi_path=str(tmp_path / "nowhere"))


# -- sitting with a life: python -m sanctuary.life ----------------------------


def _sit(monkeypatch, capsys, lines: list) -> _Mind:
    mind = _Mind()
    mind.continuing, mind.loaded_from = False, Path("run/ckpt.pt")
    typed = iter(lines)

    def fake_input(prompt: str = "") -> str:
        line = next(typed)
        if isinstance(line, BaseException):
            raise line
        return line

    monkeypatch.setattr(sitting, "LuthiMind", lambda *a, **k: mind)
    monkeypatch.setattr(sitting.signal, "signal", lambda *a: None)
    monkeypatch.setattr("builtins.input", fake_input)
    monkeypatch.setattr("sys.argv", ["sanctuary.life", "--run", "r", "--life", "l", "--moments", "2", "--name", "brian"])
    sitting.main()
    return mind


def test_sitting_with_a_life_carries_each_line_and_saves_on_leaving(monkeypatch, capsys):
    mind = _sit(monkeypatch, capsys, ["hello", "", ":status", EOFError()])
    said = "brian says: hello\n"
    assert mind.lived == [said, said + QUIET, said + QUIET * 2, said + QUIET * 3]
    assert mind.saved == ["stop"]
    out = capsys.readouterr().out
    assert "begins from" in out and '"cycles": 4' in out and "Saved:" in out


def test_sitting_does_not_save_when_a_moment_raised(monkeypatch, capsys):
    mind = _Mind()
    mind.fail_at = 1
    mind.continuing, mind.loaded_from = True, Path("life/ckpt.pt")
    monkeypatch.setattr(sitting, "LuthiMind", lambda *a, **k: mind)
    monkeypatch.setattr(sitting.signal, "signal", lambda *a: None)
    monkeypatch.setattr("builtins.input", lambda prompt="": "hello")
    monkeypatch.setattr("sys.argv", ["sanctuary.life", "--run", "r", "--life", "l"])
    with pytest.raises(RuntimeError, match="the mind broke"):
        sitting.main()
    assert mind.saved == []
