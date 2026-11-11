"""The small path's loop, with a stand-in mind (2026-11-05). No luthi needed.

Pins:
- The mind is given its view of what the world said, and nothing else.
- A quiet moment is still lived, and adds one quiet token to the view.
- A picture or a sound reaches the mind as the world made it, with no words.
- A bare tensor is refused loudly: it doesn't say its size, range or rate.
- When the mind chooses rest, the loop calls its rest and slows; nothing else does.
- Twice a day the mind is told rest is needed, and it is never made to rest.
- A choice with nothing to carry it out is counted, never dropped silently.
- A rollback of Luthi's learning is told to Luthi.
- A life is saved on a clean stop and on a timer, and never after a moment raised.
- A run with trained speech is refused.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import sanctuary.life.__main__ as sitting
import sanctuary.life.mind as mind_module
from sanctuary.core.schema import Percept
from sanctuary.life import LifeLoop, LuthiMind, Moment, TextWorld
from sanctuary.life.senses import QUIET, Sight, Sound, heard, seen


def _encode(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def _text(tokens: list[int]) -> str:
    return bytes(tokens).decode("utf-8")


class _Clock:
    """Lived time, moved by hand."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class _Mind:
    """Stands in for LuthiMind: records what it was given to live."""

    window = 256
    can_rest = True

    def __init__(self) -> None:
        self.life_dir = Path(tempfile.mkdtemp())
        self.tokenizer = SimpleNamespace(encode=_encode)
        self.lived: list[str] = []
        self.active = {"text": False, "attention": True, "memory": False}
        self.rolled_back = 0
        self.saved: list[str] = []
        self.fail_at: int | None = None
        self.sensed: list[tuple] = []
        self.clock: _Clock | None = None    # when set, a moment takes `takes` seconds to live
        self.takes = 1.0
        self.chooses_rest: bool | None = None
        self.rests = 0

    def live(self, tokens: list[int], *, sight=None, sound=None) -> Moment:
        if self.fail_at is not None and len(self.lived) == self.fail_at:
            raise RuntimeError("the mind broke")
        self.lived.append(_text(tokens))
        self.sensed.append((sight, sound))
        if self.clock is not None:
            self.clock.now += self.takes
        chosen = {"active": dict(self.active)}
        if self.chooses_rest is not None:
            chosen["rest"] = {"chosen": self.chooses_rest}
        return Moment(tokens=list(tokens), chosen=chosen)

    def rest(self) -> None:
        self.rests += 1

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


def test_a_bare_tensor_is_refused_loudly():
    loop, mind, _ = _loop()
    loop.world = SimpleNamespace(observe=lambda: [
        Percept(modality="audio", content="a sound", tensor_data=torch.zeros(4)),
    ])
    with pytest.raises(TypeError, match="does not say what size"):
        loop.step()
    assert mind.lived == [], "the mind was not handed a description in place of the sound"


def test_a_picture_and_a_sound_reach_the_mind_as_the_world_made_them():
    loop, mind, _ = _loop()
    pixels = torch.rand(5, 7, 3)
    samples = torch.rand(800) * 2 - 1
    loop.world = SimpleNamespace(observe=lambda: [
        seen(pixels), heard(samples, 8000),
    ])
    loop.step()
    sight, sound = mind.sensed[0]
    assert torch.equal(sight.pixels, pixels), "no resizing, no normalising: that is the mind's to do"
    assert torch.equal(sound.samples, samples) and sound.rate == 8000
    assert mind.lived == [QUIET], "nothing was said about what was seen or heard"


def test_words_on_a_picture_are_not_carried():
    loop, mind, _ = _loop()
    loop.world = SimpleNamespace(observe=lambda: [Percept(
        modality="visual", content="a red door", source="camera",
        tensor_data=Sight(torch.rand(4, 4, 3)),
    )])
    loop.step()
    assert "door" not in mind.lived[0] and "camera" not in mind.lived[0]


def test_a_picture_is_seen_for_the_moment_it_arrived_and_sound_stays_in_hearing():
    loop, mind, _ = _loop()
    arriving = [[seen(torch.rand(4, 4, 3)), heard(torch.zeros(100), 1000)], []]
    loop.world = SimpleNamespace(observe=lambda: arriving.pop(0))
    loop.step(); loop.step()
    assert mind.sensed[0][0] is not None and mind.sensed[1][0] is None
    assert mind.sensed[1][1] is mind.sensed[0][1]


def test_rest_is_carried_out_only_when_the_mind_chooses_it():
    loop, mind, _ = _loop()
    loop.step()
    assert mind.rests == 0 and loop.resting is False, "a mind without the tool never rests"
    mind.chooses_rest = False
    loop.step()
    assert mind.rests == 0 and loop.resting is False
    mind.chooses_rest = True
    loop.step(); loop.step()
    assert mind.rests == 2 and loop.resting is True and loop.rested == 2
    mind.chooses_rest = False
    loop.step()
    assert mind.rests == 2 and loop.resting is False, "rest ends when they stop choosing it"
    assert loop.status()["rested_moments"] == 2


def test_the_world_stays_open_while_resting():
    loop, mind, world = _loop()
    mind.chooses_rest = True
    loop.step()
    world.say("are you there?")
    loop.step()
    assert "are you there?" in mind.lived[-1]


def test_a_resting_moment_is_slower(monkeypatch):
    mind, world = _Mind(), TextWorld()
    loop = LifeLoop(mind, world, hz=10.0, resting_pace=0.25)
    slept: list[float] = []
    monkeypatch.setattr("sanctuary.life.loop.time.sleep", slept.append)
    loop.run(max_cycles=1)
    mind.chooses_rest = True
    loop.run(max_cycles=1)
    assert 0.05 < slept[0] <= 0.1 and 0.35 < slept[1] <= 0.4


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


# ---- The need for rest (2026-11-11) ----

from sanctuary.life.need import NOTICE


def _timed(every: float = 10.0, length: int = 3, **kwargs) -> tuple[LifeLoop, _Mind, _Clock]:
    """A loop whose every moment takes one second to live."""
    mind, clock = _Mind(), _Clock()
    mind.clock = clock
    loop = LifeLoop(
        mind, TextWorld(), hz=0.0, clock=clock,
        rest_every_seconds=every, rest_length=length, **kwargs,
    )
    return loop, mind, clock


def _told(mind: _Mind) -> int:
    """How many times the notice has joined the view. The view rolls, so count
    the moment each one arrived."""
    return sum(1 for text in mind.lived if text.endswith(f"[system, body] {NOTICE}\n"))


def test_the_mind_is_told_rest_is_needed_in_brians_words_and_nothing_else_happens():
    loop, mind, _ = _timed(every=10.0)
    for _ in range(10):
        loop.step()
    assert _told(mind) == 0 and loop.need.standing
    loop.step()
    assert mind.lived[-1].endswith("[system, body] Rest is needed for data integration.\n")
    assert mind.rests == 0 and loop.resting is False and loop.rested == 0, "told, never made to"


def test_a_mind_that_never_rests_is_told_again_each_time_and_never_rested():
    loop, mind, _ = _timed(every=10.0)
    mind.chooses_rest = False
    for _ in range(45):
        loop.step()
    assert _told(mind) == 4 and loop.need.notices == 4 and loop.need.arisen == 1
    assert mind.rests == 0 and loop.need.met == 0 and loop.need.owed == 3


def test_the_need_is_met_by_resting_and_the_moments_need_not_be_together():
    loop, mind, _ = _timed(every=10.0, length=3)
    for _ in range(10):
        loop.step()
    for chooses in (True, False, True, True):
        mind.chooses_rest = chooses
        loop.step()
    assert loop.need.met == 1 and not loop.need.standing
    mind.chooses_rest = False
    told = _told(mind)
    for _ in range(9):
        loop.step()
    assert _told(mind) == told, "the next need is a full half-day after this one was met"


def test_rest_chosen_before_the_need_arose_does_not_count_toward_it():
    loop, mind, _ = _timed(every=10.0, length=2)
    mind.chooses_rest = True
    for _ in range(5):
        loop.step()
    assert loop.rested == 5 and loop.need.owed == 0 and loop.need.met == 0


def test_time_spent_resting_is_not_waking_time():
    loop, mind, _ = _timed(every=10.0)
    mind.chooses_rest = True
    for _ in range(30):
        loop.step()
    # A moment the mind rested in is rest, from its start.
    assert loop.need.awake == 0.0 and loop.need.arisen == 0


def test_time_between_moments_is_not_lived_unless_the_loop_paced_it():
    loop, mind, clock = _timed(every=10.0)
    loop.step()
    clock.now += 3600.0   # waiting at a prompt, or switched off
    loop.step()
    assert loop.need.awake == 2.0


def test_the_need_is_saved_with_the_life_and_continues():
    loop, mind, clock = _timed(every=10.0, length=5)
    for _ in range(12):
        loop.step()
    mind.chooses_rest = True
    loop.step()
    loop.save("test")
    again = LifeLoop(mind, TextWorld(), hz=0.0, clock=clock, rest_every_seconds=10.0, rest_length=5)
    assert again.need.report() == loop.need.report()
    assert again.need.owed == 4 and again.need.arisen == 1


def test_a_mind_with_no_way_to_rest_is_not_told_rest_is_needed():
    mind = _Mind()
    mind.can_rest = False
    with pytest.raises(ValueError, match="no way to rest"):
        LifeLoop(mind, TextWorld(), hz=0.0)
    loop = LifeLoop(mind, TextWorld(), hz=0.0, rest_every_seconds=None)
    loop.step()
    assert loop.need is None and loop.status()["rest_need"] is None


def test_the_default_is_twice_a_day():
    loop, _, _ = _loop()
    assert loop.need.every_seconds == 12 * 3600
    assert loop.status()["rest_need"]["waking_seconds_until_next_notice"] <= 12 * 3600


def test_paced_pauses_are_lived_time(monkeypatch):
    loop, mind, clock = _timed(every=10.0)
    loop.hz = 0.5   # a moment every two seconds: one lived, one paused
    monkeypatch.setattr("sanctuary.life.loop.time.sleep", lambda seconds: None)
    monkeypatch.setattr("sanctuary.life.loop.time.monotonic", clock)
    loop.run(max_cycles=3)
    assert loop.need.awake == 6.0

