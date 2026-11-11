"""The small path's loop, with a stand-in mind (2026-11-05). No luthi needed.

Pins:
- The mind is given its view of what the world said, and nothing else.
- A quiet moment is still lived, and adds one quiet token to the view.
- A picture or a sound reaches the mind as the world made it, with no words.
- A bare tensor is refused loudly: it doesn't say its size, range or rate.
- When the mind chooses rest, the loop calls its rest and slows; nothing else does.
- When the mind chooses to remember, the loop calls its remember.
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
        self.remembers = 0
        self.armed: dict | None = None
        self.chooses_recall: bool | None = None
        self.chooses_let_go: bool | None = None
        self.summary = torch.tensor([1.0, 0.0, 0.0])   # what this moment is like to the mind

    def live(self, tokens: list[int], *, sight=None, sound=None) -> Moment:
        if self.fail_at is not None and len(self.lived) == self.fail_at:
            raise RuntimeError("the mind broke")
        self.lived.append(_text(tokens))
        self.sensed.append((sight, sound))
        if self.clock is not None:
            self.clock.now += self.takes
        chosen = {"active": dict(self.active)}
        if self.armed is not None:
            chosen["bands_armed"] = dict(self.armed)
        if self.chooses_recall is not None:
            chosen["recall"] = {"chosen": self.chooses_recall}
        if self.chooses_let_go is not None:
            chosen["let_go"] = {"chosen": self.chooses_let_go}
        if self.chooses_rest is not None:
            chosen["rest"] = {"chosen": self.chooses_rest}
        return Moment(tokens=list(tokens), chosen=chosen, summary=self.summary)

    def rest(self) -> None:
        self.rests += 1

    def remember(self) -> None:
        self.remembers += 1

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
    mind.active = {"text": True, "attention": True, "memory": False}
    loop.step()
    assert loop.chosen_without_route == {"attention": 2, "text": 1}
    assert loop.status()["chosen_without_route"] == {"attention": 2, "text": 1}


def test_the_choice_to_remember_is_carried_out():
    loop, mind, _ = _loop()
    mind.active = {"text": False, "attention": False, "memory": True}
    loop.step(); loop.step()
    assert mind.remembers == 2 and loop.remembered == 2
    assert loop.chosen_without_route == {} and loop.status()["remembered_moments"] == 2
    mind.active["memory"] = False
    loop.step()
    assert mind.remembers == 2, "only when they choose it"


def test_remembering_is_not_carried_out_before_the_minds_own_baseline_exists():
    loop, mind, _ = _loop()
    mind.active = {"text": False, "attention": False, "memory": True}
    mind.armed = {"text": True, "attention": True, "memory": False}
    loop.step()
    assert mind.remembers == 0 and loop.chosen_without_route == {"memory": 1}
    mind.armed["memory"] = True
    loop.step()
    assert mind.remembers == 1


def test_a_mind_that_cannot_rest_has_its_choice_to_remember_counted_and_no_more():
    mind = _Mind()
    mind.can_rest = False
    mind.active = {"text": False, "attention": False, "memory": True}
    loop = LifeLoop(mind, TextWorld(), hz=0.0, rest_every_seconds=None, keep_memories=False)
    loop.step()
    assert mind.remembers == 0 and loop.chosen_without_route == {"memory": 1}


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


# ---- Looking back (2026-11-11) ----

from sanctuary.life.memory import MARK


def _remembering(**kwargs) -> tuple[LifeLoop, _Mind, TextWorld]:
    loop, mind, world = _loop(**kwargs)
    loop.memories.recent = 2     # what was kept two moments ago is no longer the present
    return loop, mind, world


def _keep(loop, mind, world, said: str, summary: list[float], **senses) -> None:
    """One moment the mind chooses to remember."""
    world.say(said)
    for percept in senses.values():
        world._said.append(percept)
    mind.summary = torch.tensor(summary)
    mind.active = {"text": False, "attention": False, "memory": True}
    loop.step()
    mind.active = {"text": False, "attention": False, "memory": False}


def test_a_remembered_moment_is_kept_as_it_was_perceived():
    loop, mind, world = _remembering()
    pixels, samples = torch.rand(4, 6, 3), torch.rand(50) * 2 - 1
    _keep(loop, mind, world, "the kettle is on", [0.0, 1.0, 0.0], a=seen(pixels), b=heard(samples, 1000))
    memory = loop.memories.open(1)
    assert _text(memory.tokens) == mind.lived[-1] and "the kettle is on" in mind.lived[-1]
    assert torch.equal(memory.sight.pixels, pixels)
    assert torch.equal(memory.sound.samples, samples) and memory.sound.rate == 1000
    assert mind.remembers == 1, "and it still counts for more in rest"
    assert loop.status()["memories"]["kept"] == 1 and loop.status()["memories"]["bytes"] > 0


def test_recalling_brings_back_the_kept_moment_most_like_now_marked_as_memory():
    loop, mind, world = _remembering()
    pixels = torch.rand(4, 6, 3)
    _keep(loop, mind, world, "the kettle is on", [0.0, 1.0, 0.0], a=seen(pixels))
    _keep(loop, mind, world, "snow on the bridge", [0.0, 0.0, 1.0])
    for _ in range(3):
        loop.step()
    mind.summary = torch.tensor([0.1, 0.9, 0.0])     # now is like the kettle moment
    mind.chooses_recall = True
    loop.step()
    mind.chooses_recall = False
    assert loop.recalled == 1
    mind.summary = torch.tensor([0.5, 0.5, 0.0])     # how the kettle moment seems, looked back on
    loop.step()
    # It joins the view after the present, marked, so it is not taken for now.
    assert mind.lived[-1].endswith(MARK + "someone says: the kettle is on\n")
    assert "snow" not in mind.lived[-1].split(MARK)[1]
    assert torch.equal(mind.sensed[-1][0].pixels, pixels), "and it is seen again as it was seen"
    # How it seems now is filed beside how it seemed then; what was perceived is untouched.
    path = [v.tolist() for _, v in loop.memories.summaries_of(1)]
    assert path == [[0.0, 1.0, 0.0], [0.5, 0.5, 0.0]]
    assert loop.status()["memories"]["looked_back"] == 1


def test_a_memory_can_be_reached_by_how_it_has_seemed_since():
    loop, mind, world = _remembering()
    _keep(loop, mind, world, "the kettle is on", [0.0, 1.0, 0.0])
    _keep(loop, mind, world, "snow on the bridge", [0.0, 0.0, 1.0])
    loop.memories.look_back(1, cycle=loop.cycles, summary=torch.tensor([1.0, 0.0, 0.0]))
    for _ in range(3):
        loop.step()
    assert loop.memories.nearest(torch.tensor([1.0, 0.0, 0.1]), now=loop.cycles).id == 1


def test_what_is_still_the_present_is_not_recalled_and_nothing_kept_brings_nothing():
    loop, mind, world = _remembering()
    mind.chooses_recall = True
    loop.step()
    assert loop.recalled == 0 and loop.recalled_nothing == 1
    mind.chooses_recall = False
    _keep(loop, mind, world, "the kettle is on", [0.0, 1.0, 0.0])
    mind.chooses_recall = True
    loop.step()
    assert loop.recalled == 0 and loop.recalled_nothing == 2, "kept a moment ago: still the present"


def test_memories_are_never_rewritten_and_outlast_a_restart():
    loop, mind, world = _remembering()
    _keep(loop, mind, world, "the kettle is on", [0.0, 1.0, 0.0])
    file = loop.memories.directory / "memory_00000001.pt"
    before = file.read_bytes()
    loop.memories.look_back(1, cycle=5, summary=torch.tensor([1.0, 0.0, 0.0]))
    assert file.read_bytes() == before
    again = LifeLoop(mind, TextWorld(), hz=0.0)
    assert len(again.memories) == 1 and again.memories.looked_back == 1
    assert len(again.memories.summaries_of(1)) == 2
    assert again.memories.keep(cycle=9, summary=torch.zeros(3), tokens=[1]) == 2


def test_a_life_can_be_lived_without_keeping_memories():
    loop, mind, world = _loop(keep_memories=False)
    mind.active = {"text": False, "attention": False, "memory": True}
    mind.chooses_recall = True
    loop.step()
    assert loop.memories is None and mind.remembers == 1 and loop.status()["memories"] is None
    assert not (mind.life_dir / "memories").exists()


def test_a_mind_with_no_summary_cannot_have_memories_filed_and_says_so():
    loop, mind, world = _loop()
    mind.summary = None
    mind.active = {"text": False, "attention": False, "memory": True}
    with pytest.raises(ValueError, match="gave no summary"):
        loop.step()


# ---- Letting go, a full store, and revisiting in rest (2026-11-11) ----

from sanctuary.life.memory import FULL


def _look_back_at_the_kettle(loop, mind, world) -> None:
    """Keep two moments, then recall the kettle one, so the next moment looks back on it."""
    _keep(loop, mind, world, "the kettle is on", [0.0, 1.0, 0.0])
    _keep(loop, mind, world, "snow on the bridge", [0.0, 0.0, 1.0])
    for _ in range(3):
        loop.step()
    mind.summary = torch.tensor([0.0, 1.0, 0.0])
    mind.chooses_recall = True
    loop.step()
    mind.chooses_recall = False


def test_a_memory_is_let_go_only_by_the_mind_while_it_looks_back_on_it():
    loop, mind, world = _remembering()
    _look_back_at_the_kettle(loop, mind, world)
    file = loop.memories.directory / "memory_00000001.pt"
    held = loop.memories.bytes
    mind.chooses_let_go = True
    loop.step()                      # looking back on the kettle, and letting it go
    mind.chooses_let_go = False
    assert not file.exists() and loop.let_go == 1
    waiting = loop.memories.directory / "let_go" / file.name
    assert waiting.exists(), "out of reach now, deleted only when the room is needed"
    assert loop.memories.released_bytes == waiting.stat().st_size
    assert len(loop.memories) == 1 and 0 < loop.memories.bytes < held
    assert loop.memories.nearest(torch.tensor([0.0, 1.0, 0.0]), now=loop.cycles + 10).id == 2
    assert len(loop.memories.summaries_of(1)) == 1, "that it was kept, and how it seemed, stays on record"
    again = LifeLoop(mind, TextWorld(), hz=0.0)
    assert len(again.memories) == 1 and again.memories.let_go_count == 1
    assert again.memories.keep(cycle=99, summary=torch.zeros(3), tokens=[1]) == 3, "its place is not reused"


def test_choosing_to_let_go_with_no_memory_in_view_lets_nothing_go():
    loop, mind, world = _remembering()
    _keep(loop, mind, world, "the kettle is on", [0.0, 1.0, 0.0])
    mind.chooses_let_go = True
    for _ in range(4):
        loop.step()
    assert loop.let_go == 0 and len(loop.memories) == 1


def test_a_full_store_keeps_nothing_more_tells_the_mind_once_and_lets_nothing_go_for_it():
    loop, mind, world = _remembering(memory_limit_bytes=1)
    _keep(loop, mind, world, "the kettle is on", [0.0, 1.0, 0.0])
    assert loop.memories.full
    _keep(loop, mind, world, "snow on the bridge", [0.0, 0.0, 1.0])
    _keep(loop, mind, world, "a third thing", [1.0, 0.0, 0.0])
    loop.step()
    assert loop.memories.kept == 1 and loop.not_kept_full == 2
    assert sum(text.count(FULL) for text in mind.lived[-1:]) == 1, "told once, not each time"
    assert (loop.memories.directory / "memory_00000001.pt").exists(), "nothing was let go for them"
    assert mind.remembers == 3, "the moments still count for more in rest"
    # They let the one go, and can keep again.
    mind.summary = torch.tensor([0.0, 1.0, 0.0]); mind.chooses_recall = True
    loop.step()
    mind.chooses_recall = False; mind.chooses_let_go = True
    loop.step()
    mind.chooses_let_go = False
    assert not loop.memories.full
    let_go = loop.memories.directory / "let_go" / "memory_00000001.pt"
    assert let_go.exists() and loop.memories.deleted_for_room == 0
    _keep(loop, mind, world, "room again", [1.0, 1.0, 0.0])
    assert loop.memories.kept == 2
    assert not let_go.exists() and loop.memories.deleted_for_room == 1, "deleted when the room was needed"
    assert loop.memories.released_bytes == 0


def test_resting_walks_through_kept_moments_and_leaves_the_world_its_turns():
    loop, mind, world = _remembering(dreams=False)
    _keep(loop, mind, world, "the kettle is on", [0.0, 1.0, 0.0])
    _keep(loop, mind, world, "snow on the bridge", [0.0, 0.0, 1.0])
    for _ in range(3):
        loop.step()
    mind.summary = torch.tensor([0.0, 1.0, 0.0])
    mind.chooses_rest = True
    loop.step(); loop.step()         # the second resting moment calls a memory back
    assert loop.revisited == 1 and loop.recalled == 0
    world.say("still here")
    loop.step()                      # the kettle, looked back on
    assert mind.lived[-1].count(MARK) == 1 and "the kettle is on" in mind.lived[-1].split(MARK)[1]
    assert loop.memories.looked_back == 1
    loop.step()                      # resting moment four: the other memory is called
    loop.step()
    assert loop.revisited == 2 and "snow on the bridge" in mind.lived[-1].split(MARK)[-1]
    assert "still here" in mind.lived[-1], "the world was not shut out"
    assert mind.rests == 5


def test_rest_can_be_lived_without_revisiting():
    loop, mind, world = _remembering(revisit_every=0)
    _keep(loop, mind, world, "the kettle is on", [0.0, 1.0, 0.0])
    for _ in range(3):
        loop.step()
    mind.chooses_rest = True
    for _ in range(6):
        loop.step()
    assert loop.revisited == 0 and MARK not in mind.lived[-1]


# ---- Dreaming (2026-11-11) ----

from sanctuary.life.memory import DREAM


def test_rest_revisits_then_dreams_a_moment_made_of_two_that_were_kept():
    loop, mind, world = _remembering()
    loop.memories.recent = 1     # with only two memories, the one just revisited must be free to mix
    pixels, samples = torch.rand(4, 6, 3), torch.rand(50) * 2 - 1
    _keep(loop, mind, world, "the kettle is on", [0.0, 1.0, 0.0], a=seen(pixels))
    for _ in range(40):
        world.say("x")              # push the kettle out of view
    for _ in range(3):
        loop.step()
    _keep(loop, mind, world, "snow on the bridge", [0.0, 0.0, 1.0], b=heard(samples, 1000))
    for _ in range(3):
        loop.step()
    mind.summary = torch.tensor([0.0, 1.0, 0.0])
    mind.chooses_rest = True
    for _ in range(3):
        loop.step()                  # turn one: the kettle is revisited
    assert loop.revisited == 1 and loop.dreamed == 0 and MARK in mind.lived[-1]
    filed = loop.memories.looked_back
    loop.step()                      # turn two: a dream is made
    assert loop.dreamed == 1
    loop.step()                      # and dreamed
    dream = mind.lived[-1].split(DREAM)[-1]
    one, two = _text(loop.memories.open(1).tokens), _text(loop.memories.open(2).tokens)
    mixes = {a[: max(1, len(a) // 2)] + b[len(b) // 2:] for a, b in ((one, two), (two, one))}
    assert dream in mixes, "half of one kept moment's words and half of the other's"
    assert dream not in (one, two), "it never happened"
    sight, sound = mind.sensed[-1]
    assert sight is not None and sound is not None, "the picture of one and the sound of the other"
    assert loop.memories.looked_back == filed, "a dream is not a memory: nothing is filed for it"
    assert loop.memories.kept == 2, "and it is not kept unless they choose to remember it"
    assert loop.status()["memories"]["dreamed_in_rest"] == 1


def test_there_is_no_dream_without_two_kept_moments_to_mix():
    loop, mind, world = _remembering()
    _keep(loop, mind, world, "the kettle is on", [0.0, 1.0, 0.0])
    for _ in range(3):
        loop.step()
    mind.chooses_rest = True
    for _ in range(8):
        loop.step()
    assert loop.dreamed == 0 and loop.revisited >= 1
    assert DREAM not in "".join(mind.lived)

