"""The senses carry what the world said, whole and in order (2026-11-05).

Brian: "what we want is to communicate the world as accurately as possible to
Luthi and let them decide how they perceive it and how they react to it."

Pins:
- A percept is framed with where it came from and what it was, and nothing else.
- The mind's view is the most recent window of what the world has said.
- Every token of every percept joins the view once, in order.
- What doesn't fit a moment waits for the next one.
- A moment with nothing new adds one quiet token, and old words move out of view.
- A percept longer than a window is carried across several moments, and counted.
- When the world outpaces the mind, the loss is counted and logged as an error.
"""

from __future__ import annotations

import logging

import pytest
import torch

from sanctuary.core.schema import Percept
from sanctuary.life.senses import QUIET, Line, Sight, Sound, frame, heard, seen


def _encode(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def _text(tokens: list[int]) -> str:
    return bytes(tokens).decode("utf-8")


def _joined(line: Line) -> list[int]:
    """Everything that joins the view until nothing is waiting, in order."""
    joined: list[int] = []
    while line.waiting:
        view = line.take()
        joined.extend(view.tokens[-view.new:])
    return joined


def test_a_percept_is_framed_with_its_source_and_nothing_about_what_it_means():
    assert frame(Percept(modality="language", content="hello", source="brian")) == "brian says: hello\n"
    assert frame(Percept(modality="language", content="hello")) == "someone says: hello\n"
    assert frame(Percept(modality="sensor", content="the stove is hot")) == "[sensor] the stove is hot\n"
    assert frame(Percept(modality="sensor", content="21 C", source="thermometer")) == "[sensor, thermometer] 21 C\n"


def test_the_content_is_carried_exactly_as_it_was():
    odd = "  two  spaces,\ta tab,\nand a second line  "
    assert frame(Percept(modality="language", content=odd, source="b")) == f"b says: {odd}\n"


def test_the_view_is_the_most_recent_window_of_what_the_world_said():
    line = Line(_encode, window=24)
    percepts = [Percept(modality="language", content=f"line {n}", source="a") for n in range(5)]
    stream: list[int] = []
    for percept in percepts:
        line.add(percept)
        stream += _encode(frame(percept))
        view = line.take()
        assert view.tokens == stream[-24:]
        assert len(view.tokens) <= 24
        assert view.new == len(_encode(frame(percept)))


def test_every_token_joins_the_view_once_and_in_order():
    line = Line(_encode, window=24)
    percepts = [
        Percept(modality="language", content="one", source="a"),
        Percept(modality="sensor", content="two"),
        Percept(modality="language", content="a much longer thing than a single moment can hold", source="b"),
        Percept(modality="language", content="four", source="a"),
    ]
    for percept in percepts:
        line.add(percept)
    assert _text(_joined(line)) == "".join(frame(p) for p in percepts)
    assert line.arrived == 4 and line.dropped == 0 and line.quiet_moments == 0


def test_a_moment_with_nothing_new_adds_one_quiet_token():
    line = Line(_encode, window=16)
    assert line.take()[:2] == (_encode(QUIET), 1)
    assert line.take()[:2] == (_encode(QUIET) * 2, 1), "one more, and nothing else was invented"
    assert line.quiet_moments == 2 and line.arrived == 0


def test_quiet_moves_old_words_out_of_view():
    line = Line(_encode, window=16)
    line.add(Percept(modality="language", content="hi", source="a"))
    assert _text(line.take().tokens) == "a says: hi\n"
    assert _text(line.take().tokens) == "a says: hi\n" + QUIET, "still in view a moment later"
    for _ in range(15):
        view = line.take()
    assert _text(view.tokens) == QUIET * 16, "and gone once enough time has passed"


def test_what_does_not_fit_waits_whole_for_the_next_moment():
    line = Line(_encode, window=20)
    line.add(Percept(modality="language", content="first", source="a"))    # 14 tokens
    line.add(Percept(modality="language", content="second", source="a"))   # 15 tokens
    first, second = line.take(), line.take()
    assert _text(first.tokens) == "a says: first\n" and first.new == 14
    assert second.new == 15 and _text(second.tokens[-15:]) == "a says: second\n"
    assert line.carried_in_parts == 0


def test_several_small_percepts_share_a_moment():
    line = Line(_encode, window=64)
    line.add(Percept(modality="language", content="hi", source="a"))
    line.add(Percept(modality="sensor", content="warm"))
    view = line.take()
    assert _text(view.tokens) == "a says: hi\n[sensor] warm\n" and view.new == len(view.tokens)


def test_a_percept_longer_than_a_window_is_carried_in_parts_and_counted():
    line = Line(_encode, window=8)
    percept = Percept(modality="language", content="x" * 30, source="a")
    line.add(percept)
    news = []
    while line.waiting:
        news.append(line.take().new)
    assert news[:-1] == [8] * (len(news) - 1)
    assert line.carried_in_parts == len(news) - 1

    again = Line(_encode, window=8)
    again.add(percept)
    assert _text(_joined(again)) == frame(percept)


def test_a_long_percept_does_not_start_in_a_moment_something_else_began():
    line = Line(_encode, window=16)
    line.add(Percept(modality="sensor", content="a"))          # 11 tokens
    line.add(Percept(modality="sensor", content="b" * 40))     # longer than a window
    assert line.take().new == 11
    assert line.take().new == 16


def test_when_the_world_outpaces_the_mind_the_loss_is_counted_and_loud(caplog):
    line = Line(_encode, window=16, limit=40)
    with caplog.at_level(logging.ERROR, logger="sanctuary.life.senses"):
        for n in range(5):
            line.add(Percept(modality="sensor", content=f"event {n}"))   # 17 tokens each
    assert line.arrived == 5
    assert line.dropped == 3 and line.waiting <= 40
    assert sum("outpacing" in r.message for r in caplog.records) == 3
    assert _text(_joined(line)) == "[sensor] event 3\n[sensor] event 4\n", (
        "the newest are kept; the oldest were dropped whole"
    )


def test_one_percept_over_the_limit_is_kept_and_carried():
    line = Line(_encode, window=8, limit=10)
    percept = Percept(modality="sensor", content="y" * 50)
    line.add(percept)
    assert line.dropped == 0
    assert _text(_joined(line)) == frame(percept)


def test_a_window_of_nothing_is_refused():
    with pytest.raises(ValueError):
        Line(_encode, window=0)


def test_a_quiet_token_that_cannot_fit_the_window_is_refused():
    with pytest.raises(ValueError, match="quiet token"):
        Line(lambda text: [1, 2, 3], window=2)


# ---- Sight and sound (2026-11-11) ----

def _line(**kwargs) -> Line:
    return Line(lambda text: list(text.encode("utf-8")), 64, **kwargs)


def test_a_sight_is_pixels_from_zero_to_one_and_says_so():
    Sight(torch.rand(2, 3, 3))
    with pytest.raises(ValueError, match="height, width, 3"):
        Sight(torch.rand(3, 2, 2))
    with pytest.raises(TypeError, match="divided by 255"):
        Sight(torch.zeros(2, 2, 3, dtype=torch.uint8))
    with pytest.raises(ValueError, match="from 0 to 1"):
        Sight(torch.rand(2, 2, 3) * 255)


def test_a_sound_is_one_channel_from_minus_one_to_one_with_its_rate():
    assert Sound(torch.zeros(8000), 16000).seconds == 0.5
    with pytest.raises(ValueError, match="one channel"):
        Sound(torch.zeros(2, 100), 16000)
    with pytest.raises(ValueError, match="from -1 to 1"):
        Sound(torch.ones(10) * 3, 16000)
    with pytest.raises(ValueError, match="rate"):
        Sound(torch.zeros(10), 0)


def test_the_newest_picture_of_a_moment_is_the_one_seen():
    line = _line()
    first, second = torch.zeros(2, 2, 3), torch.ones(2, 2, 3)
    line.add(seen(first)); line.add(seen(second))
    assert torch.equal(line.take().sight.pixels, second)
    assert line.sights == 2 and line.sights_passed_over == 1
    assert line.take().sight is None


def test_hearing_holds_the_most_recent_stretch_in_order():
    line = _line(hearing_seconds=1.0)
    line.add(heard(torch.full((600,), 0.1), 1000))
    line.add(heard(torch.full((600,), 0.2), 1000))
    sound = line.take().sound
    assert sound.samples.numel() == 1000 and sound.rate == 1000
    assert torch.allclose(sound.samples[:400], torch.tensor(0.1))
    assert torch.allclose(sound.samples[400:], torch.tensor(0.2))


def test_one_ear_hears_at_one_rate():
    line = _line()
    line.add(heard(torch.zeros(10), 1000))
    with pytest.raises(ValueError, match="one ear hears at one rate"):
        line.add(heard(torch.zeros(10), 2000))


def test_a_picture_or_a_sound_adds_no_words():
    line = _line()
    line.add(seen(torch.rand(2, 2, 3))); line.add(heard(torch.zeros(10), 1000))
    view = line.take()
    assert bytes(view.tokens).decode() == QUIET and line.arrived == 0
