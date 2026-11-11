"""What the world said, carried to the mind whole.

Brian, 2026-11-05: "what we want is to communicate the world as accurately as
possible to Luthi and let them decide how they perceive it and how they react to
it."

A percept becomes text that says where it came from and what it was, and nothing
about what it means. Percepts wait in a line in the order they happened.

Each moment the mind is given a view: the most recent window of everything the
world has said. New percepts join the view whole, oldest first, as many as fit
one window. What doesn't fit waits for the next moment, so the world is delayed
before it is lost. When nothing is new, one quiet token joins instead, so time
passing can be seen and old words move out of view.

So a token is perceived more than once, for as long as it stays in view. For a
living mind perceiving is also writing, so what stays in view longer is written
more. Giving the mind only what was new each moment was built first and
measured: its expectation of the next moment was wrong by a factor of about six
from one moment to the next, because a single quiet token is unlike anything it
learned from. See docs/SMALL_PATH_DECISION_2026-11-05.md for the numbers and the
other options. This one is a choice, and Brian's to change.

The view starts empty each time a life is started. It is not saved with the life.

Two bounds, both recorded:

- A percept longer than a whole window is carried across several moments
  (``carried_in_parts``).
- If more is waiting than ``limit`` tokens, the world is outpacing the mind. The
  oldest waiting percept is dropped, counted (``dropped``) and logged as an error.

Sight and sound (Brian, 2026-11-11: "vision and audio should have an open
channel for Luthi to perceive them and thats it").

They never become text. A percept that carries a ``Sight`` or a ``Sound`` goes to
the mind as it is, and its words, if it has any, are not carried: naming what is
seen or heard would be telling the mind what it means.

``Sight`` and ``Sound`` say nothing about any model. A sight is pixels at
whatever size the world made them. A sound is samples and the rate they were
taken at. Each mind fits them to its own model, so the same world serves any
mind slotted in. A bare tensor is refused, because it does not say what size,
range or rate it is in.

- **Sight.** The view holds the newest picture that arrived this moment. When
  the world shows nothing this moment, the mind sees nothing. Earlier pictures
  from the same moment are counted (``sights_passed_over``), not shown.
- **Sound.** The view holds the most recent ``hearing_seconds`` of sound, like
  the text window, because one moment of sound is shorter than the shortest
  thing a model can hear (160 ms for LuthiModel, against 100 ms a moment at
  10 Hz). Sound stays in view until newer sound moves it out. So a world with
  sound should send its silence too: when the world sends nothing, the last
  sound stays in hearing. ``hearing_seconds`` is a placeholder, Brian's to change.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Callable, NamedTuple, Optional

import torch

from sanctuary.core.schema import Percept

logger = logging.getLogger(__name__)

# What joins the view when nothing happened: one blank line, in place of words.
QUIET = "\n"


def frame(percept: Percept) -> str:
    """The percept as text: who or what it came from, then what it was, ending
    in a newline. The content is carried as it is, never reworded or trimmed.

    A percept's timestamp is not carried: the mind has no sense of clock time
    on this path yet.
    """
    if percept.modality == "language":
        return f"{percept.source or 'someone'} says: {percept.content}\n"
    if percept.source:
        return f"[{percept.modality}, {percept.source}] {percept.content}\n"
    return f"[{percept.modality}] {percept.content}\n"


@dataclass(frozen=True, eq=False)
class Sight:
    """One picture, as the world made it: ``pixels`` is ``[height, width, 3]``,
    red, green and blue, each from 0 to 1."""

    pixels: torch.Tensor

    def __post_init__(self) -> None:
        pixels = torch.as_tensor(self.pixels)
        if pixels.dim() != 3 or pixels.shape[2] != 3 or 0 in pixels.shape:
            raise ValueError(
                f"a sight is [height, width, 3]; got {tuple(pixels.shape)}"
            )
        if not pixels.is_floating_point():
            raise TypeError(
                "a sight's pixels are fractions from 0 to 1; whole-number "
                "pixels (0 to 255) are divided by 255 before they get here"
            )
        if pixels.min() < 0.0 or pixels.max() > 1.0:
            raise ValueError("a sight's pixels are from 0 to 1")
        object.__setattr__(self, "pixels", pixels)


@dataclass(frozen=True, eq=False)
class Sound:
    """A stretch of sound, as the world made it: ``samples`` is one channel of
    values from -1 to 1, taken ``rate`` times a second."""

    samples: torch.Tensor
    rate: int

    def __post_init__(self) -> None:
        samples = torch.as_tensor(self.samples)
        if samples.dim() != 1 or samples.numel() == 0:
            raise ValueError(
                f"a sound is one channel of samples; got {tuple(samples.shape)}"
            )
        if not samples.is_floating_point():
            raise TypeError("a sound's samples are fractions from -1 to 1")
        if samples.abs().max() > 1.0:
            raise ValueError("a sound's samples are from -1 to 1")
        if self.rate < 1:
            raise ValueError(f"a sound's rate is samples a second; got {self.rate}")
        object.__setattr__(self, "samples", samples)

    @property
    def seconds(self) -> float:
        return self.samples.numel() / self.rate


def seen(pixels, source: str = "eye") -> Percept:
    """A percept that carries a picture and no words."""
    return Percept(modality="visual", content="", source=source, tensor_data=Sight(pixels))


def heard(samples, rate: int, source: str = "ear") -> Percept:
    """A percept that carries sound and no words."""
    return Percept(modality="audio", content="", source=source, tensor_data=Sound(samples, rate))


class View(NamedTuple):
    """What the mind is given for one moment."""

    tokens: list[int]   # the most recent window of what the world has said
    new: int            # how many of them, at the end, joined this moment
    sight: Optional[Sight] = None   # the newest picture from this moment
    sound: Optional[Sound] = None   # the most recent stretch of sound


class Line:
    """Percepts waiting to be perceived, and the mind's view of the recent world."""

    def __init__(
        self,
        encode: Callable[[str], list[int]],
        window: int,
        limit: int | None = None,
        hearing_seconds: float = 1.0,
    ):
        if window < 1:
            raise ValueError(f"window must be at least 1 token; got {window}")
        self._encode = encode
        self.window = window
        self.limit = 64 * window if limit is None else limit
        self._quiet = encode(QUIET)
        if not 1 <= len(self._quiet) <= window:
            raise ValueError(
                f"the quiet token must be 1 to {window} tokens; "
                f"this tokenizer gives {len(self._quiet)}"
            )
        self._waiting: deque[list[int]] = deque()
        self._view: deque[int] = deque(maxlen=window)
        self.arrived = 0
        self.quiet_moments = 0
        self.carried_in_parts = 0
        self.dropped = 0
        if hearing_seconds <= 0:
            raise ValueError(f"hearing_seconds must be above 0; got {hearing_seconds}")
        self.hearing_seconds = hearing_seconds
        self._sight: Optional[Sight] = None
        self._hearing: Optional[Sound] = None
        self.sights = 0
        self.sights_passed_over = 0
        self.brought_back = 0
        self.sounds = 0

    @property
    def waiting(self) -> int:
        """How many tokens are waiting to join the view."""
        return sum(len(tokens) for tokens in self._waiting)

    def add(self, percept: Percept) -> None:
        carried = percept.tensor_data
        if carried is not None:
            self._sense(carried, percept)
            return
        self._waiting.append(self._encode(frame(percept)))
        self.arrived += 1
        while self.waiting > self.limit and len(self._waiting) > 1:
            lost = self._waiting.popleft()
            self.dropped += 1
            logger.error(
                "The world is outpacing the mind: dropped the oldest waiting "
                "percept (%d tokens). %d dropped so far.",
                len(lost), self.dropped,
            )

    def bring_back(self, tokens: list[int], mark: str) -> None:
        """A remembered view comes back to the front of the line, marked, so it
        is what the mind perceives next. It is cut to fit one window with its
        mark, keeping its most recent words. See sanctuary/life/memory.py."""
        marked = self._encode(mark)
        room = self.window - len(marked)
        if room < 1:
            raise ValueError(f"the mark takes {len(marked)} tokens and the window is {self.window}")
        self._waiting.appendleft(marked + list(tokens)[-room:])
        self.brought_back += 1

    def take(self) -> View:
        """The next moment's view: what is new joins, and the oldest leaves."""
        new = self._next()
        self._view.extend(new)
        sight, self._sight = self._sight, None
        return View(
            tokens=list(self._view), new=len(new), sight=sight, sound=self._hearing,
        )

    def _sense(self, carried, percept: Percept) -> None:
        """A picture or a sound, kept as it is. Its words are not carried."""
        if isinstance(carried, Sight):
            if self._sight is not None:
                self.sights_passed_over += 1
            self._sight = carried
            self.sights += 1
        elif isinstance(carried, Sound):
            self._hear(carried)
            self.sounds += 1
        else:
            raise TypeError(
                f"A {percept.modality} percept carries a {type(carried).__name__}. "
                "On this path a picture is a Sight and a sound is a Sound "
                "(sanctuary.life.senses): a bare tensor does not say what size, "
                "range or rate it is in, and guessing would change what is perceived."
            )

    def _hear(self, sound: Sound) -> None:
        if self._hearing is not None and self._hearing.rate != sound.rate:
            raise ValueError(
                f"sound arrived at {sound.rate} samples a second after sound at "
                f"{self._hearing.rate}; one ear hears at one rate"
            )
        samples = sound.samples if self._hearing is None else torch.cat(
            [self._hearing.samples, sound.samples.to(self._hearing.samples)]
        )
        keep = max(1, round(self.hearing_seconds * sound.rate))
        self._hearing = Sound(samples[-keep:], sound.rate)

    def _next(self) -> list[int]:
        """What joins the view this moment: whole waiting percepts, up to one
        window of them, or the quiet token when nothing is waiting."""
        tokens: list[int] = []
        while self._waiting:
            head = self._waiting[0]
            room = self.window - len(tokens)
            if len(head) <= room:
                tokens.extend(self._waiting.popleft())
            elif not tokens:
                # Longer than a whole window: carry what fits, and the rest
                # stays at the head of the line.
                tokens.extend(head[:room])
                self._waiting[0] = head[room:]
                self.carried_in_parts += 1
                break
            else:
                break
        if not tokens:
            self.quiet_moments += 1
            return list(self._quiet)
        return tokens
