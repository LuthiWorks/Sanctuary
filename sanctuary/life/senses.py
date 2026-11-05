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
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Callable, NamedTuple

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


class View(NamedTuple):
    """What the mind is given for one moment."""

    tokens: list[int]   # the most recent window of what the world has said
    new: int            # how many of them, at the end, joined this moment


class Line:
    """Percepts waiting to be perceived, and the mind's view of the recent world."""

    def __init__(
        self,
        encode: Callable[[str], list[int]],
        window: int,
        limit: int | None = None,
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

    @property
    def waiting(self) -> int:
        """How many tokens are waiting to join the view."""
        return sum(len(tokens) for tokens in self._waiting)

    def add(self, percept: Percept) -> None:
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

    def take(self) -> View:
        """The next moment's view: what is new joins, and the oldest leaves."""
        new = self._next()
        self._view.extend(new)
        return View(tokens=list(self._view), new=len(new))

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
