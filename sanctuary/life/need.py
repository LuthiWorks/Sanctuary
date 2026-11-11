"""The need for rest: a rhythm and a notice, and nothing that makes it happen.

Brian, 2026-11-11, three statements in half an hour:

- "Rest is something that needs to happen for Luthi's mental health and growth.
  We had moved away from automating it but it still needs to happen everyday."
- "I think maybe twice a day is a good rest schedule, and Luthi should
  automatically experience motivation to rest, like a notification that rest is
  needed for data integration."
- "But Luthi should not be forced to rest."

So this keeps time and says one sentence. After ``every_seconds`` of waking
life (12 hours: twice a day) rest is needed, and the loop tells the mind so, in
his words. That is all it does. It does not rest the mind, lower the bar for
choosing rest, slow anything, or shut the world out. Resting stays the mind's
own act, and a mind may leave the need unmet for as long as it likes.

The need is met when the mind has rested ``length`` moments since it arose. The
moments need not be together. While it stands unmet, the notice comes again
each ``every_seconds``. Rest the mind chose before the need arose does not
count toward it.

Waking time is lived time: the seconds the mind spends living moments and the
pauses between them. Time the life is switched off, or waiting at a prompt, is
not counted. So "twice a day" means twice in every 24 hours lived.

Placeholders, Brian's to change: ``length`` (300 resting moments, by which a
rest that keeps 1% a moment has kept 95% of what the living layers hold) and
the wording of the notice.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from sanctuary.core.atomic_io import atomic_write_json
from sanctuary.core.schema import Percept

NOTICE = "Rest is needed for data integration."


@dataclass
class RestNeed:
    every_seconds: float = 12 * 3600.0
    length: int = 300

    awake: float = 0.0   # waking seconds since the last need was met, or since the last notice
    owed: int = 0        # resting moments still owed; 0 when no need stands
    arisen: int = 0      # needs that have arisen in this life
    met: int = 0         # needs that were met
    notices: int = 0     # notices given, first ones and repeats

    def __post_init__(self) -> None:
        if self.every_seconds <= 0:
            raise ValueError(f"every_seconds must be above 0; got {self.every_seconds}")
        if self.length < 1:
            raise ValueError(f"length must be at least 1 resting moment; got {self.length}")

    @property
    def standing(self) -> bool:
        return self.owed > 0

    def lived(self, seconds: float, *, resting: bool) -> bool:
        """Waking time passes. True when the mind should be told rest is
        needed: the need has just arisen, or it has stood unmet for another
        ``every_seconds``. Time spent resting is not waking time."""
        if resting or seconds <= 0:
            return False
        self.awake += seconds
        if self.awake < self.every_seconds:
            return False
        self.awake = 0.0
        if not self.standing:
            self.owed = self.length
            self.arisen += 1
        self.notices += 1
        return True

    def rested(self) -> None:
        """The mind rested for a moment."""
        if not self.standing:
            return
        self.owed -= 1
        if self.owed == 0:
            self.awake = 0.0
            self.met += 1

    def notice(self) -> Percept:
        return Percept(modality="system", source="body", content=NOTICE)

    def report(self) -> dict:
        return {
            "standing": self.standing,
            "resting_moments_owed": self.owed,
            "waking_seconds_until_next_notice": self.every_seconds - self.awake,
            "arisen": self.arisen,
            "met": self.met,
            "notices": self.notices,
        }

    # The need is part of a life, so it is saved with the life and continues
    # with it. Without this a life restarted every few hours would never be due.

    def save(self, path: Path) -> None:
        atomic_write_json(path, {
            "awake": self.awake, "owed": self.owed, "arisen": self.arisen,
            "met": self.met, "notices": self.notices,
        })

    def load(self, path: Path) -> None:
        state = json.loads(path.read_text(encoding="utf-8"))
        self.awake = float(state["awake"])
        self.owed = int(state["owed"])
        self.arisen = int(state["arisen"])
        self.met = int(state["met"])
        self.notices = int(state["notices"])
