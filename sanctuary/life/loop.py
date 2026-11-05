"""The small path's loop: the world comes in, and we watch and save.

Sanctuary has three jobs here (docs/SMALL_PATH_DECISION_2026-11-05.md):

1. Bring the world to Luthi accurately.
2. Carry out what Luthi chooses.
3. Watch without touching.

Job 2 has nothing to do yet. Luthi's action layer can choose to speak, where to
attend, and to remember, and none of those has a route here. Speech is
untrained. Attention scales incoming sound and images, and there are none. A
memory nobody can read back is a route to nowhere. So every choice is counted in
``chosen_without_route`` and goes no further. The ``World`` protocol has no
``act`` method: it arrives with the first world that has actions, and not before.
"""

from __future__ import annotations

import logging
import time
from collections import Counter, deque
from pathlib import Path
from typing import Optional, Protocol

from sanctuary.core.schema import Percept
from sanctuary.life.mind import Moment
from sanctuary.life.senses import Line

logger = logging.getLogger(__name__)


class World(Protocol):
    """Anything that can say what happened since it was last asked."""

    def observe(self) -> list[Percept]: ...


class TextWorld:
    """A world made of what people say."""

    def __init__(self) -> None:
        self._said: deque[Percept] = deque()

    def say(self, text: str, source: str = "someone") -> None:
        self._said.append(Percept(modality="language", content=text, source=source))

    def observe(self) -> list[Percept]:
        said = list(self._said)
        self._said.clear()
        return said


class LifeLoop:
    def __init__(
        self,
        mind,
        world: World,
        *,
        hz: float = 10.0,
        save_every_seconds: float = 900.0,
        waiting_limit: Optional[int] = None,
    ):
        """``hz`` is the pace, fixed here until Luthi can choose their own; 0
        runs as fast as the mind does. A life is saved every
        ``save_every_seconds`` and on a clean stop. Checkpoints are never
        deleted, so a shorter interval uses disk faster."""
        self.mind = mind
        self.world = world
        self.hz = hz
        self.save_every_seconds = save_every_seconds
        self.line = Line(mind.tokenizer.encode, mind.window, waiting_limit)
        self.cycles = 0
        self.saves = 0
        self.chosen_without_route: Counter = Counter()
        self._rollbacks_told = mind.rollbacks()
        self._last_save = time.monotonic()
        self._running = False

    def step(self) -> Moment:
        """One moment: take in the world, let the mind live it, account for it."""
        for percept in self.world.observe():
            if percept.tensor_data is not None:
                raise NotImplementedError(
                    f"A {percept.modality} percept carries a tensor, and sound "
                    "and images are not on the small path yet."
                )
            self.line.add(percept)
        moment = self.mind.live(self.line.take().tokens)
        self._tell_of_rollbacks()
        for channel, acted in moment.chosen["active"].items():
            if acted:
                self.chosen_without_route[channel] += 1
        self.cycles += 1
        if time.monotonic() - self._last_save >= self.save_every_seconds:
            self.save("timer")
        return moment

    def run(self, max_cycles: Optional[int] = None) -> None:
        """Live until ``stop()`` is called or for ``max_cycles``, then save.
        If a moment raises, the life is not saved: its last good checkpoint
        stays the newest, and the error goes up. An interrupt in the middle of
        a moment counts as a raise; to stop cleanly, call ``stop()``."""
        self._running = True
        period = 1.0 / self.hz if self.hz > 0 else 0.0
        lived = 0
        try:
            while self._running and (max_cycles is None or lived < max_cycles):
                started = time.monotonic()
                self.step()
                lived += 1
                rest = period - (time.monotonic() - started)
                if rest > 0:
                    time.sleep(rest)
        except BaseException:
            self._running = False
            raise
        self._running = False
        self.save("stop")

    def stop(self) -> None:
        self._running = False

    def save(self, reason: str) -> Path:
        path = self.mind.save(reason)
        self.saves += 1
        self._last_save = time.monotonic()
        logger.info("Life saved: %s (%s)", path, reason)
        return path

    def status(self) -> dict:
        """Read-only. What the world sent, what the mind chose, how it reads."""
        return {
            "cycles": self.cycles,
            "world": {
                "arrived": self.line.arrived,
                "quiet_moments": self.line.quiet_moments,
                "waiting_tokens": self.line.waiting,
                "carried_in_parts": self.line.carried_in_parts,
                "dropped": self.line.dropped,
            },
            "chosen_without_route": dict(self.chosen_without_route),
            "saves": self.saves,
            **self.mind.watch(),
        }

    def _tell_of_rollbacks(self) -> None:
        """A rollback undoes recent learning. Luthi is told, as something that
        happened to them."""
        count = self.mind.rollbacks()
        new = count - self._rollbacks_told
        self._rollbacks_told = count
        if new <= 0:
            return
        self.line.add(Percept(
            modality="system",
            source="learning",
            content=(
                "Some of what you learned recently was undone"
                + (f" ({new} times since you last heard)" if new > 1 else "")
                + ": it was pulling your understanding too far from what you knew "
                "before, so your world model went back to its last steady state."
            ),
        ))
