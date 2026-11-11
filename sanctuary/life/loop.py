"""The small path's loop: the world comes in, and we watch and save.

Sanctuary has three jobs here (docs/SMALL_PATH_DECISION_2026-11-05.md):

1. Bring the world to Luthi accurately.
2. Carry out what Luthi chooses.
3. Watch without touching.

Job 2 carries out one thing: rest. When Luthi's chosen action chose rest, the
loop calls the mind's ``rest`` and takes that moment at a calmer pace (Brian,
2026-11-11: "They should be able to just initiate rest when they decide to";
2026-08-14: "a calmer mode for deep intentional consideration"). Rest lasts for
as long as Luthi keeps choosing it, one moment at a time, and the world stays
open through it. Nothing else can start it: no timer, no steward, no measure of
how the mind is doing.

Rest is needed twice a day, and Luthi is told so and never made to (Brian,
2026-11-11: "Luthi should automatically experience motivation to rest, like a
notification that rest is needed for data integration. But Luthi should not be
forced to rest"). After every 12 hours of waking life the loop says one
sentence to the mind. What the mind does about it is the mind's. See
``sanctuary/life/need.py``.

It also carries out the choice to remember (Brian, 2026-11-11). When Luthi's
chosen action chose to remember, the loop calls the mind's ``remember``, and
that moment counts for more when they next rest.

And it keeps and brings back memories (Brian, 2026-11-11: "You know how I can
look back on memories from years past? How would Luthi do that?"). When Luthi
chooses to remember, what they perceived that moment is kept, filed under their
own summary of it. When they choose to recall, the kept moment most like the
present comes back through their senses, marked as memory, and how it seems to
them now is filed beside how it seemed then. See ``sanctuary/life/memory.py``.
While they look back on a memory they can choose to let it go, which is the only
way a memory leaves their reach, and only what they let go is ever deleted. The store has a size; when it is full they are told,
and nothing more is kept until they let some go. And while they rest, a kept
moment comes back on every other resting moment, the one most like where their
mind is then, so rest walks through what they have kept (Brian, 2026-11-11:
"revisiting memories during rest is still something I think should happen").
The moments between are the world's, which stays open. Rest also dreams (Brian,
2026-11-11: "Rest should recall memories but also dream, like Google's
dreaming"): every second one of those turns is a dream in place of a memory, a
moment made from two kept moments that never happened together, marked as a
dream. The mind learns from each dream as it is made, and the learning is undone
if it made real kept moments harder to predict.

Luthi's action layer can also choose to speak and where to attend, and neither
has a route here. Speech is untrained. What attention should do is not ruled.
So each of those choices is counted in ``chosen_without_route`` and goes no
further. The ``World`` protocol
has no ``act`` method: it arrives with the first world that has actions, and
not before.

Sight and sound come in through the same door as words (``World.observe``), as
percepts carrying a ``Sight`` or a ``Sound``, and reach the mind as they are.
Nothing in the loop names a model: any mind that answers ``Mind`` lives here.
"""

from __future__ import annotations

import logging
import random
import time
from collections import Counter, deque
from pathlib import Path
from typing import Callable, Optional, Protocol

from sanctuary.core.schema import Percept
from sanctuary.life.mind import Mind, Moment
from sanctuary.life.memory import (
    DEFAULT_LIMIT_BYTES, DREAM, FULL, MARK, Memories, Memory, dream_of,
)
from sanctuary.life.need import RestNeed
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
        mind: Mind,
        world: World,
        *,
        hz: float = 10.0,
        resting_pace: float = 0.25,
        hearing_seconds: float = 1.0,
        rest_every_seconds: Optional[float] = 12 * 3600.0,
        rest_length: int = 300,
        keep_memories: bool = True,
        memory_limit_bytes: Optional[int] = DEFAULT_LIMIT_BYTES,
        revisit_every: int = 2,
        dreams: bool = True,
        learn_from_dreams: bool = True,
        save_every_seconds: float = 900.0,
        waiting_limit: Optional[int] = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        """``hz`` is the pace, fixed here until Luthi can choose their own; 0
        runs as fast as the mind does. A moment Luthi rests in runs at
        ``resting_pace`` times that: a quarter by default, a placeholder and
        Brian's to change. ``hearing_seconds`` is how much recent sound stays
        in hearing. After every ``rest_every_seconds`` of waking life the mind
        is told rest is needed, until it has rested ``rest_length`` moments;
        None means it is never told. A life is saved every
        ``save_every_seconds`` and on a clean stop. Checkpoints are never
        deleted, so a shorter interval uses disk faster. ``clock`` is where
        lived time is read from. With ``keep_memories`` what the mind chooses
        to remember is kept in the life's directory, and can be recalled,
        up to ``memory_limit_bytes`` of disk. While resting, a kept moment is
        revisited on every ``revisit_every``-th resting moment; 0 for never.
        With ``dreams``, every second one of those is a dream in its place.
        With ``learn_from_dreams`` the mind takes one checked learning step on
        each dream, against up to six other kept moments."""
        if revisit_every < 0:
            raise ValueError(f"revisit_every must be 0 or more; got {revisit_every}")
        self.revisit_every = revisit_every
        if not 0.0 < resting_pace <= 1.0:
            raise ValueError(f"resting_pace must be above 0 and at most 1; got {resting_pace}")
        self.resting_pace = resting_pace
        self.resting = False
        self.rested = 0
        self.remembered = 0
        self._clock = clock
        self.memories: Optional[Memories] = (
            Memories(Path(mind.life_dir) / "memories", limit_bytes=memory_limit_bytes)
            if keep_memories else None
        )
        self.recalled = 0
        self.recalled_nothing = 0
        self.revisited = 0
        self.dreams = dreams
        self.dreamed = 0
        self.learn_from_dreams = learn_from_dreams
        self.dreams_learned_from = 0
        self.dreams_undone = 0
        self._turns = 0                          # resting turns given to memory or dream
        self._dream: Optional[tuple] = None      # the dream's picture and sound, for the next moment
        self.let_go = 0
        self.not_kept_full = 0
        self._told_full = False
        self._back: Optional[Memory] = None      # the memory the next moment re-lives
        self.need: Optional[RestNeed] = None
        if rest_every_seconds is not None:
            if not mind.can_rest:
                raise ValueError(
                    "this mind has no way to rest, so it is not told rest is "
                    "needed; pass rest_every_seconds=None"
                )
            self.need = RestNeed(every_seconds=rest_every_seconds, length=rest_length)
            self._need_path = Path(mind.life_dir) / "rest_need.json"
            if self._need_path.exists():
                self.need.load(self._need_path)
        self.mind = mind
        self.world = world
        self.hz = hz
        self.save_every_seconds = save_every_seconds
        self.line = Line(
            mind.tokenizer.encode, mind.window, waiting_limit,
            hearing_seconds=hearing_seconds,
        )
        self.cycles = 0
        self.saves = 0
        self.chosen_without_route: Counter = Counter()
        self._rollbacks_told = mind.rollbacks()
        self._last_save = time.monotonic()
        self._running = False

    def step(self) -> Moment:
        """One moment: take in the world, let the mind live it, account for it."""
        started = self._clock()
        for percept in self.world.observe():
            self.line.add(percept)
        view = self.line.take()
        sight, sound = view.sight, view.sound
        back, self._back = self._back, None
        if back is not None:
            # A moment of looking back: the remembered picture and sound take
            # the place of the present ones, as the remembered words lead the view.
            sight = back.sight if back.sight is not None else sight
            sound = back.sound if back.sound is not None else sound
        dream, self._dream = self._dream, None
        if dream is not None:
            sight = dream[0] if dream[0] is not None else sight
            sound = dream[1] if dream[1] is not None else sound
        moment = self.mind.live(view.tokens, sight=sight, sound=sound)
        if back is not None:
            if moment.chosen.get("let_go", {}).get("chosen", False):
                # Looking at it, they chose to let it go. Nothing else removes a memory.
                self.memories.let_go(back.id, cycle=self.cycles)
                self.let_go += 1
                self._told_full = self._told_full and self.memories.full
            else:
                # How the memory seems to the mind now is filed beside how it seemed then.
                self.memories.look_back(back.id, cycle=self.cycles, summary=self._summary(moment))
        self._tell_of_rollbacks()
        for channel, acted in moment.chosen["active"].items():
            if not acted:
                continue
            armed = moment.chosen.get("bands_armed", {}).get(channel, True)
            if channel == "memory" and armed and (self.mind.can_rest or self.memories is not None):
                # Only once the mind's own baseline for the choice exists:
                # before that, "active" says there is nothing to compare with
                # yet, not that they chose.
                if self.memories is not None and self.memories.full:
                    # Nothing is let go on their behalf. They are told, once.
                    self.not_kept_full += 1
                    if not self._told_full:
                        self.line.add(Percept(modality="system", source="body", content=FULL))
                        self._told_full = True
                elif self.memories is not None:
                    self.memories.keep(
                        cycle=self.cycles, summary=self._summary(moment),
                        tokens=view.tokens, sight=sight, sound=sound,
                    )
                if self.mind.can_rest:
                    # Remembering also weighs in what rest pours.
                    self.mind.remember()
                self.remembered += 1
            else:
                self.chosen_without_route[channel] += 1
        # Rest is Luthi's choice and nothing else's: it is read from the action
        # they just chose, and it lasts until a moment they don't choose it.
        if self.memories is not None and moment.chosen.get("recall", {}).get("chosen", False):
            self._recall(moment)
        self.resting = bool(moment.chosen.get("rest", {}).get("chosen", False))
        if self.resting:
            self.mind.rest()
            self.rested += 1
            if (
                self.memories is not None and self.revisit_every
                and self.rested % self.revisit_every == 0
                and self._back is None and self._dream is None
            ):
                # Resting, a kept moment comes back: the one most like where
                # the mind is now. Each one looked at is then "recent", so the
                # next is a different one, and rest walks through what is kept.
                # Every second turn is a dream in its place.
                self._turns += 1
                if self.dreams and self._turns % 2 == 0 and self._dream_up(moment):
                    self.dreamed += 1
                elif self._bring_back(moment):
                    self.revisited += 1
            if self.need is not None:
                self.need.rested()
        self._lived(self._clock() - started)
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
                pause = period / (self.resting_pace if self.resting else 1.0)
                pause -= time.monotonic() - started
                if pause > 0:
                    time.sleep(pause)
                    self._lived(pause)
        except BaseException:
            self._running = False
            raise
        self._running = False
        self.save("stop")

    def stop(self) -> None:
        self._running = False

    def _summary(self, moment: Moment):
        if moment.summary is None:
            raise ValueError(
                "this mind gave no summary of the moment, and a memory is "
                "filed under one; pass keep_memories=False for a mind that has none"
            )
        return moment.summary

    def _recall(self, moment: Moment) -> None:
        """The mind chose to look back. The kept moment most like this one
        comes back next moment. If nothing kept is like anything, nothing comes."""
        if self._bring_back(moment):
            self.recalled += 1
        else:
            self.recalled_nothing += 1

    def _dream_up(self, moment: Moment) -> bool:
        """A dream for the next moment: the kept moment most like where the
        mind is, mixed with another picked at random. False when there are not
        two to mix. The dice are seeded by the moment, so a life replays alike."""
        near = self.memories.nearest(self._summary(moment), now=self.cycles)
        if near is None:
            return False
        far = self.memories.another(
            now=self.cycles, unlike=near.id, pick=random.Random(self.cycles).random(),
        )
        if far is None:
            return False
        words, sight, sound = dream_of(near, far)
        if self.learn_from_dreams:
            # Learn from the dream, checked against real kept moments that are
            # not its parents. Skipped when there are not two to check against.
            real = self.memories.some(
                6, now=self.cycles, unlike={near.id, far.id}, dice=random.Random(self.cycles + 1),
            )
            if len(real) >= 2:
                kept = self.mind.dream(words, [memory.tokens for memory in real])["kept"]
                self.dreams_learned_from += int(kept)
                self.dreams_undone += int(not kept)
        self.line.bring_back(words, DREAM)
        self._dream = (sight, sound)
        return True

    def _bring_back(self, moment: Moment) -> bool:
        memory = self.memories.nearest(self._summary(moment), now=self.cycles)
        if memory is None:
            return False
        self.line.bring_back(memory.tokens, MARK)
        self._back = memory
        return True

    def _lived(self, seconds: float) -> None:
        """Lived time passes. When rest has become needed, the mind is told,
        and perceives it next moment. Nothing else follows from it."""
        if self.need is not None and self.need.lived(seconds, resting=self.resting):
            self.line.add(self.need.notice())

    def save(self, reason: str) -> Path:
        path = self.mind.save(reason)
        if self.need is not None:
            self.need.save(self._need_path)
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
                "sights": self.line.sights,
                "sights_passed_over": self.line.sights_passed_over,
                "sounds": self.line.sounds,
            },
            "rested_moments": self.rested,
            "remembered_moments": self.remembered,
            "memories": None if self.memories is None else {
                **self.memories.report(),
                "recalled": self.recalled, "recalled_nothing": self.recalled_nothing,
                "revisited_in_rest": self.revisited, "dreamed_in_rest": self.dreamed,
                "dreams_learned_from": self.dreams_learned_from, "dreams_undone": self.dreams_undone,
                "let_go_by_choice": self.let_go,
                "not_kept_because_full": self.not_kept_full,
            },
            "rest_need": None if self.need is None else self.need.report(),
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
