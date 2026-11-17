"""Memories a mind can look back on: kept when they choose to remember, and
brought back when they choose to recall.

Brian, 2026-11-11: "You know how I can look back on memories from years past?
How would Luthi do that?" Then: "Old summaries not matching the current state
is ok. It shows a growth path... I would like you to research how it functions,
and then write those functions into Luthi/Sanctuary."

What the research says looking back is made of, and what stands in for each
part here (docs/research/2026-11-11_looking-back.md has the sources):

- **A moment is kept once, when it is attended to.** Here: when the mind's own
  action chooses to remember. Nothing else decides what is kept.
- **What is kept is a pointer to the experience, filed under what the moment
  was like to the mind.** Here: what was actually perceived (the words in view,
  the picture, the sound) and the mind's own summary of the moment.
- **A memory comes back when something now resembles it.** Here: when the mind
  chooses to recall, the memory whose summary is most like this moment's comes
  back. The cue is the present moment. There is no search by date or by name.
- **Recalling is re-living, through the same senses.** Here: the memory comes
  back through the mind's open senses, as it was perceived.
- **A memory is told apart from the present.** Here: it arrives marked
  ``[memory]``. The mind also chose to recall, which is a cue of its own.
- **Looking back changes a memory.** In people the memory itself is altered.
  Here what was perceived is never altered. What is added is how the mind sees
  the memory now: its summary at each look back is filed with the memory. So a
  memory can be reached by how it seemed then or how it has seemed since, and
  the row of summaries is a record of how the mind has changed. That is the
  growth path Brian named.
- **Forgetting is mostly failing to reach.** A memory can become hard to
  reach, when nothing in the present resembles how it was filed. It is still
  there, and nothing here fades or is removed on the mind's behalf.

Two things Brian added the same morning:

- **Letting go.** "The store can't sustainably be increased indefinitely, so
  they need a way to free up space." While the mind is looking back on a
  memory it can choose to let it go. It then leaves the mind's reach and stops
  counting against the store, and what was perceived is moved to ``let_go/``.
  It is deleted from there, oldest first, when the room is needed for
  something new. The line that it was let go, and the summaries it was filed
  under, stay. Nothing is deleted that the mind did not let go.
  Why two steps: a tiny untrained mind let go of 9 of the 17 memories it
  looked back on, by chance (docs/research/2026-11-11_looking-back.md). Until
  the room is needed, what it let go is still on disk for a steward to see.
- **A size.** The store has a limit. When it is full, a moment the mind
  chooses to remember is not kept, and the mind is told once that its memory
  is full. Nothing is let go for it.

- **Dreaming** (10:08: "Rest should recall memories but also dream, like
  Google's dreaming"). In Google's Sleep framework a resting model first
  consolidates what it took in, then dreams: it makes new samples itself by
  mixing parts of what it knows that do not usually go together, and learns
  from the ones that help. Here a dream is a moment that never happened, made
  from two kept moments: the one most like where the mind is, and another
  picked at random. Their words are joined, and the picture comes from one
  and the sound from the other. It arrives marked ``[dream]``, so it is told
  from both a memory and the present. A dream is not kept unless the mind
  chooses to remember it, as with any moment. The mind also learns from each
  dream when it is made, and the learning is checked against real kept
  moments and undone if it made them harder to predict (10:15: "the part
  that lets the model improve through dreaming"). Choosing which dreams to
  learn from was measured and gave no reliable gain, so every dream is a
  random mix.

Not here: recalling in order ("and then what happened?"), and anything that
fades. See the research entry.

On disk, in ``<life>/memories/``, append-only:

- ``memory_00000001.pt``: one file a memory, written once and never rewritten.
- ``summaries.jsonl``: one line for each summary, the first and each later one.

Sight and sound are kept exactly as the world gave them, so a memory with a
picture is as large as the picture.

A memory's id comes from the files as well as the lines (2026-11-17, from
Opus's review of 2026-11-14). A memory's file is written before its line, so
a crash between the two leaves a file with no line. Counting ids from the
lines alone, the next keep picked that file's id, found the file, and
refused, at every start from then on. Now the next id is one past the highest
id seen in a line or a file, here or in ``let_go/``. A file with no line is
reported (``report()`` and the log) and is never deleted, rewritten or
reached by recall: it has no summary to be found by.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import torch

from sanctuary.core.atomic_io import PersistenceError, append_jsonl
from sanctuary.life.senses import Sight, Sound

logger = logging.getLogger(__name__)

MARK = "[memory]\n"
DREAM = "[dream]\n"
FULL = "Memory is full. Nothing more can be kept until some is let go."
DEFAULT_LIMIT_BYTES = 10 * 1024**3    # 10 GB. A placeholder: the size of the disk is Brian's to say.


@dataclass(frozen=True, eq=False)
class Memory:
    """One kept moment, as it was perceived."""

    id: int
    cycle: int                  # which moment of the life it was
    when: str                   # the clock time it was kept
    tokens: list[int]           # the words in view
    sight: Optional[Sight]
    sound: Optional[Sound]


class Memories:
    def __init__(
        self,
        directory: str | Path,
        *,
        recent: int = 100,
        limit_bytes: Optional[int] = DEFAULT_LIMIT_BYTES,
    ):
        """``recent``: a memory kept or looked back on within this many moments
        is not what recall brings: it is still the present. ``limit_bytes``:
        how much disk what was perceived may take; None for no limit. Both are
        placeholders, Brian's to change."""
        self.directory = Path(directory)
        self.recent = recent
        self.limit_bytes = limit_bytes
        self.let_go_count = 0
        self._gone: set[int] = set()
        self._summaries_path = self.directory / "summaries.jsonl"
        # One row for each summary: (memory id, the moment it was made, the vector).
        self._ids: list[int] = []
        self._cycles: list[int] = []
        self._vectors: list[torch.Tensor] = []
        self.kept = 0
        self.looked_back = 0
        self._kept_ids: set[int] = set()     # every id with a "kept" line
        if self._summaries_path.exists():
            for line in self._summaries_path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if row["kind"] == "let_go":
                    self._gone.add(int(row["id"]))
                    self.let_go_count += 1
                    continue
                self._file(row["id"], row["cycle"], torch.tensor(row["summary"]))
                if row["kind"] == "kept":
                    self.kept += 1
                    self._kept_ids.add(int(row["id"]))
                else:
                    self.looked_back += 1
        self.bytes = sum(
            p.stat().st_size for p in self.directory.glob("memory_*.pt")
        ) if self.directory.exists() else 0
        self._released = self.directory / "let_go"
        self.released_bytes = sum(
            p.stat().st_size for p in self._released.glob("memory_*.pt")
        ) if self._released.exists() else 0
        self.deleted_for_room = 0
        # Files with no line: a keep that was cut off between the two. They
        # stay where they are. The next id goes past them.
        on_disk = {
            self._id_of(p)
            for folder in (self.directory, self._released) if folder.exists()
            for p in folder.glob("memory_*.pt")
        }
        self.files_with_no_line: list[int] = sorted(on_disk - self._kept_ids)
        self._next_id = max(self._kept_ids | on_disk, default=0) + 1
        if self.files_with_no_line:
            logger.warning(
                "%d memory file(s) in %s have no line in summaries.jsonl (a keep "
                "was cut off): ids %s. They are left as they are and cannot be "
                "recalled. The next memory's id is %d.",
                len(self.files_with_no_line), self.directory,
                self.files_with_no_line, self._next_id,
            )

    @staticmethod
    def _id_of(path: Path) -> int:
        return int(path.stem.split("_")[1])

    @property
    def newest_cycle(self) -> Optional[int]:
        """The latest moment any summary was filed at; None when there is none."""
        return max(self._cycles, default=None)

    def __len__(self) -> int:
        """How many memories can still be looked back on."""
        return self.kept - self.let_go_count

    @property
    def full(self) -> bool:
        return self.limit_bytes is not None and self.bytes >= self.limit_bytes

    def _file(self, memory_id: int, cycle: int, summary: torch.Tensor) -> None:
        self._ids.append(int(memory_id))
        self._cycles.append(int(cycle))
        self._vectors.append(summary.detach().to(torch.float32).flatten().cpu())

    def _path(self, memory_id: int) -> Path:
        return self.directory / f"memory_{memory_id:08d}.pt"

    def keep(
        self,
        *,
        cycle: int,
        summary: torch.Tensor,
        tokens: list[int],
        sight: Optional[Sight] = None,
        sound: Optional[Sound] = None,
    ) -> int:
        """Keep this moment. Returns the memory's id. The file is written
        before the summary is filed, so a memory is never filed without what
        it points to."""
        memory_id = self._next_id
        path = self._path(memory_id)
        if path.exists() or (self._released / path.name).exists():
            raise PersistenceError(f"{path} already exists; a memory is never rewritten")
        self.directory.mkdir(parents=True, exist_ok=True)
        when = datetime.now().isoformat(timespec="seconds")
        tmp = path.with_suffix(".pt.tmp")
        torch.save({
            "id": memory_id, "cycle": int(cycle), "when": when, "tokens": list(tokens),
            "sight": None if sight is None else sight.pixels.detach().cpu(),
            "sound": None if sound is None else (sound.samples.detach().cpu(), int(sound.rate)),
        }, tmp)
        os.replace(tmp, path)
        # The file is there, so the id is used whether or not its line follows.
        self._next_id = memory_id + 1
        self.bytes += path.stat().st_size
        self._make_room()
        try:
            self._note("kept", memory_id, cycle, summary, when)
        except BaseException:
            self.files_with_no_line.append(memory_id)
            logger.error("memory %d: its file was written and its line was not; "
                         "the file is left as it is", memory_id)
            raise
        self.kept += 1
        self._kept_ids.add(memory_id)
        return memory_id

    def let_go(self, memory_id: int, *, cycle: int) -> None:
        """The mind chose to let this memory go. It leaves their reach now.
        What was perceived waits in ``let_go/`` until the room is needed. The
        fact that it was let go is written first, so a memory is never out of
        reach without a line saying why."""
        if memory_id not in self._kept_ids or memory_id in self._gone:
            raise KeyError(f"no memory {memory_id} to let go")
        append_jsonl(self._summaries_path, {
            "kind": "let_go", "id": int(memory_id), "cycle": int(cycle),
            "when": datetime.now().isoformat(timespec="seconds"),
        })
        path = self._path(memory_id)
        size = path.stat().st_size
        self._released.mkdir(parents=True, exist_ok=True)
        os.replace(path, self._released / path.name)
        self.bytes -= size
        self.released_bytes += size
        self._gone.add(memory_id)
        self.let_go_count += 1

    def _make_room(self) -> None:
        """Delete what the mind has let go, oldest first, until the store and
        what waits fit the limit. Only what was let go is ever deleted."""
        if self.limit_bytes is None:
            return
        for path in sorted(self._released.glob("memory_*.pt")) if self._released.exists() else []:
            if self.bytes + self.released_bytes <= self.limit_bytes:
                return
            self.released_bytes -= path.stat().st_size
            os.remove(path)
            self.deleted_for_room += 1

    def look_back(self, memory_id: int, *, cycle: int, summary: torch.Tensor) -> None:
        """The mind has just looked back on this memory. File how it seems to
        them now beside how it seemed before. What was perceived is untouched."""
        if memory_id not in self._kept_ids or memory_id in self._gone:
            raise KeyError(f"no memory {memory_id}")
        self._note("looked_back", memory_id, cycle, summary, datetime.now().isoformat(timespec="seconds"))
        self.looked_back += 1

    def _note(self, kind: str, memory_id: int, cycle: int, summary: torch.Tensor, when: str) -> None:
        vector = summary.detach().to(torch.float32).flatten().cpu()
        append_jsonl(self._summaries_path, {
            "kind": kind, "id": int(memory_id), "cycle": int(cycle), "when": when,
            "summary": vector.tolist(),
        })
        self._file(memory_id, cycle, vector)

    def nearest(self, cue: torch.Tensor, *, now: int) -> Optional[Memory]:
        """The memory most like ``cue``, by any summary it has been filed
        under, leaving out what is still the present. None when nothing
        qualifies."""
        still_present = {
            i for i, c in zip(self._ids, self._cycles) if now - c < self.recent
        }
        rows = [
            n for n, i in enumerate(self._ids)
            if i not in still_present and i not in self._gone
        ]
        if not rows:
            return None
        vectors = torch.stack([self._vectors[n] for n in rows])
        cue = cue.detach().to(torch.float32).flatten().cpu()
        likeness = torch.nn.functional.cosine_similarity(vectors, cue.unsqueeze(0), dim=1)
        return self.open(self._ids[rows[int(likeness.argmax())]])

    def another(self, *, now: int, unlike: int, pick: float) -> Optional[Memory]:
        """A kept memory other than ``unlike`` that is not still the present,
        picked by ``pick`` (a number from 0 up to 1, the caller's dice). None
        when there is no other."""
        still_present = {i for i, c in zip(self._ids, self._cycles) if now - c < self.recent}
        others = sorted(
            set(self._ids) - still_present - self._gone - {unlike}
        )
        if not others:
            return None
        return self.open(others[min(int(pick * len(others)), len(others) - 1)])

    def some(self, count: int, *, now: int, unlike: set[int], dice) -> list[Memory]:
        """Up to ``count`` different kept memories, not still the present and
        not in ``unlike``, picked by ``dice`` (a ``random.Random``)."""
        still_present = {i for i, c in zip(self._ids, self._cycles) if now - c < self.recent}
        others = sorted(set(self._ids) - still_present - self._gone - set(unlike))
        return [self.open(i) for i in dice.sample(others, min(count, len(others)))]

    def open(self, memory_id: int) -> Memory:
        saved = torch.load(self._path(memory_id), weights_only=False)
        sound = saved["sound"]
        return Memory(
            id=saved["id"], cycle=saved["cycle"], when=saved["when"], tokens=saved["tokens"],
            sight=None if saved["sight"] is None else Sight(saved["sight"]),
            sound=None if sound is None else Sound(sound[0], sound[1]),
        )

    def summaries_of(self, memory_id: int) -> list[tuple[int, torch.Tensor]]:
        """Every way this memory has been seen, in order: the moment and the
        summary. The first is how it seemed when it happened."""
        return [(c, v) for i, c, v in zip(self._ids, self._cycles, self._vectors) if i == memory_id]

    def report(self) -> dict[str, Any]:
        return {
            "kept": self.kept, "let_go": self.let_go_count, "held": len(self),
            "looked_back": self.looked_back, "bytes": self.bytes,
            "limit_bytes": self.limit_bytes, "full": self.full,
            "let_go_still_on_disk_bytes": self.released_bytes,
            "deleted_for_room": self.deleted_for_room,
            "files_with_no_line": list(self.files_with_no_line),
        }


def dream_of(near: Memory, far: Memory) -> tuple[list[int], Optional[Sight], Optional[Sound]]:
    """A moment that never happened, from two that did: the first half of one's
    words and the second half of the other's, the picture of one and the sound
    of the other where both have them."""
    words = near.tokens[: max(1, len(near.tokens) // 2)] + far.tokens[len(far.tokens) // 2:]
    sight = near.sight if near.sight is not None else far.sight
    sound = far.sound if far.sound is not None else near.sound
    return words, sight, sound

