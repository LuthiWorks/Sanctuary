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
- **Nothing is deleted.** A memory can become hard to reach, when nothing in
  the present resembles how it was filed. It is still there.

Not here: recalling in order ("and then what happened?"), revisiting memories
in rest, and anything that fades. See the research entry.

On disk, in ``<life>/memories/``, append-only:

- ``memory_00000001.pt``: one file a memory, written once and never rewritten.
- ``summaries.jsonl``: one line for each summary, the first and each later one.

Sight and sound are kept exactly as the world gave them, so a memory with a
picture is as large as the picture.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import torch

from sanctuary.core.atomic_io import PersistenceError, append_jsonl
from sanctuary.life.senses import Sight, Sound

MARK = "[memory]\n"


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
    def __init__(self, directory: str | Path, *, recent: int = 100):
        """``recent``: a memory kept or looked back on within this many moments
        is not what recall brings: it is still the present. A placeholder,
        Brian's to change."""
        self.directory = Path(directory)
        self.recent = recent
        self._summaries_path = self.directory / "summaries.jsonl"
        # One row for each summary: (memory id, the moment it was made, the vector).
        self._ids: list[int] = []
        self._cycles: list[int] = []
        self._vectors: list[torch.Tensor] = []
        self.kept = 0
        self.looked_back = 0
        if self._summaries_path.exists():
            for line in self._summaries_path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                self._file(row["id"], row["cycle"], torch.tensor(row["summary"]))
                if row["kind"] == "kept":
                    self.kept += 1
                else:
                    self.looked_back += 1

    def __len__(self) -> int:
        return self.kept

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
        memory_id = self.kept + 1
        path = self._path(memory_id)
        if path.exists():
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
        self._note("kept", memory_id, cycle, summary, when)
        self.kept += 1
        return memory_id

    def look_back(self, memory_id: int, *, cycle: int, summary: torch.Tensor) -> None:
        """The mind has just looked back on this memory. File how it seems to
        them now beside how it seemed before. What was perceived is untouched."""
        if not 1 <= memory_id <= self.kept:
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
        rows = [n for n, i in enumerate(self._ids) if i not in still_present]
        if not rows:
            return None
        vectors = torch.stack([self._vectors[n] for n in rows])
        cue = cue.detach().to(torch.float32).flatten().cpu()
        likeness = torch.nn.functional.cosine_similarity(vectors, cue.unsqueeze(0), dim=1)
        return self.open(self._ids[rows[int(likeness.argmax())]])

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
        files = list(self.directory.glob("memory_*.pt")) if self.directory.exists() else []
        return {
            "kept": self.kept, "looked_back": self.looked_back,
            "bytes": sum(p.stat().st_size for p in files),
        }
