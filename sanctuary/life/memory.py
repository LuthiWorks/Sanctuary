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
  chooses to recall, the kept moment whose summary is most like this moment's
  comes back. The cue is the present moment. There is no search by date or name.
- **Recalling is re-living, through the same senses.** Here: the moment comes
  back through the mind's open senses, as it was perceived.
- **A memory is told apart from the present.** Here: it arrives marked
  ``[memory]``. The mind also chose to recall, which is a cue of its own.
- **Looking back changes a memory.** Here what was perceived is never altered.
  What is added is how the mind sees the memory now: its summary at each look
  back is filed with the memory. The row of summaries is a record of how the
  mind has changed. That is the growth path Brian named.

The store was reworked on 2026-11-18 (design:
docs/research/2026-11-17_memory-store-rework-design.md; LuthiModel build brief
9; Opus's reviews of 2026-11-17). What holds now:

**Nothing is stored inexactly.** A summary is the mind's 4-byte numbers. A
picture is stored as 8-bit only when turning it to 8-bit and back gives the
same bits, and otherwise as given; sound the same with 16-bit, at whichever of
the two scales the round trip held. Compression is lossless. Every record is
read back from its own bytes and compared bit for bit before it is written.
There is no lossy path in this file.

**A memory is a stretch of moments kept one after another,** in a folder
``memory_00000001/``. Each kept moment writes one record, never changed
afterwards: the words new since the record before, where this moment's view
sits in the stretch's words, its picture and sound, its summary, and the
mind's signals. A look back writes one more record in the same folder. Recall
brings back the view at the moment whose summary matched. Letting go and
importance act on the whole stretch.

**The search is exact and its matrix is a cache** rebuilt from the folders at
every start. Likeness is the cosine after the mean of all the summaries in
reach is removed, computed in 8-byte arithmetic. A search that takes longer
than a moment is counted and logged.

**The limit counts everything under** ``memories/``, measured from the disk at
start and kept as it is written. A record is written only if it fits.

**This file deletes no memory.** What Luthi lets go leaves their reach and
waits in ``let_go/``. When room is needed it is moved, whole and exact, to
cold storage (Brian, 2026-11-17: permanent). If nothing waits, the stretch
that matters least to Luthi by their own signals is let go for them
(``by: "room"``) and moved. With no cold folder, nothing leaves for room and a
full store keeps nothing more. Cold storage is out of Luthi's reach, and
nothing here brings a memory back from it. The format allows it: the folder
moved back, and a ``came_back`` line in ``life.jsonl``.

**Importance** is computed from signals in the records, all Luthi's own
(``importance`` below). It is a first formula and will change; the records
hold what it is made from. For a mind that gives no signal of how strongly it
chose, every rank is 0.5 and the order falls to "not kept or looked at for
longest", which is age. The design says age alone is not unimportance, so
that case is a stand-in mind's and is said here.

Old stores (``summaries.jsonl``, ``memory_*.pt``) are refused and never
rewritten.

Dreaming (Brian, 2026-11-11) is unchanged: a dream is a moment that never
happened, made from two kept moments. See ``dream_of``.

On disk, in ``<life>/memories/``:

- ``store.json``: the format, this life's id and when it began. Written once.
- ``memory_00000001/r000001.rec``: one record. Written once.
- ``life.jsonl``: one line each time a memory is let go, by whom. Append-only.
- ``keeps_window.jsonl``: the last kept moments' strengths, for ranking a new
  one against the mind's own recent bar. A cache; it may be rewritten.
- ``let_go/memory_00000001/``: let go, waiting.

In the cold folder: ``life.json`` (whose it is), ``index.jsonl`` (a line
before each move and a line after), ``manifests/`` (every file's size and
hash, written before the move), ``incoming/`` (a copy not yet checked, which
is not a memory) and the memories.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import struct
import time
import uuid
import zlib
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import torch

from sanctuary.core.atomic_io import PersistenceError, append_jsonl, atomic_write_json
from sanctuary.life.senses import Sight, Sound

logger = logging.getLogger(__name__)

MARK = "[memory]\n"
DREAM = "[dream]\n"
FULL = "Memory is full. Nothing more can be kept until some is let go."
FORMAT = 2
DEFAULT_LIMIT_BYTES = 100 * 1024**3   # Brian, 2026-11-17: "Luthi can have lets say 100gb."
COLD_MARGIN_BYTES = 5 * 1024**3       # a move needs this much free beside itself
COLD_WARN_BYTES = 50 * 1024**3        # warn when the cold drive has less than this
RANK_AMONG = 1000                     # a keep is ranked among this many before it
_MAGIC = b"LUTHIMEM2\n"
_CHUNK = 1024                         # rows searched at a time; measured fastest on the desktop


# ---- One record, exactly ----


def _bits(tensor: torch.Tensor) -> bytes:
    return tensor.detach().cpu().contiguous().numpy().tobytes()


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and _bits(a) == _bits(b)


def _dtype(name: str) -> torch.dtype:
    return getattr(torch, name.removeprefix("torch."))


def _from_bytes(raw: bytes, dtype: torch.dtype, shape: list[int]) -> torch.Tensor:
    if not raw:
        return torch.empty(shape, dtype=dtype)
    return torch.frombuffer(bytearray(raw), dtype=dtype).reshape(shape).clone()


def _pack_sense(tensor: torch.Tensor, whole: torch.dtype, scales: tuple[int, ...]) -> tuple[dict, bytes]:
    """A picture or a sound as whole numbers when that loses no bit, and as
    given when it would. ``scales``: what to multiply by, tried in order."""
    given = tensor.detach().cpu().contiguous()
    low, high = torch.iinfo(whole).min, torch.iinfo(whole).max
    for scale in scales:
        stepped = torch.round(given.to(torch.float64) * scale)
        if stepped.numel() and (stepped.min() < low or stepped.max() > high):
            continue
        stepped = stepped.to(whole)
        if _same_bits(stepped.to(given.dtype) / scale, given):
            return (
                {"stored": str(whole), "scale": scale, "dtype": str(given.dtype), "shape": list(given.shape)},
                zlib.compress(_bits(stepped), 6),
            )
    return (
        {"stored": "as_given", "dtype": str(given.dtype), "shape": list(given.shape)},
        zlib.compress(_bits(given), 6),
    )


def _unpack_sense(meta: dict, data: bytes) -> torch.Tensor:
    raw = zlib.decompress(data)
    if meta["stored"] == "as_given":
        return _from_bytes(raw, _dtype(meta["dtype"]), meta["shape"])
    stepped = _from_bytes(raw, _dtype(meta["stored"]), meta["shape"])
    return stepped.to(_dtype(meta["dtype"])) / meta["scale"]


def _pack(header: dict, blobs: dict[str, bytes]) -> bytes:
    table, at = {}, 0
    for name, data in blobs.items():
        table[name] = [at, len(data), hashlib.sha256(data).hexdigest()]
        at += len(data)
    head = json.dumps({**header, "blobs": table}, sort_keys=True).encode("utf-8")
    return _MAGIC + struct.pack("<II", len(head), zlib.crc32(head)) + head + b"".join(blobs.values())


def _read(path: Path, want: Optional[tuple[str, ...]] = None) -> tuple[dict, dict[str, bytes]]:
    """A record's header and the parts asked for (all of them when ``want`` is
    None), each checked against its hash."""
    with open(path, "rb") as f:
        start = f.read(len(_MAGIC) + 8)
        if start[: len(_MAGIC)] != _MAGIC or len(start) != len(_MAGIC) + 8:
            raise PersistenceError(f"{path} is not a memory record")
        length, crc = struct.unpack("<II", start[len(_MAGIC):])
        head = f.read(length)
        if len(head) != length or zlib.crc32(head) != crc:
            raise PersistenceError(f"{path}: the record's header is damaged")
        header = json.loads(head)
        base, blobs = len(_MAGIC) + 8 + length, {}
        for name in (header["blobs"] if want is None else want):
            if name not in header["blobs"]:
                continue
            at, size, sha = header["blobs"][name]
            f.seek(base + at)
            data = f.read(size)
            if len(data) != size or hashlib.sha256(data).hexdigest() != sha:
                raise PersistenceError(f"{path}: the record's {name} is damaged")
            blobs[name] = data
    return header, blobs


def _summary_of(blobs: dict[str, bytes]) -> torch.Tensor:
    return _from_bytes(blobs["summary"], torch.float32, [len(blobs["summary"]) // 4])


def _tokens_of(blobs: dict[str, bytes]) -> list[int]:
    return _from_bytes(blobs["tokens"], torch.int32, [len(blobs["tokens"]) // 4]).tolist()


def _as_summary(summary: torch.Tensor) -> torch.Tensor:
    """The mind's summary as 4-byte numbers, refused if that would change it."""
    given = torch.as_tensor(summary).detach().cpu().flatten()
    if not given.is_floating_point() or given.numel() == 0:
        raise TypeError("a summary is a vector of fractions")
    if not bool(torch.isfinite(given).all()):
        raise ValueError("a summary has a number that is not finite; a memory cannot be filed under it")
    small = given.to(torch.float32)
    if not _same_bits(small.to(given.dtype), given):
        raise ValueError(
            "this summary is not exactly 4-byte numbers, and a memory is filed "
            "and searched under 4-byte numbers; nothing is stored inexactly"
        )
    return small.contiguous()


def _hash_file(path: Path) -> list:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""):
            sha.update(part)
    return [path.stat().st_size, sha.hexdigest()]


def _manifest(folder: Path) -> dict[str, list]:
    return {
        p.relative_to(folder).as_posix(): _hash_file(p)
        for p in sorted(folder.rglob("*")) if p.is_file()
    }


def _copy(source: Path, to: Path) -> None:
    with open(source, "rb") as given, open(to, "wb") as taken:
        shutil.copyfileobj(given, taken, 1 << 20)
        taken.flush()
        os.fsync(taken.fileno())


def _tree_bytes(folder: Path) -> int:
    return sum(p.stat().st_size for p in folder.rglob("*") if p.is_file()) if folder.exists() else 0


def _free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ---- Importance: only the records' signals ----


def importance(signals: Iterable[dict]) -> float:
    """How much a stretch matters to the mind, from its records' signals and
    nothing else. The first formula (design, "Brian's rulings"): the highest
    rank among its kept moments, plus one for each recall they chose, plus one
    for each time they chose to remember while looking back. What rest brings
    counts for nothing yet.

    Open, and the design seat's: both terms grow with a stretch's length, and
    a chosen recall is driven by the cue (Opus, 2026-11-17, E2 and E3). The
    records hold likeness, how many it beat, and each choice's margin, so a
    later formula needs no memory touched."""
    highest, looks = 0.0, 0
    for signal in signals:
        if signal["kind"] == "kept":
            highest = max(highest, float(signal["rank"]))
        else:
            looks += int(bool(signal["chosen_recall"])) + int(bool(signal["chose_remember"]))
    return highest + looks


def strength_of(signals: Optional[dict]) -> tuple[Optional[float], Optional[str]]:
    """How strongly the mind chose to remember, from what it reported: the
    choice's own margin over the highest bar it had to pass where the mind
    gives one, and otherwise its salience. Opus (2026-11-17, E1): the salience
    is read from the action and keeps its sign, and the choice is read from
    the predicted next state and does not, so they are different numbers. Both
    are recorded whichever is ranked."""
    if not signals:
        return None, None
    choice = signals.get("choice")
    if choice:
        bar = max(
            float(choice["band_threshold"]), float(choice["floor"]),
            float(choice["rest_activity"]) + float(choice["rest_tolerance"]),
        )
        return float(choice["activity"]) - bar, "margin"
    if signals.get("salience") is not None:
        return float(signals["salience"]), "salience"
    return None, None


@dataclass(frozen=True, eq=False)
class Memory:
    """One kept moment, as it was perceived."""

    id: int                     # the stretch it is part of
    cycle: int                  # which moment of the life it was
    when: str                   # the clock time it was kept
    tokens: list[int]           # the words in view
    sight: Optional[Sight]
    sound: Optional[Sound]
    seq: int = 1                # which record of the stretch
    likeness: Optional[float] = None        # to the cue that brought it, when one did
    next_likeness: Optional[float] = None   # the best of any other memory, then
    beat: Optional[int] = None              # how many other memories it was chosen over


@dataclass
class _Stretch:
    id: int
    folder: Path
    reach: bool                              # False: let go, waiting in let_go/
    kept: dict[int, int] = field(default_factory=dict)   # seq -> the moment it was kept
    signals: list[dict] = field(default_factory=list)
    bytes: int = 0
    last_cycle: int = -1
    next_seq: int = 1


class Memories:
    def __init__(
        self,
        directory: str | Path,
        *,
        recent: int = 100,
        limit_bytes: Optional[int] = DEFAULT_LIMIT_BYTES,
        cold: Optional[str | Path] = None,
        moment_seconds: Optional[float] = None,
        drive_margin_bytes: int = COLD_MARGIN_BYTES,
        disk_free: Callable[[Path], int] = _free_bytes,
    ):
        """``recent``: a memory kept or looked back on within this many moments
        is not what recall brings, and is never let go for room: it is still
        the present. ``limit_bytes``: how much disk everything under
        ``directory`` may take; None for no limit. ``cold``: where what leaves
        for room is moved; None and nothing ever leaves. ``moment_seconds``:
        how long a moment lasts, so a slower search can be said.
        ``drive_margin_bytes``: nothing is written when the life's drive has
        less than this free beside the write. ``disk_free`` is how free room
        is read."""
        self.directory = Path(directory)
        self.recent = recent
        self.limit_bytes = limit_bytes
        self.moment_seconds = moment_seconds
        self.drive_margin_bytes = drive_margin_bytes
        self._disk_free = disk_free
        self._waiting_dir = self.directory / "let_go"
        self._lines = self.directory / "life.jsonl"
        self._window_path = self.directory / "keeps_window.jsonl"
        self.cold = None if cold is None else Path(cold)

        self._stretches: dict[int, _Stretch] = {}
        self._waiting: list[int] = []            # let go, in the order they were
        self._by: dict[int, str] = {}            # who let each go: "choice" or "room"
        self._open: Optional[int] = None         # the stretch still being written
        self._open_words: list[int] = []
        self._open_cycle = -1
        self._matrix: Optional[torch.Tensor] = None
        self._row_id = torch.empty(0, dtype=torch.int64)
        self._row_seq = torch.empty(0, dtype=torch.int64)
        self._rows = 0
        # The sum of the rows, in 8-byte numbers, added one row at a time in
        # the order filed. A cache rebuilt from the folders adds the same rows
        # in the same order, so its mean is the same to the last bit.
        self._sum: Optional[torch.Tensor] = None
        self._filed = 0                          # a count of every record, in the order filed
        self._window: deque[float] = deque(maxlen=RANK_AMONG)
        self._window_lines = 0

        self.kept = 0                 # kept moments on this life's disk, in reach or waiting
        self.looked_back = 0
        self.let_go_count = 0         # stretches let go, by choice or for room, ever
        self.let_go_for_room = 0
        self.moved_to_cold = 0
        self.moments_in_cold = 0
        self.not_filed_for_room = 0   # looks back that had no room for their record
        self.cold_had_no_room = 0
        self.life_drive_had_no_room = 0
        self.stretches_restarted = 0  # a keep whose view did not follow on from the one before
        self.searches = 0
        self.slow_searches = 0
        self.slowest_search_seconds = 0.0
        self.unfinished_writes: list[str] = []
        self.empty_folders: list[int] = []
        self.identity: Optional[dict] = None
        self.bytes = 0                # everything under the directory, as on disk
        self._newest_cycle: Optional[int] = None
        self._highest_id = 0
        self._warned_cold_low = False

        if self.directory.exists():
            self._open_store()
        if self.cold is not None:
            self._open_cold()

    # ---- Opening ----

    def _open_store(self) -> None:
        old = [p.name for p in self.directory.glob("summaries.jsonl")]
        old += [p.name for folder in (self.directory, self._waiting_dir) if folder.exists()
                for p in folder.glob("memory_*.pt")]
        if old:
            raise PersistenceError(
                f"{self.directory} is a memory store from before 2026-11-18 "
                f"({old[0]} and {len(old) - 1} more). It is not read by this "
                "code and is never rewritten; open it with the code of its day."
            )
        store = self.directory / "store.json"
        if not store.exists():
            if any(self.directory.iterdir()):
                raise PersistenceError(f"{self.directory} has files and no store.json; not a memory store")
            return
        self.identity = json.loads(store.read_text(encoding="utf-8"))
        if self.identity.get("format") != FORMAT:
            raise PersistenceError(f"{store}: format {self.identity.get('format')}, and this code reads {FORMAT}")

        out: dict[int, bool] = {}                # id -> let go (True) or came back (False), by its last line
        order: list[int] = []
        if self._lines.exists():
            for text in self._lines.read_text(encoding="utf-8").splitlines():
                line = json.loads(text)
                memory_id = int(line["id"])
                self._highest_id = max(self._highest_id, memory_id)
                if line["kind"] == "let_go":
                    out[memory_id] = True
                    self._by[memory_id] = line["by"]
                    order.append(memory_id)
                    self.let_go_count += 1
                    self.let_go_for_room += int(line["by"] == "room")
                elif line["kind"] == "came_back":
                    out[memory_id] = False
                else:
                    raise PersistenceError(f"{self._lines}: a line of kind {line['kind']!r}")

        for path in sorted(self.directory.glob("memory_*")):
            memory_id = int(path.name.split("_")[1])
            if out.get(memory_id, False):
                # Let go, and the move to let_go/ was cut off. Finish it: the
                # line is what says it left their reach (and a folder in reach
                # whose last line is a let-go stays out of reach: Opus, D5).
                self._waiting_dir.mkdir(exist_ok=True)
                os.rename(path, self._waiting_dir / path.name)
                logger.warning("memory %d had a let-go line and was still in reach; moved to let_go/", memory_id)
        rows: list[tuple[int, int, int, torch.Tensor]] = []     # filed, id, seq whose view, summary
        for folder, reach in ((self.directory, True), (self._waiting_dir, False)):
            for path in sorted(folder.glob("memory_*")) if folder.exists() else []:
                memory_id = int(path.name.split("_")[1])
                self._highest_id = max(self._highest_id, memory_id)
                if not reach and not out.get(memory_id, False):
                    logger.warning(
                        "memory %d is in let_go/ and its last line does not say it was let go; "
                        "it stays out of reach until its folder is moved back", memory_id,
                    )
                stretch = _Stretch(id=memory_id, folder=path, reach=reach)
                for record in sorted(path.iterdir()):
                    stretch.bytes += record.stat().st_size
                    if record.suffix != ".rec":
                        self.unfinished_writes.append(f"{path.name}/{record.name}")
                        stretch.next_seq = max(stretch.next_seq, int(record.name[1:7]) + 1)
                        continue
                    header, blobs = _read(record, ("summary",))
                    seq = int(header["seq"])
                    stretch.next_seq = max(stretch.next_seq, seq + 1)
                    stretch.last_cycle = max(stretch.last_cycle, int(header["cycle"]))
                    stretch.signals.append({"kind": header["kind"], **header["signals"]})
                    self._filed = max(self._filed, int(header["filed"]))
                    self._newest_cycle = max(self._newest_cycle or 0, int(header["cycle"]))
                    if header["kind"] == "kept":
                        stretch.kept[seq] = int(header["cycle"])
                        self.kept += 1
                        of = seq
                    else:
                        self.looked_back += 1
                        of = int(header["signals"]["of_seq"])
                    if reach:
                        rows.append((int(header["filed"]), memory_id, of, _summary_of(blobs)))
                if not stretch.kept:
                    self.empty_folders.append(memory_id)
                    continue
                self._stretches[memory_id] = stretch
        self._waiting = [i for i in dict.fromkeys(order) if i in self._stretches and not self._stretches[i].reach]
        self._waiting += sorted(i for i, s in self._stretches.items() if not s.reach and i not in self._waiting)
        for _, memory_id, of, vector in sorted(rows, key=lambda r: r[0]):
            self._add_row(memory_id, of, vector)
        if self._window_path.exists():
            lines = self._window_path.read_text(encoding="utf-8").splitlines()
            self._window_lines = len(lines)
            for text in lines[-RANK_AMONG:]:
                self._window.append(float(json.loads(text)["value"]))
        self.bytes = _tree_bytes(self.directory)
        if self.unfinished_writes or self.empty_folders:
            logger.warning(
                "%s: %d write(s) cut off (%s) and %d folder(s) with no kept record (%s). "
                "They are left as they are and cannot be recalled.",
                self.directory, len(self.unfinished_writes), self.unfinished_writes[:5],
                len(self.empty_folders), self.empty_folders[:5],
            )

    def _begin(self) -> None:
        """The first write of a life: say what this store is."""
        self.directory.mkdir(parents=True, exist_ok=True)
        self.identity = {"format": FORMAT, "life_id": uuid.uuid4().hex, "began": _now(),
                         "path": str(self.directory.resolve())}
        atomic_write_json(self.directory / "store.json", self.identity)
        self.bytes = _tree_bytes(self.directory)
        if self.cold is not None:
            self._claim_cold()

    # ---- Cold storage ----

    def _open_cold(self) -> None:
        try:
            self.cold.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise PersistenceError(f"the cold folder {self.cold} cannot be made: {e}") from e
        if self.identity is not None:
            self._claim_cold()

    def _claim_cold(self) -> None:
        """A cold folder holds one life's memories. Ids start at 1 in every
        life, so two lives in one folder would collide (Opus, D1)."""
        mark = self.cold / "life.json"
        if mark.exists():
            owner = json.loads(mark.read_text(encoding="utf-8"))
            if owner["life_id"] != self.identity["life_id"]:
                raise PersistenceError(
                    f"the cold folder {self.cold} belongs to another life "
                    f"(begun {owner['began']} at {owner['path']}); this life began "
                    f"{self.identity['began']}. Give this life a cold folder of its own."
                )
        elif any(self.cold.iterdir()):
            raise PersistenceError(f"the cold folder {self.cold} has files and no life.json; not this life's")
        else:
            atomic_write_json(mark, {k: self.identity[k] for k in ("life_id", "began", "path")})
        self._settle_cold()

    def _cold_lines(self) -> list[dict]:
        index = self.cold / "index.jsonl"
        if not index.exists():
            return []
        return [json.loads(text) for text in index.read_text(encoding="utf-8").splitlines()]

    def _settle_cold(self) -> None:
        """Finish or undo a move that was cut off, so that every memory is in
        the life or in cold and never half in both (Opus, D3)."""
        last: dict[int, dict] = {}
        for line in self._cold_lines():
            last[int(line["id"])] = line
            self._highest_id = max(self._highest_id, int(line["id"]))
        moments: dict[int, int] = {}
        for line in self._cold_lines():
            if line["state"] == "moving":
                moments[int(line["id"])] = int(line["moments"])
        self.moved_to_cold = sum(1 for line in last.values() if line["state"] == "moved")
        self.moments_in_cold = sum(moments[i] for i, line in last.items() if line["state"] == "moved")
        for memory_id, line in sorted(last.items()):
            name = f"memory_{memory_id:08d}"
            there, here = self.cold / name, self._waiting_dir / name
            incoming = self.cold / "incoming" / name
            if line["state"] == "undone" or (line["state"] == "moved" and not here.exists()):
                continue
            if line["state"] == "moving" and not there.exists():
                if not here.exists():
                    raise PersistenceError(
                        f"memory {memory_id} was being moved to cold and is in neither place "
                        f"({here}, {there}). Nothing was changed; this needs a person."
                    )
                if incoming.exists():
                    shutil.rmtree(incoming)      # a copy that was never checked is not a memory
                append_jsonl(self.cold / "index.jsonl", {"state": "undone", "id": memory_id, "when": _now()})
                logger.warning("memory %d: a move to cold was cut off before it arrived; undone", memory_id)
                continue
            wanted = json.loads((self.cold / "manifests" / f"{name}.json").read_text(encoding="utf-8"))
            if _manifest(there) != wanted:
                raise PersistenceError(
                    f"memory {memory_id} in cold ({there}) is not what was moved. "
                    "Nothing was changed; this needs a person."
                )
            if line["state"] == "moving":
                append_jsonl(self.cold / "index.jsonl", {"state": "moved", "id": memory_id, "when": _now()})
                self.moved_to_cold += 1
                self.moments_in_cold += moments[memory_id]
                logger.warning("memory %d: a move to cold was cut off after it arrived; finished", memory_id)
            if here.exists():
                # The cold copy is whole and checked; what is left here is the original.
                shutil.rmtree(here)
                self.bytes = _tree_bytes(self.directory)
            if memory_id in self._stretches:
                self.kept -= len(self._stretches[memory_id].kept)
                del self._stretches[memory_id]
                self._waiting = [i for i in self._waiting if i != memory_id]

    def _to_cold(self, memory_id: int, *, cycle: int) -> bool:
        """Move a memory that was let go to cold storage, whole and exact.
        False, and nothing moved, when the cold drive has no room for it."""
        stretch = self._stretches[memory_id]
        if stretch.reach:
            raise PersistenceError(f"memory {memory_id} was not let go; only what was let go leaves for room")
        name = stretch.folder.name
        there, incoming = self.cold / name, self.cold / "incoming" / name
        free = self._disk_free(self.cold)
        if free < stretch.bytes + COLD_MARGIN_BYTES:
            self.cold_had_no_room += 1
            logger.error(
                "cold storage %s has %.1f GB free and a move needs %.1f GB beside itself; "
                "nothing leaves for room, and the store is full",
                self.cold, free / 1024**3, COLD_MARGIN_BYTES / 1024**3,
            )
            return False
        if free < COLD_WARN_BYTES and not self._warned_cold_low:
            self._warned_cold_low = True
            logger.warning("cold storage %s has under %d GB free (%.1f)", self.cold,
                           COLD_WARN_BYTES // 1024**3, free / 1024**3)
        if there.exists():
            raise PersistenceError(f"{there} already exists; nothing in cold is ever overwritten")
        files = _manifest(stretch.folder)
        manifest = self.cold / "manifests" / f"{name}.json"
        if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != files:
            raise PersistenceError(f"{manifest} is from an earlier move and does not match memory {memory_id}")
        atomic_write_json(manifest, files)
        append_jsonl(self.cold / "index.jsonl", {
            "state": "moving", "id": memory_id, "by": self._by[memory_id], "when": _now(), "cycle": int(cycle),
            "moments": len(stretch.kept), "first_moment": min(stretch.kept.values()),
            "last_moment": stretch.last_cycle,
            "importance": importance(stretch.signals),
            "bytes": stretch.bytes,
        })
        if self._same_drive(stretch.folder, self.cold):
            os.rename(stretch.folder, there)
        else:
            if incoming.exists():
                shutil.rmtree(incoming)          # left by a move that was cut off; never checked
            for relative in files:
                (incoming / relative).parent.mkdir(parents=True, exist_ok=True)
                _copy(stretch.folder / relative, incoming / relative)
            if _manifest(incoming) != files:
                raise PersistenceError(
                    f"memory {memory_id}: the copy in {incoming} does not match the original. "
                    "The original is untouched."
                )
            os.rename(incoming, there)
        append_jsonl(self.cold / "index.jsonl", {"state": "moved", "id": memory_id, "when": _now()})
        if stretch.folder.exists():
            shutil.rmtree(stretch.folder)        # the original, once its copy is checked and in place
        self.bytes -= stretch.bytes
        self.kept -= len(stretch.kept)
        self.moved_to_cold += 1
        self.moments_in_cold += len(stretch.kept)
        del self._stretches[memory_id]
        self._waiting.remove(memory_id)
        return True

    @staticmethod
    def _same_drive(a: Path, b: Path) -> bool:
        """Asked of the two folders' device numbers before anything moves
        (Opus, D6), not found out from a rename's error."""
        return os.stat(a).st_dev == os.stat(b).st_dev

    # ---- Room ----

    @property
    def _reserve(self) -> int:
        """Room held back from records so a let-go's line can always be written."""
        return 0 if self.limit_bytes is None else min(1 << 20, self.limit_bytes // 8)

    def _room_for(self, need: int, *, cycle: int, in_view: Iterable[int]) -> bool:
        """Make room for a record of ``need`` bytes. False when there is none
        to be had: the store is full."""
        place = self.directory if self.directory.exists() else self.directory.parent
        free = self._disk_free(place)
        if free < need + self.drive_margin_bytes:
            self.life_drive_had_no_room += 1
            logger.error(
                "the life's drive (%s) has %.2f GB free, under the %.2f GB kept clear; nothing is written",
                place, free / 1024**3, self.drive_margin_bytes / 1024**3,
            )
            return False
        if self.limit_bytes is None:
            return True
        ceiling = self.limit_bytes - self._reserve
        while self.bytes + need > ceiling:
            if self.cold is None:
                return False
            if not self._waiting:
                # A little more than is needed, so that this is not done at every keep.
                short = self.bytes + need - ceiling + self.limit_bytes // 100
                if not self._let_go_for_room(short, cycle=cycle, in_view=in_view):
                    return False
            if not self._to_cold(self._waiting[0], cycle=cycle):
                return False
        return True

    def _let_go_for_room(self, short: int, *, cycle: int, in_view: Iterable[int]) -> bool:
        """Let go, for the mind, of what matters least to them, until ``short``
        bytes are waiting. Never the stretch being written, one in view, or one
        still the present. False when every stretch is one of those."""
        held = set(in_view) | ({self._open} if self._open is not None else set())
        able = sorted(
            (importance(s.signals), s.last_cycle, s.id)
            for s in self._stretches.values()
            if s.reach and s.id not in held and cycle - s.last_cycle >= self.recent
        )
        freed = 0
        for weight, _, memory_id in able:
            if freed >= short:
                break
            freed += self._stretches[memory_id].bytes
            self.let_go(memory_id, cycle=cycle, by="room", importance_then=weight)
        return freed > 0

    # ---- Keeping ----

    @property
    def newest_cycle(self) -> Optional[int]:
        """The latest moment any record was filed at; None when there is none."""
        return self._newest_cycle

    def __len__(self) -> int:
        """How many memories can still be looked back on."""
        return sum(1 for s in self._stretches.values() if s.reach)

    def _path(self, memory_id: int) -> Path:
        return self.directory / f"memory_{memory_id:08d}"

    def _rank(self, value: Optional[float]) -> tuple[float, int]:
        """Where this choice sits among the mind's last kept moments: the share
        of them it was stronger than, counting an equal one as half. With none
        before it, or no strength given, 0.5."""
        among = len(self._window)
        if value is None or among == 0:
            return 0.5, among if value is not None else 0
        below = sum(1 for v in self._window if v < value)
        equal = sum(1 for v in self._window if v == value)
        return (below + 0.5 * equal) / among, among

    def keep(
        self,
        *,
        cycle: int,
        summary: torch.Tensor,
        tokens: list[int],
        new: Optional[int] = None,
        sight: Optional[Sight] = None,
        sound: Optional[Sound] = None,
        signals: Optional[dict] = None,
        in_view: Iterable[int] = (),
    ) -> Optional[int]:
        """Keep this moment. Returns its memory's id, or None when there is no
        room and none to be made: nothing was written.

        ``new`` is how many of ``tokens``, at the end, joined the view this
        moment. A moment kept straight after the last one, whose view follows
        on from it, joins the same memory and stores only its new words.
        ``signals`` is what the mind reported of the choice (``salience``,
        ``choice``); it is recorded as given. ``in_view`` is the memory being
        looked back on, if one is: it is not let go for room."""
        vector = _as_summary(summary)
        if self._matrix is not None and vector.numel() != self._matrix.shape[1]:
            raise ValueError(f"a summary of {vector.numel()} numbers, in a store of {self._matrix.shape[1]}")
        tokens = [int(t) for t in tokens]
        joins = (
            self._open is not None and cycle == self._open_cycle + 1 and new is not None
            and 0 <= new <= len(tokens) and len(self._open_words) + new >= len(tokens)
            and (self._open_words + tokens[len(tokens) - new:])[-len(tokens):] == tokens
        )
        if self._open is not None and cycle == self._open_cycle + 1 and not joins:
            self.stretches_restarted += 1
        fresh = tokens[len(tokens) - new:] if joins else tokens
        words = (len(self._open_words) if joins else 0) + len(fresh)
        memory_id = self._open if joins else max(self._highest_id, 0) + 1
        seq = self._stretches[memory_id].next_seq if joins else 1

        value, by = strength_of(signals)
        rank, among = self._rank(value)
        noted = {**(signals or {}), "value": value, "ranked_by": by, "rank": rank, "ranked_among": among}
        blobs = {"summary": _bits(vector), "tokens": _bits(torch.tensor(fresh, dtype=torch.int64).to(torch.int32))}
        if _tokens_of(blobs) != fresh:
            raise ValueError("a token does not fit 4 bytes")
        senses: dict[str, Any] = {}
        if sight is not None:
            senses["sight"], blobs["sight"] = _pack_sense(sight.pixels, torch.uint8, (255,))
        if sound is not None:
            meta, blobs["sound"] = _pack_sense(sound.samples, torch.int16, (32768, 32767))
            senses["sound"] = {**meta, "rate": int(sound.rate)}
        when = _now()
        record = _pack({
            "format": FORMAT, "kind": "kept", "id": memory_id, "seq": seq, "cycle": int(cycle),
            "when": when, "filed": self._filed + 1, "view": [words - len(tokens), words],
            "signals": noted, **senses,
        }, blobs)
        # What will be read back is compared with what was given, before anything is written.
        if not (
            _same_bits(_summary_of(blobs), vector)
            and (sight is None or _same_bits(_unpack_sense(senses["sight"], blobs["sight"]), sight.pixels.detach().cpu()))
            and (sound is None or _same_bits(_unpack_sense(senses["sound"], blobs["sound"]), sound.samples.detach().cpu()))
        ):
            raise PersistenceError("a record would not read back as it was given; nothing was written")

        if self.identity is None:
            self._begin()
        if not self._room_for(len(record) + 256, cycle=cycle, in_view=in_view):
            return None
        folder = self._path(memory_id)
        if not joins:
            if folder.exists() or (self._waiting_dir / folder.name).exists():
                raise PersistenceError(f"{folder} already exists; a memory is never rewritten")
            folder.mkdir(parents=True)
            self._highest_id = memory_id
            self._stretches[memory_id] = _Stretch(id=memory_id, folder=folder, reach=True)
            self._open, self._open_words = memory_id, []
        stretch = self._stretches[memory_id]
        self._write(folder / f"r{seq:06d}.rec", record)
        stretch.next_seq = seq + 1
        stretch.bytes += len(record)
        stretch.kept[seq] = int(cycle)
        stretch.last_cycle = int(cycle)
        stretch.signals.append({"kind": "kept", **noted})
        self._open_words += fresh
        self._open_cycle = int(cycle)
        self._filed += 1
        self._newest_cycle = max(self._newest_cycle or 0, int(cycle))
        self._add_row(memory_id, seq, vector)
        self.kept += 1
        if value is not None:
            self._note_strength(cycle, value, by)
        return memory_id

    def _write(self, path: Path, record: bytes) -> None:
        """One record, written whole or not at all, and never over another."""
        if path.exists():
            raise PersistenceError(f"{path} already exists; a record is never rewritten")
        tmp = path.with_suffix(".rec.tmp")
        try:
            with open(tmp, "wb") as f:
                f.write(record)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except OSError as e:
            raise PersistenceError(f"writing {path} failed: {e}") from e
        self.bytes += len(record)

    def _note_strength(self, cycle: int, value: float, by: Optional[str]) -> None:
        size = self._window_path.stat().st_size if self._window_path.exists() else 0
        append_jsonl(self._window_path, {"cycle": int(cycle), "value": value, "by": by})
        self._window.append(value)
        self._window_lines += 1
        if self._window_lines > 8 * RANK_AMONG:
            # A cache: cut back to what ranking reads.
            kept = self._window_path.read_text(encoding="utf-8").splitlines()[-RANK_AMONG:]
            tmp = self._window_path.with_suffix(".jsonl.tmp")
            tmp.write_text("\n".join(kept) + "\n", encoding="utf-8")
            os.replace(tmp, self._window_path)
            self._window_lines = len(kept)
        self.bytes += self._window_path.stat().st_size - size

    def look_back(
        self,
        memory_id: int,
        *,
        cycle: int,
        summary: torch.Tensor,
        of_seq: Optional[int] = None,
        chosen_recall: bool = False,
        chose_remember: bool = False,
        likeness: Optional[float] = None,
        next_likeness: Optional[float] = None,
        beat: Optional[int] = None,
        in_view: Iterable[int] = (),
    ) -> bool:
        """The mind has just looked back on this memory. File how it seems to
        them now beside how it seemed before. What was perceived is untouched.
        Recorded with it: which moment they looked at, whether they chose to
        recall it or rest brought it, whether looking at it they chose to
        remember, and how it was found. False when there was no room for the
        record: it was not filed."""
        stretch = self._stretches.get(memory_id)
        if stretch is None or not stretch.reach:
            raise KeyError(f"no memory {memory_id}")
        vector = _as_summary(summary)
        if vector.numel() != self._matrix.shape[1]:
            raise ValueError(f"a summary of {vector.numel()} numbers, in a store of {self._matrix.shape[1]}")
        of_seq = min(stretch.kept) if of_seq is None else int(of_seq)
        if of_seq not in stretch.kept:
            raise KeyError(f"memory {memory_id} has no kept moment {of_seq}")
        noted = {
            "of_seq": of_seq, "chosen_recall": bool(chosen_recall), "chose_remember": bool(chose_remember),
            "likeness": likeness, "next_likeness": next_likeness, "beat": beat,
        }
        seq = stretch.next_seq
        record = _pack({
            "format": FORMAT, "kind": "looked_back", "id": memory_id, "seq": seq, "cycle": int(cycle),
            "when": _now(), "filed": self._filed + 1, "signals": noted,
        }, {"summary": _bits(vector)})
        if not self._room_for(len(record), cycle=cycle, in_view={memory_id, *in_view}):
            self.not_filed_for_room += 1
            return False
        self._write(stretch.folder / f"r{seq:06d}.rec", record)
        stretch.next_seq = seq + 1
        stretch.bytes += len(record)
        stretch.last_cycle = max(stretch.last_cycle, int(cycle))
        stretch.signals.append({"kind": "looked_back", **noted})
        self._filed += 1
        self._newest_cycle = max(self._newest_cycle or 0, int(cycle))
        self._add_row(memory_id, of_seq, vector)
        self.looked_back += 1
        return True

    def let_go(self, memory_id: int, *, cycle: int, by: str = "choice",
               importance_then: Optional[float] = None) -> None:
        """This memory leaves the mind's reach now: by their choice, or for
        room (``by``). The whole stretch waits in ``let_go/`` until the room
        is needed. That it was let go is written first, so a memory is never
        out of reach without a line saying why."""
        if by not in ("choice", "room"):
            raise ValueError(f"a memory is let go by choice or for room; got {by!r}")
        stretch = self._stretches.get(memory_id)
        if stretch is None or not stretch.reach:
            raise KeyError(f"no memory {memory_id} to let go")
        weight = importance(stretch.signals) if importance_then is None else importance_then
        line = {
            "kind": "let_go", "id": int(memory_id), "by": by, "cycle": int(cycle), "when": _now(),
            "importance": weight, "moments": len(stretch.kept), "bytes": stretch.bytes,
        }
        size = len(json.dumps(line)) + 2
        if self.limit_bytes is not None and self.bytes + size > self.limit_bytes:
            raise PersistenceError(
                f"{self.directory} has no room left even for the line that a memory was let go "
                f"({self.bytes} of {self.limit_bytes} bytes); the store needs a cold folder or a larger limit"
            )
        before = self._lines.stat().st_size if self._lines.exists() else 0
        append_jsonl(self._lines, line)
        self.bytes += self._lines.stat().st_size - before
        self._waiting_dir.mkdir(parents=True, exist_ok=True)
        os.rename(stretch.folder, self._waiting_dir / stretch.folder.name)
        stretch.folder = self._waiting_dir / stretch.folder.name
        stretch.reach = False
        self._by[memory_id] = by
        self._waiting.append(memory_id)
        self.let_go_count += 1
        self.let_go_for_room += int(by == "room")
        if self._open == memory_id:
            self._open = None
        self._drop_rows(memory_id)

    # ---- The search ----

    def _add_row(self, memory_id: int, seq: int, vector: torch.Tensor) -> None:
        if self._matrix is None:
            self._matrix = torch.empty(1024, vector.numel(), dtype=torch.float32)
            self._row_id = torch.empty(1024, dtype=torch.int64)
            self._row_seq = torch.empty(1024, dtype=torch.int64)
        if vector.numel() != self._matrix.shape[1]:
            raise PersistenceError(
                f"memory {memory_id}: a summary of {vector.numel()} numbers in a store of {self._matrix.shape[1]}"
            )
        if self._rows == self._matrix.shape[0]:
            more = self._rows * 2
            for name, wide in (("_matrix", True), ("_row_id", False), ("_row_seq", False)):
                old = getattr(self, name)
                grown = torch.empty((more, old.shape[1]) if wide else more, dtype=old.dtype)
                grown[: self._rows] = old
                setattr(self, name, grown)
        self._matrix[self._rows] = vector
        self._row_id[self._rows] = memory_id
        self._row_seq[self._rows] = seq
        self._rows += 1
        wide = vector.to(torch.float64)
        self._sum = wide if self._sum is None else self._sum + wide

    def _drop_rows(self, memory_id: int) -> None:
        stay = self._row_id[: self._rows] != memory_id
        count = int(stay.sum())
        self._matrix[:count] = self._matrix[: self._rows][stay]
        self._row_id[:count] = self._row_id[: self._rows][stay]
        self._row_seq[:count] = self._row_seq[: self._rows][stay]
        self._rows = count
        # The sum again, one row at a time in order, as a rebuilt cache has it.
        total = torch.zeros(self._matrix.shape[1], dtype=torch.float64)
        for at in range(0, count, _CHUNK):
            part = self._matrix[at: min(at + _CHUNK, count)].to(torch.float64)
            part[0] += total
            total = part.cumsum(dim=0)[-1].clone()
        self._sum = total if count else None

    def _likeness(self, cue: torch.Tensor) -> torch.Tensor:
        """The cosine between the cue and every summary in reach, after the
        mean of those summaries is removed from both, in 8-byte arithmetic. A
        summary that is the mean itself is like nothing: 0."""
        mean = self._sum / self._rows
        centred = cue.to(torch.float64) - mean
        out = torch.empty(self._rows, dtype=torch.float64)
        for at in range(0, self._rows, _CHUNK):
            rows = self._matrix[at: min(at + _CHUNK, self._rows)].to(torch.float64) - mean
            norms = rows.norm(dim=1) * centred.norm()
            out[at: at + rows.shape[0]] = torch.where(norms > 0, (rows @ centred) / norms, torch.zeros_like(norms))
        return out

    def _still_present(self, now: int) -> set[int]:
        return {s.id for s in self._stretches.values() if s.reach and now - s.last_cycle < self.recent}

    def nearest(self, cue: torch.Tensor, *, now: int) -> Optional[Memory]:
        """The kept moment most like ``cue``, by any summary its memory has
        been filed under, leaving out what is still the present. None when
        nothing qualifies. A look back's summary brings the moment that look
        back was of."""
        if self._rows == 0:
            return None
        started = time.perf_counter()
        vector = _as_summary(cue)
        if vector.numel() != self._matrix.shape[1]:
            raise ValueError(f"a cue of {vector.numel()} numbers, in a store of {self._matrix.shape[1]}")
        ids = self._row_id[: self._rows]
        present = self._still_present(now)
        able = ~torch.isin(ids, torch.tensor(sorted(present), dtype=torch.int64)) if present else torch.ones_like(ids, dtype=torch.bool)
        if not bool(able.any()):
            return None
        likeness = self._likeness(vector).masked_fill(~able, float("-inf"))
        best = int(likeness.argmax())
        memory_id = int(ids[best])
        others = able & (ids != memory_id)
        beat = int(torch.unique(ids[others]).numel())
        runner_up = float(likeness[others].max()) if beat else None
        took = time.perf_counter() - started
        self.searches += 1
        self.slowest_search_seconds = max(self.slowest_search_seconds, took)
        if self.moment_seconds is not None and took > self.moment_seconds:
            self.slow_searches += 1
            if self.slow_searches == 1 or self.slow_searches % 100 == 0:
                logger.error(
                    "a search of %d summaries took %.3f s and a moment is %.3f s (%d slow so far)",
                    self._rows, took, self.moment_seconds, self.slow_searches,
                )
        return self.open(memory_id, int(self._row_seq[best]), likeness=float(likeness[best]),
                         next_likeness=runner_up, beat=beat)

    def _moments(self, now: int, unlike: set[int]) -> list[tuple[int, int]]:
        """Every kept moment that is not still the present and not in ``unlike``."""
        present = self._still_present(now)
        return sorted(
            (s.id, seq) for s in self._stretches.values()
            if s.reach and s.id not in present and s.id not in unlike for seq in s.kept
        )

    def another(self, *, now: int, unlike: int, pick: float) -> Optional[Memory]:
        """A kept moment of a memory other than ``unlike`` that is not still
        the present, picked by ``pick`` (a number from 0 up to 1, the caller's
        dice). None when there is no other."""
        others = self._moments(now, {unlike})
        if not others:
            return None
        return self.open(*others[min(int(pick * len(others)), len(others) - 1)])

    def some(self, count: int, *, now: int, unlike: set[int], dice) -> list[Memory]:
        """One kept moment from each of up to ``count`` different memories, not
        still the present and not in ``unlike``, picked by ``dice`` (a
        ``random.Random``)."""
        by_memory: dict[int, list[int]] = {}
        for memory_id, seq in self._moments(now, set(unlike)):
            by_memory.setdefault(memory_id, []).append(seq)
        picked = dice.sample(sorted(by_memory), min(count, len(by_memory)))
        return [self.open(i, dice.choice(by_memory[i])) for i in picked]

    def open(self, memory_id: int, seq: Optional[int] = None, **found) -> Memory:
        """A kept moment exactly as it was: its view, its picture, its sound.
        ``seq`` is which record of the memory; its first kept moment when not
        said."""
        stretch = self._stretches.get(memory_id)
        if stretch is None or not stretch.reach:
            raise KeyError(f"no memory {memory_id}")
        seq = min(stretch.kept) if seq is None else int(seq)
        if seq not in stretch.kept:
            raise KeyError(f"memory {memory_id} has no kept moment {seq}")
        words: list[int] = []
        for earlier in sorted(s for s in stretch.kept if s < seq):
            words += _tokens_of(_read(stretch.folder / f"r{earlier:06d}.rec", ("tokens",))[1])
        header, blobs = _read(stretch.folder / f"r{seq:06d}.rec")
        words += _tokens_of(blobs)
        start, end = header["view"]
        if end != len(words) or start < 0:
            raise PersistenceError(f"memory {memory_id}, record {seq}: its view is not in the stretch's words")
        sound = header.get("sound")
        return Memory(
            id=memory_id, seq=seq, cycle=int(header["cycle"]), when=header["when"], tokens=words[start:end],
            sight=Sight(_unpack_sense(header["sight"], blobs["sight"])) if "sight" in header else None,
            sound=Sound(_unpack_sense(sound, blobs["sound"]), sound["rate"]) if sound else None,
            **found,
        )

    def summaries_of(self, memory_id: int) -> list[tuple[int, torch.Tensor]]:
        """Every way this memory has been seen, in the order filed: the moment
        and the summary. Read from its records, while it is on this life's
        disk, in reach or waiting."""
        stretch = self._stretches.get(memory_id)
        if stretch is None:
            raise KeyError(f"no memory {memory_id} on this life's disk")
        seen = []
        for record in sorted(stretch.folder.glob("r*.rec")):
            header, blobs = _read(record, ("summary",))
            seen.append((int(header["filed"]), int(header["cycle"]), _summary_of(blobs)))
        return [(cycle, vector) for _, cycle, vector in sorted(seen, key=lambda s: s[0])]

    def signals_of(self, memory_id: int) -> list[dict]:
        """The mind's signals in this memory's records, as importance reads them."""
        return [dict(s) for s in self._stretches[memory_id].signals]

    def report(self) -> dict[str, Any]:
        reach = [s for s in self._stretches.values() if s.reach]
        return {
            "held": len(reach), "held_moments": sum(len(s.kept) for s in reach),
            "kept": self.kept + self.moments_in_cold, "kept_moments_on_disk": self.kept,
            "moments_in_cold": self.moments_in_cold, "looked_back": self.looked_back,
            "let_go": self.let_go_count, "let_go_for_room": self.let_go_for_room,
            "let_go_waiting": len(self._waiting),
            "let_go_waiting_bytes": sum(self._stretches[i].bytes for i in self._waiting),
            "moved_to_cold": self.moved_to_cold, "cold": None if self.cold is None else str(self.cold),
            "bytes": self.bytes, "limit_bytes": self.limit_bytes,
            "summaries_searched": self._rows, "searches": self.searches,
            "slow_searches": self.slow_searches, "slowest_search_seconds": self.slowest_search_seconds,
            "not_filed_for_room": self.not_filed_for_room,
            "cold_had_no_room": self.cold_had_no_room,
            "life_drive_had_no_room": self.life_drive_had_no_room,
            "stretches_restarted": self.stretches_restarted,
            "unfinished_writes": list(self.unfinished_writes),
            "empty_folders": list(self.empty_folders),
        }


def dream_of(near: Memory, far: Memory) -> tuple[list[int], Optional[Sight], Optional[Sound]]:
    """A moment that never happened, from two that did: the first half of one's
    words and the second half of the other's, the picture of one and the sound
    of the other where both have them."""
    words = near.tokens[: max(1, len(near.tokens) // 2)] + far.tokens[len(far.tokens) // 2:]
    sight = near.sight if near.sight is not None else far.sight
    sound = far.sound if far.sound is not None else near.sound
    return words, sight, sound
