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

**A memory is a stretch of moments kept one after another,** in one file,
``memory_00000001.mem``. Each kept moment appends one record, never changed
afterwards: the words new since the record before, where this moment's view
sits in the stretch's words, its picture and sound, its summary, and the
mind's signals. A look back appends one more record to the same file. Recall
brings back the view at the moment whose summary matched. Letting go and
importance act on the whole stretch.

**A record is a frame, and a frame fills whole clusters** (2026-11-18, build
brief 10 with Opus's 1.6; one file a record made 288,000 files a day). A frame
is a small header (its record's length and hash), the record, and zeros up to
the next multiple of the drive's cluster size, which is read once when the
life begins and written in ``store.json``. So an append writes only clusters
no earlier frame is in, and a cut-off append can harm only itself. A torn last
frame is said at opening; the frames before it are whole; that stretch is
closed and nothing is appended to it again.

**The search is exact and its matrix is a cache** rebuilt from the files at
every start. Likeness is the cosine after the mean of all the summaries in
reach is removed, computed in 8-byte arithmetic. A search that takes longer
than a moment is counted and logged.

**The search is screened, and the answer is still the full search's** (build
brief 10, part 2, with Opus's 2.1 to 2.6). One pass in 4-byte arithmetic gives
every row an upper and a lower bound on its likeness. Only rows whose upper
bound reaches the best lower bound among the rows the search may return are
computed in 8-byte arithmetic, by the same code on the same chunks of rows as
the full search, so their values are the full search's to the last bit. Every
hundredth search also runs the full search, and a difference raises.

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
(``importance`` below). It is the second formula and may change again; the
records hold what it is made from.

Old stores (``summaries.jsonl`` and ``memory_*.pt``; and the folders of
records of the first rework, format 2) are refused and never rewritten.

Dreaming (Brian, 2026-11-11) is unchanged: a dream is a moment that never
happened, made from two kept moments. See ``dream_of``.

On disk, in ``<life>/memories/``:

- ``store.json``: the format, this life's id, when it began, and the frame
  size. Written once.
- ``memory_00000001.mem``: one stretch, a frame for each record. Only ever
  appended to.
- ``life.jsonl``: one line each time a memory is let go, by whom. Append-only.
- ``keeps_window.jsonl``: the last kept moments' strengths, for ranking a new
  one against the mind's own recent bar. A cache; it may be rewritten.
- ``let_go/memory_00000001.mem``: let go, waiting.

In the cold folder: ``life.json`` (whose it is), ``index.jsonl`` (a line
before each move and a line after), ``manifests/`` (the file's size and hash,
written before the move), ``incoming/`` (a copy not yet checked, which is not
a memory) and the memories.
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
FORMAT = 3
DEFAULT_LIMIT_BYTES = 100 * 1024**3   # Brian, 2026-11-17: "Luthi can have lets say 100gb."
COLD_MARGIN_BYTES = 5 * 1024**3       # a move needs this much free beside itself
COLD_WARN_BYTES = 50 * 1024**3        # warn when the cold drive has less than this
RANK_AMONG = 1000                     # a keep is ranked among this many before it
_MAGIC = b"LUTHIMEM2\n"
_FRAME = b"LUTHIFR1"                  # a frame: this, its record's length, the frame size, the record's hash
_FRAME_HEAD = struct.Struct("<8sQI32s")
_FRAME_BYTES = _FRAME_HEAD.size + 4   # and a checksum of those
SUFFIX = ".mem"
_CHUNK = 1024                         # rows searched at a time; measured fastest on the desktop
_UNIT = 2.0 ** -24                    # half the gap between neighbouring 4-byte numbers, relative
CHECK_EVERY = 100                     # every this-many-th screened search also runs the full search


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


def _cluster(path: Path) -> int:
    """The size of a cluster on the drive ``path`` is on: what a file's size
    is rounded up to, and what one write to a file's end can touch."""
    if os.name == "nt":
        import ctypes
        root = Path(path).resolve().anchor
        per_cluster, per_sector, free, total = (ctypes.c_ulong() for _ in range(4))
        if not ctypes.windll.kernel32.GetDiskFreeSpaceW(
            ctypes.c_wchar_p(root), ctypes.byref(per_cluster), ctypes.byref(per_sector),
            ctypes.byref(free), ctypes.byref(total),
        ):
            raise PersistenceError(f"the cluster size of {root} could not be read: {ctypes.WinError()}")
        return int(per_cluster.value) * int(per_sector.value)
    return int(os.statvfs(path).f_frsize)


def _framed(length: int, pad: int) -> int:
    """How many bytes a record of ``length`` takes as a frame."""
    return -(-(_FRAME_BYTES + length) // pad) * pad


def _frame(record: bytes, pad: int) -> bytes:
    head = _FRAME_HEAD.pack(_FRAME, len(record), pad, hashlib.sha256(record).digest())
    whole = head + struct.pack("<I", zlib.crc32(head)) + record
    return whole + b"\0" * (_framed(len(record), pad) - len(whole))


def _frame_head(f, at: int) -> Optional[tuple[int, int, bytes]]:
    """The frame that starts at ``at``: its record's length, its frame size
    and its record's hash. None when no whole frame header is there."""
    f.seek(at)
    raw = f.read(_FRAME_BYTES)
    if len(raw) != _FRAME_BYTES or raw[:8] != _FRAME or zlib.crc32(raw[:-4]) != struct.unpack("<I", raw[-4:])[0]:
        return None
    _, length, pad, sha = _FRAME_HEAD.unpack(raw[:-4])
    return length, pad, sha


def _frames(f, size: int, pad: int, label) -> tuple[list[int], Optional[str]]:
    """Where each whole frame of a stretch's file starts, and what is wrong
    with the file's end when it is torn (None when it is whole).

    Only the last frame can be torn by a cut-off append, so only the last
    frame's whole record is read against its hash here; the others are checked
    part by part as they are read. A torn frame with a whole frame after it is
    not a cut-off append: that is damage, and it raises."""
    starts, at = [], 0
    while at < size:
        head = _frame_head(f, at)
        if head is None or at + _FRAME_BYTES + head[0] > size:
            for later in range(at + pad, size, pad):
                if _frame_head(f, later) is not None:
                    raise PersistenceError(
                        f"{label}: the frame at byte {at} is damaged and a whole frame follows it at "
                        f"{later}. A cut-off append cannot do that; this needs a person."
                    )
            return starts, f"torn at byte {at} of {size}"
        length, _, sha = head
        after = at + _framed(length, pad)
        if after >= size:
            f.seek(at + _FRAME_BYTES)
            if hashlib.sha256(f.read(length)).digest() != sha:
                return starts, f"torn at byte {at} of {size}: the last record is not what its frame says"
            starts.append(at)
            if after > size:
                return starts, f"the last frame's padding is cut short ({size} bytes, {after} whole)"
            return starts, None
        starts.append(at)
        at = after
    return starts, None


def _read_from(f, at: int, want: Optional[tuple[str, ...]], label) -> tuple[dict, dict[str, bytes]]:
    """The record in the frame at ``at``: its header and the parts asked for
    (all of them when ``want`` is None), each checked against its hash."""
    if _frame_head(f, at) is None:
        raise PersistenceError(f"{label}: no frame at byte {at}")
    base = at + _FRAME_BYTES
    f.seek(base)
    start = f.read(len(_MAGIC) + 8)
    if start[: len(_MAGIC)] != _MAGIC or len(start) != len(_MAGIC) + 8:
        raise PersistenceError(f"{label}, byte {at}: not a memory record")
    length, crc = struct.unpack("<II", start[len(_MAGIC):])
    head = f.read(length)
    if len(head) != length or zlib.crc32(head) != crc:
        raise PersistenceError(f"{label}, byte {at}: the record's header is damaged")
    header = json.loads(head)
    base, blobs = base + len(_MAGIC) + 8 + length, {}
    for name in (header["blobs"] if want is None else want):
        if name not in header["blobs"]:
            continue
        where, size, sha = header["blobs"][name]
        f.seek(base + where)
        data = f.read(size)
        if len(data) != size or hashlib.sha256(data).hexdigest() != sha:
            raise PersistenceError(f"{label}, byte {at}: the record's {name} is damaged")
        blobs[name] = data
    return header, blobs


def _read(path: Path, want: Optional[tuple[str, ...]] = None, at: int = 0) -> tuple[dict, dict[str, bytes]]:
    with open(path, "rb") as f:
        return _read_from(f, at, want, path)


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


def _manifest(file: Path) -> dict[str, list]:
    """A stretch's file by name: its size and hash."""
    return {file.name: _hash_file(file)}


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
    nothing else. The second formula (build brief 10 and the design seat's
    ruling of 2026-11-18, after Opus's E2, E3, 3.1 and 3.2). Three parts, each
    between 0 and 1 so that none swamps another:

    - **the mean rank of its kept moments.** The mean and not the highest: the
      highest of many ranks is high by count alone, which favoured a stretch
      for being long;
    - **0.5 if they ever chose a recall that brought it.** Once, however many
      times. They choose to look back and the store chooses which memory, and
      each look files a summary that makes the same memory easier to find, so
      counting every time would feed itself;
    - **how often, looking at it by their own choice, they chose to remember:**
      (r + 1/2) / (n + 1), with n the looks they chose and r the times they
      chose to remember during one. A rate and not a count, because every look
      is one more chance, and how many looks there were says how often the
      present resembled the memory and nothing about the memory. With no look
      it is 1/2.

    **A look that rest brought counts for nothing,** in any part: they did not
    choose to look. Choosing to let go during one does nothing either (the
    loop), so the two choices are treated alike.

    Two properties to know. **Short stretches spread wider** (Opus, 3.3): a
    stretch of one moment can have any mean rank, and a long stretch's mean is
    held near the middle, so both ends of the order are mostly short stretches
    and letting go for room works from the low end. The mean is kept anyway,
    because drawing it toward 1/2 would make equal ranks at different lengths
    unequal. **For a mind that gives no signal of how strongly it chose,**
    every rank is 0.5, every stretch never looked at ties, and the tie falls
    to "not kept or looked at for longest", which is age. The design says age
    alone is not unimportance; that case is a stand-in mind's, and the order
    it gives says nothing about what matters.

    The records hold likeness, how many it beat, and each choice's margin, so
    a later formula needs no memory touched."""
    ranks, chosen, remembered, recalled = [], 0, 0, False
    for signal in signals:
        if signal["kind"] == "kept":
            ranks.append(float(signal["rank"]))
        elif signal["chosen_recall"]:
            recalled = True
            chosen += 1
            remembered += int(bool(signal["chose_remember"]))
    if not ranks:
        raise ValueError("a stretch with no kept moment has no importance")
    return sum(ranks) / len(ranks) + (0.5 if recalled else 0.0) + (remembered + 0.5) / (chosen + 1)


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
    path: Path                               # its one file
    reach: bool                              # False: let go, waiting in let_go/
    kept: dict[int, int] = field(default_factory=dict)   # seq -> the moment it was kept
    signals: list[dict] = field(default_factory=list)
    starts: list[int] = field(default_factory=list)      # where record seq's frame starts, at seq - 1
    bytes: int = 0                           # the file's size
    last_cycle: int = -1
    closed: Optional[str] = None             # why nothing is appended to it again, when that is so

    @property
    def next_seq(self) -> int:
        return len(self.starts) + 1


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
        screened: bool = True,
    ):
        """``recent``: a memory kept or looked back on within this many moments
        is not what recall brings, and is never let go for room: it is still
        the present. ``limit_bytes``: how much disk everything under
        ``directory`` may take; None for no limit. ``cold``: where what leaves
        for room is moved; None and nothing ever leaves. ``moment_seconds``:
        how long a moment lasts, so a slower search can be said.
        ``drive_margin_bytes``: nothing is written when the life's drive has
        less than this free beside the write. ``disk_free`` is how free room
        is read. ``screened``: False and every search computes every row."""
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
        # For the screen: a reference mean, and each row's length and squared
        # distance from that mean, in 8-byte numbers. Made at the first search
        # after a start or a let-go, and again when the rows have doubled.
        self._ref_mean: Optional[torch.Tensor] = None
        self._ref_base = 0
        self._row_len = torch.empty(0, dtype=torch.float64)
        self._row_far = torch.empty(0, dtype=torch.float64)
        self.screened = screened
        self.screened_searches = 0
        self.full_searches = 0                   # searches that computed every row: asked for, or the screen left too many
        self.checked_searches = 0                # screened searches also run in full and compared
        self.rows_computed_exactly = 0           # over all screened searches
        self.most_candidates = 0
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
        self.not_filed_closed = 0     # looks back at a stretch whose file's end is torn
        self.torn: list[str] = []     # stretches whose last frame was cut off, and where
        self.empty_stretches: list[int] = []   # files with no whole kept record
        self.identity: Optional[dict] = None
        self._pad = 0                 # the frame size, from store.json
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
            raise PersistenceError(
                f"{store}: format {self.identity.get('format')}, and this code reads {FORMAT}. "
                "The store is not read and is never rewritten; open it with the code of its day."
            )
        self._pad = int(self.identity["frame"])
        cluster = _cluster(self.directory)
        if self._pad % cluster:
            raise PersistenceError(
                f"{self.directory}: this life's frames are {self._pad} bytes and this drive's clusters are "
                f"{cluster}. An append here could rewrite part of a cluster that holds an earlier frame. "
                "Nothing was changed; this needs a person."
            )

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

        for name, _ in self._listing(self.directory):
            memory_id = int(name[len("memory_"):-len(SUFFIX)])
            if out.get(memory_id, False):
                # Let go, and the move to let_go/ was cut off. Finish it: the
                # line is what says it left their reach (and a file in reach
                # whose last line is a let-go stays out of reach: Opus, D5).
                self._waiting_dir.mkdir(exist_ok=True)
                os.rename(self.directory / name, self._waiting_dir / name)
                logger.warning("memory %d had a let-go line and was still in reach; moved to let_go/", memory_id)
        rows: list[tuple[int, int, int, torch.Tensor]] = []     # filed, id, seq whose view, summary
        for folder, reach in ((self.directory, True), (self._waiting_dir, False)):
            for name, size in self._listing(folder):
                memory_id = int(name[len("memory_"):-len(SUFFIX)])
                path = folder / name
                self._highest_id = max(self._highest_id, memory_id)
                if not reach and not out.get(memory_id, False):
                    logger.warning(
                        "memory %d is in let_go/ and its last line does not say it was let go; "
                        "it stays out of reach until its file is moved back", memory_id,
                    )
                stretch = _Stretch(id=memory_id, path=path, reach=reach, bytes=size)
                with open(path, "rb") as f:
                    starts, torn = _frames(f, size, self._pad, path)
                    if torn is not None:
                        stretch.closed = torn
                        self.torn.append(f"{name}: {torn}")
                    for seq, at in enumerate(starts, start=1):
                        header, blobs = _read_from(f, at, ("summary",), path)
                        if int(header["seq"]) != seq or int(header["id"]) != memory_id:
                            raise PersistenceError(
                                f"{path}, byte {at}: record {header['seq']} of memory {header['id']}, "
                                f"where record {seq} of memory {memory_id} belongs"
                            )
                        stretch.starts.append(at)
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
                    self.empty_stretches.append(memory_id)
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
        if self.torn or self.empty_stretches:
            logger.error(
                "%s: %d stretch(es) with the end of the file torn (%s), of which %d have no whole kept "
                "record (%s). The records before a torn end are whole and can be recalled. Nothing is "
                "appended to these files again, and nothing of them was changed.",
                self.directory, len(self.torn), self.torn[:5],
                len(self.empty_stretches), self.empty_stretches[:5],
            )

    @staticmethod
    def _listing(folder: Path) -> list[tuple[str, int]]:
        """The stretches' files in a folder, by name, with their sizes. No
        file is opened."""
        if not folder.exists():
            return []
        with os.scandir(folder) as entries:
            return sorted(
                (e.name, e.stat().st_size) for e in entries
                if e.name.startswith("memory_") and e.name.endswith(SUFFIX)
            )

    def _begin(self) -> None:
        """The first write of a life: say what this store is."""
        self.directory.mkdir(parents=True, exist_ok=True)
        self._pad = _cluster(self.directory)
        self.identity = {"format": FORMAT, "life_id": uuid.uuid4().hex, "began": _now(),
                         "path": str(self.directory.resolve()), "frame": self._pad}
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
            name = f"memory_{memory_id:08d}{SUFFIX}"
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
                    os.remove(incoming)          # a copy that was never checked is not a memory
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
                os.remove(here)
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
        name = stretch.path.name
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
        files = _manifest(stretch.path)
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
        if self._same_drive(stretch.path, self.cold):
            os.rename(stretch.path, there)
        else:
            if incoming.exists():
                os.remove(incoming)              # left by a move that was cut off; never checked
            incoming.parent.mkdir(parents=True, exist_ok=True)
            _copy(stretch.path, incoming)
            if _manifest(incoming) != files:
                raise PersistenceError(
                    f"memory {memory_id}: the copy in {incoming} does not match the original. "
                    "The original is untouched."
                )
            os.rename(incoming, there)
        append_jsonl(self.cold / "index.jsonl", {"state": "moved", "id": memory_id, "when": _now()})
        if stretch.path.exists():
            os.remove(stretch.path)              # the original, once its copy is checked and in place
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
        return self.directory / f"memory_{memory_id:08d}{SUFFIX}"

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
        if not self._room_for(_framed(len(record), self._pad) + 256, cycle=cycle, in_view=in_view):
            return None
        if joins:
            stretch = self._stretches[memory_id]
        else:
            path = self._path(memory_id)
            if path.exists() or (self._waiting_dir / path.name).exists():
                raise PersistenceError(f"{path} already exists; a memory is never rewritten")
            self._highest_id = memory_id         # used, whether or not the write below is cut off
            stretch = _Stretch(id=memory_id, path=path, reach=True)
            self._open = None
        self._append(stretch, record)
        if not joins:
            self._stretches[memory_id] = stretch
            self._open, self._open_words = memory_id, []
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

    def _append(self, stretch: _Stretch, record: bytes) -> None:
        """One record, as a frame at the end of its stretch's file. Nothing
        already in the file is written again: the file is opened to append,
        and the frame starts on a cluster of its own. If the write is cut
        off, nothing more is appended to this stretch."""
        if stretch.closed is not None:
            raise PersistenceError(f"{stretch.path} is closed ({stretch.closed}); nothing is appended to it")
        framed = _frame(record, self._pad)
        try:
            with open(stretch.path, "ab" if stretch.starts else "xb") as f:
                at = f.seek(0, os.SEEK_END)
                if at != stretch.bytes or at % self._pad:
                    raise PersistenceError(
                        f"{stretch.path} is {at} bytes and was left at {stretch.bytes}; "
                        "something else has written to it"
                    )
                f.write(framed)
                f.flush()
                os.fsync(f.fileno())
        except (OSError, PersistenceError) as e:
            stretch.closed = f"an append failed: {e}"
            if self._open == stretch.id:
                self._open = None
            self.bytes = _tree_bytes(self.directory)
            if isinstance(e, PersistenceError):
                raise
            raise PersistenceError(f"appending to {stretch.path} failed: {e}") from e
        stretch.starts.append(at)
        stretch.bytes += len(framed)
        self.bytes += len(framed)

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
        remember, and how it was found. False when the record was not filed:
        there was no room for it, or the end of the memory's file is torn."""
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
        if stretch.closed is not None:
            # The end of its file is torn. What is whole in it can be looked
            # at; nothing is added to it. Counted, and said at opening.
            self.not_filed_closed += 1
            return False
        seq = stretch.next_seq
        record = _pack({
            "format": FORMAT, "kind": "looked_back", "id": memory_id, "seq": seq, "cycle": int(cycle),
            "when": _now(), "filed": self._filed + 1, "signals": noted,
        }, {"summary": _bits(vector)})
        if not self._room_for(_framed(len(record), self._pad), cycle=cycle, in_view={memory_id, *in_view}):
            self.not_filed_for_room += 1
            return False
        self._append(stretch, record)
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
        os.rename(stretch.path, self._waiting_dir / stretch.path.name)
        stretch.path = self._waiting_dir / stretch.path.name
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
        wide = vector.to(torch.float64)
        if self._ref_mean is not None:
            if self._row_len.shape[0] <= self._rows:
                for name in ("_row_len", "_row_far"):
                    grown = torch.empty(self._matrix.shape[0], dtype=torch.float64)
                    grown[: self._rows] = getattr(self, name)[: self._rows]
                    setattr(self, name, grown)
            self._row_len[self._rows] = wide.norm()
            self._row_far[self._rows] = (wide - self._ref_mean).pow(2).sum()
        self._rows += 1
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
        self._ref_mean = None                    # the rows have moved; the screen's reference is made again

    def _refer(self) -> None:
        """Take the mean as it is now for the screen's reference, and each
        row's length and squared distance from it, in 8-byte arithmetic."""
        self._ref_mean = (self._sum / self._rows).clone()
        self._ref_base = self._rows
        self._row_len = torch.empty(self._matrix.shape[0], dtype=torch.float64)
        self._row_far = torch.empty(self._matrix.shape[0], dtype=torch.float64)
        for at in range(0, self._rows, _CHUNK):
            rows = self._matrix[at: min(at + _CHUNK, self._rows)].to(torch.float64)
            self._row_len[at: at + rows.shape[0]] = rows.norm(dim=1)
            self._row_far[at: at + rows.shape[0]] = (rows - self._ref_mean).pow(2).sum(dim=1)

    def _bounds(self, mean: torch.Tensor, centred: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """An upper and a lower bound on every row's likeness to the cue, from
        one pass over the matrix in 4-byte arithmetic. Bounds, never the
        answer: a row is only ever ruled out by them.

        With r a row, m the mean now, m0 the reference mean, c the cue less m
        and d = m - m0, the likeness is  (r.c - m.c) / (|r - m| |c|),  and
        |r - m|^2 = |r - m0|^2 - 2 (r.d - m0.d) + |d|^2.  The pass gives r.c
        and r.d in 4-byte arithmetic. A 4-byte dot of n terms is within
        n 2^-24 |r| |x| of the true one, in any order of summing, and rounding
        x to 4 bytes moves it by at most 2^-24 |r| |x| more. Everything else
        is 8-byte arithmetic, whose own rounding is allowed for by the last
        terms of each error below. Where a row cannot be told from the mean
        itself, its bounds are 1 and -1.

        Opus 2.2: each bound's formula is chosen by the sign of that bound's
        own numerator. Opus 2.6: the pass is on the CPU in plain 4-byte
        numbers; anything else would void the bound, so it is asserted."""
        rows = self._matrix[: self._rows]
        if rows.dtype != torch.float32 or rows.device.type != "cpu":
            raise PersistenceError(f"the screen needs 4-byte numbers on the CPU; got {rows.dtype} on {rows.device}")
        width = rows.shape[1]
        moved = mean - self._ref_mean
        pair = torch.stack([centred.to(torch.float32), moved.to(torch.float32)], dim=1)
        dots = (rows @ pair).to(torch.float64)
        length, far = self._row_len[: self._rows], self._row_far[: self._rows]
        cue_len, moved_len = float(centred.norm()), float(moved.norm())
        mean_len, ref_len = float(mean.norm()), float(self._ref_mean.norm())
        rounding = (width + 3) * _UNIT * 1.01
        tiny = 1e-12

        off = rounding * length * cue_len + tiny * (length + mean_len) * cue_len + 1e-300
        top = dots[:, 0] - float(mean @ centred) + off
        bottom = dots[:, 0] - float(mean @ centred) - off

        cross = dots[:, 1] - float(self._ref_mean @ moved)
        cross_off = rounding * length * moved_len + tiny * (length + ref_len) * moved_len
        far_off = tiny * (length + ref_len).pow(2)
        middle = far - 2 * cross + moved_len ** 2
        least = (middle - 2 * cross_off - far_off).clamp_min(0.0).sqrt()
        most = (middle + 2 * cross_off + far_off).clamp_min(0.0).sqrt()
        known = least > 0
        safe = least.clamp_min(1e-300) * cue_len
        slack = torch.where(known, tiny * (length + mean_len) / least.clamp_min(1e-300), torch.ones_like(least)) + tiny
        up = torch.where(top >= 0, top / safe, top / (most * cue_len).clamp_min(1e-300)) + slack
        low = torch.where(bottom >= 0, bottom / (most * cue_len).clamp_min(1e-300), bottom / safe) - slack
        if cue_len == 0.0:
            return torch.ones_like(up), -torch.ones_like(low)
        up = torch.where(known, up, torch.ones_like(up)).clamp(max=1.0)
        low = torch.where(known, low, -torch.ones_like(low)).clamp(min=-1.0)
        return up, low

    def _likeness(self, cue: torch.Tensor) -> torch.Tensor:
        """The cosine between the cue and every summary in reach, after the
        mean of those summaries is removed from both, in 8-byte arithmetic. A
        summary that is the mean itself is like nothing: 0."""
        mean = self._sum / self._rows
        centred = cue.to(torch.float64) - mean
        out = torch.empty(self._rows, dtype=torch.float64)
        for at in range(0, self._rows, _CHUNK):
            self._likeness_of_chunk(at, mean, centred, out)
        return out

    def _likeness_of_chunk(self, at: int, mean: torch.Tensor, centred: torch.Tensor, out: torch.Tensor) -> None:
        """The likeness of the rows from ``at`` to the end of their chunk,
        written into ``out``. The full search and the screened one both come
        here with the same chunks, so a row's value does not depend on which
        of them asked (Opus 2.4: a row's 8-byte product must not change with
        the rows computed beside it; here the rows beside it never change)."""
        rows = self._matrix[at: min(at + _CHUNK, self._rows)].to(torch.float64) - mean
        norms = rows.norm(dim=1) * centred.norm()
        out[at: at + rows.shape[0]] = torch.where(norms > 0, (rows @ centred) / norms, torch.zeros_like(norms))

    def _find(self, vector: torch.Tensor, able: torch.Tensor, ids: torch.Tensor, *, screened: bool):
        """The best row the search may return, its likeness, the best likeness
        among other memories (None when there is none), how many other
        memories it beat, and how many rows were computed exactly."""
        if not screened:
            likeness = self._likeness(vector).masked_fill(~able, float("-inf"))
            computed = self._rows
        else:
            if self._ref_mean is None or self._rows >= 2 * self._ref_base:
                self._refer()
            mean = self._sum / self._rows
            centred = vector.to(torch.float64) - mean
            up, low = self._bounds(mean, centred)
            likeness = torch.full((self._rows,), float("-inf"), dtype=torch.float64)
            done: set[int] = set()

            def compute(candidates: torch.Tensor) -> None:
                # Whole chunks, in row order, so that equal likenesses fall to
                # the lowest row as in the full search (Opus 2.5).
                for chunk in torch.unique(candidates // _CHUNK).tolist():
                    if chunk not in done:
                        done.add(chunk)
                        self._likeness_of_chunk(chunk * _CHUNK, mean, centred, likeness)

            # The bar is the best lower bound among the rows the search may
            # return (Opus 2.1): what is still the present is often nearer.
            bar = low.masked_fill(~able, float("-inf")).max()
            first = (able & (up >= bar)).nonzero().flatten()
            compute(first)
            likeness.masked_fill_(~able, float("-inf"))
            computed = int(first.numel())
        best = int(likeness.argmax())
        memory_id = int(ids[best])
        others = able & (ids != memory_id)
        beat = int(torch.unique(ids[others]).numel())
        if screened and beat:
            bar = low.masked_fill(~others, float("-inf")).max()
            second = (others & (up >= bar)).nonzero().flatten()
            compute(second)
            likeness.masked_fill_(~able, float("-inf"))
            computed += int(second.numel())
        runner_up = float(likeness[others].max()) if beat else None
        return best, float(likeness[best]), runner_up, beat, computed

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
        if self.screened:
            found = self._find(vector, able, ids, screened=True)
            self.screened_searches += 1
            self.rows_computed_exactly += found[4]
            self.most_candidates = max(self.most_candidates, found[4])
            if self.screened_searches % CHECK_EVERY == 0:
                full = self._find(vector, able, ids, screened=False)
                self.checked_searches += 1
                if full[:4] != found[:4]:
                    raise PersistenceError(
                        f"the screened search found row {found[0]} (likeness {found[1]!r}, next {found[2]!r}, "
                        f"beat {found[3]}) and the full search found row {full[0]} ({full[1]!r}, {full[2]!r}, "
                        f"{full[3]}). The screen's bounds do not hold; open the store with screened=False."
                    )
        else:
            found = self._find(vector, able, ids, screened=False)
            self.full_searches += 1
        best, likeness_of_best, runner_up, beat, _ = found
        memory_id = int(ids[best])
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
        return self.open(memory_id, int(self._row_seq[best]), likeness=likeness_of_best,
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
        with open(stretch.path, "rb") as f:
            for earlier in sorted(s for s in stretch.kept if s < seq):
                words += _tokens_of(_read_from(f, stretch.starts[earlier - 1], ("tokens",), stretch.path)[1])
            header, blobs = _read_from(f, stretch.starts[seq - 1], None, stretch.path)
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
        with open(stretch.path, "rb") as f:
            for at in stretch.starts:
                header, blobs = _read_from(f, at, ("summary",), stretch.path)
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
            "screened_searches": self.screened_searches, "full_searches": self.full_searches,
            "checked_searches": self.checked_searches, "rows_computed_exactly": self.rows_computed_exactly,
            "most_candidates": self.most_candidates,
            "not_filed_for_room": self.not_filed_for_room,
            "not_filed_closed": self.not_filed_closed,
            "cold_had_no_room": self.cold_had_no_room,
            "life_drive_had_no_room": self.life_drive_had_no_room,
            "stretches_restarted": self.stretches_restarted,
            "torn": list(self.torn),
            "empty_stretches": list(self.empty_stretches),
            "frame_bytes": self._pad,
        }


def dream_of(near: Memory, far: Memory) -> tuple[list[int], Optional[Sight], Optional[Sound]]:
    """A moment that never happened, from two that did: the first half of one's
    words and the second half of the other's, the picture of one and the sound
    of the other where both have them."""
    words = near.tokens[: max(1, len(near.tokens) // 2)] + far.tokens[len(far.tokens) // 2:]
    sight = near.sight if near.sight is not None else far.sight
    sound = far.sound if far.sound is not None else near.sound
    return words, sight, sound
