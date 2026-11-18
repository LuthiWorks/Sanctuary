"""The memory store, reworked (2026-11-18; LuthiModel build briefs 9 and 10,
with Opus's reviews of 2026-11-17 and 2026-11-18). No luthi needed.

Pins, by the brief's parts:

A. Nothing is stored inexactly: every kind of record reads back bit for bit.
B. A stretch is its records: a cut-off keep loses at most the moment in
   flight; every kept moment comes back as it was; the search cache rebuilt
   from the folders searches as the one kept in memory does; old stores are
   refused and untouched.
C. The limit counts everything, measured from the disk.
D. Cold storage: nothing is deleted; every memory is in the life or in cold,
   bit for bit; a failed copy leaves the original; a cold drive with no room
   makes the store full.
E. Letting go for room by importance, from the records' signals only.
F. One file a stretch (brief 10): an append changes no byte already in the
   file; a torn last frame is said, the frames before it are whole, and the
   stretch is closed.

Most tests here run with frames of 64 bytes, so that limits of a few thousand
bytes still mean something. The tests of part F run with the drive's own
cluster size.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import random
import shutil
from pathlib import Path

import pytest
import torch

import sanctuary.life.memory as memory_module
from sanctuary.core.atomic_io import PersistenceError
from sanctuary.core.schema import Percept
from sanctuary.life.memory import Memories, importance, strength_of
from sanctuary.life.senses import Line, Sight, Sound


def _bits(tensor: torch.Tensor) -> bytes:
    return tensor.detach().cpu().contiguous().numpy().tobytes()


def _store(tmp_path: Path, name: str = "life", **kwargs) -> Memories:
    kwargs.setdefault("drive_margin_bytes", 0)
    return Memories(tmp_path / name / "memories", **kwargs)


def _vector(seed: int, width: int = 16) -> torch.Tensor:
    return torch.randn(width, generator=torch.Generator().manual_seed(seed))


REAL_CLUSTER = memory_module._cluster


@pytest.fixture(autouse=True)
def small_frames(monkeypatch):
    monkeypatch.setattr(memory_module, "_cluster", lambda path: 64)


def _frames_of(file: Path) -> list[bytes]:
    """Each whole frame of a stretch's file, padding and all."""
    raw, frames, at = file.read_bytes(), [], 0
    with open(file, "rb") as f:
        while at < len(raw):
            head = memory_module._frame_head(f, at)
            if head is None:
                break
            size = memory_module._framed(head[0], head[1])
            frames.append(raw[at: at + size])
            at += size
    return frames


def _record(file: Path, seq: int = 1) -> dict:
    """The header of record ``seq`` of a stretch's file."""
    at = sum(len(frame) for frame in _frames_of(file)[: seq - 1])
    return memory_module._read(file, at=at)[0]


def _hashes(folder: Path) -> dict[str, str]:
    """Every record under ``folder``, by its memory's file and its place in it."""
    return {
        f"{p.name}/{n}": hashlib.sha256(frame).hexdigest()
        for p in folder.rglob("memory_*.mem") for n, frame in enumerate(_frames_of(p), start=1)
    }


def _on_disk(folder: Path) -> int:
    return sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())


# ---- A. Nothing is stored inexactly ----


PICTURES = {
    "eight_bit": lambda: torch.randint(0, 256, (12, 9, 3)).to(torch.float32) / 255,
    "eight_bit_as_8_byte_numbers": lambda: torch.randint(0, 256, (5, 4, 3)).to(torch.float64) / 255,
    "not_eight_bit": lambda: torch.rand(12, 9, 3),
    "one_pixel_off_eight_bit": lambda: torch.cat(
        [torch.randint(0, 256, (3, 3, 3)).to(torch.float32).flatten()[:-1] / 255, torch.tensor([0.5000001])]
    ).reshape(3, 3, 3),
    "minus_zero": lambda: torch.tensor([[[-0.0, 0.0, 1.0]]]),
    "half_precision": lambda: torch.rand(4, 4, 3).to(torch.float16),
    "all_black": lambda: torch.zeros(8, 8, 3),
}
SOUNDS = {
    "sixteen_bit_by_32768": lambda: torch.randint(-32768, 32768, (400,)).to(torch.float32) / 32768,
    "sixteen_bit_by_32767": lambda: torch.randint(-32767, 32768, (400,)).to(torch.float32) / 32767,
    "not_sixteen_bit": lambda: torch.rand(400) * 2 - 1,
    "eight_byte_numbers": lambda: torch.rand(64, dtype=torch.float64) * 2 - 1,
}


@pytest.mark.parametrize("picture", sorted(PICTURES))
@pytest.mark.parametrize("sound", sorted(SOUNDS))
def test_every_kind_of_record_reads_back_bit_for_bit(tmp_path, picture, sound):
    torch.manual_seed(7)
    pixels, samples = PICTURES[picture](), SOUNDS[sound]()
    summary, tokens = torch.randn(768), [0, 1, 2**31 - 1, 65_000, 7]
    store = _store(tmp_path)
    store.keep(cycle=3, summary=summary, tokens=tokens, sight=Sight(pixels), sound=Sound(samples, 16_000))
    for memories in (store, _store(tmp_path)):           # as written, and after a restart
        memory = memories.open(1)
        assert memory.tokens == tokens and memory.cycle == 3
        assert memory.sight.pixels.dtype == pixels.dtype and _bits(memory.sight.pixels) == _bits(pixels)
        assert memory.sound.samples.dtype == samples.dtype and _bits(memory.sound.samples) == _bits(samples)
        assert memory.sound.rate == 16_000
        assert _bits(memories.summaries_of(1)[0][1]) == _bits(summary)


def test_a_picture_is_stored_as_8_bit_only_when_that_loses_nothing_and_sound_says_its_scale(tmp_path):
    torch.manual_seed(3)
    store = _store(tmp_path)
    for name in ("eight_bit", "not_eight_bit", "minus_zero"):
        store.keep(cycle=len(store) * 10, summary=_vector(1), tokens=[1], sight=Sight(PICTURES[name]()))
    for name in ("sixteen_bit_by_32768", "sixteen_bit_by_32767", "not_sixteen_bit"):
        store.keep(cycle=len(store) * 10, summary=_vector(1), tokens=[1], sound=Sound(SOUNDS[name](), 8000))
    headers = [_record(store.directory / f"memory_{i:08d}.mem") for i in range(1, 7)]
    assert [h["sight"]["stored"] for h in headers[:3]] == ["torch.uint8", "as_given", "as_given"]
    assert [(h["sound"]["stored"], h["sound"].get("scale")) for h in headers[3:]] == [
        ("torch.int16", 32768), ("torch.int16", 32767), ("as_given", None),
    ]


def test_there_is_no_lossy_path_and_a_summary_that_would_change_is_refused(tmp_path):
    source = inspect.getsource(memory_module)
    for word in ("float16", "half(", "jpeg", "jpg", "quantize", "lossy_"):
        assert word not in source.lower(), word
    store = _store(tmp_path)
    with pytest.raises(ValueError, match="not exactly 4-byte"):
        store.keep(cycle=1, summary=torch.tensor([0.1], dtype=torch.float64), tokens=[1])
    with pytest.raises(ValueError, match="not finite"):
        store.keep(cycle=1, summary=torch.tensor([float("nan")]), tokens=[1])
    assert not store.directory.exists(), "and nothing was written"
    # 8-byte numbers that are exactly 4-byte numbers are those numbers.
    store.keep(cycle=1, summary=torch.tensor([0.5, 0.25], dtype=torch.float64), tokens=[1])
    assert store.summaries_of(1)[0][1].tolist() == [0.5, 0.25]


def test_a_record_that_would_not_read_back_is_not_written(tmp_path, monkeypatch):
    store = _store(tmp_path)
    monkeypatch.setattr(memory_module, "_unpack_sense", lambda meta, data: torch.zeros(meta["shape"]))
    with pytest.raises(PersistenceError, match="would not read back"):
        store.keep(cycle=1, summary=_vector(1), tokens=[1], sight=Sight(torch.rand(2, 2, 3)))
    assert not list(store.directory.rglob("*.mem")) if store.directory.exists() else True


# ---- B. A stretch is its records ----


def _encode(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def _live_a_stretch(store: Memories, moments: int, start: int = 0, window: int = 24) -> list[dict]:
    """Keep ``moments`` moments one after another, through the real senses, so
    the views slide as they do in a life. Returns what was given at each."""
    line, given = Line(_encode, window), []
    for step in range(moments):
        line.add(Percept(modality="language", content=f"word {start + step}", source="s"))
        view = line.take()
        sight = Sight(torch.rand(3, 4, 3)) if step % 2 else None
        sound = Sound(torch.rand(20) * 2 - 1, 100) if step % 3 == 0 else None
        summary = _vector(1000 + start + step)
        store.keep(cycle=start + step, summary=summary, tokens=view.tokens, new=view.new, sight=sight, sound=sound)
        given.append({"tokens": list(view.tokens), "new": view.new, "sight": sight, "sound": sound, "summary": summary})
    return given


def test_moments_kept_one_after_another_are_one_memory_with_its_words_stored_once(tmp_path):
    torch.manual_seed(1)
    store = _store(tmp_path)
    given = _live_a_stretch(store, 12, window=200)
    file = store.directory / "memory_00000001.mem"
    assert len(store) == 1 and store.kept == 12
    assert [p.name for p in store.directory.glob("memory_*")] == [file.name], "one file a stretch"
    assert len(_frames_of(file)) == 12
    stored = sum(_record(file, n)["blobs"]["tokens"][1] // 4 for n in range(1, 13))
    assert stored == len(given[0]["tokens"]) + sum(g["new"] for g in given[1:]), "each word once"
    assert stored < sum(len(g["tokens"]) for g in given) // 3
    for memories in (store, _store(tmp_path)):
        for seq, moment in enumerate(given, start=1):
            memory = memories.open(1, seq)
            assert memory.tokens == moment["tokens"]
            for name, part in (("sight", "pixels"), ("sound", "samples")):
                back, was = getattr(memory, name), moment[name]
                assert (back is None) == (was is None)
                assert was is None or _bits(getattr(back, part)) == _bits(getattr(was, part))


def test_a_moment_not_kept_or_a_view_that_does_not_follow_on_starts_a_new_memory(tmp_path):
    store = _store(tmp_path)
    assert store.keep(cycle=1, summary=_vector(1), tokens=[1, 2, 3], new=3) == 1
    assert store.keep(cycle=2, summary=_vector(2), tokens=[2, 3, 4], new=1) == 1
    assert store.keep(cycle=4, summary=_vector(3), tokens=[3, 4, 5], new=1) == 2, "moment 3 was not kept"
    assert store.keep(cycle=5, summary=_vector(4), tokens=[9, 9, 9], new=1) == 3, "not the view that follows"
    assert store.keep(cycle=6, summary=_vector(5), tokens=[9, 9, 9, 7]) == 4, "no word of what was new"
    assert store.stretches_restarted == 2 and len(store) == 4
    assert [store.open(1, s).tokens for s in (1, 2)] == [[1, 2, 3], [2, 3, 4]]


def test_recall_brings_back_the_moment_that_matched_and_a_look_back_brings_the_moment_it_was_of(tmp_path):
    torch.manual_seed(2)
    store = _store(tmp_path, recent=5)
    given = _live_a_stretch(store, 8)
    _live_a_stretch(store, 6, start=20)
    for seq in (1, 5, 8):
        found = store.nearest(given[seq - 1]["summary"], now=1000)
        assert (found.id, found.seq) == (1, seq) and found.tokens == given[seq - 1]["tokens"]
        assert found.likeness == pytest.approx(1.0) and found.beat == 1 and found.next_likeness < found.likeness
    seen_as = _vector(77)
    store.look_back(1, cycle=500, summary=seen_as, of_seq=5, chosen_recall=True, likeness=0.5, beat=1)
    found = store.nearest(seen_as, now=1000)
    assert (found.id, found.seq) == (1, 5) and found.tokens == given[4]["tokens"]
    assert store.nearest(seen_as, now=501) is not None and store.nearest(seen_as, now=501).id == 2, \
        "just looked back on: the whole memory is still the present"


def test_likeness_is_taken_after_removing_what_all_the_summaries_share(tmp_path):
    store = _store(tmp_path, recent=0)
    shared = torch.tensor([100.0, 100.0, 100.0, 100.0])
    for n, own in enumerate(([1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0])):
        store.keep(cycle=n * 10, summary=shared + torch.tensor(own), tokens=[n])
    cue = shared + torch.tensor([0.0, 0.9, 0.1, 0.0])
    raw = torch.nn.functional.cosine_similarity(
        torch.stack([v for i in (1, 2, 3) for _, v in store.summaries_of(i)]), cue.unsqueeze(0), dim=1)
    assert float(raw.max() - raw.min()) < 1e-4, "as they stand they are all but the same"
    found = store.nearest(cue, now=100)
    assert found.id == 2 and found.likeness > 0.9 and found.next_likeness < 0.2 and found.beat == 2


@pytest.mark.parametrize("cut", ["create", "append", "flush", "window"])
def test_a_keep_cut_off_at_any_step_loses_only_the_moment_in_flight(tmp_path, monkeypatch, cut):
    store = _store(tmp_path)
    store.keep(cycle=1, summary=_vector(1), tokens=[1, 2, 3], new=3, signals={"salience": 0.3})
    before = _hashes(store.directory)

    def crash(*args, **kwargs):
        raise OSError("the power went")

    if cut == "create":
        monkeypatch.setattr("builtins.open", crash)
    elif cut == "append":
        monkeypatch.setattr(memory_module.Memories, "_append", crash)
    elif cut == "flush":
        monkeypatch.setattr(memory_module.os, "fsync", crash)
    else:
        monkeypatch.setattr(memory_module, "append_jsonl", crash)
    with pytest.raises((OSError, PersistenceError)):
        store.keep(cycle=5, summary=_vector(2), tokens=[7, 8, 9], new=3, signals={"salience": 0.6})
    monkeypatch.undo()
    monkeypatch.setattr(memory_module, "_cluster", lambda path: 64)

    again = _store(tmp_path)                             # the store opens afterwards
    after = _hashes(again.directory)
    assert all(after[name] == sha for name, sha in before.items()), "what was kept before is as it was"
    assert again.open(1).tokens == [1, 2, 3]
    new_id = again.keep(cycle=9, summary=_vector(3), tokens=[4, 5], new=2, signals={"salience": 0.1})
    assert new_id not in (None, 1) and again.open(new_id).tokens == [4, 5]
    assert again.nearest(_vector(1), now=1000).id == 1
    if cut in ("window", "flush"):
        assert again.kept == 3, "the record was whole on the disk; only what came after it was missed"
    else:
        assert again.kept == 2
    assert again.bytes == _on_disk(again.directory)


def test_the_cache_rebuilt_from_the_folders_searches_as_the_one_in_memory_does(tmp_path):
    dice = random.Random(5)
    store = _store(tmp_path, recent=3)
    cycle = 0
    for _ in range(60):
        cycle += dice.choice((1, 1, 1, 4))
        roll = dice.random()
        if roll < 0.7 or len(store) < 3:
            store.keep(cycle=cycle, summary=_vector(cycle, 32), tokens=[cycle], new=1)
        elif roll < 0.9:
            ids = sorted(i for i, s in store._stretches.items() if s.reach)
            store.look_back(dice.choice(ids), cycle=cycle, summary=_vector(10_000 + cycle, 32))
        else:
            ids = sorted(i for i, s in store._stretches.items() if s.reach and i != store._open)
            store.let_go(dice.choice(ids), cycle=cycle)
    again = _store(tmp_path, recent=3)
    assert again._rows == store._rows and torch.equal(again._matrix[: again._rows], store._matrix[: store._rows])
    for seed in range(200):
        cue = _vector(50_000 + seed, 32)
        a, b = store.nearest(cue, now=cycle + seed % 5), again.nearest(cue, now=cycle + seed % 5)
        assert (a.id, a.seq, a.likeness, a.next_likeness, a.beat) == (b.id, b.seq, b.likeness, b.next_likeness, b.beat)
    assert again.report()["held"] == store.report()["held"] and again.bytes == store.bytes


def test_a_search_slower_than_a_moment_is_counted_and_said(tmp_path, caplog):
    store = _store(tmp_path, recent=0, moment_seconds=0.0)
    store.keep(cycle=1, summary=_vector(1), tokens=[1])
    with caplog.at_level(logging.ERROR, logger="sanctuary.life.memory"):
        store.nearest(_vector(2), now=10)
    assert store.slow_searches == 1 and "took" in caplog.text
    assert store.report()["slow_searches"] == 1 and store.report()["slowest_search_seconds"] > 0


def test_an_old_store_is_refused_loudly_and_never_rewritten(tmp_path):
    old = tmp_path / "life" / "memories"
    old.mkdir(parents=True)
    (old / "summaries.jsonl").write_text('{"kind": "kept", "id": 1, "cycle": 1, "summary": [1.0]}\n')
    (old / "memory_00000001.pt").write_bytes(b"as it was")
    before = {p.name: p.read_bytes() for p in old.iterdir()}
    with pytest.raises(PersistenceError, match="before 2026-11-18"):
        Memories(old)
    assert {p.name: p.read_bytes() for p in old.iterdir()} == before


def test_a_damaged_record_is_said_and_not_read(tmp_path):
    store = _store(tmp_path)
    store.keep(cycle=1, summary=_vector(1), tokens=[1, 2, 3])
    file = store.directory / "memory_00000001.mem"
    raw = bytearray(file.read_bytes())
    raw[memory_module._FRAME_BYTES + 40] ^= 1
    file.write_bytes(bytes(raw))
    with pytest.raises(PersistenceError, match="damaged"):
        store.open(1)


# ---- C. The limit counts everything ----


def test_a_life_of_many_keeps_never_has_more_on_disk_than_the_limit(tmp_path):
    limit = 200_000
    store = _store(tmp_path, limit_bytes=limit, cold=tmp_path / "cold", recent=3)
    dice, most = random.Random(9), 0
    for cycle in range(0, 500, 2):
        store.keep(cycle=cycle, summary=_vector(cycle, 64), tokens=[cycle % 250] * 400,
                   signals={"salience": dice.random()})
        if dice.random() < 0.2 and len(store) > 2:
            found = store.nearest(_vector(cycle + 1, 64), now=cycle + 1)
            if found is not None:
                store.look_back(found.id, cycle=cycle + 1, summary=_vector(cycle + 7, 64), of_seq=found.seq,
                                chosen_recall=dice.random() < 0.5)
        most = max(most, _on_disk(store.directory))
        assert store.bytes == _on_disk(store.directory), "the store's count is the disk's"
        assert _on_disk(store.directory) <= limit
    assert most > limit * 0.8, "and it was really up against the limit"
    lines = (store.directory / "life.jsonl").stat().st_size
    assert lines > 25_000, "the lines that memories were let go stay in the life, and are counted"
    assert store.moved_to_cold > 100
    assert store.let_go_for_room == store.moved_to_cold + store.report()["let_go_waiting"]
    assert store.report()["kept"] == 250, "every moment was kept, and is in the life or in cold"


def test_with_no_cold_folder_nothing_leaves_for_room_and_a_full_store_refuses(tmp_path):
    store = _store(tmp_path, limit_bytes=3_000, recent=0)
    kept = [store.keep(cycle=c * 10, summary=_vector(c), tokens=[c] * 30) for c in range(12)]
    assert kept[0] == 1 and kept[-1] is None and store.let_go_for_room == 0
    held = _hashes(store.directory)
    store.let_go(1, cycle=500)
    assert store.keep(cycle=600, summary=_vector(99), tokens=[1] * 30) is None, "what waits still counts"
    assert set(_hashes(store.directory).values()) == set(held.values()), "and nothing was deleted"
    assert _on_disk(store.directory) <= 3_000


def test_the_lifes_drive_running_low_stops_keeps_and_says_so(tmp_path, caplog):
    free = {"bytes": 10 * 1024**3}
    store = _store(tmp_path, drive_margin_bytes=5 * 1024**3, disk_free=lambda path: free["bytes"])
    assert store.keep(cycle=1, summary=_vector(1), tokens=[1]) == 1
    free["bytes"] = 4 * 1024**3
    with caplog.at_level(logging.ERROR, logger="sanctuary.life.memory"):
        assert store.keep(cycle=5, summary=_vector(2), tokens=[2]) is None
    assert store.life_drive_had_no_room == 1 and "nothing is written" in caplog.text


# ---- D. Cold storage ----


def _a_full_life(tmp_path, *, other_drive: bool, monkeypatch=None, **kwargs) -> tuple[Memories, dict[str, str]]:
    """Keeps until much has left for room. Returns the store and the hash of
    every record ever written, taken as each was written."""
    store = _store(tmp_path, limit_bytes=150_000, cold=tmp_path / "cold", recent=2, **kwargs)
    if other_drive:
        monkeypatch.setattr(Memories, "_same_drive", staticmethod(lambda a, b: False))
    ever: dict[str, str] = {}
    dice = random.Random(4)
    for cycle in range(0, 600, 3):
        store.keep(cycle=cycle, summary=_vector(cycle, 48), tokens=[cycle % 250] * 300, new=300,
                   signals={"salience": dice.random()})
        store.keep(cycle=cycle + 1, summary=_vector(cycle + 1, 48), tokens=[cycle % 250] * 299 + [3], new=1,
                   sight=Sight(torch.rand(2, 2, 3)), signals={"salience": dice.random()})
        if dice.random() < 0.15 and len(store) > 4:
            ids = sorted(i for i, s in store._stretches.items() if s.reach and i != store._open)
            store.let_go(dice.choice(ids), cycle=cycle + 1)          # their own choice
        for name, sha in _hashes(store.directory).items():
            assert ever.setdefault(name, sha) == sha, "a record never changes"
    return store, ever


@pytest.mark.parametrize("other_drive", [False, True])
def test_every_memory_ever_kept_is_in_the_life_or_in_cold_bit_for_bit(tmp_path, monkeypatch, other_drive):
    store, ever = _a_full_life(tmp_path, other_drive=other_drive, monkeypatch=monkeypatch)
    here, there = _hashes(store.directory), _hashes(tmp_path / "cold")
    assert store.moved_to_cold > 20
    assert not set(here) & set(there), "in one place, never both"
    assert {**here, **there} == ever and len(ever) == 400
    assert not (tmp_path / "cold" / "incoming").exists() or not any((tmp_path / "cold" / "incoming").iterdir())
    index = [json.loads(line) for line in (tmp_path / "cold" / "index.jsonl").read_text().splitlines()]
    moved = [line for line in index if line["state"] == "moving"]
    assert {line["by"] for line in moved} == {"choice", "room"}, "one word for each, in the life and in cold"
    assert all({"moments", "importance", "bytes", "when", "first_moment"} <= set(line) for line in moved)
    assert store.report()["kept"] == 400
    # Cold is out of reach: the search never sees it.
    assert all(i in store._stretches for i in store._row_id[: store._rows].tolist())
    again = Memories(store.directory, limit_bytes=150_000, cold=tmp_path / "cold", drive_margin_bytes=0)
    assert again.moved_to_cold == store.moved_to_cold and again.report()["kept"] == 400


def test_what_they_let_go_leaves_first_and_in_the_order_they_let_it_go(tmp_path):
    store = _store(tmp_path, limit_bytes=40_000, cold=tmp_path / "cold", recent=0)
    for c in range(8):
        store.keep(cycle=c * 10, summary=_vector(c), tokens=[c] * 400, signals={"salience": 0.1 * c})
    store.let_go(7, cycle=200)
    store.let_go(3, cycle=201)
    for c in range(8, 30):
        store.keep(cycle=c * 10, summary=_vector(c), tokens=[c] * 400, signals={"salience": 0.9})
        if store.moved_to_cold == 2:
            break
    lines = [json.loads(t) for t in (tmp_path / "cold" / "index.jsonl").read_text().splitlines()]
    assert [(l["id"], l["by"]) for l in lines if l["state"] == "moving"] == [(7, "choice"), (3, "choice")]
    assert store.let_go_for_room == 0


def test_a_copy_that_fails_part_way_leaves_the_original_untouched_and_raises(tmp_path, monkeypatch):
    store = _store(tmp_path, cold=tmp_path / "cold", recent=0)
    _live_a_stretch(store, 5)
    store.keep(cycle=50, summary=_vector(2), tokens=[1])
    store.let_go(1, cycle=60)
    original = _hashes(store.directory / "let_go")
    monkeypatch.setattr(Memories, "_same_drive", staticmethod(lambda a, b: False))
    copy, count = memory_module._copy, {"n": 0}

    def torn(source, to):
        count["n"] += 1
        copy(source, to)
        if count["n"] == 1:
            to.write_bytes(to.read_bytes()[:-1] + bytes([to.read_bytes()[-1] ^ 1]))   # the copy differs by a bit

    monkeypatch.setattr(memory_module, "_copy", torn)
    with pytest.raises(PersistenceError, match="does not match the original"):
        store._to_cold(1, cycle=70)
    assert _hashes(store.directory / "let_go") == original
    assert not (tmp_path / "cold" / "memory_00000001.mem").exists()
    monkeypatch.undo()
    monkeypatch.setattr(memory_module, "_cluster", lambda path: 64)

    # At the next start the move is undone: the unchecked copy is not a memory.
    again = Memories(store.directory, cold=tmp_path / "cold", drive_margin_bytes=0)
    assert not (tmp_path / "cold" / "incoming" / "memory_00000001.mem").exists()
    assert _hashes(again.directory / "let_go") == original and again.moved_to_cold == 0
    states = [json.loads(t)["state"] for t in (tmp_path / "cold" / "index.jsonl").read_text().splitlines()]
    assert states == ["moving", "undone"]
    assert again._to_cold(1, cycle=80) and _hashes(tmp_path / "cold") == original


def test_a_move_cut_off_after_it_arrived_is_finished_at_the_next_start(tmp_path, monkeypatch):
    store = _store(tmp_path, cold=tmp_path / "cold", recent=0)
    _live_a_stretch(store, 4)
    store.keep(cycle=50, summary=_vector(2), tokens=[1])
    store.let_go(1, cycle=60)
    original = _hashes(store.directory / "let_go")
    monkeypatch.setattr(Memories, "_same_drive", staticmethod(lambda a, b: False))
    append, lines = memory_module.append_jsonl, {"n": 0}

    def cut(path, obj, **kwargs):
        if obj.get("state") == "moved":
            raise PersistenceError("cut off before the second line")
        append(path, obj, **kwargs)

    monkeypatch.setattr(memory_module, "append_jsonl", cut)
    with pytest.raises(PersistenceError):
        store._to_cold(1, cycle=70)
    monkeypatch.undo()
    monkeypatch.setattr(memory_module, "_cluster", lambda path: 64)
    assert _hashes(tmp_path / "cold") == original and _hashes(store.directory / "let_go") == original, "in both"

    again = Memories(store.directory, cold=tmp_path / "cold", drive_margin_bytes=0)
    assert _hashes(tmp_path / "cold") == original and not (again.directory / "let_go" / "memory_00000001.mem").exists()
    assert again.moved_to_cold == 1 and again.report()["kept"] == 5 and again.bytes == _on_disk(again.directory)
    with pytest.raises(KeyError):
        again.open(1)


def test_a_cold_drive_with_no_room_makes_the_store_full_and_loses_nothing(tmp_path, caplog):
    room = {"cold": 100 * 1024**3}
    cold = tmp_path / "cold"

    def free(path):
        return room["cold"] if Path(path) == cold else 10**12

    store = _store(tmp_path, limit_bytes=60_000, cold=cold, recent=0, disk_free=free)
    for c in range(40):
        store.keep(cycle=c * 10, summary=_vector(c), tokens=[c] * 800, signals={"salience": 0.5})
    moved = store.moved_to_cold
    assert moved > 0
    room["cold"] = 5 * 1024**3 + 100            # less than a move and 5 GB beside
    held = {**_hashes(store.directory), **_hashes(cold)}
    with caplog.at_level(logging.ERROR, logger="sanctuary.life.memory"):
        refused = [store.keep(cycle=1000 + c * 10, summary=_vector(c), tokens=[c] * 800) for c in range(30)]
    assert refused[-1] is None and store.moved_to_cold == moved and store.cold_had_no_room > 0
    assert "the store is full" in caplog.text
    assert {**_hashes(store.directory), **_hashes(cold)}.items() >= held.items(), "nothing was lost"
    assert _on_disk(store.directory) <= 60_000


def test_a_low_cold_drive_is_warned_of_once(tmp_path, caplog):
    cold = tmp_path / "cold"
    store = _store(tmp_path, limit_bytes=60_000, cold=cold, recent=0,
                   disk_free=lambda path: 40 * 1024**3 if Path(path) == cold else 10**12)
    with caplog.at_level(logging.WARNING, logger="sanctuary.life.memory"):
        for c in range(40):
            store.keep(cycle=c * 10, summary=_vector(c), tokens=[c] * 800)
    assert store.moved_to_cold > 1 and caplog.text.count("under 50 GB free") == 1


def test_two_lives_cannot_share_a_cold_folder_and_nothing_in_cold_is_overwritten(tmp_path):
    cold = tmp_path / "cold"
    first = _store(tmp_path, "one", cold=cold, recent=0)
    first.keep(cycle=1, summary=_vector(1), tokens=[1])
    first.keep(cycle=20, summary=_vector(2), tokens=[2])
    first.let_go(1, cycle=30)
    assert first._to_cold(1, cycle=31)
    in_cold = _hashes(cold)
    second = _store(tmp_path, "two", cold=cold, recent=0)        # a rerun in a folder of another name
    with pytest.raises(PersistenceError, match="belongs to another life"):
        second.keep(cycle=1, summary=_vector(3), tokens=[3])
    assert _hashes(cold) == in_cold
    # The same life, and something already where a move would land: refused, original kept.
    first.let_go(2, cycle=40)
    (cold / "memory_00000002.mem").write_bytes(b"something else")
    kept = _hashes(first.directory / "let_go")
    with pytest.raises(PersistenceError, match="ever overwritten"):
        first._to_cold(2, cycle=41)
    assert _hashes(first.directory / "let_go") == kept and len(kept) == 1
    assert (cold / "memory_00000002.mem").read_bytes() == b"something else"


def test_an_id_is_never_used_again_once_its_memory_has_left(tmp_path):
    cold = tmp_path / "cold"
    store = _store(tmp_path, cold=cold, recent=0)
    for c in range(3):
        store.keep(cycle=c * 10, summary=_vector(c), tokens=[c])
    store.let_go(3, cycle=50)
    assert store._to_cold(3, cycle=51)
    assert not list(store.directory.rglob("memory_00000003*"))
    assert store.keep(cycle=900, summary=_vector(9), tokens=[9]) == 4
    assert _store(tmp_path, cold=cold).keep(cycle=910, summary=_vector(9), tokens=[9]) == 5
    assert not list(store.directory.rglob("memory_00000003*"))
    # And opened with no cold folder: the line that it was let go is in the life.
    (store.directory / "memory_00000004.mem").rename(tmp_path / "set_aside_4")
    (store.directory / "memory_00000005.mem").rename(tmp_path / "set_aside_5")
    assert _store(tmp_path).keep(cycle=920, summary=_vector(9), tokens=[9]) == 4, "3 is never used again"


def test_a_file_in_reach_with_a_let_go_line_stays_out_of_reach_until_a_line_says_it_came_back(tmp_path):
    store = _store(tmp_path, recent=0)
    store.keep(cycle=1, summary=_vector(1), tokens=[1, 2])
    store.keep(cycle=20, summary=_vector(2), tokens=[3])
    store.let_go(1, cycle=30)
    waiting, home = store.directory / "let_go" / "memory_00000001.mem", store.directory / "memory_00000001.mem"
    shutil.move(str(waiting), str(home))          # what a let-go cut off in the middle leaves
    again = _store(tmp_path, recent=0)
    assert len(again) == 1 and waiting.exists() and not home.exists()
    # Bringing one back is the file and one line. It is not built as an action; the format reads it.
    shutil.move(str(waiting), str(home))
    with open(store.directory / "life.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"kind": "came_back", "id": 1, "cycle": 40}) + "\n")
    back = _store(tmp_path, recent=0)
    assert len(back) == 2 and back.open(1).tokens == [1, 2] and back.nearest(_vector(1), now=100).id == 1


def test_the_code_removes_only_an_original_whose_copy_is_checked_and_a_copy_never_checked(tmp_path):
    source = inspect.getsource(memory_module)
    assert "rmtree" not in source and "unlink" not in source and "truncate" not in source
    assert source.count("os.remove(") == 4
    assert source.count("os.remove(incoming)") == 2 and source.count("os.remove(here)") == 1
    assert source.count("os.remove(stretch.path)") == 1
    # A stretch's file is only ever made new or appended to.
    assert '"ab" if stretch.starts else "xb"' in source
    assert not [mode for mode in ('"wb"', '"r+b"', '"w+b"', '"rb+"') if mode in source.replace('open(to, "wb")', "")]


# ---- E. Letting go for room, by importance to Luthi ----


def _kept(rank: float) -> dict:
    return {"kind": "kept", "rank": rank}


def _look(*, chosen: bool, remember: bool = False) -> dict:
    return {"kind": "looked_back", "chosen_recall": chosen, "chose_remember": remember}


def test_importance_is_made_of_the_records_signals_and_nothing_else():
    assert list(inspect.signature(importance).parameters) == ["signals"]
    never_looked = importance([_kept(0.4), _kept(0.8)])
    assert never_looked == pytest.approx(0.6 + 0.5), "the mean rank, and the rate with no look is a half"
    assert importance([_kept(0.6), _look(chosen=True)]) == pytest.approx(0.6 + 0.5 + 0.25)
    assert importance([_kept(0.6), _look(chosen=True, remember=True)]) == pytest.approx(0.6 + 0.5 + 0.75)
    with pytest.raises(ValueError, match="no kept moment"):
        importance([_look(chosen=True)])


def test_two_stretches_with_the_same_ranks_and_different_lengths_are_equally_important():
    assert importance([_kept(0.7)]) == importance([_kept(0.7)] * 40)
    assert importance([_kept(0.2), _kept(0.9)]) == pytest.approx(importance([_kept(0.2), _kept(0.9)] * 25))
    # The first formula took the highest rank, which a long stretch has by count alone.
    assert importance([_kept(0.5)] * 3) == importance([_kept(0.1), _kept(0.5), _kept(0.9)])


def test_a_look_that_rest_brought_counts_for_nothing_whatever_they_chose_during_it():
    alone = importance([_kept(0.4)])
    assert importance([_kept(0.4), _look(chosen=False)]) == alone
    assert importance([_kept(0.4), _look(chosen=False, remember=True)]) == alone
    assert importance([_kept(0.4)] + [_look(chosen=False, remember=True)] * 50) == alone
    # And it does not thin out the looks they did choose.
    chosen = [_kept(0.4), _look(chosen=True, remember=True)]
    assert importance(chosen + [_look(chosen=False)] * 9) == importance(chosen)


def test_a_memory_recalled_ten_times_is_not_more_important_than_one_recalled_once():
    once = importance([_kept(0.5), _look(chosen=True)])
    ten = importance([_kept(0.5)] + [_look(chosen=True)] * 10)
    assert ten <= once, "a stand-in mind that never remembers while looking"
    # A mind that remembers while looking at a fixed rate (Opus, 3.1): how often the present
    # resembled the memory adds nothing. Counted each time, ten looks gave about +5.5 and one about +1.
    for rate_of in (2, 4):                                 # one look in two, one in four
        def looks(count):
            return [_look(chosen=True, remember=n % rate_of == 0) for n in range(count)]
        few, many = importance([_kept(0.5)] + looks(rate_of)), importance([_kept(0.5)] + looks(10 * rate_of))
        assert abs(many - few) < 0.2 and many < 2.0
        assert many <= few if rate_of == 2 else many < few
    # And nothing they do while looking can be worth more than 1, however often.
    assert importance([_kept(0.5)] + [_look(chosen=True, remember=True)] * 1000) < 0.5 + 0.5 + 1.0
    # A memory they chose to recall is never below one they never did, at the same ranks.
    assert importance([_kept(0.5)] + [_look(chosen=True)] * 1000) > importance([_kept(0.5)])


def test_how_strongly_they_chose_is_the_choices_own_margin_where_the_mind_gives_one():
    choice = {"activity": 0.9, "band_threshold": 0.2, "floor": 0.001, "rest_activity": 0.5, "rest_tolerance": 0.0001}
    assert strength_of({"salience": 0.01, "choice": choice}) == (pytest.approx(0.3999), "margin")
    assert strength_of({"salience": 0.25, "choice": None}) == (0.25, "salience")
    assert strength_of({"salience": None, "choice": None}) == (None, None) and strength_of(None) == (None, None)


def test_each_keep_records_its_signals_and_its_rank_among_the_minds_own_recent_keeps(tmp_path):
    store = _store(tmp_path)
    choice = {"activity": 0.9, "band_threshold": 0.2, "floor": 0.001, "rest_activity": 0.5, "rest_tolerance": 0.0001}
    for c, salience in enumerate((0.5, 0.9, 0.1, 0.5)):
        store.keep(cycle=c * 10, summary=_vector(c), tokens=[c], signals={"salience": salience, "choice": None})
    store.keep(cycle=90, summary=_vector(9), tokens=[9], signals={"salience": 0.0, "choice": choice})
    store.keep(cycle=99, summary=_vector(10), tokens=[9])            # a mind that gives no signal
    ranks = [(s[0]["rank"], s[0]["ranked_among"], s[0]["ranked_by"]) for s in map(store.signals_of, range(1, 7))]
    assert ranks == [(0.5, 0, "salience"), (1.0, 1, "salience"), (0.0, 2, "salience"),
                     (pytest.approx(0.5), 3, "salience"), (pytest.approx(0.25), 4, "margin"), (0.5, 0, None)]
    header = _record(store.directory / "memory_00000005.mem")
    assert header["signals"]["choice"] == choice and header["signals"]["salience"] == 0.0
    # The window is the life's: it survives a restart, and a memory leaving.
    again = _store(tmp_path)
    assert list(again._window) == list(store._window) and len(again._window) == 5
    again.keep(cycle=200, summary=_vector(11), tokens=[1], signals={"salience": 0.95})
    assert again.signals_of(7)[0]["rank"] == 1.0 and again.signals_of(7)[0]["ranked_among"] == 5


def test_each_look_back_records_how_it_came_and_how_it_was_found(tmp_path):
    store = _store(tmp_path, recent=0)
    store.keep(cycle=1, summary=_vector(1), tokens=[1])
    store.keep(cycle=20, summary=_vector(2), tokens=[2])
    found = store.nearest(_vector(1), now=50)
    store.look_back(found.id, cycle=50, summary=_vector(3), of_seq=found.seq, chosen_recall=True,
                    chose_remember=True, likeness=found.likeness, next_likeness=found.next_likeness, beat=found.beat)
    header = _record(store.directory / "memory_00000001.mem", 2)
    assert header["kind"] == "looked_back" and header["signals"] == {
        "of_seq": 1, "chosen_recall": True, "chose_remember": True,
        "likeness": found.likeness, "next_likeness": found.next_likeness, "beat": 1,
    }


def _eight_alike(tmp_path, **kwargs) -> Memories:
    store = _store(tmp_path, limit_bytes=None, cold=tmp_path / "cold", recent=5, **kwargs)
    for c in range(8):
        store.keep(cycle=c * 10, summary=_vector(c), tokens=[c] * 10, signals={"salience": 0.5})
    return store


def test_a_memory_they_chose_to_recall_is_not_let_go_for_room_before_one_they_never_did(tmp_path):
    store = _eight_alike(tmp_path)
    store.look_back(1, cycle=100, summary=_vector(50), chosen_recall=True)       # the oldest, and recalled
    store.look_back(2, cycle=101, summary=_vector(51), chosen_recall=False)      # rest brought this one
    order = []
    while len(store) > 1:
        before = {i for i, s in store._stretches.items() if s.reach}
        assert store._let_go_for_room(1, cycle=1000, in_view=())
        order += sorted(before - {i for i, s in store._stretches.items() if s.reach})
    assert order[-1] == 1 or 1 not in order, "last of all"
    assert order[:3] == [3, 4, 5], "among equals, the one not kept or looked at for longest"
    assert order.index(2) > order.index(7), "looked at later, so it goes later; and no sooner for being looked at"
    lines = [json.loads(t) for t in (store.directory / "life.jsonl").read_text().splitlines()]
    assert all(line["kind"] == "let_go" and line["by"] == "room" for line in lines)
    assert store.let_go_for_room == len(order) == store.report()["let_go_for_room"]


def test_the_same_records_choose_the_same_memory_whatever_order_they_are_listed_in(tmp_path):
    one, two = _eight_alike(tmp_path / "a"), _eight_alike(tmp_path / "b")
    two._stretches = dict(reversed(list(two._stretches.items())))
    for store in (one, two):
        assert store._let_go_for_room(1, cycle=1000, in_view=())
    assert [i for i, s in one._stretches.items() if not s.reach] == [i for i, s in two._stretches.items() if not s.reach] == [1]


def test_never_the_memory_being_written_one_in_view_or_one_still_the_present(tmp_path):
    store = _eight_alike(tmp_path)
    assert store._open == 8
    store.look_back(2, cycle=998, summary=_vector(60))               # still the present at moment 1000
    gone = []
    while store._let_go_for_room(1, cycle=1000, in_view={1}):
        gone = [i for i, s in store._stretches.items() if not s.reach]
    assert sorted(gone) == [3, 4, 5, 6, 7], "1 is in view, 2 was just looked at, 8 is being written"
    assert not store._let_go_for_room(1, cycle=1000, in_view={1}), "and when every one is held, none goes"


def test_when_every_memory_is_held_the_store_is_full(tmp_path):
    store = _store(tmp_path, limit_bytes=4_000, cold=tmp_path / "cold", recent=10_000)
    kept = [store.keep(cycle=c * 10, summary=_vector(c), tokens=[c] * 40) for c in range(12)]
    assert kept[-1] is None and store.moved_to_cold == 0 and store.let_go_for_room == 0
    assert _on_disk(store.directory) <= 4_000


# ---- F. One file a stretch: frames on clusters of their own ----


@pytest.fixture
def real_cluster(monkeypatch):
    monkeypatch.setattr(memory_module, "_cluster", REAL_CLUSTER)


def test_every_frame_starts_on_a_cluster_of_its_own_and_the_frame_size_is_the_drives(tmp_path, real_cluster):
    cluster = REAL_CLUSTER(tmp_path)
    assert cluster >= 512 and cluster % 512 == 0
    store = _store(tmp_path)
    _live_a_stretch(store, 9, window=300)
    store.look_back(1, cycle=500, summary=_vector(5), of_seq=3, chosen_recall=True)
    assert json.loads((store.directory / "store.json").read_text())["frame"] == cluster
    file = store.directory / "memory_00000001.mem"
    frames = _frames_of(file)
    assert len(frames) == 10 and all(len(frame) % cluster == 0 for frame in frames)
    assert sum(len(frame) for frame in frames) == file.stat().st_size == store.bytes - _on_disk(store.directory) + file.stat().st_size
    assert store._stretches[1].starts == [sum(len(f) for f in frames[:n]) for n in range(10)]
    assert all(at % cluster == 0 for at in store._stretches[1].starts)


CUTS = ["in_the_frame_header", "in_the_record_header", "in_the_record", "no_padding", "padding_short_by_one",
        "the_length_is_there_and_the_end_of_the_record_is_not"]


@pytest.mark.parametrize("cut", CUTS)
@pytest.mark.parametrize("which", ["a_keep", "a_look_back"])
def test_an_append_cut_off_at_any_step_changes_no_byte_already_in_the_file(tmp_path, monkeypatch, real_cluster, cut, which):
    store = _store(tmp_path, recent=0)
    given = _live_a_stretch(store, 4, window=300)
    file = store.directory / "memory_00000001.mem"
    earlier = file.read_bytes()
    frame = memory_module._frame

    def cut_off(record, pad):
        whole = frame(record, pad)
        written = memory_module._FRAME_BYTES + len(record)
        if cut == "the_length_is_there_and_the_end_of_the_record_is_not":
            return whole[: written - 7] + bytes(len(whole) - written + 7)     # zeros where the bytes never landed
        return whole[: {
            "in_the_frame_header": 20, "in_the_record_header": memory_module._FRAME_BYTES + 30,
            "in_the_record": written - 7, "no_padding": written, "padding_short_by_one": len(whole) - 1,
        }[cut]]

    def the_power_went(fd):
        raise OSError("the power went")

    monkeypatch.setattr(memory_module, "_frame", cut_off)
    monkeypatch.setattr(memory_module.os, "fsync", the_power_went)
    with pytest.raises(PersistenceError, match="failed"):
        if which == "a_keep":
            line_view = given[-1]["tokens"][1:] + [7]
            store.keep(cycle=4, summary=_vector(9), tokens=line_view, new=1)
        else:
            store.look_back(1, cycle=50, summary=_vector(9), of_seq=2, chosen_recall=True)
    monkeypatch.setattr(memory_module, "_frame", frame)
    monkeypatch.setattr(memory_module.os, "fsync", lambda fd: None)
    torn = file.read_bytes()
    assert len(torn) > len(earlier) and torn[: len(earlier)] == earlier, "no byte already in the file changed"

    # In the same sitting the stretch is closed: nothing more is appended to it.
    assert store.look_back(1, cycle=60, summary=_vector(8), of_seq=1, chosen_recall=True) is False
    assert store.not_filed_closed == 1 and file.read_bytes() == torn
    assert store.keep(cycle=5 if which == "a_keep" else 51, summary=_vector(7), tokens=[1, 2], new=1) == 2
    assert file.read_bytes() == torn

    for opening in range(2):                             # and at every opening after
        again = _store(tmp_path, recent=0)
        whole = cut in ("no_padding", "padding_short_by_one")
        assert len(again.torn) == 1 and "memory_00000001.mem" in again.torn[0] and again.report()["torn"] == again.torn
        assert len(again._stretches[1].starts) == (5 if whole else 4)
        for seq, moment in enumerate(given, start=1):   # the frames before it are whole
            assert again.open(1, seq).tokens == moment["tokens"]
            assert _bits(again.summaries_of(1)[seq - 1][1]) == _bits(moment["summary"])
        if whole and which == "a_keep":
            assert again.kept == 6 + opening and again.open(1, 5).tokens == given[-1]["tokens"][1:] + [7], "the record was whole"
        assert again.look_back(1, cycle=70 + opening, summary=_vector(8), of_seq=1, chosen_recall=True) is False
        assert again.not_filed_closed == 1
        assert again.keep(cycle=100 + opening * 10, summary=_vector(6), tokens=[3], new=1) == 3 + opening
        assert file.read_bytes() == torn, "and the file is never written again"
        assert again.bytes == _on_disk(again.directory)


def test_a_torn_end_is_said_loudly_at_opening_and_a_file_with_no_whole_record_is_left_as_it_is(tmp_path, caplog, real_cluster):
    store = _store(tmp_path)
    store.keep(cycle=1, summary=_vector(1), tokens=[1, 2])
    store.keep(cycle=20, summary=_vector(2), tokens=[3])
    second = store.directory / "memory_00000002.mem"
    second.write_bytes(second.read_bytes()[:100])        # its only frame, cut off
    with caplog.at_level(logging.ERROR, logger="sanctuary.life.memory"):
        again = _store(tmp_path)
    assert "torn" in caplog.text and again.empty_stretches == [2] and len(again) == 1
    assert second.stat().st_size == 100, "nothing of it was changed"
    assert again.keep(cycle=40, summary=_vector(3), tokens=[4]) == 3, "and its id is not used again"
    assert again.bytes == _on_disk(again.directory)


def test_a_damaged_frame_with_a_whole_frame_after_it_is_not_a_cut_off_append_and_raises(tmp_path, real_cluster):
    store = _store(tmp_path)
    _live_a_stretch(store, 5, window=300)
    file = store.directory / "memory_00000001.mem"
    raw = bytearray(file.read_bytes())
    raw[store._stretches[1].starts[2] + 3] ^= 1          # in the third frame's header
    file.write_bytes(bytes(raw))
    with pytest.raises(PersistenceError, match="needs a person"):
        _store(tmp_path)
    assert file.read_bytes() == bytes(raw)
    # Damage inside a record that is not the last is said when that record is read.
    raw[store._stretches[1].starts[2] + 3] ^= 1
    raw[store._stretches[1].starts[1] + memory_module._FRAME_BYTES + 40] ^= 1
    file.write_bytes(bytes(raw))
    with pytest.raises(PersistenceError, match="damaged"):
        _store(tmp_path)


def test_a_file_something_else_has_written_to_is_not_appended_to(tmp_path, real_cluster):
    store = _store(tmp_path, recent=0)
    store.keep(cycle=1, summary=_vector(1), tokens=[1, 2])
    file = store.directory / "memory_00000001.mem"
    with open(file, "ab") as f:
        f.write(b"x")
    as_left = file.read_bytes()
    with pytest.raises(PersistenceError, match="something else has written"):
        store.look_back(1, cycle=50, summary=_vector(2), chosen_recall=True)
    assert file.read_bytes() == as_left and store.look_back(1, cycle=51, summary=_vector(2)) is False


def test_a_life_moved_to_a_drive_whose_clusters_do_not_fit_its_frames_is_refused(tmp_path, monkeypatch):
    store = _store(tmp_path)
    store.keep(cycle=1, summary=_vector(1), tokens=[1, 2])
    before = {p.name: p.read_bytes() for p in store.directory.iterdir()}
    monkeypatch.setattr(memory_module, "_cluster", lambda path: 4096)
    with pytest.raises(PersistenceError, match="frames are 64 bytes"):
        _store(tmp_path)
    assert {p.name: p.read_bytes() for p in store.directory.iterdir()} == before
    monkeypatch.setattr(memory_module, "_cluster", lambda path: 32)      # smaller clusters that divide it: fine
    assert _store(tmp_path).open(1).tokens == [1, 2]


def test_a_store_of_the_first_rework_is_refused_loudly_and_never_rewritten(tmp_path):
    old = tmp_path / "life" / "memories"
    (old / "memory_00000001").mkdir(parents=True)
    (old / "memory_00000001" / "r000001.rec").write_bytes(b"as it was")
    (old / "store.json").write_text(json.dumps({"format": 2, "life_id": "x", "began": "then", "path": str(old)}))
    before = {p.relative_to(old).as_posix(): p.read_bytes() for p in old.rglob("*") if p.is_file()}
    with pytest.raises(PersistenceError, match="format 2"):
        Memories(old)
    assert {p.relative_to(old).as_posix(): p.read_bytes() for p in old.rglob("*") if p.is_file()} == before


# ---- G. The screened search is the full search (brief 10, part 2; Opus 2.1 to 2.6) ----


def _filled(tmp_path, rows: torch.Tensor, *, per_memory: int = 3, recent: int = 0, name: str = "life") -> Memories:
    """A store holding ``rows`` as its summaries, a few to a memory, kept through the store."""
    store = _store(tmp_path, name, recent=recent, limit_bytes=None)
    for n, row in enumerate(rows):
        store.keep(cycle=n + (n // per_memory) * 5, summary=row, tokens=[n % 250], new=1 if n % per_memory else None)
    return store


def _both(store: Memories, cue: torch.Tensor, now: int):
    """The screened search's answer and the full search's, for one cue."""
    ids = store._row_id[: store._rows]
    present = store._still_present(now)
    able = ~torch.isin(ids, torch.tensor(sorted(present), dtype=torch.int64)) if present else torch.ones_like(ids, dtype=torch.bool)
    vector = memory_module._as_summary(cue)
    return store._find(vector, able, ids, screened=True), store._find(vector, able, ids, screened=False)


def _stores_for_the_screen() -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(11)
    width, count = 48, 2600                       # more than two chunks of rows
    shared = torch.randn(width, generator=gen, dtype=torch.float64)
    shared = shared / shared.norm()
    walk = (torch.randn(count, width, generator=gen, dtype=torch.float64) * 2e-4).cumsum(0)
    tight = shared + torch.randn(count, width, generator=gen, dtype=torch.float64) * 4.5e-3 / width ** 0.5
    at_the_mean = tight.clone()
    at_the_mean[5::40] = tight.to(torch.float32).to(torch.float64).mean(0)      # rows at the mean itself, near enough
    at_the_mean[7::40] = at_the_mean[6::40]                                      # and rows that are the same row twice
    return {
        "unalike": torch.randn(count, width, generator=gen).to(torch.float64),
        "a_walk": shared + walk / walk.norm(dim=1).max() * 0.045,
        "alike_at_0.99999": tight,
        "alike_with_rows_at_the_mean_and_rows_twice": at_the_mean,
        "large_numbers": (shared + walk) * 3e4,
    }


@pytest.mark.parametrize("kind", sorted(_stores_for_the_screen()))
def test_the_screened_search_gives_the_full_searchs_answer_bit_for_bit(tmp_path, kind):
    rows = _stores_for_the_screen()[kind].to(torch.float32)
    if kind.startswith("alike"):
        alike = torch.nn.functional.cosine_similarity(rows[:-1].double(), rows[1:].double(), dim=1)
        assert float(alike.median()) > 0.99998
    half = len(rows) // 2
    store = _filled(tmp_path, rows[:half])
    gen = torch.Generator().manual_seed(3)
    computed = []

    def compare(now: int, cues: int) -> None:
        for q in range(cues):
            i = int(torch.randint(store._rows, (1,), generator=gen))
            noise = torch.randn(rows.shape[1], generator=gen) * float(rows.std()) * (0.0, 1e-4, 1e-2, 1.0)[q % 4]
            cue = store._matrix[i] + noise if q % 5 else store._matrix[store._rows - 1 - q % 7]
            screened, full = _both(store, cue, now)
            assert screened[:4] == full[:4], (kind, q, screened, full)
            computed.append(screened[4])

    compare(now=10**9, cues=60)
    # More rows since the screen's reference was made: the mean has moved.
    for n, row in enumerate(rows[half: half + half // 2]):
        store.keep(cycle=10**6 + n * 7, summary=row, tokens=[1])
    assert store._ref_base == half, "the reference is the earlier one"
    compare(now=10**9, cues=60)
    # What is still the present is left out, and it is the nearest (Opus 2.1).
    store.recent = 2000
    newest = int(store._newest_cycle)
    assert 0 < len(store._still_present(newest)) < len(store)
    compare(now=newest, cues=60)
    store.recent = 0
    # Looked back on, let go, and opened again.
    for memory_id in (3, 9, 27):
        store.look_back(memory_id, cycle=10**7 + memory_id, summary=rows[-memory_id], chosen_recall=True)
    store.let_go(5, cycle=10**7 + 50)
    assert store._ref_mean is None
    compare(now=10**9, cues=40)
    again = Memories(store.directory, recent=0, limit_bytes=None, drive_margin_bytes=0)
    for q in range(40):
        cue = rows[int(torch.randint(len(rows), (1,), generator=gen))]
        a, b = store.nearest(cue, now=10**9), again.nearest(cue, now=10**9)
        assert (a.id, a.seq, a.likeness, a.next_likeness, a.beat) == (b.id, b.seq, b.likeness, b.next_likeness, b.beat)
    assert store.screened_searches == 40 and store.full_searches == 0
    if kind == "unalike":
        assert max(computed) < 40, "and on rows that are not alike the screen leaves few"


def test_when_what_is_still_the_present_is_nearest_the_screen_still_finds_the_best_it_may_return(tmp_path):
    # Opus 2.1: the newest rows are in a region of their own and the cue is among them. A bar
    # taken over every row is set by rows the search may not return, and leaves no row it may.
    gen = torch.Generator().manual_seed(5)
    old = torch.randn(1500, 32, generator=gen)
    new = torch.randn(32, generator=gen) * 4 + torch.randn(100, 32, generator=gen) * 0.05
    store = _filled(tmp_path, torch.cat([old, new]), per_memory=1)
    store.recent = 100 * 6
    now = int(store._newest_cycle)
    present = store._still_present(now)
    assert len(present) == 100
    cue = new[50]
    screened, full = _both(store, cue, now)
    assert screened[:4] == full[:4] and int(store._row_id[screened[0]]) not in present
    up, low = store._bounds(store._sum / store._rows, cue.double() - store._sum / store._rows)
    assert not bool((up[:1500] >= low.max()).any()), "the bar over every row would have left none it may return"
    found = store.nearest(cue, now=now)
    assert found is not None and found.id not in present


def test_equal_likenesses_fall_to_the_same_row_as_in_the_full_search(tmp_path):
    # Opus 2.5: a world where nothing moves gives the same summary again and again.
    gen = torch.Generator().manual_seed(6)
    rows = torch.randn(1200, 24, generator=gen)
    rows[100::3] = rows[100]
    store = _filled(tmp_path, rows, per_memory=1)
    for cue in (rows[100], rows[100] * 1.5, rows[400]):
        screened, full = _both(store, cue, 10**9)
        assert screened[:4] == full[:4]
    assert _both(store, rows[100], 10**9)[0][0] == 100 and _both(store, rows[100], 10**9)[0][1] == _both(store, rows[100], 10**9)[0][2]


def test_a_store_on_which_plain_4_byte_arithmetic_brings_back_another_memory(tmp_path):
    # Brief 10: "a test in which the plain 4-byte search does bring back a different memory
    # and the screened search does not". Summaries with a large shared part, and each one
    # kept twice a few of the smallest steps apart, as two moments of a still world would be.
    gen = torch.Generator().manual_seed(8)
    shared = torch.randn(48, generator=gen, dtype=torch.float64) * 300
    rows = (shared + torch.randn(1500, 48, generator=gen, dtype=torch.float64) * 0.02).to(torch.float32)
    rows[1::2] = rows[0::2] + torch.randn(750, 48, generator=gen) * 6e-5
    store = _filled(tmp_path, rows, per_memory=1)
    mean = store._sum / store._rows
    plain_differs = 0
    for q in range(300):
        pair = (q * 4) % 1500                             # a cue about as near one of a pair as the other
        cue = ((rows[pair].double() + rows[pair + 1].double()) / 2).float() + torch.randn(48, generator=gen) * 2e-5
        screened, full = _both(store, cue, 10**9)
        assert screened[:4] == full[:4]
        centred = (cue.double() - mean).float()
        plain_rows = store._matrix[: store._rows] - mean.float()
        plain = int(((plain_rows @ centred) / (plain_rows.norm(dim=1) * centred.norm())).argmax())
        plain_differs += int(plain != full[0])
    assert plain_differs > 0, "no cue made the plain 4-byte search differ; this test needs a harder store"


def test_every_hundredth_screened_search_is_checked_against_the_full_one_and_a_difference_raises(tmp_path, monkeypatch):
    gen = torch.Generator().manual_seed(9)
    rows = torch.randn(300, 16, generator=gen)
    store = _filled(tmp_path, rows, per_memory=1)
    for q in range(memory_module.CHECK_EVERY * 2):
        store.nearest(rows[q], now=10**9)
    assert store.checked_searches == 2 and store.screened_searches == 200
    report = store.report()
    assert report["checked_searches"] == 2 and report["most_candidates"] >= 1 and report["rows_computed_exactly"] >= 200
    bounds = Memories._bounds

    def wrong(self, mean, centred):                      # bounds that rule out the true best
        up, low = bounds(self, mean, centred)
        best = int(self._likeness(centred + mean).argmax())
        up[best] = -1.0
        return up, low

    monkeypatch.setattr(Memories, "_bounds", wrong)
    with pytest.raises(PersistenceError, match="bounds do not hold"):
        for q in range(memory_module.CHECK_EVERY):
            store.nearest(rows[q], now=10**9)


def test_with_the_screen_off_every_search_computes_every_row(tmp_path):
    rows = torch.randn(200, 16, generator=torch.Generator().manual_seed(2))
    store = _filled(tmp_path, rows, per_memory=1)
    plain = Memories(store.directory, recent=0, limit_bytes=None, drive_margin_bytes=0, screened=False)
    for q in range(30):
        a, b = store.nearest(rows[q] * 1.1, now=10**9), plain.nearest(rows[q] * 1.1, now=10**9)
        assert (a.id, a.seq, a.likeness, a.next_likeness, a.beat) == (b.id, b.seq, b.likeness, b.next_likeness, b.beat)
    assert plain.full_searches == 30 and plain.screened_searches == 0 and plain._ref_mean is None


def test_the_screen_refuses_anything_but_4_byte_numbers_on_the_cpu(tmp_path):
    # Opus 2.6: a product in another precision would void the bound silently.
    rows = torch.randn(50, 8, generator=torch.Generator().manual_seed(2))
    store = _filled(tmp_path, rows, per_memory=1)
    store.nearest(rows[0], now=10**9)
    store._matrix = store._matrix.to(torch.float64)
    with pytest.raises(PersistenceError, match="4-byte numbers on the CPU"):
        store.nearest(rows[1], now=10**9)
