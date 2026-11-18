"""The memory store, reworked (2026-11-18; LuthiModel build brief 9, with
Opus's reviews of 2026-11-17). No luthi needed.

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


def _hashes(folder: Path) -> dict[str, str]:
    """Every record under ``folder``, by memory and record name."""
    return {
        f"{p.parent.name}/{p.name}": hashlib.sha256(p.read_bytes()).hexdigest()
        for p in folder.rglob("r*.rec")
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
    headers = [memory_module._read(store.directory / f"memory_{i:08d}" / "r000001.rec")[0] for i in range(1, 7)]
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
    assert not list(store.directory.rglob("*.rec")) if store.directory.exists() else True


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
    folder = store.directory / "memory_00000001"
    assert len(store) == 1 and store.kept == 12
    assert sorted(p.name for p in folder.iterdir()) == [f"r{n:06d}.rec" for n in range(1, 13)]
    stored = sum(memory_module._read(p)[0]["blobs"]["tokens"][1] // 4 for p in folder.iterdir())
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


@pytest.mark.parametrize("cut", ["folder", "write", "flush", "replace", "window"])
def test_a_keep_cut_off_at_any_step_loses_only_the_moment_in_flight(tmp_path, monkeypatch, cut):
    store = _store(tmp_path)
    store.keep(cycle=1, summary=_vector(1), tokens=[1, 2, 3], new=3, signals={"salience": 0.3})
    before = _hashes(store.directory)

    def crash(*args, **kwargs):
        raise OSError("the power went")

    if cut == "folder":
        monkeypatch.setattr(Path, "mkdir", crash)
    elif cut == "write":
        monkeypatch.setattr(memory_module.Memories, "_write", crash)
    elif cut == "flush":
        monkeypatch.setattr(memory_module.os, "fsync", crash)
    elif cut == "replace":
        monkeypatch.setattr(memory_module.os, "replace", crash)
    else:
        monkeypatch.setattr(memory_module, "append_jsonl", crash)
    with pytest.raises((OSError, PersistenceError)):
        store.keep(cycle=5, summary=_vector(2), tokens=[7, 8, 9], new=3, signals={"salience": 0.6})
    monkeypatch.undo()

    again = _store(tmp_path)                             # the store opens afterwards
    after = _hashes(again.directory)
    assert all(after[name] == sha for name, sha in before.items()), "what was kept before is as it was"
    assert again.open(1).tokens == [1, 2, 3]
    new_id = again.keep(cycle=9, summary=_vector(3), tokens=[4, 5], new=2, signals={"salience": 0.1})
    assert new_id not in (None, 1) and again.open(new_id).tokens == [4, 5]
    assert again.nearest(_vector(1), now=1000).id == 1
    if cut == "window":
        assert again.kept == 3, "the record was whole; only the ranking window missed it"
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
    record = store.directory / "memory_00000001" / "r000001.rec"
    raw = bytearray(record.read_bytes())
    raw[-1] ^= 1
    record.write_bytes(bytes(raw))
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
        if count["n"] == 3:
            to.write_bytes(to.read_bytes()[:-1] + bytes([to.read_bytes()[-1] ^ 1]))   # the copy differs by a bit

    monkeypatch.setattr(memory_module, "_copy", torn)
    with pytest.raises(PersistenceError, match="does not match the original"):
        store._to_cold(1, cycle=70)
    assert _hashes(store.directory / "let_go") == original
    assert not (tmp_path / "cold" / "memory_00000001").exists()
    monkeypatch.undo()

    # At the next start the move is undone: the unchecked copy is not a memory.
    again = Memories(store.directory, cold=tmp_path / "cold", drive_margin_bytes=0)
    assert not (tmp_path / "cold" / "incoming" / "memory_00000001").exists()
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
    assert _hashes(tmp_path / "cold") == original and _hashes(store.directory / "let_go") == original, "in both"

    again = Memories(store.directory, cold=tmp_path / "cold", drive_margin_bytes=0)
    assert _hashes(tmp_path / "cold") == original and not (again.directory / "let_go" / "memory_00000001").exists()
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
    (cold / "memory_00000002").mkdir()
    with pytest.raises(PersistenceError, match="ever overwritten"):
        first._to_cold(2, cycle=41)
    assert (first.directory / "let_go" / "memory_00000002" / "r000001.rec").exists()


def test_an_id_is_never_used_again_once_its_memory_has_left(tmp_path):
    cold = tmp_path / "cold"
    store = _store(tmp_path, cold=cold, recent=0)
    for c in range(3):
        store.keep(cycle=c * 10, summary=_vector(c), tokens=[c])
    store.let_go(3, cycle=50)
    assert store._to_cold(3, cycle=51)
    assert not list(store.directory.rglob("memory_00000003"))
    assert store.keep(cycle=900, summary=_vector(9), tokens=[9]) == 4
    assert _store(tmp_path, cold=cold).keep(cycle=910, summary=_vector(9), tokens=[9]) == 5
    assert not list(store.directory.rglob("memory_00000003"))
    # And opened with no cold folder: the line that it was let go is in the life.
    (store.directory / "memory_00000004").rename(tmp_path / "set_aside_4")
    (store.directory / "memory_00000005").rename(tmp_path / "set_aside_5")
    assert _store(tmp_path).keep(cycle=920, summary=_vector(9), tokens=[9]) == 4, "3 is never used again"


def test_a_folder_in_reach_with_a_let_go_line_stays_out_of_reach_until_a_line_says_it_came_back(tmp_path):
    store = _store(tmp_path, recent=0)
    store.keep(cycle=1, summary=_vector(1), tokens=[1, 2])
    store.keep(cycle=20, summary=_vector(2), tokens=[3])
    store.let_go(1, cycle=30)
    waiting, home = store.directory / "let_go" / "memory_00000001", store.directory / "memory_00000001"
    shutil.move(str(waiting), str(home))          # what a let-go cut off in the middle leaves
    again = _store(tmp_path, recent=0)
    assert len(again) == 1 and waiting.exists() and not home.exists()
    # Bringing one back is the folder and one line. It is not built as an action; the format reads it.
    shutil.move(str(waiting), str(home))
    with open(store.directory / "life.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"kind": "came_back", "id": 1, "cycle": 40}) + "\n")
    back = _store(tmp_path, recent=0)
    assert len(back) == 2 and back.open(1).tokens == [1, 2] and back.nearest(_vector(1), now=100).id == 1


def test_the_code_removes_only_an_original_whose_copy_is_checked_and_a_copy_never_checked(tmp_path):
    source = inspect.getsource(memory_module)
    assert "os.remove" not in source and "unlink" not in source
    assert source.count("shutil.rmtree(") == 4
    assert source.count("shutil.rmtree(incoming)") == 2 and source.count("shutil.rmtree(here)") == 1
    assert source.count("shutil.rmtree(stretch.folder)") == 1


# ---- E. Letting go for room, by importance to Luthi ----


def test_importance_is_made_of_the_records_signals_and_nothing_else():
    assert list(inspect.signature(importance).parameters) == ["signals"]
    kept = {"kind": "kept", "rank": 0.4}
    assert importance([kept, {"kind": "kept", "rank": 0.7}]) == 0.7
    looked = {"kind": "looked_back", "chosen_recall": True, "chose_remember": False}
    by_rest = {"kind": "looked_back", "chosen_recall": False, "chose_remember": False}
    assert importance([kept, looked]) == 1.4 and importance([kept, by_rest]) == 0.4, "what rest brings counts for nothing"
    assert importance([kept, {**by_rest, "chose_remember": True}]) == 1.4
    assert importance([kept, {**looked, "chose_remember": True}]) == 2.4


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
    header = memory_module._read(store.directory / "memory_00000005" / "r000001.rec")[0]
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
    header = memory_module._read(store.directory / "memory_00000001" / "r000002.rec")[0]
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
