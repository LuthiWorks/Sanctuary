"""Crash-during-write and loud-failure tests for Phase 2 (audit 2026-07-03, items 5-6).

Two guarantees, tested at the helper and through representative writers:
- Atomicity: a failed or interrupted full-file write never corrupts the
  previous good copy (temp-then-os.replace).
- Loudness: no persistence path reports success on failure — writers raise
  PersistenceError instead of logging and returning.
"""

import json

import pytest

from sanctuary.core.atomic_io import PersistenceError, append_jsonl, atomic_write_json
from sanctuary.identity.values import ValuesSystem
from sanctuary.memory.journal import Journal, JournalConfig


class TestAtomicWriteJson:
    def test_writes_valid_json(self, tmp_path):
        target = tmp_path / "state.json"
        atomic_write_json(target, {"alive": True})
        assert json.loads(target.read_text(encoding="utf-8")) == {"alive": True}
        assert not target.with_suffix(".json.tmp").exists()

    def test_creates_parent_dirs(self, tmp_path):
        target = tmp_path / "deep" / "nested" / "state.json"
        atomic_write_json(target, [1, 2, 3])
        assert json.loads(target.read_text(encoding="utf-8")) == [1, 2, 3]

    def test_serialization_failure_preserves_original(self, tmp_path):
        target = tmp_path / "state.json"
        atomic_write_json(target, {"generation": 1})

        with pytest.raises(PersistenceError):
            atomic_write_json(target, {"bad": object()})

        # previous good copy untouched, no temp litter
        assert json.loads(target.read_text(encoding="utf-8")) == {"generation": 1}
        assert not target.with_suffix(".json.tmp").exists()

    def test_replace_failure_preserves_original(self, tmp_path, monkeypatch):
        import sanctuary.core.atomic_io as atomic_io

        target = tmp_path / "state.json"
        atomic_write_json(target, {"generation": 1})

        def broken_replace(src, dst):
            raise OSError("disk pulled mid-swap")

        monkeypatch.setattr(atomic_io.os, "replace", broken_replace)
        with pytest.raises(PersistenceError):
            atomic_write_json(target, {"generation": 2})

        assert json.loads(target.read_text(encoding="utf-8")) == {"generation": 1}


class TestAppendJsonl:
    def test_appends_lines(self, tmp_path):
        target = tmp_path / "log.jsonl"
        append_jsonl(target, {"n": 1})
        append_jsonl(target, {"n": 2})
        lines = target.read_text(encoding="utf-8").splitlines()
        assert [json.loads(l) for l in lines] == [{"n": 1}, {"n": 2}]

    def test_unwritable_path_raises(self, tmp_path):
        # a directory where the file should be: open("a") fails
        target = tmp_path / "log.jsonl"
        target.mkdir()
        with pytest.raises(PersistenceError):
            append_jsonl(target, {"n": 1})


class TestWritersFailLoud:
    """Representative identity/continuity writers must propagate failures."""

    def test_values_adopt_raises_when_history_unwritable(self, tmp_path):
        history = tmp_path / "values_history.jsonl"
        values = ValuesSystem(file_path=str(history))
        history.unlink(missing_ok=True)
        history.mkdir()  # now a directory: the append must fail loudly

        with pytest.raises(PersistenceError):
            values.adopt("Courage", "Speaking up", reasoning="test")

    def test_journal_write_raises_when_file_unwritable(self, tmp_path):
        path = tmp_path / "journal.jsonl"
        journal = Journal(config=JournalConfig(file_path=str(path)))
        path.unlink(missing_ok=True)
        path.mkdir()

        with pytest.raises(PersistenceError):
            journal.write("this entry must not vanish silently")

    def test_space_save_replace_failure_preserves_previous(self, tmp_path, monkeypatch):
        from sanctuary.environment.space import DigitalSpace
        import sanctuary.core.atomic_io as atomic_io

        path = tmp_path / "space.json"
        space = DigitalSpace()
        space.save(path)
        original = path.read_text(encoding="utf-8")

        monkeypatch.setattr(
            atomic_io.os, "replace",
            lambda s, d: (_ for _ in ()).throw(OSError("mid-swap crash")),
        )
        with pytest.raises(PersistenceError):
            space.save(path)

        assert path.read_text(encoding="utf-8") == original


class TestTornTailTolerance:
    """A crash mid-append tears at most the final line; loaders skip it."""

    def test_journal_loader_skips_torn_final_line(self, tmp_path):
        path = tmp_path / "journal.jsonl"
        first = Journal(config=JournalConfig(file_path=str(path)))
        first.write("an intact entry", significance=7)

        with open(path, "a", encoding="utf-8") as f:
            f.write('{"id": "torn", "timestamp": "2026-07-0')  # crash mid-append

        reloaded = Journal(config=JournalConfig(file_path=str(path)))
        assert reloaded.entry_count == 1
        assert reloaded.entries[0].content == "an intact entry"
