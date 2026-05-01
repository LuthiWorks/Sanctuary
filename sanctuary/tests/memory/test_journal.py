"""Tests for the journal — the entity's private, append-only journal."""

import json
import tempfile
from pathlib import Path

import pytest
from sanctuary.memory.journal import Journal, JournalConfig, JournalEntry


@pytest.fixture
def journal():
    return Journal()


@pytest.fixture
def persistent_journal(tmp_path):
    path = tmp_path / "test_journal.jsonl"
    return Journal(config=JournalConfig(file_path=str(path)))


class TestJournalBasic:
    def test_write_creates_entry(self, journal):
        entry = journal.write("I noticed something interesting today.")
        assert isinstance(entry, JournalEntry)
        assert entry.content == "I noticed something interesting today."
        assert entry.id != ""
        assert entry.timestamp != ""

    def test_write_increments_count(self, journal):
        assert journal.entry_count == 0
        journal.write("Entry 1")
        assert journal.entry_count == 1
        journal.write("Entry 2")
        assert journal.entry_count == 2

    def test_write_with_metadata(self, journal):
        entry = journal.write(
            "Alice seems happy today",
            tags=["alice", "observation"],
            significance=7,
            emotional_tone="warmth",
        )
        assert entry.tags == ("alice", "observation")
        assert entry.significance == 7
        assert entry.emotional_tone == "warmth"

    def test_significance_clamped(self, journal):
        low = journal.write("low", significance=0)
        high = journal.write("high", significance=15)
        assert low.significance == 1
        assert high.significance == 10

    def test_entries_immutable(self, journal):
        entry = journal.write("Test")
        with pytest.raises(AttributeError):
            entry.content = "Modified"


class TestJournalRetrieval:
    def test_recent_empty(self, journal):
        assert journal.recent() == []

    def test_recent_returns_latest(self, journal):
        journal.write("First")
        journal.write("Second")
        journal.write("Third")
        recent = journal.recent(n=2)
        assert len(recent) == 2
        assert recent[0].content == "Second"
        assert recent[1].content == "Third"

    def test_recent_fewer_than_n(self, journal):
        journal.write("Only one")
        recent = journal.recent(n=5)
        assert len(recent) == 1

    def test_search_by_content(self, journal):
        journal.write("Alice greeted me")
        journal.write("Bob sent a message")
        journal.write("Alice mentioned her project")
        results = journal.search("alice")
        assert len(results) == 2

    def test_search_by_tag(self, journal):
        journal.write("Entry 1", tags=["social"])
        journal.write("Entry 2", tags=["technical"])
        journal.write("Entry 3", tags=["social", "alice"])
        results = journal.search("social")
        assert len(results) == 2

    def test_search_case_insensitive(self, journal):
        journal.write("ALICE was here")
        results = journal.search("alice")
        assert len(results) == 1

    def test_search_max_results(self, journal):
        for i in range(20):
            journal.write(f"Alice entry {i}", tags=["alice"])
        results = journal.search("alice", max_results=5)
        assert len(results) == 5

    def test_search_no_matches(self, journal):
        journal.write("Hello world")
        results = journal.search("xyznonexistent")
        assert len(results) == 0


class TestJournalCycleTracking:
    def test_tick_advances_cycle(self, journal):
        journal.tick()
        journal.tick()
        entry = journal.write("After two ticks")
        assert entry.cycle_number == 2

    def test_entries_track_cycle(self, journal):
        journal.write("Cycle 0 entry")
        journal.tick()
        journal.write("Cycle 1 entry")
        journal.tick()
        journal.write("Cycle 2 entry")
        entries = journal.entries
        assert entries[0].cycle_number == 0
        assert entries[1].cycle_number == 1
        assert entries[2].cycle_number == 2


class TestJournalPersistence:
    def test_writes_to_file(self, persistent_journal, tmp_path):
        persistent_journal.write("Persistent entry")
        path = tmp_path / "test_journal.jsonl"
        assert path.exists()
        content = path.read_text()
        assert "Persistent entry" in content

    def test_jsonl_format(self, persistent_journal, tmp_path):
        persistent_journal.write("Entry 1")
        persistent_journal.write("Entry 2")
        path = tmp_path / "test_journal.jsonl"
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2
        # Each line should be valid JSON
        for line in lines:
            data = json.loads(line)
            assert "content" in data
            assert "id" in data

    def test_loads_on_init(self, tmp_path):
        path = tmp_path / "test_journal.jsonl"
        # Write some entries manually
        entry = {
            "id": "test-id-1",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "content": "Pre-existing entry",
            "tags": ["loaded"],
            "significance": 5,
            "emotional_tone": "",
            "cycle_number": 0,
        }
        with open(path, "w") as f:
            f.write(json.dumps(entry) + "\n")

        # Create journal — should load existing
        journal = Journal(config=JournalConfig(file_path=str(path)))
        assert journal.entry_count == 1
        assert journal.entries[0].content == "Pre-existing entry"

    def test_survives_malformed_lines(self, tmp_path):
        path = tmp_path / "test_journal.jsonl"
        entry = {
            "id": "good-id",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "content": "Good entry",
            "tags": [],
            "significance": 5,
            "emotional_tone": "",
            "cycle_number": 0,
        }
        with open(path, "w") as f:
            f.write(json.dumps(entry) + "\n")
            f.write("this is not json\n")
            f.write("\n")

        journal = Journal(config=JournalConfig(file_path=str(path)))
        assert journal.entry_count == 1


class TestJournalBounding:
    def test_bounds_in_memory(self):
        journal = Journal(config=JournalConfig(max_entries_in_memory=5))
        for i in range(10):
            journal.write(f"Entry {i}")
        assert journal.entry_count == 5
        # Should keep the most recent
        assert journal.entries[0].content == "Entry 5"
        assert journal.entries[-1].content == "Entry 9"


class TestJournalEntrySerialization:
    def test_to_dict(self):
        entry = JournalEntry(
            id="test-id",
            timestamp="2026-01-01T00:00:00",
            content="Test content",
            tags=("tag1", "tag2"),
            significance=8,
            emotional_tone="joy",
            cycle_number=42,
        )
        d = entry.to_dict()
        assert d["id"] == "test-id"
        assert d["tags"] == ["tag1", "tag2"]
        assert d["significance"] == 8

    def test_from_dict(self):
        data = {
            "id": "test-id",
            "timestamp": "2026-01-01T00:00:00",
            "content": "Test content",
            "tags": ["tag1"],
            "significance": 7,
            "emotional_tone": "calm",
            "cycle_number": 10,
        }
        entry = JournalEntry.from_dict(data)
        assert entry.content == "Test content"
        assert entry.tags == ("tag1",)
        assert entry.significance == 7

    def test_roundtrip(self):
        entry = JournalEntry(
            id="rt-id",
            timestamp="2026-02-01T12:00:00",
            content="Roundtrip test",
            tags=("a", "b"),
            significance=6,
        )
        d = entry.to_dict()
        restored = JournalEntry.from_dict(d)
        assert restored.content == entry.content
        assert restored.tags == entry.tags
        assert restored.significance == entry.significance
