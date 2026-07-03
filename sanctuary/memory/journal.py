"""Journal — the entity's private, append-only journal.

The journal stores the entity's reflections, observations, and private thoughts.
Entries come from MemoryOp(type="journal") in CognitiveOutput. They are
append-only (never modified or deleted) and persist as JSONL for crash safety.

The journal is not chat logs — it is the entity's own voice, written by its
own choice, about what it considers worth recording.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4

from sanctuary.core.atomic_io import append_jsonl

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class JournalEntry:
    """A single journal entry — immutable once created.

    Attributes:
        id: Unique identifier.
        timestamp: When the entry was written (UTC).
        content: The journal text.
        tags: Semantic labels.
        significance: Importance rating (1-10).
        emotional_tone: Emotional context when written.
        cycle_number: Which cognitive cycle produced this entry.
    """

    id: str
    timestamp: str
    content: str
    tags: tuple[str, ...] = ()
    significance: int = 5
    emotional_tone: str = ""
    cycle_number: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "content": self.content,
            "tags": list(self.tags),
            "significance": self.significance,
            "emotional_tone": self.emotional_tone,
            "cycle_number": self.cycle_number,
        }

    @classmethod
    def from_dict(cls, data: dict) -> JournalEntry:
        return cls(
            id=data["id"],
            timestamp=data["timestamp"],
            content=data["content"],
            tags=tuple(data.get("tags", [])),
            significance=data.get("significance", 5),
            emotional_tone=data.get("emotional_tone", ""),
            cycle_number=data.get("cycle_number", 0),
        )


@dataclass
class JournalConfig:
    """Configuration for the journal."""

    max_entries_in_memory: int = 200
    file_path: Optional[str] = None  # None = in-memory only


class Journal:
    """Append-only journal for the entity's private reflections.

    Persists entries as JSONL (one JSON object per line) for crash safety.
    Each write is a single append — no rewriting the whole file.

    Usage::

        journal = Journal(config=JournalConfig(file_path="data/journal.jsonl"))
        entry = journal.write("I notice patterns in how Alice communicates...",
                              tags=["alice", "observation"], significance=6)
        recent = journal.recent(n=5)
        results = journal.search("alice")
    """

    def __init__(self, config: Optional[JournalConfig] = None):
        self._config = config or JournalConfig()
        self._entries: list[JournalEntry] = []
        self._cycle_count = 0
        self._file_path: Optional[Path] = None

        if self._config.file_path:
            self._file_path = Path(self._config.file_path)
            self._file_path.parent.mkdir(parents=True, exist_ok=True)
            self._load_existing()

    def write(
        self,
        content: str,
        tags: Optional[list[str]] = None,
        significance: int = 5,
        emotional_tone: str = "",
    ) -> JournalEntry:
        """Write a new journal entry.

        Args:
            content: The journal text.
            tags: Semantic labels for categorization.
            significance: Importance rating (1-10).
            emotional_tone: Emotional context.

        Returns:
            The created JournalEntry.
        """
        entry = JournalEntry(
            id=str(uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            content=content,
            tags=tuple(tags or []),
            significance=max(1, min(10, significance)),
            emotional_tone=emotional_tone,
            cycle_number=self._cycle_count,
        )

        self._entries.append(entry)

        # Persist to disk
        if self._file_path:
            self._append_to_file(entry)

        # Bound in-memory list
        if len(self._entries) > self._config.max_entries_in_memory:
            self._entries = self._entries[-self._config.max_entries_in_memory :]

        logger.debug("Journal entry written: %s (significance=%d)", entry.id, significance)
        return entry

    def recent(self, n: int = 5) -> list[JournalEntry]:
        """Return the N most recent journal entries."""
        return list(self._entries[-n:])

    def search(self, query: str, max_results: int = 10) -> list[JournalEntry]:
        """Simple text search across journal entries.

        For semantic search, use the memory surfacer with a vector store.
        This is keyword-based search for direct lookups.
        """
        query_lower = query.lower()
        results = []
        # Search in reverse (most recent first)
        for entry in reversed(self._entries):
            if query_lower in entry.content.lower():
                results.append(entry)
            elif any(query_lower in tag.lower() for tag in entry.tags):
                results.append(entry)
            if len(results) >= max_results:
                break
        return results

    def tick(self) -> None:
        """Advance the cycle counter. Called each cognitive cycle."""
        self._cycle_count += 1

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> list[JournalEntry]:
        """All entries currently in memory (read-only view)."""
        return list(self._entries)

    def _append_to_file(self, entry: JournalEntry) -> None:
        """Append a single entry as a JSON line, fsync'd.

        Raises PersistenceError on failure — a journal entry that was never
        persisted must not report success.
        """
        append_jsonl(self._file_path, entry.to_dict())

    def _load_existing(self) -> None:
        """Load existing entries from the JSONL file."""
        if not self._file_path or not self._file_path.exists():
            return

        loaded = 0
        try:
            with open(self._file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        self._entries.append(JournalEntry.from_dict(data))
                        loaded += 1
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning("Skipping malformed journal line: %s", e)

            # Trim to max in-memory size
            if len(self._entries) > self._config.max_entries_in_memory:
                self._entries = self._entries[-self._config.max_entries_in_memory :]

            logger.info("Loaded %d journal entries from %s", loaded, self._file_path)
        except Exception as e:
            logger.error("Failed to load journal: %s", e)
