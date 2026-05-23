"""Memory surfacer — retrieves relevant memories for the cognitive cycle.

Given the current context (recent inner speech, percepts, emotional state),
the surfacer queries persistent storage and returns the top-K most relevant
memories as SurfacedMemory objects for inclusion in CognitiveInput.

The surfacer does not decide what matters — it retrieves what is semantically
close to the current moment. The entity decides what to attend to.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Protocol

from sanctuary.core.schema import SurfacedMemory

logger = logging.getLogger(__name__)


class MemoryStoreProtocol(Protocol):
    """Interface for any backend that can answer semantic queries.

    The surfacer works with anything that implements this — a ChromaDB
    wrapper, an in-memory test store, or any other backend that exposes
    semantic recall, find_associated, and journal access.
    """

    async def recall(
        self,
        query: str,
        n_results: int = 5,
        min_significance: Optional[int] = None,
    ) -> list: ...


@dataclass
class SurfacerConfig:
    """Configuration for memory surfacing."""

    max_results: int = 5
    min_significance: int = 1
    cooldown_cycles: int = 3  # Don't re-surface the same memory within N cycles
    context_window: int = 3  # Number of recent thoughts to include in query


class MemorySurfacer:
    """Surfaces relevant memories for the cognitive cycle.

    Each cycle, the surfacer is called with the current context string
    (assembled from recent inner speech). It queries the memory store
    and returns SurfacedMemory objects, deduplicating against recently
    surfaced memories to avoid repetition.

    Usage::

        surfacer = MemorySurfacer(store=my_memory_backend)
        memories = await surfacer.surface("Alice greeted me warmly")
    """

    def __init__(
        self,
        store: Optional[MemoryStoreProtocol] = None,
        config: Optional[SurfacerConfig] = None,
    ):
        self._store = store
        self._config = config or SurfacerConfig()
        self._recently_surfaced: dict[str, int] = {}  # content_hash -> cycle_surfaced
        self._cycle_count = 0

    @property
    def store(self) -> Optional[MemoryStoreProtocol]:
        return self._store

    @store.setter
    def store(self, value: MemoryStoreProtocol) -> None:
        self._store = value

    async def surface(self, context: str) -> list[SurfacedMemory]:
        """Retrieve relevant memories for the current context.

        Args:
            context: Recent inner speech / percept summary for semantic search.

        Returns:
            List of SurfacedMemory objects for inclusion in CognitiveInput.
            Returns empty list if no store is configured or context is empty.
        """
        self._cycle_count += 1

        if not self._store or not context.strip():
            return []

        try:
            raw_results = await self._store.recall(
                query=context,
                n_results=self._config.max_results + 3,  # Fetch extra for dedup
                min_significance=self._config.min_significance,
            )
        except Exception as e:
            raise RuntimeError(f"Memory surfacing failed: {e}") from e

        # Convert raw results to SurfacedMemory and deduplicate
        surfaced: list[SurfacedMemory] = []
        for entry in raw_results:
            memory = self._entry_to_surfaced(entry)
            if memory is None:
                continue

            content_hash = self._hash(memory.content)
            last_surfaced = self._recently_surfaced.get(content_hash)
            if (
                last_surfaced is not None
                and self._cycle_count - last_surfaced < self._config.cooldown_cycles
            ):
                continue  # Skip recently surfaced

            surfaced.append(memory)
            self._recently_surfaced[content_hash] = self._cycle_count

            if len(surfaced) >= self._config.max_results:
                break

        # Prune cooldown tracker to prevent unbounded growth
        self._prune_cooldown_tracker()

        logger.debug(
            "Surfaced %d memories for context: %s",
            len(surfaced),
            context[:80],
        )
        return surfaced

    def _entry_to_surfaced(self, entry) -> Optional[SurfacedMemory]:
        """Convert a raw memory entry to a SurfacedMemory schema object.

        Handles both legacy JournalEntry objects and plain dicts.
        """
        try:
            if hasattr(entry, "content") and hasattr(entry, "significance_score"):
                # Legacy JournalEntry-like object
                emotional_tone = ""
                if hasattr(entry, "emotional_signature") and entry.emotional_signature:
                    sigs = entry.emotional_signature
                    if isinstance(sigs, list):
                        emotional_tone = ", ".join(
                            s.value if hasattr(s, "value") else str(s) for s in sigs
                        )
                    else:
                        emotional_tone = str(sigs)

                when = ""
                if hasattr(entry, "timestamp"):
                    when = str(entry.timestamp)

                return SurfacedMemory(
                    content=entry.content[:500],  # Truncate for context budget
                    significance=entry.significance_score,
                    emotional_tone=emotional_tone,
                    when=when,
                )

            if isinstance(entry, dict):
                return SurfacedMemory(
                    content=str(entry.get("content", ""))[:500],
                    significance=int(entry.get("significance", entry.get("significance_score", 5))),
                    emotional_tone=str(entry.get("emotional_tone", entry.get("emotional_signature", ""))),
                    when=str(entry.get("when", entry.get("timestamp", ""))),
                )

            # Fallback: treat as string
            return SurfacedMemory(content=str(entry)[:500], significance=5)

        except Exception as e:
            logger.warning("Failed to convert memory entry: %s", e)
            return None

    def _hash(self, content: str) -> str:
        """Simple content hash for deduplication."""
        return content[:100]

    def _prune_cooldown_tracker(self) -> None:
        """Remove entries older than cooldown window."""
        cutoff = self._cycle_count - self._config.cooldown_cycles * 2
        self._recently_surfaced = {
            k: v for k, v in self._recently_surfaced.items() if v >= cutoff
        }
