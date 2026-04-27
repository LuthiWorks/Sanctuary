"""MemorySubstrate — the memory system for the experiential core.

Implements MemoryProtocol from the cognitive cycle. Bridges:
  - Memory surfacing (context → relevant memories for CognitiveInput)
  - Memory writing (MemoryOps from CognitiveOutput → persistent storage)
  - Journal (private LLM reflections)
  - Prospective memory (future intentions)

Operates with any store implementing a store()/recall() interface.
Default is InMemoryStore for testing; production stores can be
injected via the constructor.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from sanctuary.core.schema import MemoryOp, SurfacedMemory
from sanctuary.memory.journal import Journal, JournalConfig
from sanctuary.memory.prospective import ProspectiveConfig, ProspectiveMemory
from sanctuary.memory.surfacer import MemorySurfacer, SurfacerConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# In-memory store for testing (no ChromaDB dependency)
# ---------------------------------------------------------------------------


class InMemoryStore:
    """Simple in-memory store that satisfies MemoryStoreProtocol.

    Stores entries as dicts and does basic substring matching for recall.
    Used for testing and development without ChromaDB.
    """

    def __init__(self):
        self._entries: list[dict] = []

    async def recall(
        self,
        query: str,
        n_results: int = 5,
        min_significance: Optional[int] = None,
    ) -> list[dict]:
        """Retrieve entries matching the query by substring."""
        query_lower = query.lower()
        results = []
        for entry in reversed(self._entries):  # Most recent first
            sig = entry.get("significance", entry.get("significance_score", 5))
            if min_significance and sig < min_significance:
                continue
            content = str(entry.get("content", "")).lower()
            tags = " ".join(str(t) for t in entry.get("tags", []))
            if query_lower in content or query_lower in tags.lower():
                results.append(entry)
            if len(results) >= n_results:
                break
        return results

    def store(self, entry: dict) -> None:
        """Store a new entry."""
        self._entries.append(entry)

    @property
    def entry_count(self) -> int:
        return len(self._entries)


# ---------------------------------------------------------------------------
# MemorySubstrate configuration
# ---------------------------------------------------------------------------


@dataclass
class MemorySubstrateConfig:
    """Configuration for the memory substrate."""

    surfacer: SurfacerConfig = field(default_factory=SurfacerConfig)
    journal: JournalConfig = field(default_factory=JournalConfig)
    prospective: ProspectiveConfig = field(default_factory=ProspectiveConfig)
    use_in_memory_store: bool = True  # False when injecting a custom store


# ---------------------------------------------------------------------------
# MemorySubstrate — implements MemoryProtocol
# ---------------------------------------------------------------------------


class MemorySubstrate:
    """The memory system for the experiential core.

    Implements MemoryProtocol (surface + queue_retrieval) and adds
    execute_ops for handling CognitiveOutput memory operations.

    Usage with in-memory store (testing)::

        memory = MemorySubstrate()
        memories = await memory.surface("recent context")
        await memory.execute_ops([MemoryOp(type="journal", content="...")])

    Usage with custom store::

        memory = MemorySubstrate(
            config=MemorySubstrateConfig(use_in_memory_store=False),
            store=my_store,  # must implement store() and recall()
        )
    """

    def __init__(
        self,
        config: Optional[MemorySubstrateConfig] = None,
        store=None,
    ):
        self._config = config or MemorySubstrateConfig()

        # Set up the backing store
        if store is not None:
            self._store = store
        elif self._config.use_in_memory_store:
            self._store = InMemoryStore()
        else:
            self._store = None

        # Initialize subsystems
        self._surfacer = MemorySurfacer(
            store=self._store, config=self._config.surfacer
        )
        self._journal = Journal(config=self._config.journal)
        self._prospective = ProspectiveMemory(config=self._config.prospective)

        # Retrieval queue: explicit retrieval requests from the LLM
        self._retrieval_queue: list[str] = []

        self._cycle_count = 0

        logger.info(
            "MemorySubstrate initialized (store=%s)",
            type(self._store).__name__ if self._store else "None",
        )

    # -----------------------------------------------------------------
    # MemoryProtocol implementation
    # -----------------------------------------------------------------

    async def surface(self, context: str) -> list[SurfacedMemory]:
        """Surface relevant memories for the cognitive cycle.

        Combines:
        1. Context-based surfacing from the main store
        2. Triggered prospective intentions
        3. Queued retrieval results from the previous cycle

        Returns a unified list of SurfacedMemory objects.
        """
        memories: list[SurfacedMemory] = []

        # 1. Context-based surfacing
        surfaced = await self._surfacer.surface(context)
        memories.extend(surfaced)

        # 2. Prospective memory triggers
        is_idle = not bool(context.strip())
        triggered = self._prospective.check(context=context, is_idle=is_idle)
        memories.extend(triggered)

        # 3. Queued retrievals from last cycle's MemoryOps
        if self._retrieval_queue:
            for query in self._retrieval_queue:
                try:
                    retrieved = await self._surfacer.surface(query)
                    memories.extend(retrieved)
                except Exception as e:
                    logger.error("Queued retrieval failed for %r, skipping: %s", query[:80], e)
            self._retrieval_queue.clear()

        return memories

    async def queue_retrieval(self, query: str) -> None:
        """Queue an explicit retrieval request from the LLM.

        The retrieval will be executed on the next surface() call.
        """
        if query.strip():
            self._retrieval_queue.append(query)

    # -----------------------------------------------------------------
    # Memory operation execution (from CognitiveOutput)
    # -----------------------------------------------------------------

    async def execute_ops(
        self,
        ops: list[MemoryOp],
        emotional_tone: str = "",
    ) -> list[str]:
        """Execute memory operations from the LLM's cognitive output.

        Called by the cognitive cycle after scaffold validation.

        Args:
            ops: Validated memory operations from CognitiveOutput.
            emotional_tone: Current felt quality for journal entries.

        Returns:
            List of result messages (for logging/debugging).
        """
        results: list[str] = []

        for op in ops:
            try:
                if op.type == "write_episodic":
                    result = await self._write_episodic(op, emotional_tone)
                elif op.type == "write_semantic":
                    result = await self._write_semantic(op)
                elif op.type == "journal":
                    result = self._write_journal(op, emotional_tone)
                elif op.type == "retrieve":
                    await self.queue_retrieval(op.query)
                    result = f"Queued retrieval: {op.query[:50]}"
                else:
                    result = f"Unknown memory op type: {op.type}"
                    logger.warning(result)
                results.append(result)
            except Exception as e:
                msg = f"Memory op failed ({op.type}): {e}"
                logger.error(msg)
                results.append(msg)

        return results

    # -----------------------------------------------------------------
    # Prospective memory API (exposed for the cycle)
    # -----------------------------------------------------------------

    def add_intention(
        self,
        content: str,
        trigger_type: str = "cycle",
        trigger_value: str = "",
        significance: int = 5,
        tags: Optional[list[str]] = None,
    ):
        """Add a prospective memory intention."""
        return self._prospective.add(
            content=content,
            trigger_type=trigger_type,
            trigger_value=trigger_value,
            significance=significance,
            tags=tags,
        )

    # -----------------------------------------------------------------
    # Accessors
    # -----------------------------------------------------------------

    @property
    def journal(self) -> Journal:
        return self._journal

    @property
    def prospective(self) -> ProspectiveMemory:
        return self._prospective

    @property
    def surfacer(self) -> MemorySurfacer:
        return self._surfacer

    @property
    def store(self):
        return self._store

    def tick(self) -> None:
        """Advance cycle counter. Called each cognitive cycle."""
        self._cycle_count += 1
        self._journal.tick()

    # -----------------------------------------------------------------
    # Internal: memory write implementations
    # -----------------------------------------------------------------

    async def _write_episodic(self, op: MemoryOp, emotional_tone: str) -> str:
        """Write an episodic memory to the store."""
        if self._store is None:
            logger.warning("No store available for episodic write")
            return "No store available"

        entry = {
            "content": op.content,
            "significance": op.significance,
            "significance_score": op.significance,
            "tags": op.tags,
            "emotional_tone": emotional_tone,
            "timestamp": _now_iso(),
            "type": "episodic",
        }
        self._store.store(entry)
        return f"Stored episodic memory: {op.content[:50]}"

    async def _write_semantic(self, op: MemoryOp) -> str:
        """Write a semantic memory (fact) to the store."""
        if self._store is None:
            logger.warning("No store available for semantic write")
            return "No store available"

        entry = {
            "content": op.content,
            "significance": op.significance,
            "significance_score": op.significance,
            "tags": op.tags,
            "timestamp": _now_iso(),
            "type": "semantic",
        }
        self._store.store(entry)
        return f"Stored semantic memory: {op.content[:50]}"

    def _write_journal(self, op: MemoryOp, emotional_tone: str) -> str:
        """Write a journal entry."""
        entry = self._journal.write(
            content=op.content,
            tags=op.tags,
            significance=op.significance,
            emotional_tone=emotional_tone,
        )
        return f"Journal entry written: {entry.id}"


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
