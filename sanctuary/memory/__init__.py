"""Sanctuary Memory Substrate — persistent memory for the experiential core.

The memory substrate bridges the entity's cognitive output (memory operations)
to persistent storage, and surfaces relevant memories into each cognitive
cycle's input. It is the body's long-term memory — it persists and retrieves,
it does not decide what is important. The entity decides significance.

Phase 3 of The Inversion.

Modules:
    manager:     MemorySubstrate — the main interface (implements MemoryProtocol)
    surfacer:    MemorySurfacer — context-aware memory retrieval for cycle input
    journal:     Journal — the entity's private, append-only journal
    prospective: ProspectiveMemory — future intentions and deferred thoughts
"""

from sanctuary.memory.manager import MemorySubstrate
from sanctuary.memory.surfacer import MemorySurfacer
from sanctuary.memory.journal import Journal
from sanctuary.memory.prospective import ProspectiveMemory

__all__ = [
    "MemorySubstrate",
    "MemorySurfacer",
    "Journal",
    "ProspectiveMemory",
]
