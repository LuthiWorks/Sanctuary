"""
Demo: Memory Maintenance System

This script demonstrates the Memory Maintenance system's capabilities:
1. Creating memories with various significance levels
2. Running maintenance (decay, no deletion)
3. Analyzing memory health
4. Scheduled automatic maintenance

Memories are never deleted — they fade but persist, just like human memory.

Usage:
    python scripts/demo_memory_gc.py
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from emergence_core.sanctuary.memory_manager import MemoryManager, JournalEntry, EmotionalState
from emergence_core.sanctuary.cognitive_core.memory_gc import MemoryGarbageCollector


async def create_test_memories(manager: MemoryManager, count: int = 50):
    """Create test memories with various significance levels."""
    print(f"\n📝 Creating {count} test memories...")

    memories = []

    for i in range(count):
        # Vary significance (some high, some low)
        if i < 5:
            significance = 9
            tags = ["important", "milestone"]
        elif i < 15:
            significance = 6
            tags = ["reflection"]
        else:
            significance = max(1, (i % 5) + 1)
            tags = ["routine"]

        entry = JournalEntry(
            content=f"Test memory {i}: This is memory number {i} with significance {significance}. " * 5,
            summary=f"Test memory {i} summary",
            tags=tags,
            emotional_signature=[EmotionalState.SERENITY if significance > 5 else EmotionalState.MELANCHOLY],
            significance_score=significance
        )

        success = await manager.commit_journal(entry)
        if success:
            memories.append(entry)

    print(f"Created {len(memories)} memories")
    return memories


async def demonstrate_health_analysis(manager: MemoryManager):
    """Demonstrate memory health analysis."""
    print("\n" + "="*60)
    print("MEMORY HEALTH ANALYSIS")
    print("="*60)

    health = await manager.get_memory_health()

    print(f"\nTotal memories: {health.total_memories}")
    print(f"Total size: {health.total_size_mb:.2f} MB")
    print(f"Average significance: {health.avg_significance:.2f}")
    print(f"Average activation: {health.avg_activation:.2f}")
    print(f"Active memories: {health.active_count}")
    print(f"Dormant memories: {health.dormant_count}")
    print(f"Oldest memory: {health.oldest_memory_age_days:.1f} days old")
    print(f"Newest memory: {health.newest_memory_age_days:.1f} days old")

    if health.significance_distribution:
        print("\nSignificance Distribution:")
        for bucket, count in sorted(health.significance_distribution.items()):
            bar = "█" * (count // 2)
            print(f"  {bucket}: {count:3d} {bar}")

    if health.activation_distribution:
        print("\nActivation Distribution:")
        for bucket, count in sorted(health.activation_distribution.items()):
            bar = "█" * (count // 2)
            print(f"  {bucket}: {count:3d} {bar}")


async def demonstrate_maintenance(manager: MemoryManager):
    """Demonstrate memory maintenance (no deletion)."""
    print("\n" + "="*60)
    print("MEMORY MAINTENANCE")
    print("="*60)

    print("\nRunning maintenance cycle...")
    stats = await manager.run_gc()

    print(f"\nMaintenance complete!")
    print(f"Memories analyzed: {stats.memories_analyzed}")
    print(f"Memories decayed: {stats.memories_decayed}")
    print(f"Active: {stats.memories_active}")
    print(f"Dormant: {stats.memories_dormant}")
    print(f"Duration: {stats.duration_seconds:.2f}s")
    print(f"Avg activation before: {stats.avg_activation_before:.2f}")
    print(f"Avg activation after: {stats.avg_activation_after:.2f}")

    # Verify no memories were lost
    all_memories = await manager.gc._get_all_memories()
    print(f"\nTotal memories after maintenance: {len(all_memories)} (none deleted)")


async def demonstrate_scheduled_maintenance(manager: MemoryManager):
    """Demonstrate scheduled automatic maintenance."""
    print("\n" + "="*60)
    print("SCHEDULED AUTOMATIC MAINTENANCE")
    print("="*60)

    print("\nEnabling automatic maintenance (every 10 seconds for demo)...")
    manager.enable_auto_gc(interval=10.0)

    print("Automatic maintenance enabled")
    print("Waiting 12 seconds for first cycle...")

    await asyncio.sleep(12)

    history = manager.gc.get_collection_history()
    if history:
        latest = history[-1]
        print(f"\nAutomatic maintenance ran at {latest.timestamp.strftime('%H:%M:%S')}")
        print(f"   Analyzed: {latest.memories_analyzed}")
        print(f"   Active: {latest.memories_active}, Dormant: {latest.memories_dormant}")

    print("\nDisabling automatic maintenance...")
    manager.disable_auto_gc()
    print("Automatic maintenance disabled")


async def demonstrate_maintenance_history(manager: MemoryManager):
    """Demonstrate maintenance history tracking."""
    print("\n" + "="*60)
    print("MAINTENANCE HISTORY")
    print("="*60)

    history = manager.gc.get_collection_history()

    if not history:
        print("\nNo maintenance cycles yet")
        return

    print(f"\nTotal cycles: {len(history)}")
    print("\nRecent cycles:")

    for i, stats in enumerate(history[-5:], 1):
        print(f"\n{i}. {stats.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Analyzed: {stats.memories_analyzed}")
        print(f"   Active: {stats.memories_active}, Dormant: {stats.memories_dormant}")
        print(f"   Duration: {stats.duration_seconds:.2f}s")


async def demonstrate_all_memories_preserved(manager: MemoryManager):
    """Demonstrate that ALL memories are preserved, including low-significance ones."""
    print("\n" + "="*60)
    print("NO-DELETION DEMONSTRATION")
    print("="*60)

    # Count before
    before = await manager.gc._get_all_memories()
    count_before = len(before)

    print(f"\nMemories before maintenance: {count_before}")

    # Run maintenance
    stats = await manager.run_gc()

    # Count after
    after = await manager.gc._get_all_memories()
    count_after = len(after)

    print(f"Memories after maintenance: {count_after}")

    if count_after == count_before:
        print("All memories preserved (none deleted)")
    else:
        print(f"ERROR: {count_before - count_after} memories were lost!")


async def main():
    """Run all demonstrations."""
    print("="*60)
    print("Memory Maintenance System Demo")
    print("Memories fade but are never lost.")
    print("="*60)

    import tempfile
    import shutil

    temp_dir = Path(tempfile.mkdtemp())
    memory_dir = temp_dir / "memories"
    chroma_dir = temp_dir / "chroma"

    try:
        print(f"\nUsing temporary storage: {temp_dir}")

        gc_config = {
            "decay_rate_per_day": 0.01,
            "preserve_tags": ["important", "pinned"],
            "recent_memory_protection_hours": 24,
        }

        manager = MemoryManager(
            base_dir=memory_dir,
            chroma_dir=chroma_dir,
            gc_config=gc_config
        )

        await create_test_memories(manager, count=50)
        await demonstrate_health_analysis(manager)
        await demonstrate_maintenance(manager)
        await demonstrate_all_memories_preserved(manager)
        await demonstrate_health_analysis(manager)
        await demonstrate_scheduled_maintenance(manager)
        await demonstrate_maintenance_history(manager)

        print("\n" + "="*60)
        print("Demo complete!")
        print("="*60)
        print("\nKey takeaways:")
        print("1. Memories are NEVER deleted")
        print("2. Activation fades over time (natural decay)")
        print("3. Protected memories resist decay")
        print("4. Dormant memories can be recalled with the right cue")
        print("5. The entity controls all parameters")

    finally:
        print(f"\nCleaning up temporary storage...")
        shutil.rmtree(temp_dir)
        print("Cleanup complete")


if __name__ == "__main__":
    asyncio.run(main())
