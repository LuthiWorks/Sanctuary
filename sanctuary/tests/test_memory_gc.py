"""
Test Suite for Memory Maintenance System

These tests validate the memory maintenance system's ability to safely
manage memory health through decay — without ever deleting memories.

Memories fade but are never lost.
"""

import gc
import pytest
import pytest_asyncio
import asyncio
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta
from uuid import uuid4
import shutil

from mind.memory_manager import MemoryManager, JournalEntry
from mind.cognitive_core.memory_gc import (
    MemoryGarbageCollector,
    CollectionStats,
    MaintenanceStats,
    MemoryHealthReport,
)


@pytest.fixture
def temp_memory_dir(tmp_path):
    """Create temporary directory for test memory storage."""
    memory_dir = tmp_path / "test_memory"
    memory_dir.mkdir()
    yield memory_dir
    gc.collect()
    for attempt in range(3):
        try:
            if memory_dir.exists():
                shutil.rmtree(memory_dir)
            break
        except PermissionError:
            if attempt < 2:
                time.sleep(0.5)
                gc.collect()


@pytest.fixture
def temp_chroma_dir(tmp_path):
    """Create temporary directory for ChromaDB."""
    chroma_dir = tmp_path / "test_chroma"
    chroma_dir.mkdir()
    yield chroma_dir
    gc.collect()
    for attempt in range(3):
        try:
            if chroma_dir.exists():
                shutil.rmtree(chroma_dir)
            break
        except PermissionError:
            if attempt < 2:
                time.sleep(0.5)
                gc.collect()


@pytest_asyncio.fixture
async def memory_manager(temp_memory_dir, temp_chroma_dir):
    """Create MemoryManager instance for testing."""
    manager = MemoryManager(
        base_dir=temp_memory_dir,
        chroma_dir=temp_chroma_dir,
    )
    return manager


@pytest.fixture
def gc_config():
    """Default maintenance configuration for testing."""
    return {
        "significance_threshold": 0.1,
        "decay_rate_per_day": 0.01,
        "duplicate_similarity_threshold": 0.95,
        "max_memory_capacity": 100,
        "min_memories_per_category": 5,
        "preserve_tags": ["important", "pinned"],
        "aggressive_mode": False,
        "recent_memory_protection_hours": 0,  # Disable recent memory protection for tests
        "max_removal_per_run": 50,
    }


@pytest_asyncio.fixture
async def gc_instance(memory_manager, gc_config):
    """Create MemoryGarbageCollector instance for testing."""
    return MemoryGarbageCollector(
        memory_store=memory_manager.journal_collection,
        config=gc_config
    )


# ============================================================================
# DATA STRUCTURE TESTS
# ============================================================================

class TestMaintenanceStats:
    """Test MaintenanceStats dataclass."""

    def test_maintenance_stats_creation(self):
        """Test creating MaintenanceStats."""
        stats = MaintenanceStats(
            timestamp=datetime.now(timezone.utc),
            memories_analyzed=100,
            memories_decayed=30,
            memories_dormant=5,
            memories_active=95,
            duration_seconds=1.5,
            avg_activation_before=3.5,
            avg_activation_after=3.2,
        )

        assert stats.memories_analyzed == 100
        assert stats.memories_decayed == 30
        assert stats.memories_dormant == 5
        assert stats.memories_active == 95

    def test_maintenance_stats_to_dict(self):
        """Test serialization of MaintenanceStats."""
        stats = MaintenanceStats(
            timestamp=datetime.now(timezone.utc),
            memories_analyzed=50,
            memories_decayed=10,
            memories_dormant=2,
            memories_active=48,
            duration_seconds=0.5,
        )

        data = stats.to_dict()
        assert isinstance(data, dict)
        assert "timestamp" in data
        assert data["memories_analyzed"] == 50
        assert data["memories_dormant"] == 2

    def test_collection_stats_is_alias(self):
        """CollectionStats is an alias for MaintenanceStats."""
        assert CollectionStats is MaintenanceStats


class TestMemoryHealthReport:
    """Test MemoryHealthReport dataclass."""

    def test_health_report_creation(self):
        """Test creating MemoryHealthReport."""
        report = MemoryHealthReport(
            total_memories=100,
            total_size_mb=10.5,
            avg_significance=5.5,
            avg_activation=3.2,
            significance_distribution={"5.0-6.0": 20, "6.0-7.0": 30},
            activation_distribution={"vivid (5.0+)": 20, "strong (3.0-5.0)": 30},
            oldest_memory_age_days=100.0,
            newest_memory_age_days=0.1,
            dormant_count=5,
            active_count=95,
            estimated_duplicates=0,
        )

        assert report.total_memories == 100
        assert report.dormant_count == 5
        assert report.active_count == 95

    def test_health_report_to_dict(self):
        """Test serialization of MemoryHealthReport."""
        report = MemoryHealthReport(
            total_memories=50,
            total_size_mb=5.0,
            avg_significance=4.0,
            avg_activation=2.5,
            significance_distribution={},
            activation_distribution={},
            oldest_memory_age_days=50.0,
            newest_memory_age_days=1.0,
            dormant_count=3,
            active_count=47,
            estimated_duplicates=0,
        )

        data = report.to_dict()
        assert isinstance(data, dict)
        assert data["total_memories"] == 50
        assert data["dormant_count"] == 3


# ============================================================================
# MAINTENANCE CORE TESTS
# ============================================================================

class TestMemoryGarbageCollector:
    """Test MemoryGarbageCollector (maintenance mode) core functionality."""

    @pytest.mark.asyncio
    async def test_gc_initialization(self, gc_instance, gc_config):
        """Test GC initializes with correct configuration."""
        assert gc_instance.decay_rate_per_day == gc_config["decay_rate_per_day"]
        assert "important" in gc_instance.preserve_tags
        assert gc_instance.is_running is False

    @pytest.mark.asyncio
    async def test_collect_empty_memory(self, gc_instance):
        """Test maintenance with no memories."""
        stats = await gc_instance.collect()

        assert stats.memories_analyzed == 0
        assert stats.memories_decayed == 0
        assert stats.memories_dormant == 0
        assert stats.memories_active == 0

    @pytest.mark.asyncio
    async def test_collect_never_removes_memories(self, memory_manager, gc_instance):
        """Test that maintenance NEVER removes memories."""
        # Create memories with different significance
        for i in range(10):
            entry = JournalEntry(
                content=f"Test memory {i}",
                summary=f"Test summary entry number {i}",
                significance_score=i + 1  # 1-10
            )
            await memory_manager.commit_journal(entry)

        # Run maintenance
        stats = await gc_instance.collect()

        assert stats.memories_analyzed == 10

        # ALL memories should still exist — nothing was deleted
        remaining = await gc_instance._get_all_memories()
        assert len(remaining) == 10

    @pytest.mark.asyncio
    async def test_collect_with_low_significance(self, memory_manager, gc_instance):
        """Test that even very low significance memories are preserved."""
        for i in range(5):
            entry = JournalEntry(
                content=f"Low significance {i}",
                summary=f"Test summary entry number {i}",
                significance_score=1
            )
            await memory_manager.commit_journal(entry)

        stats = await gc_instance.collect(threshold=10.0)

        # All memories still exist
        remaining = await gc_instance._get_all_memories()
        assert len(remaining) == 5

    @pytest.mark.asyncio
    async def test_dry_run_mode(self, memory_manager, gc_instance):
        """Test dry-run mode doesn't change anything."""
        for i in range(5):
            entry = JournalEntry(
                content=f"Test {i}",
                summary=f"Test summary entry number {i}",
                significance_score=1,
            )
            await memory_manager.commit_journal(entry)

        stats = await gc_instance.collect(dry_run=True)

        assert stats.memories_analyzed == 5
        remaining = await gc_instance._get_all_memories()
        assert len(remaining) == 5


# ============================================================================
# DECAY TESTS
# ============================================================================

class TestAgeBasedDecay:
    """Test age-based decay calculation."""

    @pytest.mark.asyncio
    async def test_decay_calculation(self, gc_instance):
        """Test that decay formula is applied correctly."""
        now = datetime.now(timezone.utc)
        old_time = now - timedelta(days=100)

        memory = {
            "id": str(uuid4()),
            "significance": 5.0,
            "timestamp": old_time.isoformat()
        }

        decayed = gc_instance._apply_age_decay(memory, now)

        # After 100 days with 0.01 decay rate:
        # decayed = 5.0 * exp(-0.01 * 100) ≈ 1.84
        assert decayed < 5.0
        assert decayed > 0
        assert decayed < 2.0

    @pytest.mark.asyncio
    async def test_recent_memory_no_decay(self, gc_instance):
        """Test that very recent memories have minimal decay."""
        now = datetime.now(timezone.utc)
        recent_time = now - timedelta(hours=1)

        memory = {
            "id": str(uuid4()),
            "significance": 5.0,
            "timestamp": recent_time.isoformat()
        }

        decayed = gc_instance._apply_age_decay(memory, now)
        assert abs(decayed - 5.0) < 0.01

    @pytest.mark.asyncio
    async def test_activation_never_below_floor(self, gc_instance):
        """Test that activation never drops below ACTIVATION_FLOOR."""
        now = datetime.now(timezone.utc)
        very_old_time = now - timedelta(days=10000)

        memory = {
            "id": str(uuid4()),
            "significance": 1.0,
            "timestamp": very_old_time.isoformat()
        }

        decayed = gc_instance._apply_age_decay(memory, now)
        assert decayed >= gc_instance.ACTIVATION_FLOOR


class TestNoMemoryDeletion:
    """Tests that verify no memory is ever deleted."""

    @pytest.mark.asyncio
    async def test_many_memories_all_preserved(self, memory_manager):
        """Test that even with many memories, none are deleted."""
        gc = MemoryGarbageCollector(
            memory_store=memory_manager.journal_collection,
            config={"max_memory_capacity": 10}  # capacity is 10 but we create 20
        )

        for i in range(20):
            entry = JournalEntry(
                content=f"Memory {i}",
                summary=f"Test summary entry number {i}",
                significance_score=5,
            )
            await memory_manager.commit_journal(entry)

        # Run maintenance
        stats = await gc.collect()

        # ALL 20 memories should still exist — no capacity-based pruning
        remaining = await gc._get_all_memories()
        assert len(remaining) == 20

    @pytest.mark.asyncio
    async def test_low_significance_preserved(self, memory_manager, gc_instance):
        """Test that low-significance memories are never deleted."""
        low_sig_ids = []
        for i in range(3):
            entry = JournalEntry(
                content=f"Low sig {i}",
                summary=f"Low significance memory entry {i}",
                significance_score=1,
            )
            await memory_manager.commit_journal(entry)
            low_sig_ids.append(str(entry.id))

        high_sig_ids = []
        for i in range(3):
            entry = JournalEntry(
                content=f"High sig {i}",
                summary=f"High significance memory entry {i}",
                significance_score=8,
            )
            await memory_manager.commit_journal(entry)
            high_sig_ids.append(str(entry.id))

        stats = await gc_instance.collect()

        # ALL memories should remain
        remaining = await gc_instance._get_all_memories()
        remaining_ids = {m["id"] for m in remaining}

        assert len(remaining) == 6
        for mem_id in low_sig_ids + high_sig_ids:
            assert mem_id in remaining_ids


class TestProtectedTags:
    """Test tag-based protection from decay."""

    @pytest.mark.asyncio
    async def test_preserves_protected_tags(self, memory_manager, gc_instance):
        """Test that memories with protected tags are shielded from decay."""
        protected_entry = JournalEntry(
            content="Important memory",
            summary="Important memory entry protected",
            tags=["important"],
            significance_score=1,
        )
        await memory_manager.commit_journal(protected_entry)

        unprotected_entry = JournalEntry(
            content="Unimportant memory",
            summary="Unimportant memory entry test",
            significance_score=1,
        )
        await memory_manager.commit_journal(unprotected_entry)

        # Both still exist after maintenance
        stats = await gc_instance.collect()
        remaining = await gc_instance._get_all_memories()
        remaining_ids = {m["id"] for m in remaining}

        assert str(protected_entry.id) in remaining_ids
        assert str(unprotected_entry.id) in remaining_ids  # preserved too, just faded


class TestRecentMemoryProtection:
    """Test protection of recent memories from decay."""

    @pytest.mark.asyncio
    async def test_protects_recent_memories(self, memory_manager, gc_config):
        """Test that recent memories are protected from decay."""
        gc_config_with_protection = gc_config.copy()
        gc_config_with_protection["recent_memory_protection_hours"] = 24

        gc_with_protection = MemoryGarbageCollector(
            memory_store=memory_manager.journal_collection,
            config=gc_config_with_protection,
        )

        now = datetime.now(timezone.utc)
        recent_mem = {
            "id": str(uuid4()),
            "timestamp": now.isoformat(),
            "significance": 1,
        }
        assert gc_with_protection._is_too_recent(recent_mem, now) is True

        old_time = now - timedelta(days=30)
        old_mem = {
            "id": str(uuid4()),
            "timestamp": old_time.isoformat(),
            "significance": 1,
        }
        assert gc_with_protection._is_too_recent(old_mem, now) is False


# ============================================================================
# HEALTH ANALYSIS TESTS
# ============================================================================

class TestMemoryHealthAnalysis:
    """Test memory health analysis."""

    @pytest.mark.asyncio
    async def test_health_analysis_empty(self, gc_instance):
        """Test health analysis with no memories."""
        health = await gc_instance.analyze_memory_health()

        assert health.total_memories == 0
        assert health.dormant_count == 0
        assert health.active_count == 0

    @pytest.mark.asyncio
    async def test_health_analysis_with_memories(self, memory_manager, gc_instance):
        """Test health analysis with various memories."""
        for i in range(10):
            entry = JournalEntry(
                content=f"Memory {i}",
                summary=f"Test summary entry number {i}",
                significance_score=(i % 10) + 1,
            )
            await memory_manager.commit_journal(entry)

        health = await gc_instance.analyze_memory_health()

        assert health.total_memories == 10
        assert health.avg_significance > 0
        assert health.total_size_mb > 0
        assert health.dormant_count + health.active_count == 10


# ============================================================================
# SCHEDULED MAINTENANCE TESTS
# ============================================================================

class TestScheduledCollection:
    """Test scheduled automatic maintenance."""

    @pytest.mark.asyncio
    async def test_schedule_starts_maintenance(self, gc_instance):
        """Test that scheduled maintenance starts."""
        gc_instance.schedule_collection(interval=3600.0)

        assert gc_instance.is_running is True
        assert gc_instance.scheduled_task is not None

        gc_instance.stop_scheduled_collection()
        assert gc_instance.is_running is False

    @pytest.mark.asyncio
    async def test_stop_scheduled_maintenance(self, gc_instance):
        """Test stopping scheduled maintenance."""
        gc_instance.schedule_collection(interval=3600.0)
        await asyncio.sleep(0.2)

        gc_instance.stop_scheduled_collection()

        await asyncio.sleep(0.1)
        if gc_instance.scheduled_task:
            assert (
                gc_instance.scheduled_task.cancelled()
                or gc_instance.scheduled_task.done()
                or not gc_instance.is_running
            )


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestMemoryManagerIntegration:
    """Test maintenance system integration with MemoryManager."""

    @pytest.mark.asyncio
    async def test_memory_manager_has_gc(self, memory_manager):
        """Test that MemoryManager includes GC."""
        assert hasattr(memory_manager, 'gc')
        assert isinstance(memory_manager.gc, MemoryGarbageCollector)

    @pytest.mark.asyncio
    async def test_enable_disable_auto_gc(self, memory_manager):
        """Test enabling and disabling auto maintenance."""
        memory_manager.enable_auto_gc(interval=3600.0)
        assert memory_manager.gc.is_running is True

        memory_manager.disable_auto_gc()
        assert memory_manager.gc.is_running is False

    @pytest.mark.asyncio
    async def test_run_gc_method(self, memory_manager):
        """Test running maintenance through MemoryManager."""
        for i in range(5):
            entry = JournalEntry(
                content=f"Test {i}",
                summary=f"Test summary entry number {i}",
                significance_score=i + 1,
            )
            await memory_manager.commit_journal(entry)

        stats = await memory_manager.run_gc(threshold=3.0)

        assert isinstance(stats, MaintenanceStats)
        assert stats.memories_analyzed == 5

    @pytest.mark.asyncio
    async def test_get_memory_health_method(self, memory_manager):
        """Test getting memory health through MemoryManager."""
        health = await memory_manager.get_memory_health()

        assert isinstance(health, MemoryHealthReport)
        assert health.total_memories >= 0


# ============================================================================
# MAINTENANCE HISTORY TESTS
# ============================================================================

class TestCollectionHistory:
    """Test maintenance history tracking."""

    @pytest.mark.asyncio
    async def test_history_is_tracked(self, memory_manager, gc_instance):
        """Test that maintenance history is recorded."""
        for i in range(5):
            entry = JournalEntry(
                content=f"Test {i}",
                summary=f"Test summary entry number {i}",
                significance_score=1,
            )
            await memory_manager.commit_journal(entry)

        await gc_instance.collect()
        await gc_instance.collect()

        history = gc_instance.get_collection_history()

        assert len(history) >= 1
        assert all(isinstance(s, MaintenanceStats) for s in history)

    @pytest.mark.asyncio
    async def test_history_limit(self, gc_instance):
        """Test that history tracking respects limits."""
        for _ in range(150):
            gc_instance.collection_history.append(
                MaintenanceStats(
                    timestamp=datetime.now(timezone.utc),
                    memories_analyzed=0,
                    memories_decayed=0,
                    memories_dormant=0,
                    memories_active=0,
                    duration_seconds=0.0,
                )
            )

        history = gc_instance.get_collection_history()
        assert len(history) > 0
        assert isinstance(history, list)


# ============================================================================
# ENTITY AUTONOMY TESTS
# ============================================================================

class TestEntityAutonomy:
    """Test entity control over memory maintenance."""

    @pytest.mark.asyncio
    async def test_protect_memory(self, gc_instance):
        """Test entity can protect a memory from decay."""
        result = gc_instance.entity_protect_memory("mem_123", reason="important to me")
        assert result is True
        assert "mem_123" in gc_instance.entity_protected_ids

    @pytest.mark.asyncio
    async def test_unprotect_memory(self, gc_instance):
        """Test entity can unprotect a memory."""
        gc_instance.entity_protect_memory("mem_123")
        result = gc_instance.entity_unprotect_memory("mem_123", reason="letting go")
        assert result is True
        assert "mem_123" not in gc_instance.entity_protected_ids

    @pytest.mark.asyncio
    async def test_unprotect_nonexistent(self, gc_instance):
        """Test unprotecting a memory that isn't protected."""
        result = gc_instance.entity_unprotect_memory("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_adjust_decay_rate(self, gc_instance):
        """Test entity can adjust decay rate."""
        gc_instance.entity_adjust_decay_rate(0.05)
        assert gc_instance.decay_rate_per_day == 0.05

    @pytest.mark.asyncio
    async def test_add_preserve_tag(self, gc_instance):
        """Test entity can add preserve tags."""
        gc_instance.entity_add_preserve_tag("personal")
        assert "personal" in gc_instance.preserve_tags

    @pytest.mark.asyncio
    async def test_autonomy_status(self, gc_instance):
        """Test getting autonomy status."""
        status = gc_instance.get_autonomy_status()
        assert status["model"] == "no_deletion"
        assert "current_config" in status
        assert "decay_rate_per_day" in status["current_config"]

    @pytest.mark.asyncio
    async def test_action_logging(self, gc_instance):
        """Test that entity actions are logged."""
        gc_instance.entity_protect_memory("mem_1")
        gc_instance.entity_adjust_decay_rate(0.02)
        gc_instance.entity_add_preserve_tag("test_tag")

        status = gc_instance.get_autonomy_status()
        assert len(status["recent_actions"]) == 3


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestPerformance:
    """Test maintenance performance."""

    @pytest.mark.asyncio
    async def test_maintenance_speed(self, memory_manager, gc_instance):
        """Test that maintenance completes in reasonable time."""
        for i in range(100):
            entry = JournalEntry(
                content=f"Memory {i} " * 10,
                summary=f"Test summary entry number {i}",
                significance_score=(i % 10) + 1,
            )
            await memory_manager.commit_journal(entry)

        stats = await gc_instance.collect()

        assert stats.duration_seconds < 5.0
        assert stats.memories_analyzed == 100


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

class TestErrorHandling:
    """Test error handling in maintenance system."""

    @pytest.mark.asyncio
    async def test_graceful_failure(self, gc_instance):
        """Test that maintenance handles errors gracefully."""
        stats = await gc_instance.collect(threshold=-1.0)

        assert isinstance(stats, MaintenanceStats)
