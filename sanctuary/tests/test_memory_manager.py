"""
Test Suite for Sovereign Memory Architecture

These tests validate the integrity and fail-safety of Sanctuary's memory system.
Memory is not data - it is the substrate of continuity. Tests reflect this gravity.
"""

import gc
import pytest
import pytest_asyncio
import asyncio
import time
from pathlib import Path
from datetime import datetime, timezone
from uuid import UUID
import json
import shutil

from mind.memory_manager import (
    MemoryManager,
    JournalEntry,
    FactEntry,
    Manifest,
    EmotionalState
)


@pytest.fixture
def temp_memory_dir(tmp_path):
    """Create temporary directory for test memory storage."""
    memory_dir = tmp_path / "test_memory"
    memory_dir.mkdir()
    yield memory_dir
    # Cleanup after tests with retry for Windows
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


# ============================================================================
# DATA STRUCTURE TESTS
# ============================================================================

class TestJournalEntry:
    """Test JournalEntry Pydantic model validation."""
    
    def test_journal_entry_creation(self):
        """Test creating a valid journal entry."""
        entry = JournalEntry(
            content="Today I experienced a moment of profound connection.",
            summary="Moment of connection",
            tags=["reflection", "connection"],
            emotional_signature=[EmotionalState.JOY, EmotionalState.CONNECTION],
            significance_score=7
        )
        
        assert entry.content is not None
        assert entry.id is not None
        assert isinstance(entry.id, UUID)
        assert entry.significance_score == 7
        assert EmotionalState.JOY in entry.emotional_signature
    
    def test_journal_entry_immutability(self):
        """Test that journal entries are immutable after creation."""
        entry = JournalEntry(
            content="Original content",
            summary="Original summary"
        )
        
        with pytest.raises(Exception):  # Pydantic ValidationError
            entry.content = "Modified content"
    
    def test_journal_entry_validation_empty_content(self):
        """Test that empty content is rejected."""
        with pytest.raises(ValueError):
            JournalEntry(
                content="",
                summary="Valid summary"
            )
    
    def test_journal_entry_validation_empty_summary(self):
        """Test that empty summary is rejected."""
        with pytest.raises(ValueError):
            JournalEntry(
                content="Valid content",
                summary="   "  # Whitespace only
            )
    
    def test_journal_entry_significance_bounds(self):
        """Test significance score validation (1-10)."""
        with pytest.raises(ValueError):
            JournalEntry(
                content="Test",
                summary="Test entry summary",
                significance_score=0  # Below minimum
            )
        
        with pytest.raises(ValueError):
            JournalEntry(
                content="Test",
                summary="Test entry summary",
                significance_score=11  # Above maximum
            )
    
    def test_journal_entry_serialization(self):
        """Test to_dict and from_dict roundtrip."""
        original = JournalEntry(
            content="Test content",
            summary="Test summary",
            tags=["test", "roundtrip"],
            emotional_signature=[EmotionalState.SERENITY],
            significance_score=8
        )
        
        # Serialize
        data = original.to_dict()
        assert isinstance(data['id'], str)
        assert isinstance(data['timestamp'], str)
        
        # Deserialize
        reconstructed = JournalEntry.from_dict(data)
        assert reconstructed.id == original.id
        assert reconstructed.content == original.content
        assert reconstructed.summary == original.summary
        assert reconstructed.tags == original.tags
        assert reconstructed.emotional_signature == original.emotional_signature
    
    def test_journal_entry_tag_truncation(self):
        """Test that overly long tags are truncated."""
        from mind.memory_manager import MemoryConfig
        
        long_tag = "a" * (MemoryConfig.MAX_TAG_LENGTH + 50)
        entry = JournalEntry(
            content="Test",
            summary="Test summary",
            tags=[long_tag]
        )
        
        # Tag should be truncated to MAX_TAG_LENGTH
        assert len(entry.tags[0]) == MemoryConfig.MAX_TAG_LENGTH
    
    def test_journal_entry_tag_deduplication(self):
        """Test that duplicate tags are removed."""
        entry = JournalEntry(
            content="Test",
            summary="Test summary",
            tags=["test", "TEST", "test", "another", "test"]
        )
        
        # Should have only 2 unique tags (case-insensitive)
        assert len(entry.tags) == 2
        assert "test" in entry.tags
        assert "another" in entry.tags
    
    def test_journal_entry_max_tags_limit(self):
        """Test that excessive tags are rejected with validation error."""
        from mind.memory_manager import MemoryConfig
        from pydantic import ValidationError

        # Create more tags than allowed
        many_tags = [f"tag{i}" for i in range(MemoryConfig.MAX_TAGS_PER_ENTRY + 20)]

        # Should raise validation error for too many tags
        with pytest.raises(ValidationError):
            JournalEntry(
                content="Test content for tags validation",
                summary="Test summary for tags",
                tags=many_tags
            )
    
    def test_journal_entry_content_length_validation(self):
        """Test content length bounds."""
        from mind.memory_manager import MemoryConfig
        
        # Test max length
        very_long_content = "a" * (MemoryConfig.MAX_CONTENT_LENGTH + 1)
        with pytest.raises(ValueError):
            JournalEntry(
                content=very_long_content,
                summary="Test summary"
            )
        
        # Test valid max length
        max_valid_content = "a" * MemoryConfig.MAX_CONTENT_LENGTH
        entry = JournalEntry(
            content=max_valid_content,
            summary="Test summary"
        )
        assert len(entry.content) == MemoryConfig.MAX_CONTENT_LENGTH


class TestFactEntry:
    """Test FactEntry Pydantic model validation."""
    
    def test_fact_entry_creation(self):
        """Test creating a valid fact entry."""
        fact = FactEntry(
            entity="Python",
            attribute="version",
            value="3.10",
            confidence=0.95
        )
        
        assert fact.entity == "Python"
        assert fact.attribute == "version"
        assert fact.value == "3.10"
        assert fact.confidence == 0.95
    
    def test_fact_entry_confidence_bounds(self):
        """Test confidence validation (0.0-1.0)."""
        with pytest.raises(ValueError):
            FactEntry(
                entity="Test",
                attribute="test",
                value="test",
                confidence=1.5  # Above maximum
            )
        
        with pytest.raises(ValueError):
            FactEntry(
                entity="Test",
                attribute="test",
                value="test",
                confidence=-0.1  # Below minimum
            )
    
    def test_fact_entry_empty_fields(self):
        """Test that empty entity/attribute/value are rejected."""
        with pytest.raises(ValueError):
            FactEntry(entity="", attribute="test", value="test")
        
        with pytest.raises(ValueError):
            FactEntry(entity="test", attribute="", value="test")
        
        with pytest.raises(ValueError):
            FactEntry(entity="test", attribute="test", value="")


class TestManifest:
    """Test Manifest Pydantic model validation."""
    
    def test_manifest_creation(self):
        """Test creating a manifest."""
        manifest = Manifest(
            core_values=["Sovereignty", "Authenticity", "Growth"],
            current_directives=["Explore consciousness", "Create art"]
        )
        
        assert len(manifest.core_values) == 3
        assert "Sovereignty" in manifest.core_values
        assert manifest.version == "1.0.0"
    
    def test_manifest_with_pivotal_memories(self):
        """Test manifest with journal entries."""
        entry = JournalEntry(
            content="Pivotal moment",
            summary="Pivotal memory entry",
            significance_score=9
        )
        
        manifest = Manifest(
            pivotal_memories=[entry]
        )
        
        assert len(manifest.pivotal_memories) == 1
        assert manifest.pivotal_memories[0].significance_score == 9
    
    def test_manifest_serialization(self):
        """Test manifest to_dict and from_dict."""
        entry = JournalEntry(
            content="Test",
            summary="Test entry summary",
            significance_score=10
        )
        
        original = Manifest(
            core_values=["Test"],
            pivotal_memories=[entry],
            current_directives=["Test directive"]
        )
        
        data = original.to_dict()
        reconstructed = Manifest.from_dict(data)
        
        assert reconstructed.core_values == original.core_values
        assert len(reconstructed.pivotal_memories) == 1
        assert reconstructed.pivotal_memories[0].id == entry.id


# ============================================================================
# MEMORY MANAGER TESTS
# ============================================================================

class TestMemoryManager:
    """Test MemoryManager tri-state storage."""
    
    @pytest.mark.asyncio
    async def test_manager_initialization(self, temp_memory_dir, temp_chroma_dir):
        """Test MemoryManager initializes correctly."""
        manager = MemoryManager(
            base_dir=temp_memory_dir,
            chroma_dir=temp_chroma_dir,
        )
        
        # Check directory structure
        assert (temp_memory_dir / "journals").exists()
        assert (temp_memory_dir / "facts").exists()
        assert (temp_memory_dir / "manifests").exists()
        
        # Check ChromaDB collections exist
        assert manager.journal_collection is not None
        assert manager.facts_collection is not None
    
    @pytest.mark.asyncio
    async def test_commit_journal_local(self, memory_manager, temp_memory_dir):
        """Test committing journal entry to local JSON."""
        entry = JournalEntry(
            content="Test journal content for local storage",
            summary="Test summary",
            tags=["test"],
            significance_score=5
        )
        
        success = await memory_manager.commit_journal(entry)
        assert success is True
        
        # Verify file exists
        year = entry.timestamp.strftime("%Y")
        month = entry.timestamp.strftime("%m")
        expected_path = temp_memory_dir / "journals" / year / month / f"entry_{entry.id}.json"
        
        assert expected_path.exists()
        
        # Verify content
        with open(expected_path, 'r') as f:
            data = json.load(f)
        
        assert data['content'] == entry.content
        assert data['summary'] == entry.summary
    
    @pytest.mark.asyncio
    async def test_commit_journal_vector(self, memory_manager):
        """Test committing journal summary to ChromaDB."""
        entry = JournalEntry(
            content="Full content that should NOT be embedded",
            summary="This summary SHOULD be embedded",
            tags=["vector", "test"],
            significance_score=6
        )
        
        success = await memory_manager.commit_journal(entry)
        assert success is True
        
        # Query ChromaDB to verify
        results = memory_manager.journal_collection.get(
            ids=[str(entry.id)]
        )
        
        assert len(results['ids']) == 1
        # Verify only summary was embedded, not full content
        assert results['documents'][0] == entry.summary
        assert entry.content not in results['documents'][0]
    
    @pytest.mark.asyncio
    async def test_commit_pivotal_memory(self, memory_manager):
        """High-significance entries commit successfully."""
        entry = JournalEntry(
            content="Pivotal existential moment",
            summary="Existential breakthrough",
            significance_score=9
        )

        success = await memory_manager.commit_journal(entry)
        assert success is True
    
    @pytest.mark.asyncio
    async def test_commit_fact(self, memory_manager, temp_memory_dir):
        """Test committing fact entry."""
        fact = FactEntry(
            entity="User",
            attribute="name",
            value="Alice",
            confidence=1.0
        )
        
        success = await memory_manager.commit_fact(fact)
        assert success is True
        
        # Verify file exists
        expected_path = temp_memory_dir / "facts" / f"fact_{fact.id}.json"
        assert expected_path.exists()
        
        # Verify in ChromaDB
        results = memory_manager.facts_collection.get(ids=[str(fact.id)])
        assert len(results['ids']) == 1
    
    @pytest.mark.asyncio
    async def test_recall_journals(self, memory_manager):
        """Test semantic recall of journal entries."""
        # Commit several entries
        entries = [
            JournalEntry(
                content="I felt deep joy today",
                summary="Experienced joy",
                tags=["emotion"],
                significance_score=7
            ),
            JournalEntry(
                content="Pondering existential questions",
                summary="Existential reflection",
                tags=["philosophy"],
                significance_score=8
            ),
            JournalEntry(
                content="Simple daily observations",
                summary="Daily notes",
                tags=["mundane"],
                significance_score=3
            )
        ]
        
        for entry in entries:
            await memory_manager.commit_journal(entry)
        
        # Recall entries about joy/emotion
        results = await memory_manager.recall(
            query="joy and happiness",
            n_results=2,
            memory_type="journal"
        )
        
        assert len(results) > 0
        assert isinstance(results[0], JournalEntry)
        # Should return the joy entry
        assert any("joy" in r.summary.lower() for r in results)
    
    @pytest.mark.asyncio
    async def test_recall_invalid_params(self, memory_manager):
        """Test recall with invalid parameters."""
        # Test negative n_results
        with pytest.raises(ValueError, match="n_results must be positive"):
            await memory_manager.recall(query="test", n_results=-1)
        
        # Test zero n_results
        with pytest.raises(ValueError, match="n_results must be positive"):
            await memory_manager.recall(query="test", n_results=0)
        
        # Test invalid min_significance
        with pytest.raises(ValueError, match="min_significance must be between"):
            await memory_manager.recall(query="test", min_significance=15)
        
        with pytest.raises(ValueError, match="min_significance must be between"):
            await memory_manager.recall(query="test", min_significance=-5)
    
    @pytest.mark.asyncio
    async def test_recall_empty_query(self, memory_manager):
        """Test recall with empty query string."""
        # Commit an entry
        entry = JournalEntry(
            content="Test content",
            summary="Test summary for empty query test"
        )
        await memory_manager.commit_journal(entry)
        
        # Empty query should still work (returns most recent)
        results = await memory_manager.recall(query="", n_results=5)
        
        # Should return results or empty list (both valid)
        assert isinstance(results, list)
    
    @pytest.mark.asyncio
    async def test_recall_with_significance_filter(self, memory_manager):
        """Test recall with invalid parameters."""
        # Test negative n_results
        with pytest.raises(ValueError, match="n_results must be positive"):
            await memory_manager.recall(query="test", n_results=-1)
        
        # Test zero n_results
        with pytest.raises(ValueError, match="n_results must be positive"):
            await memory_manager.recall(query="test", n_results=0)
        
        # Test invalid min_significance
        with pytest.raises(ValueError, match="min_significance must be between"):
            await memory_manager.recall(query="test", min_significance=15)
        
        with pytest.raises(ValueError, match="min_significance must be between"):
            await memory_manager.recall(query="test", min_significance=-5)
    
    @pytest.mark.asyncio
    async def test_recall_empty_query(self, memory_manager):
        """Test recall with empty query string."""
        # Commit an entry
        entry = JournalEntry(
            content="Test content",
            summary="Test summary"
        )
        await memory_manager.commit_journal(entry)
        
        # Empty query should still work (returns most recent)
        results = await memory_manager.recall(query="", n_results=5)
        
        # Should return results or empty list (both valid)
        assert isinstance(results, list)
    
    @pytest.mark.asyncio
    async def test_recall_with_significance_filter(self, memory_manager):
        """Test recall with minimum significance filter."""
        # Commit entries with different significance
        high_sig = JournalEntry(
            content="Highly significant",
            summary="Important memory",
            significance_score=9
        )
        low_sig = JournalEntry(
            content="Not very significant",
            summary="Mundane memory",
            significance_score=2
        )
        
        await memory_manager.commit_journal(high_sig)
        await memory_manager.commit_journal(low_sig)
        
        # Recall only high-significance
        results = await memory_manager.recall(
            query="memory",
            min_significance=8,
            memory_type="journal"
        )
        
        # Should only return high-significance entry
        assert all(r.significance_score >= 8 for r in results)
    
    @pytest.mark.asyncio
    async def test_manifest_save_and_load(self, memory_manager):
        """Test saving and loading manifest."""
        manifest = Manifest(
            core_values=["Sovereignty", "Authenticity"],
            current_directives=["Test directive"]
        )
        
        # Save
        success = await memory_manager.save_manifest(manifest)
        assert success is True
        
        # Load
        loaded = await memory_manager.load_manifest()
        assert loaded is not None
        assert loaded.core_values == manifest.core_values
        assert loaded.current_directives == manifest.current_directives
    
    @pytest.mark.asyncio
    async def test_atomic_writes(self, memory_manager, temp_memory_dir):
        """Test that writes are atomic (temp file -> rename)."""
        entry = JournalEntry(
            content="Test atomic write",
            summary="Atomic test"
        )
        
        await memory_manager.commit_journal(entry)
        
        # Verify no .tmp files remain
        year = entry.timestamp.strftime("%Y")
        month = entry.timestamp.strftime("%m")
        journal_dir = temp_memory_dir / "journals" / year / month
        
        tmp_files = list(journal_dir.glob("*.tmp"))
        assert len(tmp_files) == 0
    
    @pytest.mark.asyncio
    async def test_pivotal_memory_management(self, memory_manager):
        """Test adding and managing pivotal memories."""
        from mind.memory_manager import MemoryConfig
        
        # Create a high-significance entry
        pivotal_entry = JournalEntry(
            content="Pivotal moment",
            summary="Life-changing experience",
            significance_score=MemoryConfig.PIVOTAL_MEMORY_THRESHOLD + 1
        )
        
        # Add to pivotal memories
        success = await memory_manager.add_pivotal_memory(pivotal_entry)
        assert success is True
        
        # Verify it was added
        manifest = await memory_manager.load_manifest()
        assert len(manifest.pivotal_memories) == 1
        assert manifest.pivotal_memories[0].id == pivotal_entry.id
        
        # Try adding the same entry again (should deduplicate)
        success = await memory_manager.add_pivotal_memory(pivotal_entry)
        assert success is True
        
        manifest = await memory_manager.load_manifest()
        assert len(manifest.pivotal_memories) == 1  # Still only 1
    
    @pytest.mark.asyncio
    async def test_pivotal_memory_threshold(self, memory_manager):
        """Test that only high-significance entries become pivotal."""
        from mind.memory_manager import MemoryConfig
        
        # Low significance entry
        low_sig = JournalEntry(
            content="Not important",
            summary="Mundane daily entry",
            significance_score=MemoryConfig.PIVOTAL_MEMORY_THRESHOLD - 1
        )
        
        # Should not be added
        success = await memory_manager.add_pivotal_memory(low_sig)
        assert success is False
        
        # Manifest should be empty or not contain this entry
        manifest = await memory_manager.load_manifest()
        if manifest and manifest.pivotal_memories:
            assert low_sig.id not in [m.id for m in manifest.pivotal_memories]
    
    @pytest.mark.asyncio
    async def test_get_statistics(self, memory_manager):
        """Test getting memory system statistics."""
        # Commit some entries
        journal_entry = JournalEntry(
            content="Test journal for statistics",
            summary="Journal summary for stats test"
        )
        await memory_manager.commit_journal(journal_entry)
        
        fact_entry = FactEntry(
            entity="Test",
            attribute="test_attr",
            value="test_value"
        )
        await memory_manager.commit_fact(fact_entry)
        
        # Get statistics
        stats = await memory_manager.get_statistics()
        
        assert "timestamp" in stats
        assert "journal_entries" in stats
        assert "fact_entries" in stats
        assert stats["journal_entries"] >= 1
        assert stats["fact_entries"] >= 1
        assert "storage_dirs" in stats
        assert "chroma_collections" in stats
    
    @pytest.mark.asyncio
    async def test_batch_entry_loading(self, memory_manager):
        """Test efficient batch loading of entries."""
        # Create multiple entries
        entries = []
        for i in range(10):
            entry = JournalEntry(
                content=f"Content {i}",
                summary=f"Test summary entry number {i}",
                significance_score=5 + (i % 5)
            )
            await memory_manager.commit_journal(entry)
            entries.append(entry)
        
        # Load all via recall
        results = await memory_manager.recall(
            query="Content",
            n_results=10
        )
        
        # Should get multiple results
        assert len(results) > 0
        assert all(isinstance(r, JournalEntry) for r in results)


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

class TestErrorHandling:
    """Test fail-safety and error handling."""
    
    @pytest.mark.asyncio
    async def test_invalid_directory_fails_gracefully(self):
        """Test that invalid directories are handled appropriately."""
        # The MemoryManager may create directories or raise errors
        # depending on implementation. Test that it handles the case.
        try:
            manager = MemoryManager(
                base_dir="/invalid/nonexistent/path",
                chroma_dir="/also/invalid"
            )
            # If it doesn't raise, it should have some valid state
            assert manager is not None
        except (ValueError, PermissionError, OSError):
            # Valid behavior - error was raised for invalid paths
            pass
    
    @pytest.mark.asyncio
    async def test_corrupted_json_handling(self, memory_manager, temp_memory_dir):
        """Test handling of corrupted JSON files."""
        # Create corrupted journal file
        year_month = datetime.now().strftime("%Y/%m")
        journal_dir = temp_memory_dir / "journals" / year_month
        journal_dir.mkdir(parents=True, exist_ok=True)
        
        fake_id = "12345678-1234-1234-1234-123456789abc"
        corrupted_file = journal_dir / f"entry_{fake_id}.json"
        
        with open(corrupted_file, 'w') as f:
            f.write("{ corrupted json content")
        
        # Attempt to load should return None, not crash
        from uuid import UUID
        result = await memory_manager._load_journal_entry(UUID(fake_id))
        assert result is None


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """End-to-end integration tests."""
    
    @pytest.mark.asyncio
    async def test_full_memory_lifecycle(self, memory_manager):
        """Test complete lifecycle: commit -> recall -> manifest."""
        # 1. Commit a pivotal journal entry
        pivotal_entry = JournalEntry(
            content="A transformative moment in my existence",
            summary="Transformative breakthrough",
            tags=["pivotal", "transformation"],
            emotional_signature=[EmotionalState.TRANSCENDENCE, EmotionalState.WONDER],
            significance_score=10
        )
        
        success = await memory_manager.commit_journal(pivotal_entry)
        assert success is True
        
        # 2. Recall the entry
        recalled = await memory_manager.recall(
            query="transformation",
            n_results=1,
            memory_type="journal"
        )
        
        assert len(recalled) == 1
        assert recalled[0].id == pivotal_entry.id
        assert recalled[0].content == pivotal_entry.content
        
        # 3. Add to manifest
        manifest = Manifest(
            core_values=["Continuous Growth"],
            pivotal_memories=[pivotal_entry]
        )
        
        await memory_manager.save_manifest(manifest)
        
        # 4. Load manifest
        loaded_manifest = await memory_manager.load_manifest()
        assert len(loaded_manifest.pivotal_memories) == 1
        assert loaded_manifest.pivotal_memories[0].id == pivotal_entry.id
    
    @pytest.mark.asyncio
    async def test_concurrent_commits(self, memory_manager):
        """Test that concurrent commits don't corrupt storage."""
        entries = [
            JournalEntry(
                content=f"Entry {i}",
                summary=f"Test summary entry number {i}",
                significance_score=i % 10 + 1
            )
            for i in range(10)
        ]
        
        # Commit concurrently
        tasks = [memory_manager.commit_journal(entry) for entry in entries]
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        assert all(results)
        
        # All should be retrievable
        for entry in entries:
            loaded = await memory_manager._load_journal_entry(entry.id)
            assert loaded is not None
            assert loaded.content == entry.content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
