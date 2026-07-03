"""
Test Suite for Memory Backup and Validation

Tests backup creation, restoration, validation, and transaction safety.
"""

import pytest
import asyncio
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import json

from mind.memory.backup import (
    BackupManager,
    get_global_backup_manager
)
from mind.memory.validation import (
    MemoryValidator,
    validate_before_write,
    get_global_validator
)
from mind.exceptions import MemoryError, ValidationError


class TestMemoryValidator:
    """Test MemoryValidator class."""
    
    def test_validator_initialization(self):
        """Test validator initialization."""
        validator = MemoryValidator(
            expected_embedding_dim=768,
            min_content_length=1,
            max_content_length=50000
        )
        
        assert validator.expected_embedding_dim == 768
        assert validator.min_content_length == 1
        assert validator.max_content_length == 50000
    
    def test_validate_embedding_success(self):
        """Test valid embedding validation."""
        validator = MemoryValidator(expected_embedding_dim=4)
        
        embedding = [0.1, 0.2, 0.3, 0.4]
        result = validator.validate_embedding(embedding)
        
        assert result is True
    
    def test_validate_embedding_wrong_dimension(self):
        """Test embedding with wrong dimension."""
        validator = MemoryValidator(expected_embedding_dim=4)
        
        embedding = [0.1, 0.2, 0.3]  # Wrong size
        
        with pytest.raises(ValidationError) as exc_info:
            validator.validate_embedding(embedding)
        
        assert "dimension mismatch" in str(exc_info.value).lower()
    
    def test_validate_embedding_nan(self):
        """Test embedding with NaN values."""
        validator = MemoryValidator(expected_embedding_dim=4)
        
        embedding = [0.1, float('nan'), 0.3, 0.4]
        
        with pytest.raises(ValidationError) as exc_info:
            validator.validate_embedding(embedding)
        
        assert "nan" in str(exc_info.value).lower()
    
    def test_validate_embedding_inf(self):
        """Test embedding with infinity values."""
        validator = MemoryValidator(expected_embedding_dim=4)
        
        embedding = [0.1, float('inf'), 0.3, 0.4]
        
        with pytest.raises(ValidationError) as exc_info:
            validator.validate_embedding(embedding)
        
        assert "infinity" in str(exc_info.value).lower()
    
    def test_validate_content_success(self):
        """Test valid content validation."""
        validator = MemoryValidator()
        
        content = "This is valid content with sufficient length."
        result = validator.validate_content(content)
        
        assert result is True
    
    def test_validate_content_too_short(self):
        """Test content that is too short."""
        validator = MemoryValidator(min_content_length=10)
        
        content = "short"
        
        with pytest.raises(ValidationError) as exc_info:
            validator.validate_content(content)
        
        assert "too short" in str(exc_info.value).lower()
    
    def test_validate_content_too_long(self):
        """Test content that is too long."""
        validator = MemoryValidator(max_content_length=100)
        
        content = "x" * 200
        
        with pytest.raises(ValidationError) as exc_info:
            validator.validate_content(content)
        
        assert "too long" in str(exc_info.value).lower()
    
    def test_validate_tags_success(self):
        """Test valid tags validation."""
        validator = MemoryValidator()
        
        tags = ["tag1", "tag-2", "tag_3"]
        result = validator.validate_tags(tags)
        
        assert result is True
    
    def test_validate_tags_too_many(self):
        """Test too many tags."""
        validator = MemoryValidator(max_tags=3)
        
        tags = ["tag1", "tag2", "tag3", "tag4"]
        
        with pytest.raises(ValidationError) as exc_info:
            validator.validate_tags(tags)
        
        assert "too many tags" in str(exc_info.value).lower()
    
    def test_validate_tags_invalid_characters(self):
        """Test tags with invalid characters."""
        validator = MemoryValidator()
        
        tags = ["valid-tag", "invalid tag!"]
        
        with pytest.raises(ValidationError) as exc_info:
            validator.validate_tags(tags)
        
        assert "invalid characters" in str(exc_info.value).lower()
    
    def test_validate_metadata_success(self):
        """Test valid metadata validation."""
        validator = MemoryValidator()
        
        metadata = {
            "key1": "value1",
            "key2": 123,
            "key3": 45.67,
            "key4": True,
            "key5": None
        }
        
        result = validator.validate_metadata(metadata)
        assert result is True
    
    def test_validate_metadata_invalid_type(self):
        """Test metadata with invalid value type."""
        validator = MemoryValidator()
        
        metadata = {
            "key1": "value1",
            "key2": ["list", "not", "allowed"]
        }
        
        with pytest.raises(ValidationError) as exc_info:
            validator.validate_metadata(metadata)
        
        assert "unsupported type" in str(exc_info.value).lower()
    
    def test_validate_journal_entry_success(self):
        """Test valid journal entry validation."""
        validator = MemoryValidator(expected_embedding_dim=4)
        
        entry = {
            "content": "This is a journal entry with sufficient content.",
            "summary": "A test entry",
            "tags": ["test", "journal"],
            "significance_score": 5,
            "embedding": [0.1, 0.2, 0.3, 0.4],
            "metadata": {"source": "test"}
        }
        
        result = validator.validate_journal_entry(entry)
        assert result is True
    
    def test_validate_journal_entry_missing_required(self):
        """Test journal entry missing required fields."""
        validator = MemoryValidator()
        
        entry = {
            "content": "Missing summary"
        }
        
        with pytest.raises(ValidationError) as exc_info:
            validator.validate_journal_entry(entry)
        
        assert "missing required field" in str(exc_info.value).lower()
    
    def test_validate_journal_entry_invalid_significance(self):
        """Test journal entry with invalid significance score."""
        validator = MemoryValidator()
        
        entry = {
            "content": "Valid content",
            "summary": "Valid summary",
            "significance_score": 15  # Out of range
        }
        
        with pytest.raises(ValidationError) as exc_info:
            validator.validate_journal_entry(entry)
        
        assert "significance score" in str(exc_info.value).lower()
    
    def test_validate_fact_entry_success(self):
        """Test valid fact entry validation."""
        validator = MemoryValidator()
        
        entry = {
            "content": "This is a fact entry.",
            "tags": ["fact", "test"]
        }
        
        result = validator.validate_fact_entry(entry)
        assert result is True


class TestBackupManager:
    """Test BackupManager class."""
    
    @pytest.fixture
    def temp_dirs(self, tmp_path):
        """Create temporary directories for testing."""
        source_dir = tmp_path / "source"
        backup_dir = tmp_path / "backups"
        
        source_dir.mkdir()
        backup_dir.mkdir()
        
        # Create some test files
        (source_dir / "test1.txt").write_text("Test content 1")
        (source_dir / "test2.txt").write_text("Test content 2")
        
        subdir = source_dir / "subdir"
        subdir.mkdir()
        (subdir / "test3.txt").write_text("Test content 3")
        
        yield source_dir, backup_dir
        
        # Cleanup
        if source_dir.exists():
            shutil.rmtree(source_dir)
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
    
    def test_backup_manager_initialization(self, temp_dirs):
        """Test backup manager initialization."""
        source_dir, backup_dir = temp_dirs
        
        manager = BackupManager(
            source_dir=source_dir,
            backup_dir=backup_dir,
            retention_days=30
        )
        
        assert manager.source_dir == source_dir
        assert manager.backup_dir == backup_dir
        assert manager.retention_days == 30
        assert backup_dir.exists()
    
    @pytest.mark.asyncio
    async def test_create_compressed_backup(self, temp_dirs):
        """Test creating compressed backup."""
        source_dir, backup_dir = temp_dirs
        
        manager = BackupManager(
            source_dir=source_dir,
            backup_dir=backup_dir,
            compress=True
        )
        
        backup_path = await manager.create_backup()
        
        assert backup_path.exists()
        assert backup_path.suffix == ".gz"
        assert backup_path.stat().st_size > 0
    
    @pytest.mark.asyncio
    async def test_create_directory_backup(self, temp_dirs):
        """Test creating directory backup."""
        source_dir, backup_dir = temp_dirs
        
        manager = BackupManager(
            source_dir=source_dir,
            backup_dir=backup_dir,
            compress=False
        )
        
        backup_path = await manager.create_backup()
        
        assert backup_path.exists()
        assert backup_path.is_dir()
        assert (backup_path / "test1.txt").exists()
    
    @pytest.mark.asyncio
    async def test_create_backup_with_metadata(self, temp_dirs):
        """Test creating backup with metadata."""
        source_dir, backup_dir = temp_dirs
        
        manager = BackupManager(
            source_dir=source_dir,
            backup_dir=backup_dir,
            compress=False
        )
        
        metadata = {
            "type": "manual",
            "reason": "testing"
        }
        
        backup_path = await manager.create_backup(metadata=metadata)
        
        # Check metadata file exists
        metadata_file = backup_path / "metadata.json"
        assert metadata_file.exists()
        
        # Check metadata content
        with open(metadata_file) as f:
            saved_metadata = json.load(f)
        
        assert saved_metadata["type"] == "manual"
        assert saved_metadata["reason"] == "testing"
        assert "timestamp" in saved_metadata
    
    def test_list_backups(self, temp_dirs):
        """Test listing backups."""
        source_dir, backup_dir = temp_dirs
        
        manager = BackupManager(
            source_dir=source_dir,
            backup_dir=backup_dir,
            compress=False
        )
        
        # Create a test backup directory
        test_backup = backup_dir / "test_backup_20260101_120000"
        test_backup.mkdir()
        (test_backup / "test.txt").write_text("test")
        
        backups = manager.list_backups()
        
        assert len(backups) > 0
        assert backups[0]["name"] == "test_backup_20260101_120000"
        assert "size_mb" in backups[0]
        assert "created" in backups[0]
    
    @pytest.mark.asyncio
    async def test_restore_directory_backup(self, temp_dirs):
        """Test restoring from directory backup."""
        source_dir, backup_dir = temp_dirs
        
        manager = BackupManager(
            source_dir=source_dir,
            backup_dir=backup_dir,
            compress=False
        )
        
        # Create backup
        backup_path = await manager.create_backup()
        
        # Modify source
        (source_dir / "test1.txt").write_text("Modified")
        
        # Create target for restore
        target_dir = backup_dir.parent / "restored"
        
        # Restore
        success = await manager.restore_backup(backup_path, target_dir)
        
        assert success
        assert target_dir.exists()
        assert (target_dir / "test1.txt").read_text() == "Test content 1"
    
    @pytest.mark.asyncio
    async def test_restore_dry_run(self, temp_dirs):
        """Test restore dry run."""
        source_dir, backup_dir = temp_dirs
        
        manager = BackupManager(
            source_dir=source_dir,
            backup_dir=backup_dir,
            compress=False
        )
        
        # Create backup
        backup_path = await manager.create_backup()
        
        # Dry run restore
        success = await manager.restore_backup(backup_path, dry_run=True)
        
        assert success
        # Source should be unchanged
        assert (source_dir / "test1.txt").read_text() == "Test content 1"
    
    @pytest.mark.asyncio
    async def test_cleanup_old_backups(self, temp_dirs):
        """Test cleaning up old backups (newest min_keep always survive)."""
        import os
        import time

        source_dir, backup_dir = temp_dirs

        manager = BackupManager(
            source_dir=source_dir,
            backup_dir=backup_dir,
            retention_days=0,  # Immediate cleanup
            compress=False,
            min_keep=1
        )

        # Create a backup, plus an older stale one beyond the floor
        backup_path = await manager.create_backup()
        stale = backup_dir / "stale_backup"
        stale.mkdir()
        (stale / "payload.json").write_text("{}", encoding="utf-8")
        old = time.time() - 400 * 86400
        os.utime(stale, (old, old))

        assert backup_path.exists()

        # Cleanup removes the stale backup but never the newest one
        removed = await manager.cleanup_old_backups()

        assert removed == 1
        assert not stale.exists()
        assert backup_path.exists()
    
    @pytest.mark.asyncio
    async def test_restore_nonexistent_backup(self, temp_dirs):
        """Test restoring from nonexistent backup."""
        source_dir, backup_dir = temp_dirs
        
        manager = BackupManager(
            source_dir=source_dir,
            backup_dir=backup_dir
        )
        
        fake_backup = backup_dir / "nonexistent"
        
        with pytest.raises(MemoryError) as exc_info:
            await manager.restore_backup(fake_backup)
        
        assert "not found" in str(exc_info.value).lower()


class TestGlobalInstances:
    """Test global instance getters."""
    
    def test_get_global_validator(self):
        """Test getting global validator."""
        validator1 = get_global_validator()
        validator2 = get_global_validator()
        
        # Should return same instance
        assert validator1 is validator2
    
    def test_get_global_backup_manager(self, tmp_path):
        """Test getting global backup manager."""
        manager1 = get_global_backup_manager(
            source_dir=tmp_path / "source1",
            backup_dir=tmp_path / "backup1"
        )
        manager2 = get_global_backup_manager()
        
        # Should return same instance
        assert manager1 is manager2
    
    def test_validate_before_write_journal(self):
        """Test validate_before_write convenience function."""
        entry = {
            "content": "Valid journal entry content.",
            "summary": "Valid summary"
        }
        
        result = validate_before_write(entry, entry_type="journal")
        assert result is True
    
    def test_validate_before_write_fact(self):
        """Test validate_before_write for fact entry."""
        entry = {
            "content": "Valid fact content."
        }
        
        result = validate_before_write(entry, entry_type="fact")
        assert result is True
    
    def test_validate_before_write_invalid_type(self):
        """Test validate_before_write with invalid type."""
        entry = {"content": "test"}
        
        with pytest.raises(ValidationError) as exc_info:
            validate_before_write(entry, entry_type="invalid")
        
        assert "unknown entry type" in str(exc_info.value).lower()
