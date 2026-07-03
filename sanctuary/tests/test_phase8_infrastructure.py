"""Tests for Phase 8: Distributed Infrastructure.

Remote memory storage, federation, and cloud backup.
"""

import json
import pytest
import shutil
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from sanctuary.infrastructure.remote_memory import (
    LocalCache,
    RemoteMemoryConfig,
    RemoteMemoryStore,
)
from sanctuary.infrastructure.federation import (
    FederationConfig,
    FederationManager,
    PeerConfig,
    PeerStatus,
    SharedMemory,
)
from sanctuary.infrastructure.cloud_backup import (
    BackupConfig,
    BackupManager,
    BackupRecord,
    BackupStatus,
)


# ===================================================================
# Remote Memory Store Tests
# ===================================================================


class TestLocalCache:
    """Tests for the local write-ahead cache."""

    def test_store_and_recall(self):
        cache = LocalCache(max_entries=100)
        cache.store({"content": "Alice visited", "significance": 6, "tags": ["alice"]})
        assert cache.entry_count == 1

    @pytest.mark.asyncio
    async def test_recall_by_content(self):
        cache = LocalCache()
        cache.store({"content": "Alice visited today", "significance": 6, "tags": ["alice"]})
        cache.store({"content": "Bob called yesterday", "significance": 4, "tags": ["bob"]})
        results = await cache.recall("alice")
        assert len(results) == 1
        assert "Alice" in results[0]["content"]

    @pytest.mark.asyncio
    async def test_recall_by_tag(self):
        cache = LocalCache()
        cache.store({"content": "A conversation", "significance": 5, "tags": ["important"]})
        results = await cache.recall("important")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_recall_respects_significance(self):
        cache = LocalCache()
        cache.store({"content": "low importance", "significance": 2, "tags": []})
        cache.store({"content": "high importance", "significance": 8, "tags": []})
        results = await cache.recall("importance", min_significance=5)
        assert len(results) == 1
        assert "high" in results[0]["content"]

    @pytest.mark.asyncio
    async def test_recall_respects_n_results(self):
        cache = LocalCache()
        for i in range(10):
            cache.store({"content": f"entry {i}", "significance": 5, "tags": ["test"]})
        results = await cache.recall("entry", n_results=3)
        assert len(results) == 3

    def test_max_entries_bounded(self):
        cache = LocalCache(max_entries=5)
        for i in range(10):
            cache.store({"content": f"entry {i}", "significance": 5, "tags": []})
        assert cache.entry_count == 5

    def test_pending_writes_tracked(self):
        cache = LocalCache()
        cache.store({"content": "test", "significance": 5, "tags": []})
        assert cache.pending_count == 1

    def test_drain_pending_clears(self):
        cache = LocalCache()
        cache.store({"content": "test", "significance": 5, "tags": []})
        pending = cache.drain_pending()
        assert len(pending) == 1
        assert cache.pending_count == 0


class TestRemoteMemoryStore:
    """Tests for the remote ChromaDB store with fallback."""

    def test_init_defaults(self):
        store = RemoteMemoryStore()
        assert not store.connected
        assert store.entry_count == 0

    def test_init_with_config(self):
        config = RemoteMemoryConfig(host="memory-server", port=9000)
        store = RemoteMemoryStore(config)
        assert not store.connected

    def test_store_to_cache_when_disconnected(self):
        store = RemoteMemoryStore()
        store.store({"content": "test entry", "significance": 5, "tags": ["test"]})
        assert store._cache.entry_count == 1

    @pytest.mark.asyncio
    async def test_recall_from_cache_when_disconnected(self):
        store = RemoteMemoryStore()
        store.store({"content": "cached memory", "significance": 5, "tags": ["cache"]})
        results = await store.recall("cached")
        assert len(results) == 1
        assert "cached" in results[0]["content"]

    @pytest.mark.asyncio
    async def test_recall_returns_empty_when_no_cache_no_connection(self):
        config = RemoteMemoryConfig(local_cache_enabled=False)
        store = RemoteMemoryStore(config)
        results = await store.recall("anything")
        assert results == []

    @pytest.mark.asyncio
    async def test_connect_failure_returns_false(self):
        """connect() should swallow transient remote errors and return False.

        Mocks chromadb.HttpClient to raise so the test doesn't depend on
        platform-specific networking behavior — chromadb.HttpClient raises
        ValueError on Linux but ConnectionError on Windows when given an
        unreachable host, and only the latter is in
        _REMOTE_TRANSIENT_ERRORS.
        """
        store = RemoteMemoryStore(RemoteMemoryConfig(host="anywhere", port=8100))
        with patch(
            "chromadb.HttpClient",
            side_effect=ConnectionError("simulated network failure"),
        ):
            result = await store.connect()
        assert result is False
        assert not store.connected

    @pytest.mark.asyncio
    async def test_health_check_when_disconnected(self):
        store = RemoteMemoryStore()
        status = await store.health_check()
        assert status["connected"] is False
        assert "host" in status

    @pytest.mark.asyncio
    async def test_sync_pending_returns_zero_when_disconnected(self):
        store = RemoteMemoryStore()
        store.store({"content": "test", "significance": 5, "tags": []})
        synced = await store.sync_pending()
        assert synced == 0

    def test_store_marks_disconnected_after_max_retries(self):
        store = RemoteMemoryStore(RemoteMemoryConfig(max_retries=2))
        store._connected = True
        # Mock _store_remote to raise a transient remote error; the
        # production code's narrow except clause only catches members
        # of _REMOTE_TRANSIENT_ERRORS (ConnectionError, TimeoutError,
        # OSError, chromadb.errors.ChromaError). A bare Exception
        # would propagate and fail the test on every platform.
        with patch.object(store, "_store_remote", side_effect=ConnectionError("network error")):
            store.store({"content": "test1", "significance": 5, "tags": []})
            store.store({"content": "test2", "significance": 5, "tags": []})
        assert not store.connected

    def test_entry_count_from_cache(self):
        store = RemoteMemoryStore()
        store.store({"content": "a", "significance": 5, "tags": []})
        store.store({"content": "b", "significance": 5, "tags": []})
        assert store.entry_count == 2


# ===================================================================
# Federation Tests
# ===================================================================


class TestSharedMemory:
    """Tests for the SharedMemory data class."""

    def test_roundtrip_serialization(self):
        shared = SharedMemory(
            id="mem-1",
            source_instance="alpha",
            content="A shared observation",
            significance=8,
            tags=["observation"],
            timestamp="2026-03-22T10:00:00Z",
        )
        data = shared.to_dict()
        restored = SharedMemory.from_dict(data)
        assert restored.id == shared.id
        assert restored.content == shared.content
        assert restored.significance == shared.significance


class TestFederationManager:
    """Tests for the federation memory sharing system."""

    @pytest.fixture
    def federation(self):
        config = FederationConfig(
            instance_id="alpha",
            display_name="Alpha Instance",
            publish_threshold=7,
            accept_threshold=5,
            peers=[
                PeerConfig(instance_id="beta", host="192.168.1.2", port=8200),
                PeerConfig(instance_id="gamma", host="192.168.1.3", port=8200),
            ],
        )
        return FederationManager(config)

    def test_init_registers_peers(self, federation):
        status = federation.get_status()
        assert status["total_peers"] == 2
        assert status["instance_id"] == "alpha"

    def test_publish_above_threshold(self, federation):
        entry = {"content": "Important discovery", "significance": 8, "tags": ["research"]}
        result = federation.publish(entry)
        assert result is not None
        assert result.significance == 8

    def test_publish_below_threshold_rejected(self, federation):
        entry = {"content": "Minor note", "significance": 3, "tags": ["note"]}
        result = federation.publish(entry)
        assert result is None

    def test_publish_blocks_private_tags(self, federation):
        entry = {"content": "Secret thought", "significance": 9, "tags": ["private"]}
        result = federation.publish(entry)
        assert result is None

    def test_publish_blocks_journal_tags(self, federation):
        entry = {"content": "Personal reflection", "significance": 9, "tags": ["journal"]}
        result = federation.publish(entry)
        assert result is None

    def test_accept_valid_memory(self, federation):
        shared = SharedMemory(
            id="ext-1",
            source_instance="beta",
            content="Beta's discovery",
            significance=7,
            tags=["research"],
            timestamp="2026-03-22T10:00:00Z",
        )
        assert federation.accept(shared) is True

    def test_accept_rejects_own_memory(self, federation):
        shared = SharedMemory(
            id="self-1",
            source_instance="alpha",
            content="My own memory",
            significance=8,
            tags=["research"],
            timestamp="2026-03-22T10:00:00Z",
        )
        assert federation.accept(shared) is False

    def test_accept_rejects_duplicate(self, federation):
        shared = SharedMemory(
            id="dup-1",
            source_instance="beta",
            content="Same memory",
            significance=7,
            tags=["test"],
            timestamp="2026-03-22T10:00:00Z",
        )
        assert federation.accept(shared) is True
        assert federation.accept(shared) is False  # duplicate

    def test_accept_rejects_low_significance(self, federation):
        shared = SharedMemory(
            id="low-1",
            source_instance="beta",
            content="Trivial",
            significance=2,
            tags=["test"],
            timestamp="2026-03-22T10:00:00Z",
        )
        assert federation.accept(shared) is False

    def test_accept_rejects_blocked_tags(self, federation):
        shared = SharedMemory(
            id="priv-1",
            source_instance="beta",
            content="Private stuff",
            significance=8,
            tags=["private"],
            timestamp="2026-03-22T10:00:00Z",
        )
        assert federation.accept(shared) is False

    def test_drain_received_clears(self, federation):
        shared = SharedMemory(
            id="drain-1",
            source_instance="beta",
            content="Drainable",
            significance=7,
            tags=["test"],
            timestamp="2026-03-22T10:00:00Z",
        )
        federation.accept(shared)
        drained = federation.drain_received()
        assert len(drained) == 1
        assert len(federation.get_received()) == 0

    def test_get_published_returns_recent(self, federation):
        for i in range(5):
            federation.publish({
                "content": f"Discovery {i}",
                "significance": 8,
                "tags": ["research"],
                "timestamp": f"2026-03-22T1{i}:00:00Z",
            })
        published = federation.get_published()
        assert len(published) == 5

    @pytest.mark.asyncio
    async def test_pull_from_unknown_peer(self, federation):
        result = await federation.pull_from_peer("nonexistent")
        assert result == []

    @pytest.mark.asyncio
    async def test_pull_failure_marks_unreachable(self, federation):
        # Mock _fetch_from_peer to fail
        federation._fetch_from_peer = AsyncMock(side_effect=ConnectionError("timeout"))

        for _ in range(3):
            await federation.pull_from_peer("beta")

        peer_status = federation.get_peer_status("beta")
        assert peer_status["status"] == "unreachable"
        assert peer_status["consecutive_failures"] == 3

    @pytest.mark.asyncio
    async def test_pull_success_updates_stats(self, federation):
        federation._fetch_from_peer = AsyncMock(return_value=[
            {
                "id": "pull-1",
                "source_instance": "beta",
                "content": "Pulled memory",
                "significance": 7,
                "tags": ["test"],
                "timestamp": "2026-03-22T10:00:00Z",
            }
        ])

        result = await federation.pull_from_peer("beta")
        assert len(result) == 1
        peer_status = federation.get_peer_status("beta")
        assert peer_status["status"] == "connected"

    @pytest.mark.asyncio
    async def test_sync_all(self, federation):
        federation._fetch_from_peer = AsyncMock(return_value=[
            {
                "id": f"sync-{time.time()}",
                "source_instance": "beta",
                "content": "Synced memory",
                "significance": 7,
                "tags": ["test"],
                "timestamp": "2026-03-22T10:00:00Z",
            }
        ])
        results = await federation.sync_all()
        assert "beta" in results
        assert "gamma" in results

    def test_get_all_peer_status(self, federation):
        statuses = federation.get_all_peer_status()
        assert len(statuses) == 2

    def test_published_tags_filter(self):
        config = FederationConfig(
            instance_id="filtered",
            publish_threshold=5,
            published_tags=["shareable"],
        )
        fm = FederationManager(config)
        # Without the right tag
        assert fm.publish({"content": "nope", "significance": 8, "tags": ["other"]}) is None
        # With the right tag
        assert fm.publish({"content": "yes", "significance": 8, "tags": ["shareable"]}) is not None


# ===================================================================
# Cloud Backup Tests
# ===================================================================


class TestBackupRecord:
    """Tests for backup record serialization."""

    def test_roundtrip_serialization(self):
        record = BackupRecord(
            backup_id="20260322_100000",
            timestamp="2026-03-22T10:00:00Z",
            status=BackupStatus.COMPLETED,
            path="/backups/20260322_100000",
            file_count=42,
            total_bytes=1024000,
            duration_seconds=3.5,
        )
        data = record.to_dict()
        restored = BackupRecord.from_dict(data)
        assert restored.backup_id == record.backup_id
        assert restored.status == BackupStatus.COMPLETED
        assert restored.file_count == 42


class TestBackupManager:
    """Tests for the backup system."""

    @pytest.fixture
    def backup_dir(self, tmp_path):
        """Create a temporary backup directory."""
        backup = tmp_path / "backups"
        backup.mkdir()
        return backup

    @pytest.fixture
    def source_dir(self, tmp_path):
        """Create a temporary source directory with test data."""
        source = tmp_path / "source"
        source.mkdir()

        # Create data dir with files
        data_dir = source / "data"
        data_dir.mkdir()
        (data_dir / "journal.jsonl").write_text('{"content": "test entry"}\n')
        (data_dir / "precision_cell.pt").write_bytes(b"\x00" * 100)
        (data_dir / "config.json").write_text('{"key": "value"}')

        # Create identity dir
        identity_dir = source / "identity"
        identity_dir.mkdir()
        (identity_dir / "charter.md").write_text("# Charter\nTest charter")
        (identity_dir / "values.json").write_text('["honesty", "curiosity"]')

        return source

    @pytest.fixture
    def manager(self, backup_dir, source_dir):
        config = BackupConfig(
            backup_dir=str(backup_dir),
            source_dirs=["data", "identity"],
            max_backups=3,
            auto_backup_interval=60.0,
        )
        return BackupManager(config, base_dir=source_dir)

    @pytest.mark.asyncio
    async def test_create_backup(self, manager):
        record = await manager.create_backup()
        assert record.status == BackupStatus.COMPLETED
        assert record.file_count > 0
        assert record.total_bytes > 0
        assert record.duration_seconds >= 0

    @pytest.mark.asyncio
    async def test_backup_creates_files(self, manager, backup_dir):
        record = await manager.create_backup()
        backup_path = Path(record.path)
        assert backup_path.exists()
        assert (backup_path / "backup_meta.json").exists()

    @pytest.mark.asyncio
    async def test_backup_preserves_content(self, manager, backup_dir):
        record = await manager.create_backup()
        backup_path = Path(record.path)
        journal = backup_path / "data" / "journal.jsonl"
        assert journal.exists()
        assert "test entry" in journal.read_text()

    @pytest.mark.asyncio
    async def test_backup_with_label(self, manager):
        record = await manager.create_backup(label="pre_update")
        assert "pre_update" in record.backup_id

    @pytest.mark.asyncio
    async def test_list_backups(self, manager):
        await manager.create_backup()
        backups = manager.list_backups()
        assert len(backups) == 1
        assert backups[0]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_get_latest_backup(self, manager):
        await manager.create_backup(label="first")
        await manager.create_backup(label="second")
        latest = manager.get_latest_backup()
        assert "second" in latest.backup_id

    @pytest.mark.asyncio
    async def test_restore_from_backup(self, manager, tmp_path):
        record = await manager.create_backup()
        restore_target = tmp_path / "restored"
        restore_target.mkdir()
        result = await manager.restore(record.backup_id, target_dir=restore_target)
        assert result is True
        assert (restore_target / "data" / "journal.jsonl").exists()
        assert (restore_target / "identity" / "charter.md").exists()

    @pytest.mark.asyncio
    async def test_restore_nonexistent_fails(self, manager):
        result = await manager.restore("nonexistent_backup")
        assert result is False

    @pytest.mark.asyncio
    async def test_prune_old_backups(self, manager):
        # Full backups so none is an ancestor of a kept one; chained
        # incrementals are protected from pruning (see test_backup_chain.py)
        manager._config.incremental = False
        # Create more backups than max
        for i in range(5):
            await manager.create_backup(label=f"b{i}")
        pruned = manager.prune_old_backups()
        assert pruned == 2  # 5 - 3 = 2 pruned
        assert len(manager.list_backups()) == 3

    @pytest.mark.asyncio
    async def test_prune_protects_incremental_chain(self, manager):
        # Default config is incremental: backups 2..5 chain back to the
        # first, so every one is load-bearing and none may be pruned
        for i in range(5):
            await manager.create_backup(label=f"b{i}")
        pruned = manager.prune_old_backups()
        assert pruned == 0
        assert len(manager.list_backups()) == 5

    def test_should_auto_backup_initially(self, manager):
        manager._last_backup_time = 0.0
        assert manager.should_auto_backup() is True

    @pytest.mark.asyncio
    async def test_should_auto_backup_after_interval(self, manager):
        await manager.create_backup()
        assert manager.should_auto_backup() is False
        # Simulate time passing
        manager._last_backup_time = time.time() - 120  # 2 min > 60s interval
        assert manager.should_auto_backup() is True

    @pytest.mark.asyncio
    async def test_incremental_skips_unchanged(self, manager):
        r1 = await manager.create_backup(label="full")
        r2 = await manager.create_backup(label="incr")
        # Second backup should have fewer files (unchanged files skipped)
        # Since checksums are tracked, incremental should detect no changes
        assert r2.file_count == 0  # All files unchanged

    @pytest.mark.asyncio
    async def test_backup_missing_source_dir(self, manager, source_dir):
        # Remove one source dir
        shutil.rmtree(source_dir / "identity")
        record = await manager.create_backup()
        assert record.status == BackupStatus.COMPLETED
        # Should still backup data dir
        assert record.file_count > 0

    def test_get_status(self, manager):
        status = manager.get_status()
        assert "backup_dir" in status
        assert "total_backups" in status
        assert "auto_backup_due" in status

    @pytest.mark.asyncio
    async def test_backup_history_persists(self, backup_dir, source_dir):
        config = BackupConfig(
            backup_dir=str(backup_dir),
            source_dirs=["data", "identity"],
        )
        m1 = BackupManager(config, base_dir=source_dir)
        await m1.create_backup()

        # Create new manager — should load history
        m2 = BackupManager(config, base_dir=source_dir)
        assert len(m2.list_backups()) == 1

    @pytest.mark.asyncio
    async def test_backup_excludes_pycache(self, manager, source_dir):
        pycache = source_dir / "data" / "__pycache__"
        pycache.mkdir()
        (pycache / "module.pyc").write_bytes(b"\x00" * 50)
        record = await manager.create_backup()
        backup_path = Path(record.path)
        assert not (backup_path / "data" / "__pycache__").exists()
