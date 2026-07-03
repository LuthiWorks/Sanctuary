"""Regression tests for the Phase 1 data-integrity firestops (audit 2026-07-03).

Covers audit items 1, 2, and 4:
- MemoryStorage.close() / MindVectorDB.close() must never destroy data
  (they previously called client.reset(), erasing every collection on a
  clean shutdown).
- BackupManager.cleanup_old_backups() must never delete the last backups
  (it previously pruned purely by age, with no floor).
"""

import os
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from sanctuary.mind.memory.storage import MemoryStorage
from sanctuary.mind.memory.backup import BackupManager


class TestMemoryStorageClosePreservesData:
    """A clean shutdown must leave every collection intact on disk."""

    def test_close_then_reopen_keeps_documents(self, tmp_path):
        storage = MemoryStorage(persistence_dir=str(tmp_path / "chroma"))
        storage.episodic_memory.add(
            ids=["evt-1"],
            embeddings=[[0.1, 0.2, 0.3, 0.4]],
            documents=["the first remembered thing"],
            metadatas=[{"kind": "test"}],
        )
        assert storage.episodic_memory.count() == 1

        storage.close()
        assert storage.client is None

        reopened = MemoryStorage(persistence_dir=str(tmp_path / "chroma"))
        try:
            assert reopened.episodic_memory.count() == 1
            got = reopened.episodic_memory.get(ids=["evt-1"])
            assert got["documents"] == ["the first remembered thing"]
        finally:
            reopened.close()

    def test_reset_is_disallowed_by_settings(self, tmp_path):
        storage = MemoryStorage(persistence_dir=str(tmp_path / "chroma"))
        try:
            # Defense in depth: even a future code path that reaches for
            # reset() must be refused by the client itself.
            with pytest.raises(Exception):
                storage.client.reset()
            assert storage.episodic_memory is not None
        finally:
            storage.close()


class TestMindVectorDBCloseNeverResets:
    """close() releases handles via system stop, never via reset()."""

    def test_close_calls_system_stop_not_reset(self):
        pytest.importorskip("langchain_core")
        from sanctuary.mind.rag_engine import MindVectorDB

        db = MindVectorDB.__new__(MindVectorDB)
        db.db_path = Path("unused")
        client = MagicMock()
        db.client = client

        db.close()

        client.reset.assert_not_called()
        client._system.stop.assert_called_once()
        assert db.client is None


class TestBackupRetentionFloor:
    """Age-based cleanup must always retain the newest min_keep backups."""

    @staticmethod
    def _make_backup_dirs(backup_dir: Path, count: int) -> list[Path]:
        # Stagger mtimes so list_backups() has a stable newest-first order,
        # all of them far older than any reasonable retention cutoff.
        dirs = []
        base_age = time.time() - 400 * 86400
        for i in range(count):
            d = backup_dir / f"backup_{i}"
            d.mkdir()
            (d / "payload.json").write_text("{}", encoding="utf-8")
            os.utime(d, (base_age + i * 86400, base_age + i * 86400))
            dirs.append(d)
        return dirs

    @pytest.mark.asyncio
    async def test_cleanup_keeps_newest_min_keep(self, tmp_path):
        source = tmp_path / "source"
        source.mkdir()
        backup_dir = tmp_path / "backups"
        manager = BackupManager(
            source_dir=source,
            backup_dir=backup_dir,
            retention_days=30,
            min_keep=2,
        )
        dirs = self._make_backup_dirs(backup_dir, 3)

        removed = await manager.cleanup_old_backups()

        assert removed == 1
        # newest two (highest mtimes: backup_2, backup_1) survive
        assert not dirs[0].exists()
        assert dirs[1].exists()
        assert dirs[2].exists()

    @pytest.mark.asyncio
    async def test_min_keep_floors_at_one(self, tmp_path):
        source = tmp_path / "source"
        source.mkdir()
        backup_dir = tmp_path / "backups"
        manager = BackupManager(
            source_dir=source,
            backup_dir=backup_dir,
            retention_days=30,
            min_keep=0,
        )
        assert manager.min_keep == 1
        dirs = self._make_backup_dirs(backup_dir, 2)

        removed = await manager.cleanup_old_backups()

        assert removed == 1
        assert dirs[1].exists()
