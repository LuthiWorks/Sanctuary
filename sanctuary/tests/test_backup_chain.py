"""Regression tests for incremental-backup chain integrity (audit 2026-07-03, item 3).

An incremental backup holds only files changed since its parent; files stable
since an earlier backup exist only in that earlier backup. These tests pin the
three guarantees the fix introduced:
- restore layers the full ancestor chain (previously it read one directory and
  silently produced an incomplete set),
- a broken chain refuses to restore rather than restoring partially,
- pruning never deletes an ancestor a kept backup depends on.
"""

import json

import pytest

from sanctuary.infrastructure.cloud_backup import (
    BackupConfig,
    BackupManager,
    BackupStatus,
)


def make_manager(tmp_path, max_backups=10):
    base = tmp_path / "base"
    (base / "data").mkdir(parents=True)
    config = BackupConfig(
        backup_dir=str(tmp_path / "backups"),
        source_dirs=["data"],
        incremental=True,
        max_backups=max_backups,
    )
    return BackupManager(config, base_dir=base), base


def write_json(base, name, payload):
    (base / "data" / name).write_text(json.dumps(payload), encoding="utf-8")


class TestChainRestore:
    @pytest.mark.asyncio
    async def test_first_backup_is_full_second_is_incremental(self, tmp_path):
        manager, base = make_manager(tmp_path)
        write_json(base, "a.json", {"v": 1})
        write_json(base, "b.json", {"stable": True})

        first = await manager.create_backup(label="one")
        assert first.status == BackupStatus.COMPLETED
        assert first.incremental is False
        assert first.parent_id is None

        write_json(base, "a.json", {"v": 2})
        second = await manager.create_backup(label="two")
        assert second.status == BackupStatus.COMPLETED
        assert second.incremental is True
        assert second.parent_id == first.backup_id
        # the unchanged file lives only in the parent backup
        assert not (tmp_path / "backups" / second.backup_id / "data" / "b.json").exists()

    @pytest.mark.asyncio
    async def test_restore_layers_ancestor_chain(self, tmp_path):
        manager, base = make_manager(tmp_path)
        write_json(base, "a.json", {"v": 1})
        write_json(base, "b.json", {"stable": True})
        await manager.create_backup(label="one")

        write_json(base, "a.json", {"v": 2})
        second = await manager.create_backup(label="two")

        # simulate total loss of live state
        (base / "data" / "a.json").unlink()
        (base / "data" / "b.json").unlink()

        assert await manager.restore(second.backup_id) is True

        a = json.loads((base / "data" / "a.json").read_text(encoding="utf-8"))
        b = json.loads((base / "data" / "b.json").read_text(encoding="utf-8"))
        assert a == {"v": 2}          # from the incremental
        assert b == {"stable": True}  # from the full parent — the old bug lost this

    @pytest.mark.asyncio
    async def test_broken_chain_refuses_partial_restore(self, tmp_path):
        manager, base = make_manager(tmp_path)
        write_json(base, "a.json", {"v": 1})
        first = await manager.create_backup(label="one")
        write_json(base, "a.json", {"v": 2})
        second = await manager.create_backup(label="two")

        # parent vanishes from history: the chain is no longer provable
        manager._history = [r for r in manager._history if r.backup_id != first.backup_id]

        assert await manager.restore(second.backup_id) is False

    @pytest.mark.asyncio
    async def test_missing_parent_directory_refuses_restore(self, tmp_path):
        import shutil

        manager, base = make_manager(tmp_path)
        write_json(base, "a.json", {"v": 1})
        first = await manager.create_backup(label="one")
        write_json(base, "a.json", {"v": 2})
        second = await manager.create_backup(label="two")

        shutil.rmtree(tmp_path / "backups" / first.backup_id)

        assert await manager.restore(second.backup_id) is False


class TestChainAwarePrune:
    @pytest.mark.asyncio
    async def test_prune_keeps_ancestors_of_kept_incrementals(self, tmp_path):
        manager, base = make_manager(tmp_path, max_backups=1)
        write_json(base, "a.json", {"v": 1})
        write_json(base, "b.json", {"stable": True})
        first = await manager.create_backup(label="one")
        write_json(base, "a.json", {"v": 2})
        second = await manager.create_backup(label="two")

        pruned = manager.prune_old_backups()

        # the full parent is over quota but load-bearing: it must survive
        assert pruned == 0
        assert (tmp_path / "backups" / first.backup_id).exists()
        assert await manager.restore(second.backup_id) is True

    @pytest.mark.asyncio
    async def test_prune_still_prunes_superseded_chains(self, tmp_path):
        manager, base = make_manager(tmp_path, max_backups=1)
        write_json(base, "a.json", {"v": 1})
        first = await manager.create_backup(label="one")
        write_json(base, "a.json", {"v": 2})
        second = await manager.create_backup(label="two")

        # a fresh manager has an empty checksum cache, so its backup is full
        # and the old chain is no longer needed
        manager2 = BackupManager(manager._config, base_dir=base)
        third = await manager2.create_backup(label="three")
        assert third.incremental is False

        pruned = manager2.prune_old_backups()

        assert pruned == 2
        assert not (tmp_path / "backups" / first.backup_id).exists()
        assert not (tmp_path / "backups" / second.backup_id).exists()
        assert (tmp_path / "backups" / third.backup_id).exists()
