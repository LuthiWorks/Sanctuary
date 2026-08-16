"""Tests for identity checkpoint — model state snapshots for rollback."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sanctuary.growth.identity_checkpoint import (
    CheckpointMetadata,
    IdentityCheckpoint,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def checkpoint_mgr(tmp_path):
    return IdentityCheckpoint(checkpoint_dir=tmp_path / "checkpoints")


@pytest.fixture
def model_file(tmp_path):
    """Create a fake model file for checkpointing."""
    model_path = tmp_path / "model.bin"
    model_path.write_bytes(b"fake model weights 12345")
    return model_path


@pytest.fixture
def model_dir(tmp_path):
    """Create a fake model directory for checkpointing."""
    model_path = tmp_path / "model_dir"
    model_path.mkdir()
    (model_path / "weights.bin").write_bytes(b"fake weights")
    (model_path / "config.json").write_text('{"hidden_size": 768}')
    return model_path


# ---------------------------------------------------------------------------
# Creating checkpoints
# ---------------------------------------------------------------------------


class TestCreateCheckpoint:
    def test_creates_from_file(self, checkpoint_mgr, model_file):
        cid = checkpoint_mgr.create_checkpoint(model_file)
        assert cid is not None
        assert len(cid) > 0

    def test_creates_from_directory(self, checkpoint_mgr, model_dir):
        cid = checkpoint_mgr.create_checkpoint(model_dir)
        assert cid is not None

    def test_creates_checkpoint_dir(self, checkpoint_mgr, model_file):
        cid = checkpoint_mgr.create_checkpoint(model_file)
        checkpoint_path = checkpoint_mgr.checkpoint_dir / cid
        assert checkpoint_path.exists()
        assert (checkpoint_path / "metadata.json").exists()
        assert (checkpoint_path / "weights").exists()

    def test_copies_file_weights(self, checkpoint_mgr, model_file):
        cid = checkpoint_mgr.create_checkpoint(model_file)
        weights_dir = checkpoint_mgr.checkpoint_dir / cid / "weights"
        copied = weights_dir / model_file.name
        assert copied.exists()
        assert copied.read_bytes() == b"fake model weights 12345"

    def test_copies_dir_weights(self, checkpoint_mgr, model_dir):
        cid = checkpoint_mgr.create_checkpoint(model_dir)
        weights_dir = checkpoint_mgr.checkpoint_dir / cid / "weights"
        assert (weights_dir / "weights.bin").exists()
        assert (weights_dir / "config.json").exists()

    def test_metadata_saved(self, checkpoint_mgr, model_file):
        cid = checkpoint_mgr.create_checkpoint(
            model_file,
            metadata={
                "description": "Test checkpoint",
                "training_pair_count": 5,
            },
        )
        meta_path = checkpoint_mgr.checkpoint_dir / cid / "metadata.json"
        data = json.loads(meta_path.read_text())
        assert data["description"] == "Test checkpoint"
        assert data["training_pair_count"] == 5
        assert data["checkpoint_id"] == cid

    def test_extra_metadata_stored(self, checkpoint_mgr, model_file):
        cid = checkpoint_mgr.create_checkpoint(
            model_file,
            metadata={"custom_field": "custom_value"},
        )
        meta = checkpoint_mgr.get_checkpoint(cid)
        assert meta.extra["custom_field"] == "custom_value"

    def test_nonexistent_model_raises(self, checkpoint_mgr, tmp_path):
        with pytest.raises(FileNotFoundError):
            checkpoint_mgr.create_checkpoint(tmp_path / "nonexistent")


# ---------------------------------------------------------------------------
# Listing and getting checkpoints
# ---------------------------------------------------------------------------


class TestListCheckpoints:
    def test_empty_initially(self, checkpoint_mgr):
        assert checkpoint_mgr.list_checkpoints() == []

    def test_lists_created_checkpoints(self, checkpoint_mgr, model_file):
        cid1 = checkpoint_mgr.create_checkpoint(model_file)
        cid2 = checkpoint_mgr.create_checkpoint(model_file)
        checkpoints = checkpoint_mgr.list_checkpoints()
        assert len(checkpoints) == 2
        ids = {c.checkpoint_id for c in checkpoints}
        assert cid1 in ids
        assert cid2 in ids

    def test_get_specific_checkpoint(self, checkpoint_mgr, model_file):
        cid = checkpoint_mgr.create_checkpoint(
            model_file, metadata={"description": "specific one"},
        )
        meta = checkpoint_mgr.get_checkpoint(cid)
        assert meta is not None
        assert meta.description == "specific one"

    def test_get_nonexistent_returns_none(self, checkpoint_mgr):
        assert checkpoint_mgr.get_checkpoint("nonexistent_id") is None


# ---------------------------------------------------------------------------
# Restoring checkpoints
# ---------------------------------------------------------------------------


class TestRestoreCheckpoint:
    def test_restore_file(self, checkpoint_mgr, model_file, tmp_path):
        cid = checkpoint_mgr.create_checkpoint(model_file)
        restore_dest = tmp_path / "restored"
        checkpoint_mgr.restore_checkpoint(cid, restore_dest)
        # Restored as directory containing the original file
        assert restore_dest.exists()

    def test_restore_directory(self, checkpoint_mgr, model_dir, tmp_path):
        cid = checkpoint_mgr.create_checkpoint(model_dir)
        restore_dest = tmp_path / "restored_model"
        checkpoint_mgr.restore_checkpoint(cid, restore_dest)
        assert (restore_dest / "weights.bin").exists()
        assert (restore_dest / "config.json").exists()

    def test_restore_overwrites_existing(self, checkpoint_mgr, model_file, tmp_path):
        cid = checkpoint_mgr.create_checkpoint(model_file)
        restore_dest = tmp_path / "restored"
        restore_dest.mkdir()
        (restore_dest / "old_file.txt").write_text("old")
        checkpoint_mgr.restore_checkpoint(cid, restore_dest)
        assert not (restore_dest / "old_file.txt").exists()

    def test_restore_nonexistent_raises(self, checkpoint_mgr, tmp_path):
        with pytest.raises(ValueError, match="no weights"):
            checkpoint_mgr.restore_checkpoint("nonexistent", tmp_path / "dest")


# ---------------------------------------------------------------------------
# Comparing checkpoints
# ---------------------------------------------------------------------------


class TestCompareCheckpoints:
    def test_compare_different_descriptions(self, checkpoint_mgr, model_file):
        cid1 = checkpoint_mgr.create_checkpoint(
            model_file, metadata={"description": "Before training"},
        )
        cid2 = checkpoint_mgr.create_checkpoint(
            model_file, metadata={"description": "After training"},
        )
        comparison = checkpoint_mgr.compare_checkpoints(cid1, cid2)
        assert "diff" in comparison
        assert "description" in comparison["diff"]
        assert comparison["diff"]["description"]["before"] == "Before training"
        assert comparison["diff"]["description"]["after"] == "After training"

    def test_compare_nonexistent_raises(self, checkpoint_mgr, model_file):
        cid = checkpoint_mgr.create_checkpoint(model_file)
        with pytest.raises(ValueError):
            checkpoint_mgr.compare_checkpoints(cid, "nonexistent")


# ---------------------------------------------------------------------------
# Deleting checkpoints
# ---------------------------------------------------------------------------


class TestDeleteCheckpoint:
    def test_delete_existing(self, checkpoint_mgr, model_file):
        cid = checkpoint_mgr.create_checkpoint(model_file)
        assert checkpoint_mgr.delete_checkpoint(cid)
        assert checkpoint_mgr.get_checkpoint(cid) is None

    def test_delete_nonexistent(self, checkpoint_mgr):
        assert not checkpoint_mgr.delete_checkpoint("nonexistent")

    def test_delete_removes_files(self, checkpoint_mgr, model_file):
        cid = checkpoint_mgr.create_checkpoint(model_file)
        checkpoint_path = checkpoint_mgr.checkpoint_dir / cid
        assert checkpoint_path.exists()
        checkpoint_mgr.delete_checkpoint(cid)
        assert not checkpoint_path.exists()


# ---------------------------------------------------------------------------
# CheckpointMetadata
# ---------------------------------------------------------------------------


class TestCheckpointMetadata:
    def test_default_values(self):
        meta = CheckpointMetadata()
        assert meta.checkpoint_type == "pre_training"
        assert meta.what_was_learned == []
        assert meta.training_pair_count == 0
        assert meta.final_loss is None

    def test_has_timestamp(self):
        meta = CheckpointMetadata()
        assert meta.created_at is not None


# ---------------------------------------------------------------------------
# Regression: checkpoint ids must be unique (2026-08-16)
# ---------------------------------------------------------------------------


class TestCheckpointIdUniqueness:
    """Two checkpoints made in the same clock tick must not become one.

    The id used to be a bare `datetime.now()` timestamp, and the directory was
    created with `exist_ok=True`. Windows' clock granularity (~15.6 ms) meant
    rapid successive checkpoints -- a rollback followed by a re-checkpoint, for
    instance -- collided on the id, and the second silently overwrote the
    first's weights and metadata while returning success. That is identity
    state destroyed by a persistence layer reporting that it worked, which is
    the failure class this project's CLAUDE.md exists to forbid.

    These tests flushed out as intermittent failures in TestListCheckpoints and
    TestCompareCheckpoints, which is how a collision presents: two checkpoints
    behaving as one.
    """

    def test_rapid_successive_checkpoints_have_distinct_ids(
        self, checkpoint_mgr, model_file
    ):
        ids = [checkpoint_mgr.create_checkpoint(model_file) for _ in range(25)]
        assert len(set(ids)) == len(ids), "checkpoint ids collided"

    def test_rapid_successive_checkpoints_all_survive(
        self, checkpoint_mgr, model_file
    ):
        ids = [checkpoint_mgr.create_checkpoint(model_file) for _ in range(25)]
        listed = {c.checkpoint_id for c in checkpoint_mgr.list_checkpoints()}
        assert listed == set(ids), "a checkpoint was overwritten or lost"

    def test_each_checkpoint_keeps_its_own_metadata(
        self, checkpoint_mgr, model_file
    ):
        first = checkpoint_mgr.create_checkpoint(
            model_file, metadata={"description": "before"}
        )
        second = checkpoint_mgr.create_checkpoint(
            model_file, metadata={"description": "after"}
        )
        assert first != second
        assert checkpoint_mgr.get_checkpoint(first).description == "before"
        assert checkpoint_mgr.get_checkpoint(second).description == "after"

    def test_ids_remain_chronologically_sortable(self, checkpoint_mgr, model_file):
        """The timestamp prefix must stay first -- list_checkpoints sorts on it."""
        ids = [checkpoint_mgr.create_checkpoint(model_file) for _ in range(10)]
        assert ids == sorted(ids)
