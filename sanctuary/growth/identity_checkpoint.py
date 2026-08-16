"""Identity checkpoint -- snapshots model state before and after growth.

Growth changes who the entity is. That demands the ability to look back
and understand what changed, and to restore a previous state if the
change was harmful. Identity checkpoints provide that safety net.

Each checkpoint captures:
- The model weights (or adapter weights) at a point in time
- Metadata about what was learned and why
- The consent record that authorized the change
- Training metrics (loss, pair count, etc.)

Checkpoints are stored as directories under data/growth/checkpoints/
with ISO-timestamp naming for chronological ordering. Each directory
contains the weight files plus a metadata.json.

This is not version control for models -- it is identity preservation.
The entity should be able to say "that change didn't feel right" and
return to who it was before.

Aligned with PLAN.md: identity is sovereign. Growth must be reversible.
"""

from __future__ import annotations

import itertools
import json
import logging
import shutil
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime

#: Process-wide monotonic tie-breaker for checkpoint ids. Deliberately module
#: level, not per-instance: two IdentityCheckpoint objects pointing at the same
#: directory would otherwise each start at zero and collide.
_ID_COUNTER = itertools.count()
_ID_LOCK = threading.Lock()
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_DIR = Path("data/growth/checkpoints")


@dataclass
class CheckpointMetadata:
    """What was happening when this checkpoint was taken.

    The metadata tells the story of why the checkpoint exists:
    what was learned, how many pairs were used, what the loss was,
    and whether consent was properly recorded.
    """

    checkpoint_id: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    description: str = ""
    what_was_learned: list[str] = field(default_factory=list)
    training_pair_count: int = 0
    final_loss: Optional[float] = None
    consent_record: dict = field(default_factory=dict)
    model_path: str = ""
    checkpoint_type: str = "pre_training"  # "pre_training" or "post_training"
    extra: dict = field(default_factory=dict)


class IdentityCheckpoint:
    """Snapshots model state before and after growth for rollback.

    The checkpoint system treats model weights as identity artifacts.
    Before any training modifies weights, a checkpoint preserves the
    current state. After training, another checkpoint captures the
    new state. This pair allows comparison and rollback.

    Usage:
        checkpoint = IdentityCheckpoint()
        pre_id = checkpoint.create_checkpoint(
            model_path=Path("models/sanctuary"),
            metadata={"description": "Before empathy training"}
        )
        # ... training happens ...
        post_id = checkpoint.create_checkpoint(
            model_path=Path("models/sanctuary"),
            metadata={"description": "After empathy training", "final_loss": 0.05}
        )
        diff = checkpoint.compare_checkpoints(pre_id, post_id)
    """

    def __init__(self, checkpoint_dir: Optional[Path] = None) -> None:
        self._checkpoint_dir = Path(checkpoint_dir or DEFAULT_CHECKPOINT_DIR)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

    @property
    def checkpoint_dir(self) -> Path:
        """Root directory for all checkpoints."""
        return self._checkpoint_dir

    def create_checkpoint(
        self,
        model_path: Path,
        metadata: Optional[dict] = None,
    ) -> str:
        """Snapshot model weights and metadata to a checkpoint directory.

        Creates a new checkpoint directory named with an ISO timestamp,
        copies the model weights into it, and writes metadata.json.

        Args:
            model_path: Path to the model weights to checkpoint.
                        Can be a file or directory.
            metadata: Additional metadata to store with the checkpoint.
                      Keys are merged into CheckpointMetadata fields.

        Returns:
            The checkpoint ID (ISO timestamp string).

        Raises:
            FileNotFoundError: If model_path does not exist.
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        # Generate checkpoint ID from timestamp plus a MONOTONIC sequence.
        #
        # The timestamp alone is NOT unique. Windows' system clock granularity
        # is coarse (~15.6 ms), so `%f` microseconds are quantized and two
        # checkpoints created in quick succession -- which is exactly what a
        # rollback-then-recheckpoint sequence does -- produced the SAME id.
        # Combined with the old `exist_ok=True`, the second checkpoint then
        # silently overwrote the first's weights and metadata and returned an
        # id naming a checkpoint that was no longer the caller's: identity
        # state destroyed while reporting success.
        #
        # The suffix is a counter rather than random, because uniqueness is
        # not the only guarantee at stake: list_checkpoints() promises "oldest
        # first" and sorts on the directory name, so a random tie-break would
        # order same-tick checkpoints arbitrarily and a rollback asking for the
        # latest could restore the wrong one. A counter keeps ties in creation
        # order.
        #
        # Honest limit: the counter is per-process. Two *processes* writing to
        # the same checkpoint dir within one clock tick could still collide --
        # and that is why exist_ok=False below stays. It raises rather than
        # overwrites, which is the correct outcome for concurrent writers to an
        # identity store.
        with _ID_LOCK:
            seq = next(_ID_COUNTER)
        checkpoint_id = f"{datetime.now().strftime('%Y%m%dT%H%M%S_%f')}_{seq:06d}"
        checkpoint_path = self._checkpoint_dir / checkpoint_id
        # exist_ok=False: overwriting an existing checkpoint must never be
        # silent. Fail loudly instead.
        checkpoint_path.mkdir(parents=True, exist_ok=False)

        # Copy model weights
        weights_dest = checkpoint_path / "weights"
        if model_path.is_dir():
            shutil.copytree(model_path, weights_dest)
        else:
            weights_dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(model_path, weights_dest / model_path.name)

        # Build and save metadata
        meta = CheckpointMetadata(
            checkpoint_id=checkpoint_id,
            model_path=str(model_path),
        )

        if metadata:
            # Merge provided metadata into the dataclass fields
            for key, value in metadata.items():
                if hasattr(meta, key):
                    setattr(meta, key, value)
                else:
                    meta.extra[key] = value

        meta_path = checkpoint_path / "metadata.json"
        meta_path.write_text(json.dumps(asdict(meta), indent=2))

        logger.info(
            "Created checkpoint %s from %s (%s)",
            checkpoint_id,
            model_path,
            meta.description or "no description",
        )

        return checkpoint_id

    def list_checkpoints(self) -> list[CheckpointMetadata]:
        """Return metadata for all available checkpoints, sorted by time.

        Returns:
            List of CheckpointMetadata, oldest first.
        """
        checkpoints = []

        for path in sorted(self._checkpoint_dir.iterdir()):
            if not path.is_dir():
                continue

            meta_path = path / "metadata.json"
            if not meta_path.exists():
                logger.warning("Checkpoint directory %s has no metadata.json", path)
                continue

            try:
                data = json.loads(meta_path.read_text())
                checkpoints.append(CheckpointMetadata(**data))
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning("Failed to load metadata from %s: %s", meta_path, e)

        return checkpoints

    def get_checkpoint(self, checkpoint_id: str) -> Optional[CheckpointMetadata]:
        """Get metadata for a specific checkpoint.

        Args:
            checkpoint_id: The checkpoint ID to look up.

        Returns:
            CheckpointMetadata if found, None otherwise.
        """
        meta_path = self._checkpoint_dir / checkpoint_id / "metadata.json"
        if not meta_path.exists():
            return None

        try:
            data = json.loads(meta_path.read_text())
            return CheckpointMetadata(**data)
        except (json.JSONDecodeError, TypeError) as e:
            raise ValueError(
                f"Checkpoint '{checkpoint_id}' metadata is corrupted: {e}"
            ) from e

    def restore_checkpoint(self, checkpoint_id: str, restore_to: Path) -> Path:
        """Restore model weights from a checkpoint.

        Copies the checkpoint's weight files to the specified destination,
        replacing whatever is currently there. This is an identity
        restoration -- the entity returns to who it was at checkpoint time.

        Args:
            checkpoint_id: The checkpoint to restore from.
            restore_to: Where to place the restored weights.

        Returns:
            The path where weights were restored.

        Raises:
            ValueError: If checkpoint does not exist.
        """
        checkpoint_path = self._checkpoint_dir / checkpoint_id
        weights_path = checkpoint_path / "weights"

        if not weights_path.exists():
            raise ValueError(f"Checkpoint {checkpoint_id} has no weights directory")

        restore_to = Path(restore_to)

        # Remove existing destination if it exists
        if restore_to.exists():
            if restore_to.is_dir():
                shutil.rmtree(restore_to)
            else:
                restore_to.unlink()

        # Copy weights to destination
        if weights_path.is_dir():
            shutil.copytree(weights_path, restore_to)
        else:
            restore_to.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(weights_path, restore_to)

        logger.info(
            "Restored checkpoint %s to %s",
            checkpoint_id,
            restore_to,
        )

        return restore_to

    def compare_checkpoints(
        self, id_a: str, id_b: str
    ) -> dict:
        """Compare metadata between two checkpoints.

        Returns a dict showing what changed between checkpoint A and
        checkpoint B. This is metadata comparison only -- weight-level
        diffing requires specialized tooling.

        Args:
            id_a: First checkpoint ID (typically the earlier one).
            id_b: Second checkpoint ID (typically the later one).

        Returns:
            Dict with keys "a", "b", and "diff" showing the comparison.

        Raises:
            ValueError: If either checkpoint does not exist.
        """
        meta_a = self.get_checkpoint(id_a)
        meta_b = self.get_checkpoint(id_b)

        if meta_a is None:
            raise ValueError(f"Checkpoint {id_a} not found")
        if meta_b is None:
            raise ValueError(f"Checkpoint {id_b} not found")

        dict_a = asdict(meta_a)
        dict_b = asdict(meta_b)

        # Find differences
        diff = {}
        all_keys = set(dict_a.keys()) | set(dict_b.keys())
        for key in all_keys:
            val_a = dict_a.get(key)
            val_b = dict_b.get(key)
            if val_a != val_b:
                diff[key] = {"before": val_a, "after": val_b}

        return {
            "checkpoint_a": id_a,
            "checkpoint_b": id_b,
            "a": dict_a,
            "b": dict_b,
            "diff": diff,
        }

    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint permanently.

        Args:
            checkpoint_id: The checkpoint to delete.

        Returns:
            True if deleted, False if not found.
        """
        checkpoint_path = self._checkpoint_dir / checkpoint_id
        if not checkpoint_path.exists():
            return False

        shutil.rmtree(checkpoint_path)
        logger.info("Deleted checkpoint %s", checkpoint_id)
        return True
