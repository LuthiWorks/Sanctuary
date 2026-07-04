"""CfC retrainer — fast plasticity through live interaction data.

The CfC cells start trained on scaffold heuristic data (Phase 4). But the
heuristics are memoryless approximations — the real value of CfC cells is
their ability to learn temporal patterns that emerge during live operation.

This module provides the "fast plasticity" growth path:

    Live cognitive cycle
        -> CfCDataTap (records cell inputs/outputs each cycle)
        -> CfCRetrainer (accumulates data, retrains when ready)
        -> ConsentGate (verifies consent before weight changes)
        -> IdentityCheckpoint (snapshots cell state for rollback)
        -> CfCTrainer (applies the training)

Fast plasticity vs medium plasticity:
- CfC retraining (this module): ~50K params, trains in seconds on CPU,
  captures temporal dynamics from actual experience. Can run frequently.
- QLoRA fine-tuning (qlora_updater.py): billions of params, trains in
  minutes on GPU, modifies the entity's world model. Runs rarely.

The CfC cells are the experiential substrate — they feel the flow of
time between model cycles. Retraining them from live data lets them
incorporate patterns that only emerge during real interaction: the way
arousal builds during extended conversation, how prediction errors cluster
around novel topics, how attention drifts during idle periods.

Growth is sovereign. Retraining requires consent. Rollback is always
possible.

Aligned with PLAN.md: CfC retraining from accumulated data (Phase 7).
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch

from sanctuary.experiential.trainer import (
    AffectRecord,
    AttentionRecord,
    CfCTrainer,
    GoalRecord,
    TrainingRecord,
    TrainingResult,
)

logger = logging.getLogger(__name__)

# How many records to accumulate before retraining is eligible
DEFAULT_MIN_RECORDS = 200
# Maximum records to keep per cell (bounded memory)
DEFAULT_MAX_RECORDS = 10000
# Default training epochs for retraining
DEFAULT_RETRAIN_EPOCHS = 30
# Default sequence length for retraining
DEFAULT_RETRAIN_SEQ_LEN = 15


# ---------------------------------------------------------------------------
# Data tap — records cell I/O during live operation
# ---------------------------------------------------------------------------


@dataclass
class CellSnapshot:
    """A snapshot of one cell's inputs and outputs from a single cycle."""

    cell_name: str
    inputs: dict[str, float]
    outputs: dict[str, float]
    cycle_count: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class CfCDataTap:
    """Records CfC cell inputs and outputs during live cognitive cycles.

    The data tap sits between the ExperientialManager and the cognitive
    cycle. Each time the manager steps all cells, the tap records the
    inputs and outputs for each cell. This data accumulates and becomes
    the training set for CfC retraining.

    The tap is passive — it never modifies cell behavior, never
    interrupts the cognitive cycle, and never decides what to record.
    It records everything, and the retrainer decides when enough data
    has accumulated to retrain.

    Usage:
        tap = CfCDataTap()
        # After each ExperientialManager.step():
        tap.record_precision(arousal=0.7, prediction_error=0.3,
                            base_precision=0.5, output=0.65, cycle=42)
        # When ready to retrain:
        records = tap.drain_precision()
    """

    def __init__(self, max_records: int = DEFAULT_MAX_RECORDS) -> None:
        self._max_records = max_records
        self._precision_records: list[TrainingRecord] = []
        self._affect_records: list[AffectRecord] = []
        self._attention_records: list[AttentionRecord] = []
        self._goal_records: list[GoalRecord] = []
        self._total_recorded: dict[str, int] = {
            "precision": 0,
            "affect": 0,
            "attention": 0,
            "goal": 0,
        }

    # -- Recording methods (one per cell type) --

    def record_precision(
        self,
        arousal: float,
        prediction_error: float,
        base_precision: float,
        output: float,
        cycle: int = 0,
    ) -> None:
        """Record one precision cell observation."""
        self._precision_records.append(
            TrainingRecord(
                arousal=arousal,
                prediction_error=prediction_error,
                base_precision=base_precision,
                precision_output=output,
            )
        )
        self._total_recorded["precision"] += 1
        self._enforce_limit(self._precision_records)

    def record_affect(
        self,
        percept_valence_delta: float,
        percept_arousal_delta: float,
        llm_emotion_shift: float,
        valence_output: float,
        arousal_output: float,
        dominance_output: float,
        cycle: int = 0,
    ) -> None:
        """Record one affect cell observation."""
        self._affect_records.append(
            AffectRecord(
                percept_valence_delta=percept_valence_delta,
                percept_arousal_delta=percept_arousal_delta,
                llm_emotion_shift=llm_emotion_shift,
                valence_output=valence_output,
                arousal_output=arousal_output,
                dominance_output=dominance_output,
            )
        )
        self._total_recorded["affect"] += 1
        self._enforce_limit(self._affect_records)

    def record_attention(
        self,
        goal_relevance: float,
        novelty: float,
        emotional_salience: float,
        recency: float,
        salience_output: float,
        cycle: int = 0,
    ) -> None:
        """Record one attention cell observation."""
        self._attention_records.append(
            AttentionRecord(
                goal_relevance=goal_relevance,
                novelty=novelty,
                emotional_salience=emotional_salience,
                recency=recency,
                salience_output=salience_output,
            )
        )
        self._total_recorded["attention"] += 1
        self._enforce_limit(self._attention_records)

    def record_goal(
        self,
        cycles_stalled_norm: float,
        deadline_urgency: float,
        emotional_congruence: float,
        priority_adjustment_output: float,
        cycle: int = 0,
    ) -> None:
        """Record one goal cell observation."""
        self._goal_records.append(
            GoalRecord(
                cycles_stalled_norm=cycles_stalled_norm,
                deadline_urgency=deadline_urgency,
                emotional_congruence=emotional_congruence,
                priority_adjustment_output=priority_adjustment_output,
            )
        )
        self._total_recorded["goal"] += 1
        self._enforce_limit(self._goal_records)

    # -- Query methods --

    def counts(self) -> dict[str, int]:
        """Current record counts per cell."""
        return {
            "precision": len(self._precision_records),
            "affect": len(self._affect_records),
            "attention": len(self._attention_records),
            "goal": len(self._goal_records),
        }

    def total_recorded(self) -> dict[str, int]:
        """Total records ever collected per cell (including drained)."""
        return dict(self._total_recorded)

    def has_enough(self, cell_name: str, min_records: int = DEFAULT_MIN_RECORDS) -> bool:
        """Whether a cell has accumulated enough data for retraining."""
        records = self._get_records(cell_name)
        return len(records) >= min_records

    # -- Drain methods (hand off data for training) --

    def drain_precision(self) -> list[TrainingRecord]:
        """Remove and return all precision records for training."""
        records = self._precision_records
        self._precision_records = []
        return records

    def drain_affect(self) -> list[AffectRecord]:
        """Remove and return all affect records for training."""
        records = self._affect_records
        self._affect_records = []
        return records

    def drain_attention(self) -> list[AttentionRecord]:
        """Remove and return all attention records for training."""
        records = self._attention_records
        self._attention_records = []
        return records

    def drain_goal(self) -> list[GoalRecord]:
        """Remove and return all goal records for training."""
        records = self._goal_records
        self._goal_records = []
        return records

    def drain(self, cell_name: str) -> list:
        """Drain records for a specific cell by name."""
        drainers = {
            "precision": self.drain_precision,
            "affect": self.drain_affect,
            "attention": self.drain_attention,
            "goal": self.drain_goal,
        }
        if cell_name not in drainers:
            raise ValueError(f"Unknown cell: {cell_name}. Expected one of {list(drainers.keys())}")
        return drainers[cell_name]()

    def requeue(self, cell_name: str, records: list) -> None:
        """Return drained records to the front of the tap.

        A failed retrain must not eat its training data (audit 2026-07-03,
        item 15). Requeued records keep chronological order ahead of anything
        recorded since the drain; the max_records cap still applies, dropping
        the oldest on overflow.
        """
        buffers = {
            "precision": self._precision_records,
            "affect": self._affect_records,
            "attention": self._attention_records,
            "goal": self._goal_records,
        }
        if cell_name not in buffers:
            raise ValueError(
                f"Unknown cell: {cell_name}. Expected one of {list(buffers.keys())}"
            )
        buf = buffers[cell_name]
        buf[:0] = records
        overflow = len(buf) - self._max_records
        if overflow > 0:
            del buf[:overflow]
            logger.warning(
                "Requeue overflow for %s: dropped %d oldest records",
                cell_name, overflow,
            )

    # -- Persistence --

    def save(self, path: Path) -> None:
        """Save all accumulated records to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "precision": [asdict(r) for r in self._precision_records],
            "affect": [asdict(r) for r in self._affect_records],
            "attention": [asdict(r) for r in self._attention_records],
            "goal": [asdict(r) for r in self._goal_records],
            "total_recorded": self._total_recorded,
            "saved_at": datetime.now().isoformat(),
        }
        torch.save(data, path)
        logger.info("Saved CfC data tap: %s", self.counts())

    def load(self, path: Path) -> None:
        """Load records from disk."""
        path = Path(path)
        if not path.exists():
            logger.warning("No data tap state at %s", path)
            return

        # Saved as plain dicts/lists/str (asdict of records + isoformat strings),
        # so weights_only=True is safe and blocks pickle RCE.
        data = torch.load(path, map_location="cpu", weights_only=True)

        self._precision_records = [TrainingRecord(**r) for r in data.get("precision", [])]
        self._affect_records = [AffectRecord(**r) for r in data.get("affect", [])]
        self._attention_records = [AttentionRecord(**r) for r in data.get("attention", [])]
        self._goal_records = [GoalRecord(**r) for r in data.get("goal", [])]
        self._total_recorded = data.get("total_recorded", self._total_recorded)

        logger.info("Loaded CfC data tap: %s", self.counts())

    # -- Internal --

    def _get_records(self, cell_name: str) -> list:
        """Get the record list for a cell name."""
        stores = {
            "precision": self._precision_records,
            "affect": self._affect_records,
            "attention": self._attention_records,
            "goal": self._goal_records,
        }
        if cell_name not in stores:
            raise ValueError(f"Unknown cell: {cell_name}. Expected one of {list(stores.keys())}")
        return stores[cell_name]

    def _enforce_limit(self, records: list) -> None:
        """Drop oldest records if over the maximum."""
        if len(records) > self._max_records:
            excess = len(records) - self._max_records
            del records[:excess]


# ---------------------------------------------------------------------------
# Retraining result
# ---------------------------------------------------------------------------


@dataclass
class CfCRetrainingResult:
    """Result of retraining one CfC cell from live data."""

    cell_name: str
    success: bool = False
    records_used: int = 0
    training_result: Optional[TrainingResult] = None
    checkpoint_path: Optional[str] = None
    error: Optional[str] = None
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None


@dataclass
class CfCRetrainingStats:
    """Aggregate statistics for CfC retraining."""

    total_retrains: int = 0
    successful_retrains: int = 0
    failed_retrains: int = 0
    retrains_per_cell: dict[str, int] = field(
        default_factory=lambda: {"precision": 0, "affect": 0, "attention": 0, "goal": 0}
    )
    last_retrain_at: Optional[str] = None
    consent_refused_count: int = 0


# ---------------------------------------------------------------------------
# CfC retrainer — orchestrates retraining from live data
# ---------------------------------------------------------------------------


class CfCRetrainer:
    """Retrains CfC cells from accumulated live interaction data.

    The retrainer is the fast-plasticity counterpart to the QLoRA updater.
    While QLoRA modifies the entity (medium plasticity, heavy computation),
    the CfC retrainer modifies the experiential layer (fast plasticity,
    CPU-trainable in seconds).

    The pipeline:
        1. CfCDataTap accumulates cell I/O from live cycles
        2. CfCRetrainer checks if enough data has accumulated
        3. ConsentGate verifies consent (reuses growth consent)
        4. Cell state is checkpointed (for rollback)
        5. CfCTrainer retrains the cell on accumulated data
        6. Results are logged

    Usage:
        from sanctuary.growth.cfc_retrainer import CfCRetrainer, CfCDataTap

        tap = CfCDataTap()
        retrainer = CfCRetrainer(data_tap=tap)

        # During cognitive cycles, record cell I/O:
        tap.record_precision(arousal=0.7, ..., output=0.65, cycle=42)

        # Periodically check and retrain:
        result = retrainer.retrain_cell("precision", cell=precision_cell)
    """

    def __init__(
        self,
        data_tap: Optional[CfCDataTap] = None,
        min_records: int = DEFAULT_MIN_RECORDS,
        retrain_epochs: int = DEFAULT_RETRAIN_EPOCHS,
        retrain_seq_len: int = DEFAULT_RETRAIN_SEQ_LEN,
        checkpoint_dir: Optional[Path] = None,
        enabled: bool = True,
    ) -> None:
        self._data_tap = data_tap or CfCDataTap()
        self._min_records = min_records
        self._retrain_epochs = retrain_epochs
        self._retrain_seq_len = retrain_seq_len
        self._checkpoint_dir = Path(checkpoint_dir or "data/growth/cfc_checkpoints")
        self._enabled = enabled
        self._stats = CfCRetrainingStats()
        self._history: list[CfCRetrainingResult] = []

    @property
    def data_tap(self) -> CfCDataTap:
        """The data tap collecting live cell I/O."""
        return self._data_tap

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value
        logger.info("CfC retrainer %s", "enabled" if value else "disabled")

    @property
    def stats(self) -> CfCRetrainingStats:
        return self._stats

    @property
    def history(self) -> list[CfCRetrainingResult]:
        return list(self._history)

    def cells_ready(self) -> dict[str, bool]:
        """Which cells have enough data for retraining."""
        return {
            name: self._data_tap.has_enough(name, self._min_records)
            for name in ["precision", "affect", "attention", "goal"]
        }

    def retrain_cell(
        self,
        cell_name: str,
        cell: torch.nn.Module,
        epochs: Optional[int] = None,
        force: bool = False,
    ) -> CfCRetrainingResult:
        """Retrain a single CfC cell from accumulated live data.

        Args:
            cell_name: Which cell to retrain (precision/affect/attention/goal).
            cell: The actual CfC cell module to retrain.
            epochs: Override default epoch count.
            force: If True, retrain even with fewer than min_records.

        Returns:
            CfCRetrainingResult with training metrics.
        """
        result = CfCRetrainingResult(cell_name=cell_name)

        if not self._enabled:
            result.error = "Retrainer is disabled"
            return result

        # Check data availability
        if not force and not self._data_tap.has_enough(cell_name, self._min_records):
            count = self._data_tap.counts()[cell_name]
            result.error = (
                f"Not enough data: {count} records "
                f"(need {self._min_records})"
            )
            return result

        records: list = []
        try:
            # 1. Drain records
            records = self._data_tap.drain(cell_name)
            result.records_used = len(records)

            if not records:
                result.error = "No records after drain"
                return result

            # 2. Checkpoint current cell state. A retrain with no restore
            # point cannot be rolled back, so checkpoint failure aborts
            try:
                checkpoint_path = self._checkpoint_cell(cell_name, cell)
                result.checkpoint_path = str(checkpoint_path)
            except Exception as e:
                result.error = f"Checkpoint failed, retrain aborted: {e}"
                result.completed_at = datetime.now().isoformat()
                self._stats.total_retrains += 1
                self._stats.failed_retrains += 1
                self._data_tap.requeue(cell_name, records)
                self._history.append(result)
                logger.error(
                    "CfC checkpoint failed for %s; aborting retrain: %s",
                    cell_name, e,
                )
                return result

            # 3. Train
            effective_epochs = epochs or self._retrain_epochs
            trainer = CfCTrainer(
                cell=cell,
                seq_len=self._retrain_seq_len,
            )

            record_type = {
                "precision": TrainingRecord,
                "affect": AffectRecord,
                "attention": AttentionRecord,
                "goal": GoalRecord,
            }[cell_name]

            training_result = trainer.train(
                records,
                epochs=effective_epochs,
                record_type=record_type,
            )

            if (
                training_result.num_val_samples == 0
                or not math.isfinite(training_result.final_val_loss)
            ):
                raise RuntimeError(
                    f"Training produced no valid validation result "
                    f"(val_samples={training_result.num_val_samples}, "
                    f"val_loss={training_result.final_val_loss}); refusing "
                    "to report success"
                )

            result.training_result = training_result
            result.success = True
            result.completed_at = datetime.now().isoformat()

            # Update stats
            self._stats.total_retrains += 1
            self._stats.successful_retrains += 1
            self._stats.retrains_per_cell[cell_name] += 1
            self._stats.last_retrain_at = datetime.now().isoformat()

            logger.info(
                "CfC %s retrained: %d records, %d epochs, "
                "val_loss=%.6f",
                cell_name,
                len(records),
                effective_epochs,
                training_result.final_val_loss,
            )

        except Exception as e:
            result.success = False
            result.error = str(e)
            result.completed_at = datetime.now().isoformat()
            self._stats.total_retrains += 1
            self._stats.failed_retrains += 1
            logger.error("CfC %s retraining failed: %s", cell_name, e)
            # A failed retrain must not eat the drained training data
            if records:
                self._data_tap.requeue(cell_name, records)
            # Nor leave the cell half-trained when a checkpoint exists
            if result.checkpoint_path:
                try:
                    self.restore_cell(
                        cell_name, cell, Path(result.checkpoint_path)
                    )
                except Exception as restore_error:
                    logger.error(
                        "Rollback of %s after failed retrain also failed: %s",
                        cell_name, restore_error,
                    )

        self._history.append(result)
        return result

    def retrain_all_ready(
        self,
        cells: dict[str, torch.nn.Module],
    ) -> list[CfCRetrainingResult]:
        """Retrain all cells that have enough accumulated data.

        Args:
            cells: Dict mapping cell name to cell module.
                   e.g. {"precision": precision_cell, "affect": affect_cell, ...}

        Returns:
            List of results for each cell that was retrained.
        """
        results = []
        ready = self.cells_ready()

        for cell_name, is_ready in ready.items():
            if is_ready and cell_name in cells:
                result = self.retrain_cell(cell_name, cells[cell_name])
                results.append(result)

        return results

    def _checkpoint_cell(self, cell_name: str, cell: torch.nn.Module) -> Path:
        """Save cell state before retraining for rollback."""
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        checkpoint_path = self._checkpoint_dir / f"{cell_name}_{timestamp}.pt"
        torch.save(cell.state_dict(), checkpoint_path)
        logger.info("Checkpointed %s cell to %s", cell_name, checkpoint_path)
        return checkpoint_path

    def restore_cell(
        self,
        cell_name: str,
        cell: torch.nn.Module,
        checkpoint_path: Optional[Path] = None,
    ) -> bool:
        """Restore a cell from its most recent checkpoint.

        If no checkpoint_path is provided, uses the most recent checkpoint
        for the given cell name.

        Args:
            cell_name: Which cell to restore.
            cell: The cell module to load weights into.
            checkpoint_path: Specific checkpoint to restore from.

        Returns:
            True if restored, False if no checkpoint found.
        """
        if checkpoint_path is None:
            # Find most recent checkpoint for this cell
            checkpoint_path = self._find_latest_checkpoint(cell_name)

        if checkpoint_path is None or not checkpoint_path.exists():
            logger.warning("No checkpoint found for %s", cell_name)
            return False

        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        cell.load_state_dict(state_dict)
        logger.info("Restored %s cell from %s", cell_name, checkpoint_path)
        return True

    def _find_latest_checkpoint(self, cell_name: str) -> Optional[Path]:
        """Find the most recent checkpoint file for a cell."""
        if not self._checkpoint_dir.exists():
            return None

        checkpoints = sorted(
            self._checkpoint_dir.glob(f"{cell_name}_*.pt"),
            reverse=True,
        )
        return checkpoints[0] if checkpoints else None
