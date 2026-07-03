"""CfC cell trainer — learns from scaffold heuristic data.

The scaffold runs first, logging input/output pairs. The trainer uses
those pairs to train CfC cells via supervised learning. Once a cell
approximates the heuristic, it can generalize beyond it.

Training workflow:
    1. Run scaffold for N cycles, collecting data via DataCollector
    2. Create training sequences from collected data
    3. Train CfC cell on sequences (MSE loss, Adam optimizer)
    4. Validate: CfC output ≈ scaffold output on held-out data
    5. Wire CfC cell into cognitive cycle (ExperientialManager)

Supports all cell types: precision, affect, attention, goal.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Protocol, Sequence, runtime_checkable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

# Training defaults
DEFAULT_EPOCHS = 50
DEFAULT_LR = 1e-3
DEFAULT_BATCH_SIZE = 16
DEFAULT_SEQ_LEN = 10
DEFAULT_TRAIN_SPLIT = 0.8


# ---------------------------------------------------------------------------
# Record types — one per cell type
# ---------------------------------------------------------------------------


@dataclass
class TrainingRecord:
    """Input/output pair from the scaffold precision heuristic."""

    arousal: float
    prediction_error: float
    base_precision: float
    precision_output: float


@dataclass
class AffectRecord:
    """Input/output pair from the scaffold affect heuristic."""

    percept_valence_delta: float
    percept_arousal_delta: float
    llm_emotion_shift: float
    valence_output: float
    arousal_output: float
    dominance_output: float


@dataclass
class AttentionRecord:
    """Input/output pair from the scaffold attention heuristic."""

    goal_relevance: float
    novelty: float
    emotional_salience: float
    recency: float
    salience_output: float


@dataclass
class GoalRecord:
    """Input/output pair from the scaffold goal dynamics heuristic."""

    cycles_stalled_norm: float
    deadline_urgency: float
    emotional_congruence: float
    priority_adjustment_output: float


# ---------------------------------------------------------------------------
# Record field mappings — defines which fields are inputs vs outputs
# ---------------------------------------------------------------------------

# Maps record type -> (input_field_names, output_field_names)
RECORD_FIELDS: dict[type, tuple[list[str], list[str]]] = {
    TrainingRecord: (
        ["arousal", "prediction_error", "base_precision"],
        ["precision_output"],
    ),
    AffectRecord: (
        ["percept_valence_delta", "percept_arousal_delta", "llm_emotion_shift"],
        ["valence_output", "arousal_output", "dominance_output"],
    ),
    AttentionRecord: (
        ["goal_relevance", "novelty", "emotional_salience", "recency"],
        ["salience_output"],
    ),
    GoalRecord: (
        ["cycles_stalled_norm", "deadline_urgency", "emotional_congruence"],
        ["priority_adjustment_output"],
    ),
}


# ---------------------------------------------------------------------------
# Trainable cell protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class TrainableCell(Protocol):
    """Any CfC cell that can be trained."""

    def forward_training(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def parameters(self) -> ...: ...

    def train(self, mode: bool = True) -> nn.Module: ...

    def eval(self) -> nn.Module: ...


# ---------------------------------------------------------------------------
# Data collector — generic for any record type
# ---------------------------------------------------------------------------


class DataCollector:
    """Collects training data from the scaffold's precision weighting system.

    Attach this to the existing PrecisionWeighting instance to passively
    log every computation as a training record.
    """

    def __init__(self):
        self._records: list[TrainingRecord] = []

    def record(
        self,
        arousal: float,
        prediction_error: float,
        base_precision: float,
        precision_output: float,
    ):
        """Log one scaffold precision computation."""
        self._records.append(
            TrainingRecord(
                arousal=arousal,
                prediction_error=prediction_error,
                base_precision=base_precision,
                precision_output=precision_output,
            )
        )

    @property
    def count(self) -> int:
        return len(self._records)

    @property
    def records(self) -> list[TrainingRecord]:
        return list(self._records)

    def clear(self):
        self._records.clear()

    def save(self, path: Path):
        data = [asdict(r) for r in self._records]
        torch.save(data, path)
        logger.info("Saved %d training records to %s", len(data), path)

    def load(self, path: Path):
        # Saved as a list of plain dicts (asdict of records), so weights_only=True
        # is safe and blocks pickle RCE.
        data = torch.load(path, map_location="cpu", weights_only=True)
        self._records = [TrainingRecord(**d) for d in data]
        logger.info("Loaded %d training records from %s", len(self._records), path)


class MultiFieldCollector:
    """Generic data collector for any record type.

    Usage:
        collector = MultiFieldCollector(AffectRecord)
        collector.record(percept_valence_delta=0.2, ..., valence_output=0.3, ...)
    """

    def __init__(self, record_type: type):
        self._record_type = record_type
        self._records: list = []

    def record(self, **kwargs):
        """Log one scaffold computation as a record of the configured type."""
        self._records.append(self._record_type(**kwargs))

    @property
    def count(self) -> int:
        return len(self._records)

    @property
    def records(self) -> list:
        return list(self._records)

    def clear(self):
        self._records.clear()

    def save(self, path: Path):
        data = [asdict(r) for r in self._records]
        torch.save(data, path)
        logger.info("Saved %d records to %s", len(data), path)

    def load(self, path: Path):
        # Saved as a list of plain dicts (asdict of records), so weights_only=True
        # is safe and blocks pickle RCE.
        data = torch.load(path, map_location="cpu", weights_only=True)
        self._records = [self._record_type(**d) for d in data]
        logger.info("Loaded %d records from %s", len(self._records), path)


# ---------------------------------------------------------------------------
# Training result
# ---------------------------------------------------------------------------


@dataclass
class TrainingResult:
    """Result of a training run."""

    epochs: int
    final_train_loss: float
    final_val_loss: float
    best_val_loss: float
    best_epoch: int
    num_train_samples: int
    num_val_samples: int


# ---------------------------------------------------------------------------
# Trainer — works with any CfC cell + record type
# ---------------------------------------------------------------------------


class CfCTrainer:
    """Trains CfC cells from scaffold data.

    Works with any cell that implements forward_training(inputs, targets)
    and any record type registered in RECORD_FIELDS.
    """

    def __init__(
        self,
        cell: nn.Module,
        learning_rate: float = DEFAULT_LR,
        batch_size: int = DEFAULT_BATCH_SIZE,
        seq_len: int = DEFAULT_SEQ_LEN,
        train_split: float = DEFAULT_TRAIN_SPLIT,
    ):
        self.cell = cell
        self.lr = learning_rate
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.train_split = train_split

    def prepare_data(
        self, records: Sequence, record_type: Optional[type] = None,
    ) -> tuple[TensorDataset, TensorDataset]:
        """Convert records into sequential training/validation datasets.

        Automatically detects the record type from the first element,
        or uses the explicitly provided record_type.
        """
        if len(records) < self.seq_len:
            raise ValueError(
                f"Need at least {self.seq_len} records, got {len(records)}"
            )

        # Determine field mapping
        rtype = record_type or type(records[0])
        if rtype not in RECORD_FIELDS:
            raise ValueError(f"Unknown record type: {rtype}. Register it in RECORD_FIELDS.")
        input_fields, output_fields = RECORD_FIELDS[rtype]

        inputs = []
        targets = []

        for i in range(len(records) - self.seq_len + 1):
            seq = records[i : i + self.seq_len]
            inp = [[getattr(r, f) for f in input_fields] for r in seq]
            tgt = [[getattr(r, f) for f in output_fields] for r in seq]
            inputs.append(inp)
            targets.append(tgt)

        inputs_t = torch.tensor(inputs, dtype=torch.float32)
        targets_t = torch.tensor(targets, dtype=torch.float32)

        n = len(inputs_t)
        split_idx = int(n * self.train_split)

        train_ds = TensorDataset(inputs_t[:split_idx], targets_t[:split_idx])
        val_ds = TensorDataset(inputs_t[split_idx:], targets_t[split_idx:])

        return train_ds, val_ds

    def train(
        self,
        records: Sequence,
        epochs: int = DEFAULT_EPOCHS,
        record_type: Optional[type] = None,
    ) -> TrainingResult:
        """Train the cell on scaffold data.

        Returns a TrainingResult with loss metrics.
        """
        train_ds, val_ds = self.prepare_data(records, record_type)
        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.batch_size, shuffle=False
        )

        optimizer = torch.optim.Adam(self.cell.parameters(), lr=self.lr)

        best_val_loss = float("inf")
        best_epoch = 0
        final_train_loss = 0.0
        final_val_loss = 0.0

        self.cell.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                _, loss = self.cell.forward_training(inputs, targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            final_train_loss = epoch_loss / max(n_batches, 1)

            self.cell.eval()
            val_loss = 0.0
            n_val = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    _, loss = self.cell.forward_training(inputs, targets)
                    val_loss += loss.item()
                    n_val += 1
            final_val_loss = val_loss / max(n_val, 1)
            self.cell.train()

            if final_val_loss < best_val_loss:
                best_val_loss = final_val_loss
                best_epoch = epoch

            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    "Epoch %d/%d — train_loss=%.6f, val_loss=%.6f",
                    epoch + 1,
                    epochs,
                    final_train_loss,
                    final_val_loss,
                )

        self.cell.eval()

        result = TrainingResult(
            epochs=epochs,
            final_train_loss=final_train_loss,
            final_val_loss=final_val_loss,
            best_val_loss=best_val_loss,
            best_epoch=best_epoch,
            num_train_samples=len(train_ds),
            num_val_samples=len(val_ds),
        )

        logger.info(
            "Training complete: best_val_loss=%.6f at epoch %d",
            best_val_loss,
            best_epoch,
        )
        return result
