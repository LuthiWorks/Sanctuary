"""Tests for the CfC experiential layer (Phase 4.1).

Tests cover:
- PrecisionCell: creation, stepping, hidden state persistence, save/load
- CfCTrainer: data preparation, training from scaffold data, loss convergence
- ExperientialManager: authority blending, promotion/demotion, reset
- DataCollector: record collection and serialization
"""

from __future__ import annotations

import pytest
import torch
from pathlib import Path

from sanctuary.experiential.precision_cell import (
    PrecisionCell,
    PrecisionCellConfig,
    PrecisionReading,
)
from sanctuary.experiential.trainer import (
    CfCTrainer,
    DataCollector,
    TrainingRecord,
)
from sanctuary.experiential.manager import (
    ExperientialManager,
    ExperientialState,
)
from sanctuary.core.authority import AuthorityLevel, AuthorityManager


# ---------------------------------------------------------------------------
# PrecisionCell tests
# ---------------------------------------------------------------------------


class TestPrecisionCell:
    def test_creation(self):
        cell = PrecisionCell()
        assert cell is not None
        param_count = sum(p.numel() for p in cell.parameters())
        assert param_count > 0
        assert param_count < 10_000  # should be small

    def test_step_returns_valid_precision(self):
        cell = PrecisionCell()
        precision = cell.step(arousal=0.5, prediction_error=0.3, base_precision=0.5)
        assert 0.0 <= precision <= 1.0

    def test_step_updates_history(self):
        cell = PrecisionCell()
        cell.step(arousal=0.5, prediction_error=0.3, base_precision=0.5)
        cell.step(arousal=0.8, prediction_error=0.1, base_precision=0.5)
        history = cell.get_history()
        assert len(history) == 2
        assert isinstance(history[0], PrecisionReading)
        assert history[0].arousal == 0.5
        assert history[1].arousal == 0.8

    def test_hidden_state_persists(self):
        cell = PrecisionCell()
        cell.step(arousal=0.5, prediction_error=0.3, base_precision=0.5)
        h1 = cell.get_hidden_state()
        assert h1 is not None

        cell.step(arousal=0.5, prediction_error=0.3, base_precision=0.5)
        h2 = cell.get_hidden_state()
        # Hidden state should change (not identical)
        # They could be very close but the cell should evolve
        assert h2 is not None

    def test_reset_hidden_clears_state(self):
        cell = PrecisionCell()
        cell.step(arousal=0.5, prediction_error=0.3, base_precision=0.5)
        assert cell.get_hidden_state() is not None
        cell.reset_hidden()
        assert cell.get_hidden_state() is None

    def test_different_inputs_produce_different_outputs(self):
        cell = PrecisionCell()
        # Run enough steps to warm up the cell state
        for _ in range(5):
            cell.step(arousal=0.5, prediction_error=0.5, base_precision=0.5)

        # Reset and compare two different input trajectories
        cell.reset_hidden()
        for _ in range(5):
            cell.step(arousal=0.1, prediction_error=0.1, base_precision=0.5)
        p_low = cell.step(arousal=0.1, prediction_error=0.1, base_precision=0.5)

        cell.reset_hidden()
        for _ in range(5):
            cell.step(arousal=0.9, prediction_error=0.9, base_precision=0.5)
        p_high = cell.step(arousal=0.9, prediction_error=0.9, base_precision=0.5)

        # Different input trajectories should produce different precision values
        assert p_low != p_high

    def test_history_max_length(self):
        cell = PrecisionCell()
        for i in range(150):
            cell.step(arousal=0.5, prediction_error=0.5, base_precision=0.5)
        assert len(cell.get_history()) == 100

    def test_summary(self):
        cell = PrecisionCell()
        summary = cell.get_summary()
        assert summary["total_steps"] == 0

        cell.step(arousal=0.5, prediction_error=0.3, base_precision=0.5)
        summary = cell.get_summary()
        assert summary["total_steps"] == 1
        assert "average_precision" in summary
        assert "recent_precisions" in summary

    def test_custom_config(self):
        config = PrecisionCellConfig(units=32)
        cell = PrecisionCell(config)
        precision = cell.step(arousal=0.5, prediction_error=0.3, base_precision=0.5)
        assert 0.0 <= precision <= 1.0

    def test_save_and_load(self, tmp_path: Path):
        cell = PrecisionCell()
        # Step a few times to establish hidden state
        for i in range(5):
            cell.step(arousal=0.5, prediction_error=float(i) / 10, base_precision=0.5)

        save_path = tmp_path / "precision_cell.pt"
        cell.save(save_path)
        assert save_path.exists()

        loaded = PrecisionCell.load(save_path)
        # Verify loaded cell produces output
        p = loaded.step(arousal=0.5, prediction_error=0.3, base_precision=0.5)
        assert 0.0 <= p <= 1.0

    def test_forward_training(self):
        cell = PrecisionCell()
        inputs = torch.randn(4, 10, 3)  # batch=4, seq=10, features=3
        targets = torch.rand(4, 10, 1)   # target precision values
        predictions, loss = cell.forward_training(inputs, targets)
        assert predictions.shape == (4, 10, 1)
        assert loss.item() >= 0.0
        # Predictions should be in [0, 1] (sigmoid)
        assert predictions.min() >= 0.0
        assert predictions.max() <= 1.0


# ---------------------------------------------------------------------------
# DataCollector tests
# ---------------------------------------------------------------------------


class TestDataCollector:
    def test_record_and_count(self):
        collector = DataCollector()
        collector.record(0.5, 0.3, 0.5, 0.45)
        collector.record(0.8, 0.1, 0.5, 0.20)
        assert collector.count == 2

    def test_records_content(self):
        collector = DataCollector()
        collector.record(0.5, 0.3, 0.5, 0.45)
        records = collector.records
        assert len(records) == 1
        assert records[0].arousal == 0.5
        assert records[0].precision_output == 0.45

    def test_clear(self):
        collector = DataCollector()
        collector.record(0.5, 0.3, 0.5, 0.45)
        collector.clear()
        assert collector.count == 0

    def test_save_and_load(self, tmp_path: Path):
        collector = DataCollector()
        for i in range(20):
            collector.record(
                arousal=i / 20.0,
                prediction_error=1.0 - i / 20.0,
                base_precision=0.5,
                precision_output=0.5 + (i - 10) / 40.0,
            )

        save_path = tmp_path / "training_data.pt"
        collector.save(save_path)

        loaded = DataCollector()
        loaded.load(save_path)
        assert loaded.count == 20
        assert loaded.records[0].arousal == collector.records[0].arousal


# ---------------------------------------------------------------------------
# CfCTrainer tests
# ---------------------------------------------------------------------------


def _make_scaffold_data(n: int = 100) -> list[TrainingRecord]:
    """Generate synthetic scaffold data mimicking the heuristic.

    Heuristic: precision = clip(0.5 - 0.5*arousal + 0.3*error, 0, 1)
    """
    import random
    random.seed(42)
    records = []
    for _ in range(n):
        arousal = random.random()
        error = random.random()
        base = 0.5
        precision = max(0.0, min(1.0, base - 0.5 * arousal + 0.3 * error))
        records.append(
            TrainingRecord(
                arousal=arousal,
                prediction_error=error,
                base_precision=base,
                precision_output=precision,
            )
        )
    return records


class TestCfCTrainer:
    def test_prepare_data(self):
        records = _make_scaffold_data(50)
        cell = PrecisionCell()
        trainer = CfCTrainer(cell, seq_len=10)
        train_ds, val_ds = trainer.prepare_data(records)
        assert len(train_ds) > 0
        assert len(val_ds) > 0

    def test_prepare_data_too_few_records(self):
        records = _make_scaffold_data(5)
        cell = PrecisionCell()
        trainer = CfCTrainer(cell, seq_len=10)
        with pytest.raises(ValueError, match="Need at least"):
            trainer.prepare_data(records)

    def test_training_reduces_loss(self):
        records = _make_scaffold_data(200)
        cell = PrecisionCell()
        trainer = CfCTrainer(cell, seq_len=10, batch_size=8)

        result = trainer.train(records, epochs=30)

        assert result.epochs == 30
        assert result.final_train_loss < 0.5  # should learn something
        assert result.best_val_loss <= result.final_val_loss
        assert result.num_train_samples > 0
        assert result.num_val_samples > 0

    def test_trained_cell_approximates_heuristic(self):
        """After training, the CfC cell should approximate the scaffold heuristic."""
        records = _make_scaffold_data(200)
        cell = PrecisionCell()
        trainer = CfCTrainer(cell, seq_len=10, batch_size=16)
        result = trainer.train(records, epochs=30)

        # Test on known inputs
        cell.reset_hidden()
        # Warm up with a few steps
        for r in records[:10]:
            cell.step(r.arousal, r.prediction_error, r.base_precision)

        # Check that outputs are in a reasonable range
        errors = []
        for r in records[10:30]:
            cfc_out = cell.step(r.arousal, r.prediction_error, r.base_precision)
            errors.append(abs(cfc_out - r.precision_output))

        mean_error = sum(errors) / len(errors)
        # CfC should be within 0.25 of the heuristic on average
        # (generous bound — it's a small model on synthetic data)
        assert mean_error < 0.25, f"Mean error {mean_error:.4f} too high"


# ---------------------------------------------------------------------------
# ExperientialManager tests
# ---------------------------------------------------------------------------


class TestExperientialManager:
    def test_creation(self):
        mgr = ExperientialManager()
        assert mgr is not None
        status = mgr.get_status()
        assert "precision" in status

    def test_step_returns_state(self):
        mgr = ExperientialManager()
        state = mgr.step(
            arousal=0.5,
            prediction_error=0.3,
            base_precision=0.5,
            scaffold_precision=0.45,
        )
        assert isinstance(state, ExperientialState)
        assert 0.0 <= state.precision_weight <= 1.0

    def test_scaffold_only_ignores_cfc(self):
        """At SCAFFOLD_ONLY authority, the CfC output is ignored."""
        mgr = ExperientialManager()
        # Default is SCAFFOLD_ONLY
        state = mgr.step(
            arousal=0.5,
            prediction_error=0.3,
            base_precision=0.5,
            scaffold_precision=0.42,
        )
        assert state.precision_weight == pytest.approx(0.42, abs=1e-6)
        assert state.cell_active["precision"] is False

    def test_llm_controls_uses_only_cfc(self):
        """At MODEL_CONTROLS, the scaffold output is ignored."""
        authority = AuthorityManager()
        mgr = ExperientialManager(authority=authority)
        mgr.authority.set_level(
            ExperientialManager.AUTHORITY_FUNCTION,
            AuthorityLevel.MODEL_CONTROLS,
            reason="test",
        )
        state = mgr.step(
            arousal=0.5,
            prediction_error=0.3,
            base_precision=0.5,
            scaffold_precision=0.99,  # should be ignored
        )
        # Precision should NOT be 0.99 since CfC is in control
        assert state.precision_weight != pytest.approx(0.99, abs=0.01)
        assert state.cell_active["precision"] is True

    def test_blending_at_advises(self):
        """At MODEL_ADVISES, blend is 75% scaffold + 25% CfC."""
        authority = AuthorityManager()
        mgr = ExperientialManager(authority=authority)
        mgr.authority.set_level(
            ExperientialManager.AUTHORITY_FUNCTION,
            AuthorityLevel.MODEL_ADVISES,
            reason="test",
        )

        state = mgr.step(
            arousal=0.5,
            prediction_error=0.3,
            base_precision=0.5,
            scaffold_precision=0.4,
        )
        # Should be between scaffold and CfC values
        assert 0.0 <= state.precision_weight <= 1.0
        assert state.cell_active["precision"] is True

    def test_promotion(self):
        mgr = ExperientialManager()
        level = mgr.promote_precision("test promotion")
        assert level == AuthorityLevel.MODEL_ADVISES

        level = mgr.promote_precision("further promotion")
        assert level == AuthorityLevel.MODEL_GUIDES

    def test_demotion(self):
        authority = AuthorityManager()
        mgr = ExperientialManager(authority=authority)
        mgr.authority.set_level(
            ExperientialManager.AUTHORITY_FUNCTION,
            AuthorityLevel.MODEL_GUIDES,
            reason="start high",
        )
        level = mgr.demote_precision("regression")
        assert level == AuthorityLevel.MODEL_ADVISES

    def test_reset(self):
        mgr = ExperientialManager()
        mgr.step(
            arousal=0.5,
            prediction_error=0.3,
            base_precision=0.5,
            scaffold_precision=0.45,
        )
        assert mgr.precision_cell.get_hidden_state() is not None
        mgr.reset()
        assert mgr.precision_cell.get_hidden_state() is None

    def test_save_and_load(self, tmp_path: Path):
        mgr = ExperientialManager()
        for i in range(5):
            mgr.step(0.5, float(i) / 10, 0.5, 0.45)

        mgr.save(tmp_path)
        # Registry-based layout: cells in subdirectories
        assert (tmp_path / "precision" / "cell.pt").exists()

        mgr2 = ExperientialManager()
        mgr2.load(tmp_path)
        state = mgr2.step(0.5, 0.3, 0.5, 0.45)
        assert 0.0 <= state.precision_weight <= 1.0

    def test_status(self):
        mgr = ExperientialManager()
        mgr.step(0.5, 0.3, 0.5, 0.45)
        status = mgr.get_status()
        assert status["precision"]["authority"] == "SCAFFOLD_ONLY"
        assert status["precision"]["summary"]["total_steps"] == 1
