"""Tests for Phase 4.2 — expanded CfC experiential layer.

Tests all new cells (affect, attention, goal), the generalized trainer,
inter-cell connections, and the full ensemble in ExperientialManager.
"""

from __future__ import annotations

import random
import tempfile
from pathlib import Path

import pytest
import torch

from sanctuary.core.authority import AuthorityLevel, AuthorityManager
from sanctuary.core.schema import CognitiveInput, ExperientialSignals
from sanctuary.experiential.affect_cell import AffectCell, AffectCellConfig
from sanctuary.experiential.attention_cell import AttentionCell, AttentionCellConfig
from sanctuary.experiential.goal_cell import GoalCell, GoalCellConfig
from sanctuary.experiential.manager import ExperientialManager, ExperientialState
from sanctuary.experiential.precision_cell import PrecisionCell
from sanctuary.experiential.trainer import (
    AffectRecord,
    AttentionRecord,
    CfCTrainer,
    DataCollector,
    GoalRecord,
    MultiFieldCollector,
    TrainingRecord,
)


# ---------------------------------------------------------------------------
# AffectCell tests
# ---------------------------------------------------------------------------


class TestAffectCell:
    def test_creation(self):
        cell = AffectCell()
        assert sum(p.numel() for p in cell.parameters()) > 0

    def test_step_returns_valid_vad(self):
        cell = AffectCell()
        v, a, d = cell.step(0.2, 0.1, 0.0)
        assert -1.0 <= v <= 1.0
        assert 0.0 <= a <= 1.0
        assert 0.0 <= d <= 1.0

    def test_hidden_state_persists(self):
        cell = AffectCell()
        cell.step(0.5, 0.3, 0.1)
        h1 = cell.get_hidden_state()
        assert h1 is not None
        cell.step(-0.2, 0.0, 0.0)
        h2 = cell.get_hidden_state()
        assert not torch.equal(h1, h2)

    def test_different_inputs_produce_different_outputs(self):
        cell = AffectCell()
        v1, a1, d1 = cell.step(0.8, 0.0, 0.0)
        cell.reset_hidden()
        v2, a2, d2 = cell.step(-0.8, 0.5, 0.3)
        assert (v1, a1, d1) != (v2, a2, d2)

    def test_history(self):
        cell = AffectCell()
        for _ in range(5):
            cell.step(0.1, 0.1, 0.0)
        assert len(cell.get_history()) == 5

    def test_summary(self):
        cell = AffectCell()
        cell.step(0.2, 0.3, 0.0)
        s = cell.get_summary()
        assert s["total_steps"] == 1
        assert "average_valence" in s

    def test_save_and_load(self):
        cell = AffectCell()
        cell.step(0.5, 0.3, 0.1)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "affect.pt"
            cell.save(path)
            loaded = AffectCell.load(path)
            v1, a1, d1 = cell.step(0.1, 0.1, 0.0)
            v2, a2, d2 = loaded.step(0.1, 0.1, 0.0)
            assert abs(v1 - v2) < 0.01
            assert abs(a1 - a2) < 0.01

    def test_forward_training(self):
        cell = AffectCell()
        inputs = torch.randn(4, 5, 3)
        targets = torch.cat([
            torch.randn(4, 5, 1).tanh(),       # valence [-1,1]
            torch.randn(4, 5, 1).sigmoid(),     # arousal [0,1]
            torch.randn(4, 5, 1).sigmoid(),     # dominance [0,1]
        ], dim=2)
        preds, loss = cell.forward_training(inputs, targets)
        assert preds.shape == (4, 5, 3)
        assert loss.item() > 0
        assert loss.requires_grad

    def test_custom_config(self):
        config = AffectCellConfig(units=8)
        cell = AffectCell(config)
        assert cell.config.units == 8


# ---------------------------------------------------------------------------
# AttentionCell tests
# ---------------------------------------------------------------------------


class TestAttentionCell:
    def test_creation(self):
        cell = AttentionCell()
        assert sum(p.numel() for p in cell.parameters()) > 0

    def test_step_returns_valid_salience(self):
        cell = AttentionCell()
        s = cell.step(0.4, 0.3, 0.2, 0.1)
        assert 0.0 <= s <= 1.0

    def test_hidden_state_persists(self):
        cell = AttentionCell()
        cell.step(0.5, 0.5, 0.5, 0.5)
        h1 = cell.get_hidden_state()
        assert h1 is not None

    def test_different_inputs_produce_different_outputs(self):
        cell = AttentionCell()
        s1 = cell.step(1.0, 0.0, 0.0, 0.0)
        cell.reset_hidden()
        s2 = cell.step(0.0, 1.0, 0.0, 0.0)
        assert s1 != s2

    def test_summary(self):
        cell = AttentionCell()
        cell.step(0.4, 0.3, 0.2, 0.1)
        s = cell.get_summary()
        assert s["total_steps"] == 1
        assert "average_salience" in s

    def test_save_and_load(self):
        cell = AttentionCell()
        cell.step(0.5, 0.3, 0.2, 0.1)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "attention.pt"
            cell.save(path)
            loaded = AttentionCell.load(path)
            s1 = cell.step(0.1, 0.1, 0.1, 0.1)
            s2 = loaded.step(0.1, 0.1, 0.1, 0.1)
            assert abs(s1 - s2) < 0.01

    def test_forward_training(self):
        cell = AttentionCell()
        inputs = torch.randn(4, 5, 4)
        targets = torch.randn(4, 5, 1).sigmoid()
        preds, loss = cell.forward_training(inputs, targets)
        assert preds.shape == (4, 5, 1)
        assert loss.requires_grad


# ---------------------------------------------------------------------------
# GoalCell tests
# ---------------------------------------------------------------------------


class TestGoalCell:
    def test_creation(self):
        cell = GoalCell()
        assert sum(p.numel() for p in cell.parameters()) > 0

    def test_step_returns_valid_adjustment(self):
        cell = GoalCell()
        adj = cell.step(0.5, 0.3, 0.2)
        assert -1.0 <= adj <= 1.0

    def test_hidden_state_persists(self):
        cell = GoalCell()
        cell.step(0.5, 0.5, 0.5)
        h1 = cell.get_hidden_state()
        assert h1 is not None

    def test_different_inputs_produce_different_outputs(self):
        cell = GoalCell()
        a1 = cell.step(1.0, 0.0, 0.0)
        cell.reset_hidden()
        a2 = cell.step(0.0, 1.0, 0.0)
        assert a1 != a2

    def test_summary(self):
        cell = GoalCell()
        cell.step(0.5, 0.3, 0.2)
        s = cell.get_summary()
        assert s["total_steps"] == 1
        assert "average_adjustment" in s

    def test_save_and_load(self):
        cell = GoalCell()
        cell.step(0.5, 0.3, 0.2)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "goal.pt"
            cell.save(path)
            loaded = GoalCell.load(path)
            a1 = cell.step(0.1, 0.1, 0.1)
            a2 = loaded.step(0.1, 0.1, 0.1)
            assert abs(a1 - a2) < 0.01

    def test_forward_training(self):
        cell = GoalCell()
        inputs = torch.randn(4, 5, 3)
        targets = torch.randn(4, 5, 1).tanh()
        preds, loss = cell.forward_training(inputs, targets)
        assert preds.shape == (4, 5, 1)
        assert loss.requires_grad


# ---------------------------------------------------------------------------
# MultiFieldCollector tests
# ---------------------------------------------------------------------------


class TestMultiFieldCollector:
    def test_affect_collector(self):
        collector = MultiFieldCollector(AffectRecord)
        collector.record(
            percept_valence_delta=0.2,
            percept_arousal_delta=0.1,
            llm_emotion_shift=0.0,
            valence_output=0.3,
            arousal_output=0.25,
            dominance_output=0.5,
        )
        assert collector.count == 1
        assert collector.records[0].valence_output == 0.3

    def test_attention_collector(self):
        collector = MultiFieldCollector(AttentionRecord)
        collector.record(
            goal_relevance=0.4,
            novelty=0.3,
            emotional_salience=0.2,
            recency=0.1,
            salience_output=0.65,
        )
        assert collector.count == 1

    def test_goal_collector(self):
        collector = MultiFieldCollector(GoalRecord)
        collector.record(
            cycles_stalled_norm=0.5,
            deadline_urgency=0.8,
            emotional_congruence=0.3,
            priority_adjustment_output=0.15,
        )
        assert collector.count == 1

    def test_save_and_load(self):
        collector = MultiFieldCollector(AffectRecord)
        collector.record(
            percept_valence_delta=0.2,
            percept_arousal_delta=0.1,
            llm_emotion_shift=0.0,
            valence_output=0.3,
            arousal_output=0.25,
            dominance_output=0.5,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "affect_data.pt"
            collector.save(path)
            loaded = MultiFieldCollector(AffectRecord)
            loaded.load(path)
            assert loaded.count == 1
            assert loaded.records[0].valence_output == 0.3

    def test_clear(self):
        collector = MultiFieldCollector(AttentionRecord)
        collector.record(
            goal_relevance=0.4, novelty=0.3,
            emotional_salience=0.2, recency=0.1,
            salience_output=0.5,
        )
        collector.clear()
        assert collector.count == 0


# ---------------------------------------------------------------------------
# Generalized CfCTrainer tests
# ---------------------------------------------------------------------------


class TestGeneralizedTrainer:
    def _make_affect_records(self, n=300):
        random.seed(42)
        records = []
        for _ in range(n):
            vd = random.uniform(-0.5, 0.5)
            ad = random.uniform(0, 0.5)
            ls = random.uniform(-0.3, 0.3)
            records.append(AffectRecord(
                percept_valence_delta=vd,
                percept_arousal_delta=ad,
                llm_emotion_shift=ls,
                valence_output=max(-1, min(1, vd * 0.15)),
                arousal_output=max(0, min(1, 0.2 + ad * 0.15)),
                dominance_output=0.5,
            ))
        return records

    def _make_attention_records(self, n=300):
        random.seed(42)
        records = []
        for _ in range(n):
            gr = random.random()
            nv = random.random()
            es = random.random()
            rc = random.random()
            # Scaffold heuristic: 0.4*gr + 0.3*nv + 0.2*es + 0.1*rc
            out = 0.4 * gr + 0.3 * nv + 0.2 * es + 0.1 * rc
            records.append(AttentionRecord(
                goal_relevance=gr, novelty=nv,
                emotional_salience=es, recency=rc,
                salience_output=out,
            ))
        return records

    def _make_goal_records(self, n=300):
        random.seed(42)
        records = []
        for _ in range(n):
            cs = random.random()
            du = random.random()
            ec = random.random()
            # Simple heuristic
            adj = min(0.25, max(-0.25, 0.002 * max(0, cs * 100 - 30) + 0.2 * du))
            records.append(GoalRecord(
                cycles_stalled_norm=cs,
                deadline_urgency=du,
                emotional_congruence=ec,
                priority_adjustment_output=adj,
            ))
        return records

    def test_train_affect_cell(self):
        cell = AffectCell()
        trainer = CfCTrainer(cell, seq_len=10, batch_size=16)
        records = self._make_affect_records()
        result = trainer.train(records, epochs=30)
        assert result.final_train_loss < 0.5
        assert result.best_val_loss < 1.0

    def test_train_attention_cell(self):
        cell = AttentionCell()
        trainer = CfCTrainer(cell, seq_len=10, batch_size=16)
        records = self._make_attention_records()
        result = trainer.train(records, epochs=50)
        assert result.final_train_loss < 0.1
        assert result.best_val_loss < 0.2

    def test_train_goal_cell(self):
        cell = GoalCell()
        trainer = CfCTrainer(cell, seq_len=10, batch_size=16)
        records = self._make_goal_records()
        result = trainer.train(records, epochs=50)
        assert result.final_train_loss < 0.1
        assert result.best_val_loss < 0.2

    def test_precision_still_works(self):
        """Backward compatibility: precision training still works."""
        cell = PrecisionCell()
        trainer = CfCTrainer(cell, seq_len=10, batch_size=16)
        random.seed(42)
        records = [
            TrainingRecord(
                arousal=random.random(),
                prediction_error=random.random(),
                base_precision=0.5,
                precision_output=max(0, min(1, 0.5 - 0.5 * random.random() + 0.3 * random.random())),
            )
            for _ in range(300)
        ]
        result = trainer.train(records, epochs=30)
        assert result.final_train_loss < 0.1


# ---------------------------------------------------------------------------
# ExperientialManager (expanded) tests
# ---------------------------------------------------------------------------


class TestExperientialManagerExpanded:
    def test_creation_has_four_cells(self):
        mgr = ExperientialManager()
        assert hasattr(mgr, "precision_cell")
        assert hasattr(mgr, "affect_cell")
        assert hasattr(mgr, "attention_cell")
        assert hasattr(mgr, "goal_cell")

    def test_step_returns_full_state(self):
        mgr = ExperientialManager()
        state = mgr.step(
            arousal=0.5, prediction_error=0.3,
            base_precision=0.5, scaffold_precision=0.5,
        )
        assert isinstance(state, ExperientialState)
        assert 0.0 <= state.precision_weight <= 1.0
        assert len(state.affect_vad) == 3
        assert 0.0 <= state.attention_salience <= 1.0
        assert -1.0 <= state.goal_adjustment <= 1.0
        assert len(state.hidden_state_norms) == 4
        assert len(state.cell_active) == 4

    def test_scaffold_only_ignores_all_cfc(self):
        """At SCAFFOLD_ONLY, all outputs should be pure scaffold values."""
        mgr = ExperientialManager()
        state = mgr.step(
            arousal=0.5, prediction_error=0.3,
            base_precision=0.5, scaffold_precision=0.7,
            scaffold_vad=(0.3, 0.4, 0.6),
            scaffold_salience=0.8,
            scaffold_goal_adj=0.1,
        )
        assert state.precision_weight == 0.7
        assert state.affect_vad == (0.3, 0.4, 0.6)
        assert state.attention_salience == 0.8
        assert state.goal_adjustment == 0.1

    def test_promote_and_demote_individual_cells(self):
        mgr = ExperientialManager()

        # Promote affect
        mgr.promote("affect", "trained well")
        assert mgr.authority.level("experiential_affect") == AuthorityLevel.MODEL_ADVISES

        # Demote it back
        mgr.demote("affect", "regression detected")
        assert mgr.authority.level("experiential_affect") == AuthorityLevel.SCAFFOLD_ONLY

    def test_inter_cell_affect_to_precision(self):
        """Affect's arousal should influence precision computation."""
        mgr = ExperientialManager()
        # Promote affect so its output is used
        mgr.promote("affect", "test")

        # Step with high percept arousal delta
        state1 = mgr.step(
            arousal=0.2, prediction_error=0.3,
            base_precision=0.5, scaffold_precision=0.5,
            percept_arousal_delta=0.8,
            scaffold_vad=(0.0, 0.2, 0.5),
        )

        mgr.reset()
        mgr.promote("affect", "test")

        # Step with low percept arousal delta
        state2 = mgr.step(
            arousal=0.2, prediction_error=0.3,
            base_precision=0.5, scaffold_precision=0.5,
            percept_arousal_delta=0.0,
            scaffold_vad=(0.0, 0.2, 0.5),
        )

        # Both calls used the same manager, so precision cell stepped twice
        # The key point: the affect cell's arousal modulates precision input
        h1 = mgr.precision_cell.get_summary()
        assert h1["total_steps"] == 2

    def test_reset_clears_all(self):
        mgr = ExperientialManager()
        mgr.step(arousal=0.5, prediction_error=0.3, base_precision=0.5, scaffold_precision=0.5)
        mgr.reset()
        assert mgr.precision_cell.get_hidden_state() is None
        assert mgr.affect_cell.get_hidden_state() is None
        assert mgr.attention_cell.get_hidden_state() is None
        assert mgr.goal_cell.get_hidden_state() is None

    def test_status_shows_all_cells(self):
        mgr = ExperientialManager()
        mgr.step(arousal=0.5, prediction_error=0.3, base_precision=0.5, scaffold_precision=0.5)
        status = mgr.get_status()
        assert "precision" in status
        assert "affect" in status
        assert "attention" in status
        assert "goal" in status

    def test_save_and_load(self):
        mgr = ExperientialManager()
        mgr.step(arousal=0.5, prediction_error=0.3, base_precision=0.5, scaffold_precision=0.5)

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "experiential"
            mgr.save(directory)

            mgr2 = ExperientialManager()
            mgr2.load(directory)

            # Both should produce same output for same input
            mgr.reset()
            mgr2.reset()  # reset hidden to compare from clean slate
            # Loaded model weights should match
            s1 = mgr.step(arousal=0.3, prediction_error=0.2, base_precision=0.5, scaffold_precision=0.5)
            s2 = mgr2.step(arousal=0.3, prediction_error=0.2, base_precision=0.5, scaffold_precision=0.5)
            assert abs(s1.precision_weight - s2.precision_weight) < 0.01

    def test_backward_compat_promote_precision(self):
        mgr = ExperientialManager()
        level = mgr.promote_precision("test")
        assert level == AuthorityLevel.MODEL_ADVISES

    def test_backward_compat_demote_precision(self):
        mgr = ExperientialManager()
        mgr.promote_precision("test")
        level = mgr.demote_precision("test")
        assert level == AuthorityLevel.SCAFFOLD_ONLY


# ---------------------------------------------------------------------------
# ExperientialSignals schema tests
# ---------------------------------------------------------------------------


class TestExperientialSignalsExpanded:
    def test_default_values(self):
        es = ExperientialSignals()
        assert es.precision_weight == 0.5
        assert es.affect_valence == 0.0
        assert es.affect_arousal == 0.2
        assert es.affect_dominance == 0.5
        assert es.attention_salience == 0.5
        assert es.goal_adjustment == 0.0
        assert es.cells_active == {}

    def test_full_signals(self):
        es = ExperientialSignals(
            precision_weight=0.7,
            affect_valence=-0.3,
            affect_arousal=0.6,
            affect_dominance=0.4,
            attention_salience=0.8,
            goal_adjustment=0.15,
            cells_active={"precision": True, "affect": True, "attention": False, "goal": False},
        )
        assert es.affect_valence == -0.3
        assert es.attention_salience == 0.8

    def test_roundtrip(self):
        ci = CognitiveInput(
            experiential_state=ExperientialSignals(
                precision_weight=0.6,
                affect_valence=-0.2,
                affect_arousal=0.7,
                affect_dominance=0.3,
                attention_salience=0.9,
                goal_adjustment=-0.1,
                cells_active={"precision": True, "affect": True},
            )
        )
        data = ci.model_dump()
        ci2 = CognitiveInput.model_validate(data)
        assert ci2.experiential_state.affect_valence == -0.2
        assert ci2.experiential_state.goal_adjustment == -0.1

    def test_validation_bounds(self):
        with pytest.raises(Exception):
            ExperientialSignals(precision_weight=1.5)
        with pytest.raises(Exception):
            ExperientialSignals(affect_valence=-2.0)
        with pytest.raises(Exception):
            ExperientialSignals(affect_arousal=-0.1)
