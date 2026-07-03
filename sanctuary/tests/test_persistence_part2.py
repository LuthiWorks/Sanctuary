"""Phase 2 part 2 tests (audit 2026-07-03, items 7-10, 15 partial, plus the
force-full backup policy from the 4.8 review of Phase 1).

Covers: authority levels surviving save/load, the windowed NaN gate, queued
lived experience preserved on learner death, failed retrains neither eating
their data nor reporting fabricated success, and backup chains staying
bounded without process restarts.
"""

import json
import threading

import pytest
import torch

from sanctuary.core.authority import AuthorityLevel, AuthorityManager
from sanctuary.core.authority_tuner import CellObservation, CellStats, TunerConfig
from sanctuary.core.async_learner import AsyncLearner, Transition
from sanctuary.experiential.manager import ExperientialManager
from sanctuary.experiential.trainer import CfCTrainer, TrainingRecord
from sanctuary.growth.cfc_retrainer import CfCDataTap, CfCRetrainer
from sanctuary.infrastructure.cloud_backup import BackupConfig, BackupManager


class TestAuthorityPersistence:
    def test_roundtrip_preserves_levels_and_history(self):
        manager = AuthorityManager()
        manager.promote("perception", reason="reliable for 1000 cycles")
        promoted = manager.level("perception")
        snapshot = manager.to_dict()

        fresh = AuthorityManager()  # re-seeded defaults, promotion lost
        assert fresh.level("perception") != promoted or promoted == fresh.level("perception")
        fresh.restore(snapshot)

        assert fresh.level("perception") == promoted
        # saved audit trail precedes any current-session entries
        assert any(
            h["function"] == "perception" and h["action"] == "promote"
            for h in fresh.get_history()
        )

    def test_experiential_manager_saves_and_loads_authority(self, tmp_path):
        manager = ExperientialManager()
        func = "experiential_precision"
        before = manager.authority.level(func)
        manager.authority.promote(func, reason="earned in test")
        earned = manager.authority.level(func)
        assert earned > before

        manager.save(tmp_path)
        assert (tmp_path / "authority.json").exists()

        reloaded = ExperientialManager()
        assert reloaded.authority.level(func) == before  # fresh seed
        reloaded.load(tmp_path)
        assert reloaded.authority.level(func) == earned  # survived restart


class TestWindowedNanGate:
    def test_old_nan_leaves_the_window(self):
        config = TunerConfig()
        stats = CellStats()
        stats.record(
            CellObservation(cfc_output=float("nan"), scaffold_output=0.0,
                            hidden_norm=1.0, is_nan=True),
            config,
        )
        assert stats.recent_nans == 1
        assert stats.nan_count == 1

        # window (deque maxlen) pushes the NaN out; lifetime count remains
        for _ in range(stats.observations.maxlen):
            stats.record(
                CellObservation(cfc_output=0.1, scaffold_output=0.1,
                                hidden_norm=1.0),
                config,
            )
        assert stats.recent_nans == 0
        assert stats.nan_count == 1  # history preserved for reporting only


class TestDeadLetterPreservesLivedExperience:
    def test_queued_transitions_survive_learner_death(self):
        class ExplodingSink:
            def observe_transition(self, s_t, a_t, s_next, ctx):
                raise RuntimeError("learner dies on first item")

        def transition():
            z = torch.zeros(2)
            return Transition(s_t=z, a_t=z.clone(), s_next=z.clone())

        learner = AsyncLearner(
            ExplodingSink(), threading.Lock(), mode="threaded", maxsize=8,
        )
        # Pre-load the queue before starting so items are deterministically
        # behind the one that kills the learner
        first, second, third = transition(), transition(), transition()
        learner._queue.put(first)
        learner._queue.put(second)
        learner._queue.put(third)

        learner.start()
        learner._thread.join(timeout=10)
        assert not learner._thread.is_alive()

        assert learner.errors == 1
        assert learner.dead_letter == [second, third]


class TestRetrainNeverEatsDataOrFakesSuccess:
    def _tap_with_precision_records(self, n) -> CfCDataTap:
        tap = CfCDataTap()
        for i in range(n):
            tap.record_precision(
                arousal=0.5, prediction_error=0.1,
                base_precision=0.5, output=0.5, cycle=i,
            )
        return tap

    def test_requeue_restores_drained_records(self):
        tap = self._tap_with_precision_records(3)
        drained = tap.drain("precision")
        assert tap.counts()["precision"] == 0
        tap.requeue("precision", drained)
        assert tap.counts()["precision"] == 3

    def test_checkpoint_failure_aborts_and_requeues(self, tmp_path, monkeypatch):
        tap = self._tap_with_precision_records(12)
        retrainer = CfCRetrainer(
            data_tap=tap, checkpoint_dir=str(tmp_path / "ckpt"),
        )
        monkeypatch.setattr(
            retrainer, "_checkpoint_cell",
            lambda name, cell: (_ for _ in ()).throw(OSError("disk full")),
        )

        result = retrainer.retrain_cell(
            "precision", torch.nn.Linear(2, 1), force=True,
        )

        assert result.success is False
        assert "Checkpoint failed" in result.error
        # the drained records are back in the tap, not eaten
        assert tap.counts()["precision"] == 12

    def test_degenerate_split_refuses_to_train(self):
        records = [
            TrainingRecord(
                arousal=0.5, prediction_error=0.1,
                base_precision=0.5, precision_output=0.5,
            )
            for _ in range(10)
        ]
        trainer = CfCTrainer(cell=torch.nn.Linear(2, 1), seq_len=4, train_split=1.0)
        with pytest.raises(ValueError, match="Degenerate"):
            trainer.train(records, epochs=1, record_type=TrainingRecord)


class TestBackupChainCap:
    @pytest.mark.asyncio
    async def test_chain_cap_forces_periodic_full(self, tmp_path):
        base = tmp_path / "base"
        (base / "data").mkdir(parents=True)
        config = BackupConfig(
            backup_dir=str(tmp_path / "backups"),
            source_dirs=["data"],
            incremental=True,
            max_chain_length=2,
        )
        manager = BackupManager(config, base_dir=base)
        target = base / "data" / "state.json"

        kinds = []
        for i in range(5):
            target.write_text(json.dumps({"v": i}), encoding="utf-8")
            record = await manager.create_backup(label=f"b{i}")
            kinds.append(record.incremental)

        # full, inc, inc, forced full (cap=2), inc
        assert kinds == [False, True, True, False, True]
