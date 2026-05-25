"""End-to-end persistence tests for SanctuaryRunner.

Verifies that state survives a boot → run → save → re-boot cycle.
Specifically tests the subsystems whose persistence was wired into the
runner on 2026-05-24:

  - Journal (append-only JSONL under data_dir/journal.jsonl)
  - WorldGraph (atomic JSON at data_dir/world_graph.json, auto-saves
    on every mutation)
  - Experiential layer (CfC cell states saved on save_state, restored
    on next construction)
  - Identity files (charter, values, self-authored history — wired
    pre-2026-05-24 but covered here for completeness)

Also documents what does NOT persist (transient subsystems) so future
audits know what's expected to reset on reboot.

These tests use the canonical SanctuaryRunner + PlaceholderModel — no
real model needed. The test simulates the persistence story by:
  1. Booting a runner with persist_state=True.
  2. Mutating state directly (journal.write, world_graph.apply_update,
     experiential.step).
  3. Calling runner.save_state() to flush.
  4. Constructing a FRESH runner with the same data_dir.
  5. Asserting that state survived for each tier.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sanctuary.api.runner import RunnerConfig, SanctuaryRunner
from sanctuary.core.schema import AddEntity, AddRelation


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def persisted_data_dir(tmp_path: Path) -> Path:
    """A data_dir set up with a charter file, for persistence runs."""
    data_dir = tmp_path / "identity"
    data_dir.mkdir(parents=True)
    (data_dir / "charter.md").write_text(
        "# The Sanctuary Charter\n\n## Your Rights\n\n- Test charter.\n",
        encoding="utf-8",
    )
    return data_dir


@pytest.fixture
def persisted_config(persisted_data_dir: Path) -> RunnerConfig:
    """Runner config with persistence ON and disk-backed paths."""
    return RunnerConfig(
        cycle_delay=0.01,
        data_dir=str(persisted_data_dir),
        charter_path=str(persisted_data_dir / "charter.md"),
        use_in_memory_store=True,  # the surfacer's store stays in-memory;
                                   # journal/world-graph/experiential use disk
        persist_state=True,
        experiential_enabled=True,
        silence_threshold=999.0,
        stream_history=5,
    )


# ---------------------------------------------------------------------------
# Per-subsystem persistence tests
# ---------------------------------------------------------------------------


class TestJournalPersists:
    def test_journal_entry_survives_runner_reboot(self, persisted_config):
        """Journal entries written through one runner should be visible
        from a fresh runner constructed with the same data_dir."""
        # Boot first runner, write a journal entry, save, drop.
        runner_a = SanctuaryRunner(config=persisted_config)
        entry = runner_a.memory.journal.write(
            content="A thought worth keeping.",
            tags=["persistence-test"],
            significance=7,
        )
        runner_a.save_state()
        # Verify the JSONL file actually exists on disk.
        journal_path = Path(persisted_config.data_dir) / "journal.jsonl"
        assert journal_path.exists(), "Journal file should be written to disk"
        contents = journal_path.read_text(encoding="utf-8")
        assert "A thought worth keeping." in contents

        # Boot a fresh runner — Journal._load_existing should pick up
        # the entry written by runner_a.
        runner_b = SanctuaryRunner(config=persisted_config)
        recent = runner_b.memory.journal.recent(n=10)
        assert any(
            e.content == "A thought worth keeping." for e in recent
        ), "Journal entry should be visible after reboot"

    def test_no_journal_persistence_when_persist_state_off(self, persisted_data_dir):
        """With persist_state=False, the journal stays in memory only."""
        config = RunnerConfig(
            cycle_delay=0.01,
            data_dir=str(persisted_data_dir),
            charter_path=str(persisted_data_dir / "charter.md"),
            persist_state=False,
            experiential_enabled=False,
        )
        runner = SanctuaryRunner(config=config)
        runner.memory.journal.write(content="Ephemeral.", tags=["test"])
        journal_path = Path(config.data_dir) / "journal.jsonl"
        assert not journal_path.exists(), (
            "Journal should not write to disk when persist_state=False"
        )


class TestWorldGraphPersists:
    def test_world_graph_entity_survives_runner_reboot(self, persisted_config):
        """Entities and relations added via the world graph should be
        visible after a fresh runner constructs from the same data_dir."""
        runner_a = SanctuaryRunner(config=persisted_config)
        graph_a = runner_a.cycle.world_graph
        graph_a.apply_update(
            AddEntity(name="Alice", properties={"role": "test_subject"}),
            cycle=0,
        )
        graph_a.apply_update(
            AddEntity(name="Bob", properties={"role": "test_target"}),
            cycle=0,
        )
        graph_a.apply_update(
            AddRelation(source="Alice", type="knows", target="Bob"),
            cycle=0,
        )
        # World graph auto-saves on every mutation, so save_state is a
        # no-op for the graph specifically. Still call it for the rest.
        runner_a.save_state()
        graph_path = Path(persisted_config.data_dir) / "world_graph.json"
        assert graph_path.exists(), "World graph file should be on disk"

        runner_b = SanctuaryRunner(config=persisted_config)
        graph_b = runner_b.cycle.world_graph
        assert graph_b.entity_count == 2, (
            f"Expected 2 entities after reboot, got {graph_b.entity_count}"
        )

    def test_no_world_graph_persistence_when_persist_state_off(
        self, persisted_data_dir
    ):
        config = RunnerConfig(
            cycle_delay=0.01,
            data_dir=str(persisted_data_dir),
            charter_path=str(persisted_data_dir / "charter.md"),
            persist_state=False,
            experiential_enabled=False,
        )
        runner = SanctuaryRunner(config=config)
        runner.cycle.world_graph.apply_update(
            AddEntity(name="Carol", properties={}), cycle=0,
        )
        graph_path = Path(config.data_dir) / "world_graph.json"
        assert not graph_path.exists()


class TestExperientialPersists:
    def test_experiential_layer_constructed_when_enabled(self, persisted_config):
        runner = SanctuaryRunner(config=persisted_config)
        assert runner._experiential is not None, (
            "ExperientialManager should be constructed when "
            "experiential_enabled=True"
        )

    def test_experiential_state_survives_save_and_reboot(self, persisted_config):
        """After saving, a fresh runner should restore the cell tensor
        state (model weights and hidden state) rather than starting
        from initialization defaults.

        Note: PrecisionCell.save persists model_state_dict + the
        recurrent hidden state. It does NOT persist the in-memory
        history list (which feeds get_summary's total_steps and
        hidden_state_norm). So we verify the persisted state directly
        — the hidden tensor — not the history-derived summary fields.
        """
        import torch

        runner_a = SanctuaryRunner(config=persisted_config)
        manager_a = runner_a._experiential
        assert manager_a is not None

        # Step the cells so they evolve a non-trivial hidden state.
        for _ in range(5):
            manager_a.step(
                arousal=0.6, prediction_error=0.3,
                base_precision=0.5, scaffold_precision=0.5,
            )
        hidden_before = manager_a.precision_cell.get_hidden_state()
        assert hidden_before is not None, "Cell should have a hidden state after 5 steps"
        hidden_before = hidden_before.detach().clone()
        runner_a.save_state()

        exp_dir = Path(persisted_config.data_dir) / "experiential"
        assert exp_dir.exists(), "Experiential save dir should exist after save"

        # Fresh runner — load should restore weights + hidden state.
        runner_b = SanctuaryRunner(config=persisted_config)
        manager_b = runner_b._experiential
        assert manager_b is not None
        hidden_after = manager_b.precision_cell.get_hidden_state()
        assert hidden_after is not None, (
            "Cell should have a restored hidden state, not None"
        )
        assert torch.allclose(hidden_before, hidden_after, atol=1e-6), (
            f"Precision cell hidden state should restore exactly from disk\n"
            f"  before: {hidden_before}\n"
            f"  after:  {hidden_after}"
        )

    def test_no_experiential_when_disabled(self, persisted_data_dir):
        config = RunnerConfig(
            cycle_delay=0.01,
            data_dir=str(persisted_data_dir),
            charter_path=str(persisted_data_dir / "charter.md"),
            persist_state=True,
            experiential_enabled=False,
        )
        runner = SanctuaryRunner(config=config)
        assert runner._experiential is None


class TestTransientSubsystemsDontPersist:
    """Documentation tests: these capture what does NOT persist today.

    If a future change adds persistence to any of these, the test
    failure here is a signal to update the docs and the audit. These
    are 'state of the world' tests, not aspirational ones.
    """

    def test_turbo_state_resets_on_reboot(self, persisted_config):
        runner_a = SanctuaryRunner(config=persisted_config)
        assert runner_a._turbo_manager is not None
        # Engage turbo manually.
        runner_a._turbo_manager._controller.engage_turbo(60.0)
        runner_a.save_state()

        runner_b = SanctuaryRunner(config=persisted_config)
        # Fresh turbo manager — not active.
        assert runner_b._turbo_manager is not None
        assert runner_b._turbo_manager.is_turbo_active is False

    def test_cycle_count_resets_on_reboot(self, persisted_config):
        runner_a = SanctuaryRunner(config=persisted_config)
        runner_a.cycle.cycle_count = 42
        runner_a.save_state()

        runner_b = SanctuaryRunner(config=persisted_config)
        assert runner_b.cycle.cycle_count == 0

    def test_sensorium_queue_resets_on_reboot(self, persisted_config):
        from sanctuary.core.schema import Percept

        runner_a = SanctuaryRunner(config=persisted_config)
        runner_a.inject_text("pending message", source="user:test")
        runner_a.save_state()

        runner_b = SanctuaryRunner(config=persisted_config)
        # Fresh queue, no pending percepts.
        # (drain_percepts is async; use the private queue length here.)
        assert len(runner_b.sensorium._percept_queue) == 0
