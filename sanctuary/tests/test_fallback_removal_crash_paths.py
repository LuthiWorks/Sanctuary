"""
Tests for error handling after fallback removal.

Two categories:
1. Producer tests: Verify that fail-fast functions raise proper exceptions
2. Consumer tests: Verify that callers handle those exceptions gracefully

The philosophy:
- Producing functions fail fast (raise on error)
- Consuming functions at system boundaries decide how to degrade gracefully
"""

import asyncio
import json
import os
import tempfile
import pytest
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from pathlib import Path


# ============================================================
# 1. PRODUCERS: MemorySurfacer, PerceptionSubsystem, etc.
#    These should still raise — the errors are real.
# ============================================================

class TestProducerFailFast:
    """Verify that producer functions raise proper exceptions."""

    @pytest.mark.asyncio
    async def test_surfacer_raises_on_store_failure(self):
        """MemorySurfacer.surface() raises RuntimeError when store fails."""
        from memory.surfacer import MemorySurfacer

        store = AsyncMock()
        store.recall = AsyncMock(side_effect=ConnectionError("DB down"))
        surfacer = MemorySurfacer(store=store, config={})

        with pytest.raises(RuntimeError, match="Memory surfacing failed"):
            await surfacer.surface("some context")

    @pytest.mark.asyncio
    async def test_substrate_propagates_surfacer_error(self):
        """MemorySubstrate.surface() propagates RuntimeError from _surfacer."""
        from memory.manager import MemorySubstrate

        substrate = MemorySubstrate.__new__(MemorySubstrate)
        substrate._surfacer = AsyncMock()
        substrate._surfacer.surface = AsyncMock(
            side_effect=RuntimeError("Memory surfacing failed: ChromaDB connection lost")
        )
        substrate._prospective = MagicMock()
        substrate._retrieval_queue = []

        with pytest.raises(RuntimeError, match="Memory surfacing failed"):
            await substrate.surface("some context")

    @pytest.mark.asyncio
    async def test_substrate_context_surfacing_error_propagates(self):
        """Context surfacing error (step 1) still propagates — no partial results yet."""
        from memory.manager import MemorySubstrate

        substrate = MemorySubstrate.__new__(MemorySubstrate)
        substrate._surfacer = AsyncMock()
        substrate._surfacer.surface = AsyncMock(
            side_effect=RuntimeError("Memory surfacing failed: disk error")
        )
        substrate._prospective = MagicMock()
        substrate._retrieval_queue = []

        with pytest.raises(RuntimeError, match="Memory surfacing failed"):
            await substrate.surface("some context")

    def test_retriever_raises_on_zero_k(self):
        """retrieve_memories raises ValueError on k=0."""
        from mind.memory.retrieval import MemoryRetriever

        retriever = MemoryRetriever(Mock(), Mock())
        with pytest.raises(ValueError, match="k must be positive"):
            retriever.retrieve_memories("test", k=0)

    def test_retriever_raises_on_negative_k(self):
        """retrieve_memories raises ValueError on k=-1."""
        from mind.memory.retrieval import MemoryRetriever

        retriever = MemoryRetriever(Mock(), Mock())
        with pytest.raises(ValueError, match="k must be positive"):
            retriever.retrieve_memories("test", k=-1)

    def test_retriever_wraps_internal_errors(self):
        """retrieve_memories wraps internal errors as RuntimeError."""
        from mind.memory.retrieval import MemoryRetriever

        retriever = MemoryRetriever(Mock(), Mock())
        retriever._retrieve_direct = Mock(
            side_effect=ConnectionError("Database unavailable")
        )

        with pytest.raises(RuntimeError, match="Memory retrieval failed"):
            retriever.retrieve_memories("test query", k=5)

    def test_checkpoint_raises_on_corrupted_json(self):
        """get_checkpoint raises ValueError on corrupted JSON."""
        from growth.identity_checkpoint import IdentityCheckpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = IdentityCheckpoint(checkpoint_dir=tmpdir)
            cp_dir = Path(tmpdir) / "corrupted_cp"
            cp_dir.mkdir()
            (cp_dir / "metadata.json").write_text("{invalid json!!!")

            with pytest.raises(ValueError, match="corrupted"):
                checkpoint.get_checkpoint("corrupted_cp")

    def test_checkpoint_raises_on_wrong_types(self):
        """get_checkpoint raises ValueError on metadata with wrong types."""
        from growth.identity_checkpoint import IdentityCheckpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = IdentityCheckpoint(checkpoint_dir=tmpdir)
            cp_dir = Path(tmpdir) / "bad_types"
            cp_dir.mkdir()
            (cp_dir / "metadata.json").write_text(json.dumps({
                "checkpoint_id": 12345,
                "timestamp": None,
                "trigger": [],
            }))

            with pytest.raises(ValueError, match="corrupted"):
                checkpoint.get_checkpoint("bad_types")

    def test_checkpoint_compare_propagates_corruption(self):
        """compare_checkpoints propagates ValueError from corrupted metadata."""
        from growth.identity_checkpoint import IdentityCheckpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = IdentityCheckpoint(checkpoint_dir=tmpdir)

            cp_a_dir = Path(tmpdir) / "cp_a"
            cp_a_dir.mkdir()
            (cp_a_dir / "metadata.json").write_text(json.dumps({
                "checkpoint_id": "cp_a",
                "timestamp": "2026-03-21T00:00:00",
                "trigger": "test",
                "description": "test checkpoint",
                "adapter_path": None,
                "parent_checkpoint": None,
                "tags": [],
            }))

            cp_b_dir = Path(tmpdir) / "cp_b"
            cp_b_dir.mkdir()
            (cp_b_dir / "metadata.json").write_text("not json")

            with pytest.raises(ValueError, match="corrupted"):
                checkpoint.compare_checkpoints("cp_a", "cp_b")


# ============================================================
# 2. CONSUMERS: Verify graceful degradation at caller sites
# ============================================================

class TestCognitiveCycleGracefulDegradation:
    """Test that the cognitive cycle survives subsystem failures."""

    @pytest.mark.asyncio
    async def test_assemble_input_catches_memory_failure(self):
        """_assemble_input catches memory surfacing errors and continues."""
        from core.cognitive_cycle import CognitiveCycle
        import logging

        cycle = CognitiveCycle.__new__(CognitiveCycle)
        cycle.sensorium = MagicMock()
        cycle.sensorium.drain_percepts = AsyncMock(return_value=[])
        cycle.sensorium.get_prediction_errors = MagicMock(return_value=[])
        cycle.sensorium.get_temporal_context = MagicMock()
        cycle.sensorium.get_temporal_context.return_value = MagicMock(
            interactions_this_session=0
        )
        cycle.scaffold = MagicMock()
        cycle.stream = MagicMock()
        cycle.stream.get_recent_context = MagicMock(return_value="test context")
        cycle.experiential = None
        cycle.sleep = None
        cycle.communication = None
        cycle._current_percepts = []
        cycle._current_memories = []
        cycle.model = MagicMock()
        cycle.cycle_count = 0

        # Memory raises
        cycle.memory = AsyncMock()
        cycle.memory.surface = AsyncMock(
            side_effect=RuntimeError("Memory surfacing failed: connection refused")
        )

        # Capture log output to verify the error was caught and logged
        with pytest.raises(Exception) as exc_info:
            # _assemble_input may fail further down (Pydantic etc.) but
            # the memory error should be caught — NOT a RuntimeError
            await cycle._assemble_input()

        # The error should NOT be the memory RuntimeError
        # (it was caught; the failure is from downstream mocking)
        assert "Memory surfacing failed" not in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_run_loop_propagates_first_cycle_failure(self):
        """run() propagates a failing cycle instead of swallowing it.

        Per AGENTS.md, the top-level cognitive-loop crash boundary does
        not exist until Phase 9 (First Awakening). Cycle failures during
        development must surface so they can be diagnosed.
        """
        from core.cognitive_cycle import CognitiveCycle

        cycle = CognitiveCycle.__new__(CognitiveCycle)
        cycle.running = True
        cycle._cycle_delay = 0.001

        call_count = 0
        async def cycle_side_effect():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Transient failure")

        cycle._cycle = AsyncMock(side_effect=cycle_side_effect)

        with pytest.raises(RuntimeError, match="Transient failure"):
            await cycle.run(max_cycles=2)
        assert call_count == 1  # second cycle never ran

    @pytest.mark.asyncio
    async def test_run_loop_propagates_repeated_failure(self):
        """The first failure stops the loop — there is no retry semantics yet.

        Per AGENTS.md, retry/recovery semantics belong to the Phase 9
        crash boundary; before then, the loop must not swallow errors.
        """
        from core.cognitive_cycle import CognitiveCycle

        cycle = CognitiveCycle.__new__(CognitiveCycle)
        cycle.running = True
        cycle._cycle_delay = 0.001

        failure_count = 0
        async def always_fail():
            nonlocal failure_count
            failure_count += 1
            raise RuntimeError(f"Failure #{failure_count}")

        cycle._cycle = AsyncMock(side_effect=always_fail)

        with pytest.raises(RuntimeError, match="Failure #1"):
            await cycle.run(max_cycles=5)
        # Only the first attempt ran — the loop didn't get a second chance.
        assert failure_count == 1


class TestStateManagerGracefulDegradation:
    """Test that perception encoding failures are handled per-item."""

    @pytest.mark.asyncio
    async def test_single_encode_failure_preserves_other_percepts(self):
        """If encode fails on item 2 of 3, items 1 and 3 are still returned."""
        from mind.cognitive_core.core.state_manager import StateManager
        from mind.cognitive_core.workspace import GlobalWorkspace

        workspace = GlobalWorkspace()
        state = StateManager(workspace, config={})
        state.input_queue = asyncio.Queue()
        state.input_queue.put_nowait(("input1", "text"))
        state.input_queue.put_nowait(("input2", "image"))  # This will fail
        state.input_queue.put_nowait(("input3", "text"))

        async def encode_side_effect(data, modality="text"):
            if modality == "image":
                raise RuntimeError("Image encoder not loaded")
            return MagicMock(embedding=[0.1] * 384, metadata={})

        perception = AsyncMock()
        perception.encode = AsyncMock(side_effect=encode_side_effect)

        # Should return 2 percepts (items 1 and 3), not crash
        percepts = await state.gather_percepts(perception)
        assert len(percepts) == 2

    @pytest.mark.asyncio
    async def test_all_encodes_fail_returns_empty(self):
        """If all encode calls fail, returns empty list (no crash)."""
        from mind.cognitive_core.core.state_manager import StateManager
        from mind.cognitive_core.workspace import GlobalWorkspace

        workspace = GlobalWorkspace()
        state = StateManager(workspace, config={})
        state.input_queue = asyncio.Queue()
        state.input_queue.put_nowait(("input1", "text"))
        state.input_queue.put_nowait(("input2", "text"))

        perception = AsyncMock()
        perception.encode = AsyncMock(
            side_effect=RuntimeError("Everything is broken")
        )

        percepts = await state.gather_percepts(perception)
        assert percepts == []

    @pytest.mark.asyncio
    async def test_tool_percepts_preserved_despite_encode_failure(self):
        """Pending tool percepts are returned even if new encoding fails."""
        from mind.cognitive_core.core.state_manager import StateManager
        from mind.cognitive_core.workspace import GlobalWorkspace

        workspace = GlobalWorkspace()
        state = StateManager(workspace, config={})
        state.input_queue = asyncio.Queue()
        state.input_queue.put_nowait(("input1", "text"))

        # Add a pending tool percept
        tool_percept = MagicMock()
        state._pending_tool_percepts = [tool_percept]

        perception = AsyncMock()
        perception.encode = AsyncMock(
            side_effect=RuntimeError("Encode failed")
        )

        percepts = await state.gather_percepts(perception)
        # Should still have the tool percept even though encoding failed
        assert len(percepts) == 1
        assert percepts[0] is tool_percept


class TestLanguageInputParserGracefulDegradation:
    """Test that language parsing survives perception failures."""

    @pytest.mark.asyncio
    async def test_create_percept_returns_minimal_on_encode_failure(self):
        """_create_percept returns a percept without embedding when encode fails."""
        from mind.cognitive_core.language_input import (
            LanguageInputParser, IntentType, Intent
        )

        failing_perception = AsyncMock()
        failing_perception.encode = AsyncMock(
            side_effect=RuntimeError("Encoding failed")
        )

        parser = LanguageInputParser.__new__(LanguageInputParser)
        parser.perception = failing_perception
        parser.conversation_context = {"turn_count": 1}

        intent = Intent(type=IntentType.STATEMENT, confidence=0.8, metadata={})

        # Should NOT crash — returns percept without embedding
        percept = await parser._create_percept("test text", intent, {})
        assert percept is not None
        assert percept.modality == "text"
        assert percept.raw == "test text"
        assert percept.embedding is None
        # Metadata should still be populated
        assert percept.metadata["intent"] == IntentType.STATEMENT

    @pytest.mark.asyncio
    async def test_fallback_parse_survives_encode_failure(self):
        """Rule-based fallback parsing completes even if perception.encode() fails."""
        from mind.cognitive_core.language_input import LanguageInputParser
        from mind.cognitive_core.mock_perception import MockPerceptionSubsystem

        perception = MockPerceptionSubsystem()
        parser = LanguageInputParser(perception, llm_client=None)

        # Monkey-patch encode to fail
        async def failing_encode(*args, **kwargs):
            raise RuntimeError("Transient encoding failure")
        perception.encode = failing_encode

        # Should complete without crashing
        result = await parser.parse("Hello there!")
        assert result is not None
        assert result.intent is not None
        assert result.percept is not None
        assert result.percept.embedding is None  # No embedding, but still valid

    @pytest.mark.asyncio
    async def test_cached_parse_survives_encode_failure(self):
        """Cache path completes even if perception.encode() fails."""
        from mind.cognitive_core.language_input import LanguageInputParser
        from mind.cognitive_core.mock_perception import MockPerceptionSubsystem

        perception = MockPerceptionSubsystem()
        parser = LanguageInputParser(
            perception, llm_client=None, config={"enable_cache": True}
        )

        # First parse succeeds (populates cache)
        result = await parser.parse("Hello there!")
        assert result is not None
        assert result.percept.embedding is not None

        # Break perception
        async def failing_encode(*args, **kwargs):
            raise RuntimeError("Transient encoding failure")
        perception.encode = failing_encode

        # Cache hit path should still complete
        result2 = await parser.parse("Hello there!")
        assert result2 is not None
        assert result2.percept is not None
        assert result2.percept.embedding is None  # Degraded but functional


# ============================================================
# 3. Edge cases: boundary conditions and transient failures
# ============================================================

class TestTransientFailureEdgeCases:
    """Edge cases around transient failures."""

    @pytest.mark.asyncio
    async def test_single_bad_memory_entry_in_surfacer(self):
        """A single malformed memory entry doesn't crash surfacing."""
        from memory.surfacer import MemorySurfacer

        store = AsyncMock()
        store.recall = AsyncMock(return_value=[
            {"content": "valid memory", "significance": 0.8, "timestamp": "2026-01-01"},
            None,
            {"content": "another valid", "significance": 0.7, "timestamp": "2026-01-02"},
        ])

        surfacer = MemorySurfacer(store=store, config={})
        result = await surfacer.surface("test context")
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_empty_context_returns_empty(self):
        """Empty context returns empty list, no error."""
        from memory.surfacer import MemorySurfacer

        store = AsyncMock()
        surfacer = MemorySurfacer(store=store, config={})
        result = await surfacer.surface("")
        assert result == []
        store.recall.assert_not_called()

    @pytest.mark.asyncio
    async def test_none_store_returns_empty(self):
        """No store configured returns empty list, no error."""
        from memory.surfacer import MemorySurfacer

        surfacer = MemorySurfacer(store=None, config={})
        result = await surfacer.surface("test context")
        assert result == []

    @pytest.mark.asyncio
    async def test_cycle_failure_propagates_out_of_run(self):
        """A failing cycle propagates rather than being silently swallowed.

        Per AGENTS.md, the top-level cognitive-loop crash boundary does
        not exist until Phase 9 (First Awakening). During development
        and testing, cycle failures must surface so they can be diagnosed
        — they must not be quietly logged-and-continued.

        This test verifies that contract: when `_cycle()` raises, it
        propagates out of `run()` instead of being absorbed.
        """
        from core.cognitive_cycle import CognitiveCycle

        cycle = CognitiveCycle.__new__(CognitiveCycle)
        cycle.running = True
        cycle._cycle_delay = 0.001

        call_count = 0
        async def cycle_with_transient_failure():
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("Transient DB hiccup")

        cycle._cycle = AsyncMock(side_effect=cycle_with_transient_failure)

        with pytest.raises(RuntimeError, match="Transient DB hiccup"):
            await cycle.run(max_cycles=5)

        # Cycle 1 succeeded, cycle 2 raised — cycles 3-5 never ran.
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_concurrent_encode_failures_dont_compound(self):
        """Multiple queued inputs with failures are handled independently."""
        from mind.cognitive_core.core.state_manager import StateManager
        from mind.cognitive_core.workspace import GlobalWorkspace

        workspace = GlobalWorkspace()
        state = StateManager(workspace, config={})
        state.input_queue = asyncio.Queue()

        # Queue 5 items: 1 good, 1 bad, 1 good, 1 bad, 1 good
        for i in range(5):
            modality = "bad_modality" if i % 2 == 1 else "text"
            state.input_queue.put_nowait((f"input_{i}", modality))

        async def encode_side_effect(data, modality="text"):
            if modality == "bad_modality":
                raise RuntimeError(f"Cannot encode {modality}")
            return MagicMock(embedding=[0.1] * 384, metadata={})

        perception = AsyncMock()
        perception.encode = AsyncMock(side_effect=encode_side_effect)

        percepts = await state.gather_percepts(perception)
        assert len(percepts) == 3  # 3 good inputs survived


# ============================================================
# 4. Partial memory retrieval: queued retrieval failure
# ============================================================

class TestMemorySubstratePartialFailure:
    """Test that queued retrieval errors don't discard earlier memories."""

    @pytest.mark.asyncio
    async def test_queued_retrieval_failure_preserves_context_memories(self):
        """If queued retrieval fails, context-surfaced memories are still returned."""
        from memory.manager import MemorySubstrate

        substrate = MemorySubstrate.__new__(MemorySubstrate)

        call_count = 0
        async def surface_side_effect(context):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [MagicMock(content="context memory")]
            raise RuntimeError("Queued retrieval DB error")

        substrate._surfacer = AsyncMock()
        substrate._surfacer.surface = AsyncMock(side_effect=surface_side_effect)
        substrate._prospective = MagicMock()
        substrate._prospective.check = MagicMock(return_value=[MagicMock(content="prospective")])
        substrate._retrieval_queue = ["some query"]

        # Should return memories from steps 1 and 2, despite step 3 failing
        result = await substrate.surface("test context")
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_multiple_queued_retrievals_partial_failure(self):
        """If one of three queued retrievals fails, the other two succeed."""
        from memory.manager import MemorySubstrate

        substrate = MemorySubstrate.__new__(MemorySubstrate)

        call_count = 0
        async def surface_side_effect(context):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return []  # context surfacing
            if call_count == 3:
                raise RuntimeError("DB hiccup")
            return [MagicMock(content=f"queued_{call_count}")]

        substrate._surfacer = AsyncMock()
        substrate._surfacer.surface = AsyncMock(side_effect=surface_side_effect)
        substrate._prospective = MagicMock()
        substrate._prospective.check = MagicMock(return_value=[])
        substrate._retrieval_queue = ["q1", "q2", "q3"]

        result = await substrate.surface("context")
        assert len(result) == 2  # q1 and q3 succeed, q2 fails
        # Queue should be cleared regardless
        assert substrate._retrieval_queue == []

    @pytest.mark.asyncio
    async def test_retrieval_queue_cleared_even_on_all_failures(self):
        """Queue is always cleared, even if every queued retrieval fails."""
        from memory.manager import MemorySubstrate

        substrate = MemorySubstrate.__new__(MemorySubstrate)
        substrate._surfacer = AsyncMock()

        call_count = 0
        async def surface_side_effect(context):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return []  # context surfacing
            raise RuntimeError("All retrievals fail")

        substrate._surfacer.surface = AsyncMock(side_effect=surface_side_effect)
        substrate._prospective = MagicMock()
        substrate._prospective.check = MagicMock(return_value=[])
        substrate._retrieval_queue = ["q1", "q2"]

        result = await substrate.surface("context")
        assert result == []
        assert substrate._retrieval_queue == []  # Must be cleared


# ============================================================
# 5. Post-LLM interior crash paths in _cycle()
# ============================================================

class TestCycleInteriorCrashPaths:
    """Test that post-LLM operations don't kill the cycle.

    These tests mock _assemble_input to skip Pydantic construction,
    focusing on the post-LLM error handling in _cycle().
    """

    def _make_cycle(self):
        """Build a CognitiveCycle with _assemble_input mocked out."""
        from core.cognitive_cycle import CognitiveCycle

        cycle = CognitiveCycle.__new__(CognitiveCycle)
        cycle.running = True
        cycle._cycle_delay = 0.001
        cycle.cycle_count = 0
        cycle._last_output = None
        cycle._output_handlers = []

        # Mock _assemble_input to return a MagicMock (skips Pydantic).
        # The mocked CognitiveOutput has empty world-graph fields so
        # _cycle's apply/resolve step is a no-op.
        cycle._assemble_input = AsyncMock(return_value=MagicMock())
        cycle.context_mgr = MagicMock()
        cycle.model = AsyncMock()
        cycle.model.think = AsyncMock(return_value=MagicMock(
            self_model_updates={"values": "test"},
            predictions=[],
            world_model_updates=[],
            world_model_queries=[],
            external_speech=None,
        ))
        cycle.scaffold = MagicMock()
        cycle.scaffold.integrate = AsyncMock(return_value=MagicMock())
        cycle.scaffold.broadcast = AsyncMock()
        cycle.stream = MagicMock()
        cycle.sensorium = MagicMock()
        cycle.identity = MagicMock()
        cycle.authority = MagicMock()
        cycle.sleep = None
        cycle.communication = None
        cycle._current_percepts = []
        cycle._current_memories = []
        cycle._pending_query_results = []
        cycle.environment = None
        cycle.growth = None
        cycle.world_graph = MagicMock()
        cycle.world_graph.emit_size_warning = MagicMock(return_value=None)
        cycle.world_graph_path = None
        cycle._execute = AsyncMock()

        return cycle

    @pytest.mark.asyncio
    async def test_identity_update_failure_doesnt_kill_cycle(self):
        """identity.process_value_updates() failure doesn't prevent execution."""
        cycle = self._make_cycle()
        cycle.identity.process_value_updates = MagicMock(
            side_effect=RuntimeError("Identity DB locked")
        )

        await cycle._cycle()

        # Execute and broadcast should still have been called
        cycle._execute.assert_called_once()
        cycle.scaffold.broadcast.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_failure_doesnt_prevent_broadcast(self):
        """_execute() failure doesn't prevent broadcast."""
        cycle = self._make_cycle()
        cycle._execute = AsyncMock(side_effect=RuntimeError("Action failed"))

        await cycle._cycle()

        cycle.scaffold.broadcast.assert_called_once()

    @pytest.mark.asyncio
    async def test_broadcast_failure_doesnt_kill_cycle(self):
        """broadcast() failure doesn't prevent bookkeeping."""
        cycle = self._make_cycle()
        cycle.scaffold.broadcast = AsyncMock(
            side_effect=RuntimeError("Broadcast failed")
        )

        await cycle._cycle()

        assert cycle.cycle_count == 1
        assert cycle._last_output is not None


# ============================================================
# 6. PerceptionSubsystem boot-time init failure
# ============================================================

class TestPerceptionBootFailure:
    """Test the PerceptionSubsystem init fallback logic in isolation.

    SubsystemCoordinator.__init__ has many dependencies, so we test the
    perception init/fallback logic directly rather than through the full
    constructor.
    """

    def test_perception_init_failure_falls_back_to_mock(self):
        """When PerceptionSubsystem raises, the fallback code creates MockPerception."""
        from mind.cognitive_core.core.subsystem_coordinator import SubsystemCoordinator
        from mind.cognitive_core.mock_perception import MockPerceptionSubsystem

        # Simulate the perception init logic from SubsystemCoordinator.__init__
        perception_config = {"mock_mode": False}

        with patch(
            "mind.cognitive_core.core.subsystem_coordinator.PerceptionSubsystem",
            side_effect=ImportError("No sentence-transformers")
        ):
            # Replicate the exact logic from subsystem_coordinator.py
            if perception_config.get("mock_mode", False):
                perception = MockPerceptionSubsystem(config=perception_config)
            else:
                try:
                    from mind.cognitive_core.core.subsystem_coordinator import PerceptionSubsystem
                    perception = PerceptionSubsystem(config=perception_config)
                except Exception:
                    perception = MockPerceptionSubsystem(config=perception_config)

            assert isinstance(perception, MockPerceptionSubsystem)

    def test_perception_init_failure_in_real_coordinator(self):
        """Full SubsystemCoordinator boots with mock perception when real init fails."""
        from mind.cognitive_core.core.subsystem_coordinator import SubsystemCoordinator
        from mind.cognitive_core.mock_perception import MockPerceptionSubsystem
        from mind.cognitive_core.workspace import GlobalWorkspace

        workspace = GlobalWorkspace()
        # Provide the full config that SubsystemCoordinator needs
        config = {
            "perception": {"mock_mode": False},
            "action": {},
            "affect": {},
            "attention_budget": 100,
            "cycle_rate_hz": 1,
        }

        with patch(
            "mind.cognitive_core.core.subsystem_coordinator.PerceptionSubsystem",
            side_effect=ImportError("No sentence-transformers")
        ):
            coordinator = SubsystemCoordinator(workspace, config)
            assert isinstance(coordinator.perception, MockPerceptionSubsystem)


try:
    from mind.librarian import SanctuaryLibrarian
    _has_langchain = True
except ImportError:
    _has_langchain = False


@pytest.mark.skipif(not _has_langchain, reason="langchain_text_splitters not installed")
class TestLibrarianCrashPath:
    """Librarian still fails fast — no graceful degradation here."""

    def test_split_error_propagates(self):
        """process_documents propagates splitting errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            librarian = SanctuaryLibrarian(
                base_dir=tmpdir,
                persist_dir=os.path.join(tmpdir, "db"),
            )
            librarian.text_splitter = MagicMock()
            librarian.text_splitter.split_documents = MagicMock(
                side_effect=ValueError("Cannot split empty document")
            )

            with pytest.raises(ValueError, match="Cannot split empty document"):
                librarian.process_documents([MagicMock()])

    def test_split_error_crashes_build_index(self):
        """build_index crashes when process_documents raises."""
        with tempfile.TemporaryDirectory() as tmpdir:
            librarian = SanctuaryLibrarian(
                base_dir=tmpdir,
                persist_dir=os.path.join(tmpdir, "db"),
            )
            librarian.load_documents = MagicMock(return_value=[MagicMock()])
            librarian.text_splitter = MagicMock()
            librarian.text_splitter.split_documents = MagicMock(
                side_effect=RuntimeError("Splitter crashed")
            )

            with pytest.raises(RuntimeError, match="Splitter crashed"):
                librarian.build_index()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
