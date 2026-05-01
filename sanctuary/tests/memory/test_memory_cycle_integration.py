"""Integration tests — memory substrate wired into the cognitive cycle."""

import pytest
from sanctuary.core.cognitive_cycle import CognitiveCycle, NullMemory
from sanctuary.core.placeholder import PlaceholderModel
from sanctuary.core.schema import CognitiveOutput, MemoryOp, Percept
from sanctuary.memory.manager import MemorySubstrate


@pytest.fixture
def model():
    return PlaceholderModel()


@pytest.fixture
def memory():
    return MemorySubstrate()


@pytest.fixture
def cycle(model, memory):
    return CognitiveCycle(model=model, memory=memory, cycle_delay=0.0)


class TestMemoryInCycle:
    @pytest.mark.asyncio
    async def test_cycle_runs_with_memory_substrate(self, cycle):
        """Cycle should execute normally with real memory substrate."""
        await cycle.run(max_cycles=3)
        assert cycle.cycle_count == 3
        assert cycle.last_output is not None

    @pytest.mark.asyncio
    async def test_memory_tick_called(self, cycle, memory):
        """Memory tick should advance each cycle."""
        await cycle.run(max_cycles=5)
        # Journal's cycle count should match
        assert memory.journal._cycle_count == 5

    @pytest.mark.asyncio
    async def test_null_memory_still_works(self, model):
        """NullMemory should satisfy the updated protocol."""
        null = NullMemory()
        cycle = CognitiveCycle(model=model, memory=null, cycle_delay=0.0)
        await cycle.run(max_cycles=3)
        assert cycle.cycle_count == 3

    @pytest.mark.asyncio
    async def test_null_memory_execute_ops(self):
        null = NullMemory()
        results = await null.execute_ops([])
        assert results == []
        null.tick()  # Should not raise


class TestMemoryOpsInCycle:
    @pytest.mark.asyncio
    async def test_journal_ops_persisted(self, model, memory):
        """When the placeholder model emits journal ops, they should persist."""

        # Create a model that emits a journal op
        class JournalingModel:
            async def think(self, input):
                return CognitiveOutput(
                    inner_speech="Writing to journal",
                    memory_ops=[
                        MemoryOp(
                            type="journal",
                            content="Today was productive",
                            significance=6,
                            tags=["reflection"],
                        )
                    ],
                )

        cycle = CognitiveCycle(
            model=JournalingModel(), memory=memory, cycle_delay=0.0
        )
        await cycle.run(max_cycles=1)
        assert memory.journal.entry_count == 1
        assert memory.journal.entries[0].content == "Today was productive"

    @pytest.mark.asyncio
    async def test_episodic_write_ops(self, memory):
        """Episodic writes from the model should be stored."""

        class EpisodicModel:
            async def think(self, input):
                return CognitiveOutput(
                    inner_speech="Storing a memory",
                    memory_ops=[
                        MemoryOp(
                            type="write_episodic",
                            content="Alice told me about her cat",
                            significance=5,
                            tags=["alice", "cat"],
                        )
                    ],
                )

        cycle = CognitiveCycle(
            model=EpisodicModel(), memory=memory, cycle_delay=0.0
        )
        await cycle.run(max_cycles=1)
        assert memory.store.entry_count == 1

    @pytest.mark.asyncio
    async def test_retrieve_ops_queued(self, memory):
        """Retrieve ops should queue for next cycle's surfacing."""

        class RetrievingModel:
            async def think(self, input):
                return CognitiveOutput(
                    inner_speech="Retrieving",
                    memory_ops=[
                        MemoryOp(type="retrieve", query="alice birthday")
                    ],
                )

        cycle = CognitiveCycle(
            model=RetrievingModel(), memory=memory, cycle_delay=0.0
        )
        await cycle.run(max_cycles=1)
        # The retrieval should be queued
        assert len(memory._retrieval_queue) == 1


class TestProspectiveInCycle:
    @pytest.mark.asyncio
    async def test_prospective_triggers_in_surfacing(self, model, memory):
        """Prospective intentions should surface when triggered."""
        memory.add_intention(
            "Check on Alice's project",
            trigger_type="cycle",
            trigger_value="2",
        )

        # Run cycles and capture surfaced memories
        surfaced_all = []

        async def capture(output):
            pass

        cycle = CognitiveCycle(model=model, memory=memory, cycle_delay=0.0)

        # After 2 cycles, the intention should have triggered
        await cycle.run(max_cycles=3)

        # The intention should now be triggered
        assert memory.prospective.pending_count == 0


class TestFullFlow:
    @pytest.mark.asyncio
    async def test_write_then_surface(self, memory):
        """Write a memory, then surface it in a later cycle."""

        class WriteThenReadModel:
            def __init__(self):
                self.call_count = 0

            async def think(self, input):
                self.call_count += 1
                if self.call_count == 1:
                    return CognitiveOutput(
                        inner_speech="Storing something about alice",
                        memory_ops=[
                            MemoryOp(
                                type="write_episodic",
                                content="Alice loves painting",
                                significance=7,
                                tags=["alice"],
                            )
                        ],
                    )
                # Later cycles: check if memory surfaces
                surfaced = [m.content for m in (input.surfaced_memories or [])]
                has_alice = any("alice" in s.lower() for s in surfaced)
                return CognitiveOutput(
                    inner_speech=f"Surfaced alice memory: {has_alice}",
                )

        model = WriteThenReadModel()
        cycle = CognitiveCycle(model=model, memory=memory, cycle_delay=0.0)

        # Cycle 1: write
        await cycle.run(max_cycles=1)
        assert memory.store.entry_count == 1

        # Cycle 2+: the stored memory should be retrievable via surfacing
        # (The surfacer uses the stream's context, which mentions "alice")
        stored = await memory.surface("alice")
        assert any("Alice" in m.content for m in stored)
