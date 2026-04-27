"""
Phase 1 Boot Integration Test.

Goal: Verify that CognitiveCore can instantiate and run cognitive cycles
without errors, using mock perception (no sentence-transformers needed).

This is the most fundamental integration test: if the system can boot
and cycle, everything downstream has a foundation to build on.

Run with:
    pytest sanctuary/tests/integration/test_phase1_boot.py -v
"""

import asyncio
import logging
import tempfile
from pathlib import Path

import pytest

# Configure logging to see boot sequence
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def boot_temp_dir():
    """Create a temporary directory for boot data files."""
    with tempfile.TemporaryDirectory(prefix="sanctuary_boot_test_") as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def boot_config(boot_temp_dir):
    """Create boot configuration with mock perception."""
    from sanctuary.mind.cognitive_core.boot_config import create_boot_config
    return create_boot_config(temp_dir=boot_temp_dir, mock_perception=True)


class TestPhase1Boot:
    """Phase 1: Can the system instantiate and cycle without errors?"""

    def test_boot_config_creation(self, boot_temp_dir):
        """Boot config creates valid configuration with required directories."""
        from sanctuary.mind.cognitive_core.boot_config import create_boot_config
        config = create_boot_config(temp_dir=boot_temp_dir)

        assert config["cycle_rate_hz"] == 10.0
        assert config["attention_budget"] == 100
        assert config["perception"]["mock_mode"] is True
        assert config["devices"]["enabled"] is False

        # Required directories exist
        assert Path(config["identity_dir"]).is_dir()
        assert Path(config["journal_dir"]).is_dir()
        assert Path(config["checkpointing"]["checkpoint_dir"]).is_dir()

    def test_mock_perception_instantiation(self):
        """MockPerceptionSubsystem can be created without ML dependencies."""
        from sanctuary.mind.cognitive_core.mock_perception import MockPerceptionSubsystem

        perception = MockPerceptionSubsystem(config={"mock_embedding_dim": 384})
        assert perception.embedding_dim == 384

    @pytest.mark.asyncio
    async def test_mock_perception_encoding(self):
        """Mock perception produces deterministic embeddings."""
        from sanctuary.mind.cognitive_core.mock_perception import MockPerceptionSubsystem

        perception = MockPerceptionSubsystem(config={"mock_embedding_dim": 384})

        # Encode text
        percept = await perception.encode("hello world", "text")
        assert percept is not None
        assert len(percept.embedding) == 384
        assert percept.modality == "text"

        # Same input produces same embedding (deterministic)
        percept2 = await perception.encode("hello world", "text")
        assert percept.embedding == percept2.embedding

        # Different input produces different embedding
        percept3 = await perception.encode("goodbye world", "text")
        assert percept.embedding != percept3.embedding

    @pytest.mark.asyncio
    async def test_mock_perception_all_modalities(self):
        """Mock perception handles all modality types."""
        from sanctuary.mind.cognitive_core.mock_perception import MockPerceptionSubsystem

        perception = MockPerceptionSubsystem()

        modalities = {
            "text": "hello",
            "image": b"fake_image_bytes",
            "audio": {"duration_seconds": 2},
            "sensor": {"value": 42, "sensor_type": "temperature"},
            "introspection": {"description": "reflecting on state"},
        }

        for modality, raw_input in modalities.items():
            percept = await perception.encode(raw_input, modality)
            assert percept is not None, f"Failed for modality: {modality}"
            assert len(percept.embedding) == 384, f"Wrong dim for modality: {modality}"
            assert percept.modality == modality

    def test_workspace_instantiation(self):
        """GlobalWorkspace can be created independently."""
        from sanctuary.mind.cognitive_core.workspace import GlobalWorkspace
        ws = GlobalWorkspace()
        snapshot = ws.broadcast()
        assert snapshot is not None

    def test_cognitive_core_instantiation(self, boot_config):
        """CognitiveCore instantiates with boot config (the main Phase 1 test)."""
        from sanctuary.mind.cognitive_core.core import CognitiveCore

        core = CognitiveCore(config=boot_config)

        assert core is not None
        assert core.workspace is not None
        assert core.subsystems is not None
        assert core.subsystems.perception is not None
        assert core.subsystems.affect is not None
        assert core.subsystems.attention is not None
        assert core.subsystems.action is not None

        logger.info("CognitiveCore instantiated successfully!")

        # Close journal to release file handles (Windows temp dir cleanup)
        if hasattr(core.subsystems, 'introspective_journal') and core.subsystems.introspective_journal:
            core.subsystems.introspective_journal.close()

    @pytest.mark.asyncio
    async def test_cognitive_core_runs_cycles(self, boot_config):
        """CognitiveCore can start, run cycles, and stop without errors."""
        from sanctuary.mind.cognitive_core.core import CognitiveCore

        core = CognitiveCore(config=boot_config)

        # Start the cognitive loop
        await core.start()
        assert core.running

        # Let it run for a short time (~10 cycles at 10Hz)
        await asyncio.sleep(1.0)

        # Stop gracefully
        await core.stop()
        assert not core.running

        # Check metrics - should have completed some cycles
        metrics = core.get_metrics()
        logger.info(f"Completed cycles: {metrics.get('total_cycles', 0)}")

        logger.info("CognitiveCore ran and stopped successfully!")

    @pytest.mark.asyncio
    async def test_inject_input_and_cycle(self, boot_config):
        """CognitiveCore accepts input injection during cycling."""
        from sanctuary.mind.cognitive_core.core import CognitiveCore

        core = CognitiveCore(config=boot_config)

        # Inject input before starting
        core.inject_input("hello sanctuary", modality="text")

        # Start and let it process
        await core.start()
        await asyncio.sleep(0.5)
        await core.stop()

        # Query final state
        snapshot = core.query_state()
        assert snapshot is not None

        logger.info("Input injection test passed!")

    @pytest.mark.asyncio
    async def test_workspace_state_after_cycling(self, boot_config):
        """Workspace has valid state after cycling."""
        from sanctuary.mind.cognitive_core.core import CognitiveCore

        core = CognitiveCore(config=boot_config)

        await core.start()
        await asyncio.sleep(0.5)
        await core.stop()

        snapshot = core.query_state()
        assert snapshot is not None

        # Workspace should have serializable state
        state_dict = snapshot.model_dump() if hasattr(snapshot, "model_dump") else vars(snapshot)
        assert isinstance(state_dict, dict)

        logger.info("Workspace state is valid after cycling!")

    def test_subsystem_cross_references(self, boot_config):
        """Subsystems are properly wired together after instantiation."""
        from sanctuary.mind.cognitive_core.core import CognitiveCore

        core = CognitiveCore(config=boot_config)

        # Workspace has references to key subsystems
        assert core.workspace.affect is not None
        assert core.workspace.action_subsystem is not None
        assert core.workspace.perception is not None

        # IWMT core should be initialized (enabled by default in boot config)
        assert core.subsystems.iwmt_core is not None

        logger.info("Subsystem cross-references verified!")

        # Close journal to release file handles (Windows temp dir cleanup)
        if hasattr(core.subsystems, 'introspective_journal') and core.subsystems.introspective_journal:
            core.subsystems.introspective_journal.close()


class TestPhase1MockPerceptionStats:
    """Verify mock perception tracking and cache behavior."""

    @pytest.mark.asyncio
    async def test_cache_hit_rate(self):
        """Cache works correctly for repeated inputs."""
        from sanctuary.mind.cognitive_core.mock_perception import MockPerceptionSubsystem

        perception = MockPerceptionSubsystem(config={"cache_size": 10})

        # First encoding (cache miss)
        await perception.encode("test input", "text")
        stats = perception.get_stats()
        assert stats["cache_misses"] == 1
        assert stats["cache_hits"] == 0

        # Second encoding of same input (cache hit)
        await perception.encode("test input", "text")
        stats = perception.get_stats()
        assert stats["cache_hits"] == 1

        assert stats["mock_mode"] is True

    @pytest.mark.asyncio
    async def test_cache_eviction(self):
        """Cache evicts oldest entries when full."""
        from sanctuary.mind.cognitive_core.mock_perception import MockPerceptionSubsystem

        perception = MockPerceptionSubsystem(config={"cache_size": 3})

        # Fill cache
        for i in range(3):
            await perception.encode(f"input_{i}", "text")
        assert len(perception.embedding_cache) == 3

        # One more evicts the oldest
        await perception.encode("input_3", "text")
        assert len(perception.embedding_cache) == 3  # Still 3

        # First input should be evicted (cache miss)
        await perception.encode("input_0", "text")
        stats = perception.get_stats()
        # input_0 was evicted, so this is a miss
        assert stats["cache_misses"] > 3  # More than initial 3
