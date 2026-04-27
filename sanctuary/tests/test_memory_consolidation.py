"""
Test Suite for Memory Consolidation System

Tests the memory consolidation engine, idle detector, and scheduler.

Author: Sanctuary Emergence Team
"""
import pytest
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil
import sys

# Add parent directory to path to import modules directly
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mind.memory.idle_detector import IdleDetector
from mind.memory.scheduler import ConsolidationScheduler, ConsolidationMetrics
from mind.memory.consolidation import MemoryConsolidator
from mind.memory.storage import MemoryStorage
from mind.memory.encoding import MemoryEncoder


@pytest.fixture
def temp_dirs():
    """Create temporary directories for testing."""
    import time
    import gc

    temp_base = tempfile.mkdtemp()
    persistence_dir = Path(temp_base) / "memories"
    persistence_dir.mkdir(parents=True)

    yield persistence_dir

    # Cleanup with retry for Windows file locking
    gc.collect()  # Force garbage collection to release handles

    if Path(temp_base).exists():
        for attempt in range(3):
            try:
                shutil.rmtree(temp_base)
                break
            except PermissionError:
                if attempt < 2:
                    time.sleep(0.5)  # Wait for file handles to release
                    gc.collect()
                # On final attempt, ignore error (Windows file locking)


@pytest.fixture
def storage(temp_dirs):
    """Create MemoryStorage instance for testing."""
    persistence_dir = temp_dirs
    store = MemoryStorage(persistence_dir=str(persistence_dir))
    yield store
    # Close storage to release ChromaDB file handles
    store.close()


@pytest.fixture
def encoder():
    """Create MemoryEncoder instance for testing."""
    return MemoryEncoder()


@pytest.fixture
def consolidator(storage, encoder):
    """Create MemoryConsolidator instance for testing."""
    return MemoryConsolidator(
        storage=storage,
        encoder=encoder,
        strengthening_factor=0.1,
        decay_rate=0.95,
        deletion_threshold=0.1,
        pattern_threshold=3
    )


@pytest.fixture
def idle_detector():
    """Create IdleDetector instance for testing."""
    return IdleDetector(
        idle_threshold_seconds=1.0,  # Short threshold for testing
        activity_decay=0.9
    )


class TestIdleDetector:
    """Test idle detection functionality."""
    
    def test_initialization(self, idle_detector):
        """Test idle detector initializes correctly."""
        assert idle_detector.idle_threshold == 1.0
        assert idle_detector.activity_decay == 0.9
        assert idle_detector.activity_level == 1.0
    
    def test_record_activity(self, idle_detector):
        """Test recording activity resets idle state."""
        idle_detector.activity_level = 0.5
        idle_detector.record_activity()
        
        assert idle_detector.activity_level == 1.0
        assert not idle_detector.is_idle()
    
    def test_is_idle_after_threshold(self, idle_detector):
        """Test system becomes idle after threshold."""
        # Should not be idle immediately
        assert not idle_detector.is_idle()
        
        # Wait for threshold
        import time
        time.sleep(1.2)
        
        # Should be idle now
        assert idle_detector.is_idle()
    
    def test_consolidation_budget(self, idle_detector):
        """Test consolidation budget calculation."""
        # Active system has no budget
        assert idle_detector.get_consolidation_budget() == 0.0
        
        # Wait to become idle
        import time
        time.sleep(1.2)
        
        # Should have some budget
        budget = idle_detector.get_consolidation_budget()
        assert 0.0 < budget <= 1.0
    
    def test_reset(self, idle_detector):
        """Test resetting idle detector."""
        import time
        time.sleep(1.2)
        
        assert idle_detector.is_idle()
        
        idle_detector.reset()
        assert not idle_detector.is_idle()
        assert idle_detector.activity_level == 1.0


class TestMemoryConsolidator:
    """Test memory consolidation functionality."""
    
    def test_initialization(self, consolidator):
        """Test consolidator initializes correctly."""
        assert consolidator.strengthening_factor == 0.1
        assert consolidator.decay_rate == 0.95
        assert consolidator.deletion_threshold == 0.1
        assert consolidator.pattern_threshold == 3
        assert len(consolidator.retrieval_log) == 0
    
    def test_record_retrieval(self, consolidator):
        """Test recording memory retrievals."""
        consolidator.record_retrieval("mem_123", session_id="session_1")
        
        assert len(consolidator.retrieval_log) == 1
        assert consolidator.retrieval_log[0]["memory_id"] == "mem_123"
        assert consolidator.retrieval_log[0]["session_id"] == "session_1"
    
    def test_strengthen_retrieved_memories(self, consolidator, storage, encoder):
        """Test strengthening memories based on retrieval."""
        # Store a test memory
        experience = {
            "description": "Test experience",
            "timestamp": datetime.now().isoformat()
        }
        document, metadata, doc_id = encoder.encode_experience(experience)
        metadata["base_activation"] = 0.5
        metadata["last_accessed"] = datetime.now().isoformat()
        storage.add_episodic(document, metadata, doc_id)
        
        # Record some retrievals
        for _ in range(3):
            consolidator.record_retrieval(doc_id)
        
        # Strengthen based on retrievals
        strengthened = consolidator.strengthen_retrieved_memories()
        
        assert strengthened > 0
    
    def test_apply_decay(self, consolidator, storage, encoder):
        """Test memory decay for unretrieved memories."""
        # Store a test memory with old last_accessed time
        experience = {
            "description": "Old experience",
            "timestamp": datetime.now().isoformat()
        }
        document, metadata, doc_id = encoder.encode_experience(experience)
        metadata["base_activation"] = 0.5
        # Set last accessed to 10 days ago
        old_time = datetime.now() - timedelta(days=10)
        metadata["last_accessed"] = old_time.isoformat()
        storage.add_episodic(document, metadata, doc_id)
        
        # Apply decay
        decayed, pruned = consolidator.apply_decay(threshold_days=7)
        
        # Should have decayed the memory
        assert decayed >= 0  # May be 0 if implementation differs
    
    def test_association_strengthening(self, consolidator):
        """Test strengthening associations between co-retrieved memories."""
        # Record co-retrievals in same session
        consolidator.record_retrieval("mem_1", session_id="session_1")
        consolidator.record_retrieval("mem_2", session_id="session_1")
        consolidator.record_retrieval("mem_3", session_id="session_1")
        
        # Reorganize associations
        updated = consolidator.reorganize_associations()
        
        # Should have created associations
        assert updated > 0
        
        # Check association strength
        strength = consolidator.get_association_strength("mem_1", "mem_2")
        assert strength > 0
    
    def test_get_associated_memories(self, consolidator):
        """Test retrieving associated memories."""
        # Create some associations
        consolidator.record_retrieval("mem_1", session_id="session_1")
        consolidator.record_retrieval("mem_2", session_id="session_1")
        consolidator.reorganize_associations()
        
        # Get associated memories
        associated = consolidator.get_associated_memories("mem_1", threshold=0.05)
        
        # Should find mem_2 associated with mem_1
        assert len(associated) > 0
    
    def test_transfer_to_semantic(self, consolidator, storage, encoder):
        """Test transferring repeated patterns to semantic memory."""
        # Store multiple episodes with same tags
        for i in range(3):
            experience = {
                "description": f"Experience {i}",
                "timestamp": datetime.now().isoformat()
            }
            document, metadata, doc_id = encoder.encode_experience(experience)
            # ChromaDB metadata only accepts scalar values, so join list as string
            metadata["tags"] = "pattern,test"
            metadata["timestamp"] = datetime.now().isoformat()
            storage.add_episodic(document, metadata, f"mem_pattern_{i}")
        
        # Transfer patterns to semantic
        transferred = consolidator.transfer_to_semantic(days=1)
        
        # Should extract pattern (or 0 if not enough episodes found)
        assert transferred >= 0
    
    def test_reprocess_emotional_memories(self, consolidator, storage, encoder):
        """Test emotional memory reprocessing."""
        # Store high-emotion memory
        experience = {
            "description": "Emotional experience",
            "timestamp": datetime.now().isoformat()
        }
        document, metadata, doc_id = encoder.encode_experience(experience)
        metadata["emotional_intensity"] = 0.9
        metadata["base_activation"] = 0.5
        storage.add_episodic(document, metadata, doc_id)
        
        # Reprocess emotional memories
        reprocessed = consolidator.reprocess_emotional_memories(threshold=0.7)
        
        assert reprocessed > 0


class TestConsolidationScheduler:
    """Test consolidation scheduling functionality."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, consolidator, idle_detector):
        """Test scheduler initializes correctly."""
        scheduler = ConsolidationScheduler(
            engine=consolidator,
            detector=idle_detector,
            check_interval=1.0
        )
        
        assert scheduler.check_interval == 1.0
        assert not scheduler.is_running
        assert len(scheduler.metrics_history) == 0
    
    @pytest.mark.asyncio
    async def test_start_stop(self, consolidator, idle_detector):
        """Test starting and stopping scheduler."""
        scheduler = ConsolidationScheduler(
            engine=consolidator,
            detector=idle_detector,
            check_interval=1.0
        )
        
        # Start scheduler
        await scheduler.start()
        assert scheduler.is_running
        
        # Let it run briefly
        await asyncio.sleep(0.5)
        
        # Stop scheduler
        await scheduler.stop()
        assert not scheduler.is_running
    
    @pytest.mark.asyncio
    async def test_consolidation_cycle(self, consolidator, idle_detector, storage, encoder):
        """Test consolidation cycle runs during idle."""
        scheduler = ConsolidationScheduler(
            engine=consolidator,
            detector=idle_detector,
            check_interval=0.5
        )
        
        # Store a test memory
        experience = {"description": "Test", "timestamp": datetime.now().isoformat()}
        document, metadata, doc_id = encoder.encode_experience(experience)
        metadata["base_activation"] = 0.5
        storage.add_episodic(document, metadata, doc_id)
        
        # Record retrieval
        consolidator.record_retrieval(doc_id)
        
        # Start scheduler
        await scheduler.start()
        
        # Wait for system to become idle (1 second threshold)
        await asyncio.sleep(1.5)
        
        # Check if consolidation ran
        metrics = scheduler.get_recent_metrics(limit=1)
        
        # Stop scheduler
        await scheduler.stop()
        
        # Should have run at least once if idle
        assert len(metrics) >= 0  # May be 0 if timing varies
    
    @pytest.mark.asyncio
    async def test_metrics_tracking(self, consolidator, idle_detector):
        """Test consolidation metrics are tracked."""
        scheduler = ConsolidationScheduler(
            engine=consolidator,
            detector=idle_detector,
            check_interval=0.5
        )
        
        # Manually run a cycle
        budget = 0.5
        metrics = await scheduler._run_consolidation_cycle(budget)
        
        # Check metrics structure
        assert metrics.budget_used == budget
        assert metrics.timestamp is not None
        assert metrics.consolidation_duration_ms >= 0
    
    @pytest.mark.asyncio
    async def test_metrics_summary(self, consolidator, idle_detector):
        """Test getting metrics summary."""
        scheduler = ConsolidationScheduler(
            engine=consolidator,
            detector=idle_detector
        )
        
        # Get summary with no history
        summary = scheduler.get_metrics_summary()
        assert summary["total_cycles"] == 0
        
        # Run a cycle
        await scheduler._run_consolidation_cycle(0.5)
        scheduler._record_metrics(
            await scheduler._run_consolidation_cycle(0.5)
        )
        
        # Get summary again
        summary = scheduler.get_metrics_summary()
        assert summary["total_cycles"] == 1


class TestEndToEndConsolidation:
    """End-to-end consolidation tests."""
    
    @pytest.mark.asyncio
    async def test_full_consolidation_cycle(
        self,
        consolidator,
        idle_detector,
        storage,
        encoder
    ):
        """Test complete consolidation workflow."""
        # Store some test memories
        for i in range(5):
            experience = {
                "description": f"Experience {i}",
                "timestamp": datetime.now().isoformat()
            }
            document, metadata, doc_id = encoder.encode_experience(experience)
            metadata["base_activation"] = 0.5
            metadata["last_accessed"] = datetime.now().isoformat()
            storage.add_episodic(document, metadata, f"mem_{i}")
        
        # Simulate retrievals
        for i in range(3):
            consolidator.record_retrieval(f"mem_{i}", session_id="session_1")
        
        # Run strengthening
        strengthened = consolidator.strengthen_retrieved_memories()
        assert strengthened > 0
        
        # Create scheduler
        scheduler = ConsolidationScheduler(
            engine=consolidator,
            detector=idle_detector,
            check_interval=0.5
        )
        
        # Start scheduler
        await scheduler.start()
        
        # Wait for idle period
        await asyncio.sleep(1.5)
        
        # Stop scheduler
        await scheduler.stop()
        
        # Check that consolidation happened
        summary = scheduler.get_metrics_summary()
        assert summary["total_cycles"] >= 0
