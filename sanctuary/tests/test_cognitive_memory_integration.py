"""
Unit tests for memory integration with cognitive core.

Tests cover:
- Memory retrieval triggered by workspace state
- Memory-to-percept conversion
- Consolidation triggers (arousal, goals, percepts)
- Memory entry creation from workspace
- Integration with CognitiveCore
"""

import pytest
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import List
from unittest.mock import Mock, AsyncMock, patch
from uuid import uuid4

from mind.cognitive_core.memory_integration import MemoryIntegration
from mind.cognitive_core.workspace import (
    GlobalWorkspace,
    Goal,
    GoalType,
    Percept,
)
from mind.memory_manager import JournalEntry, EmotionalState


@pytest.fixture
def workspace():
    """Create a test workspace."""
    return GlobalWorkspace(capacity=5)


@pytest.fixture
def mock_memory_manager():
    """Create a mock memory manager."""
    manager = Mock()
    manager.recall = AsyncMock(return_value=[])
    manager.commit_journal = AsyncMock(return_value=True)
    return manager


@pytest.fixture
def memory_integration(workspace, tmp_path):
    """Create memory integration with temporary storage."""
    config = {
        "memory_config": {
            "base_dir": str(tmp_path / "memories"),
            "chroma_dir": str(tmp_path / "chroma"),
        },
        "consolidation_threshold": 0.6,
        "retrieval_top_k": 5,
        "min_cycles": 5,  # Shorter for testing
    }

    integration = MemoryIntegration(workspace, config)
    return integration


class TestMemoryRetrieval:
    """Test memory retrieval functionality."""
    
    @pytest.mark.asyncio
    async def test_retrieve_with_goals(self, memory_integration, workspace):
        """Test memory retrieval based on active goals."""
        # Add a high-priority goal
        goal = Goal(
            type=GoalType.RESPOND_TO_USER,
            description="Tell me about quantum physics",
            priority=0.8
        )
        workspace.add_goal(goal)
        
        # Create a mock journal entry
        mock_entry = JournalEntry(
            content="Discussion about quantum mechanics and wave functions",
            summary="Quantum physics conversation",
            tags=["science", "physics"],
            significance_score=7,
        )
        
        # Mock the recall method
        memory_integration.memory_manager.recall = AsyncMock(return_value=[mock_entry])
        
        # Retrieve memories
        snapshot = workspace.broadcast()
        percepts = await memory_integration.retrieve_for_workspace(snapshot)
        
        # Verify retrieval
        assert len(percepts) == 1
        assert percepts[0].modality == "memory"
        assert "quantum" in percepts[0].raw["content"].lower()
        assert percepts[0].raw["significance"] == 7
        
        # Verify query was built from goal
        memory_integration.memory_manager.recall.assert_called_once()
        call_args = memory_integration.memory_manager.recall.call_args
        assert "quantum physics" in call_args[1]["query"].lower()
    
    @pytest.mark.asyncio
    async def test_retrieve_with_high_attention_percepts(self, memory_integration, workspace):
        """Test memory retrieval based on high-attention percepts."""
        # Add a percept with high attention
        percept = Percept(
            modality="text",
            raw="What is the meaning of consciousness?",
            complexity=10,
            metadata={"attention_score": 0.9}
        )
        workspace.active_percepts[percept.id] = percept
        
        # Mock memory entry
        mock_entry = JournalEntry(
            content="Reflection on consciousness and awareness",
            summary="Consciousness discussion",
            tags=["philosophy", "consciousness"],
            significance_score=8,
        )
        
        memory_integration.memory_manager.recall = AsyncMock(return_value=[mock_entry])
        
        # Retrieve memories
        snapshot = workspace.broadcast()
        percepts = await memory_integration.retrieve_for_workspace(snapshot)
        
        # Verify
        assert len(percepts) == 1
        assert percepts[0].modality == "memory"
        
        # Query should include the high-attention percept
        call_args = memory_integration.memory_manager.recall.call_args
        assert "consciousness" in call_args[1]["query"].lower()
    
    @pytest.mark.asyncio
    async def test_retrieve_empty_workspace(self, memory_integration, workspace):
        """Test that empty workspace produces empty query."""
        snapshot = workspace.broadcast()
        percepts = await memory_integration.retrieve_for_workspace(snapshot)
        
        # Should return empty list
        assert len(percepts) == 0
    
    @pytest.mark.asyncio
    async def test_memory_to_percept_conversion(self, memory_integration):
        """Test conversion of memory entry to percept."""
        # Create a journal entry
        entry = JournalEntry(
            content="Test memory content with important information",
            summary="Test memory summary",
            tags=["test", "important"],
            emotional_signature=[EmotionalState.JOY, EmotionalState.WONDER],
            significance_score=9,
        )
        
        # Convert to percept
        percept = memory_integration._memory_to_percept(entry)
        
        # Verify percept structure
        assert percept.modality == "memory"
        assert percept.raw["content"] == entry.content
        assert percept.raw["summary"] == entry.summary
        assert percept.raw["significance"] == 9
        assert "joy" in [e for e in percept.raw["emotional_signature"]]
        assert percept.complexity == 27  # 9 * 3
        assert percept.metadata["source"] == "long_term_memory"


class TestMemoryConsolidation:
    """Test memory consolidation functionality."""
    
    @pytest.mark.asyncio
    async def test_consolidation_high_arousal(self, memory_integration, workspace):
        """Test that high arousal triggers consolidation."""
        # Set high arousal
        workspace.emotional_state["arousal"] = 0.8
        workspace.emotional_state["valence"] = 0.3
        
        # Add a goal
        goal = Goal(
            type=GoalType.RESPOND_TO_USER,
            description="Urgent task",
            priority=0.9
        )
        workspace.add_goal(goal)
        
        # Advance cycles to meet minimum
        memory_integration.cycles_since_consolidation = 10
        
        # Mock commit_journal
        memory_integration.memory_manager.commit_journal = AsyncMock(return_value=True)
        
        # Consolidate
        snapshot = workspace.broadcast()
        await memory_integration.consolidate(snapshot)
        
        # Verify consolidation occurred
        memory_integration.memory_manager.commit_journal.assert_called_once()
        assert memory_integration.cycles_since_consolidation == 0
    
    @pytest.mark.asyncio
    async def test_consolidation_extreme_valence(self, memory_integration, workspace):
        """Test that extreme valence triggers consolidation."""
        # Set extreme negative valence
        workspace.emotional_state["valence"] = -0.7
        workspace.emotional_state["arousal"] = 0.3
        
        memory_integration.cycles_since_consolidation = 10
        memory_integration.memory_manager.commit_journal = AsyncMock(return_value=True)
        
        snapshot = workspace.broadcast()
        await memory_integration.consolidate(snapshot)
        
        # Should consolidate
        memory_integration.memory_manager.commit_journal.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_consolidation_goal_completion(self, memory_integration, workspace):
        """Test that completed goals trigger consolidation."""
        # Add a completed goal
        goal = Goal(
            type=GoalType.RESPOND_TO_USER,
            description="Completed task",
            priority=0.7,
            progress=1.0  # Completed
        )
        workspace.add_goal(goal)
        
        memory_integration.cycles_since_consolidation = 10
        memory_integration.memory_manager.commit_journal = AsyncMock(return_value=True)
        
        snapshot = workspace.broadcast()
        await memory_integration.consolidate(snapshot)
        
        # Should consolidate
        memory_integration.memory_manager.commit_journal.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_consolidation_significant_percepts(self, memory_integration, workspace):
        """Test that significant percepts trigger consolidation."""
        # Add multiple significant percepts
        for i in range(3):
            percept = Percept(
                modality="text",
                raw=f"Significant percept {i}",
                complexity=35,  # High complexity
            )
            workspace.active_percepts[percept.id] = percept
        
        memory_integration.cycles_since_consolidation = 10
        memory_integration.memory_manager.commit_journal = AsyncMock(return_value=True)
        
        snapshot = workspace.broadcast()
        await memory_integration.consolidate(snapshot)
        
        # Should consolidate
        memory_integration.memory_manager.commit_journal.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_no_consolidation_min_cycles(self, memory_integration, workspace):
        """Test that consolidation respects minimum cycle gap."""
        # Set high arousal but too few cycles
        workspace.emotional_state["arousal"] = 0.9
        memory_integration.cycles_since_consolidation = 3  # Below min_cycles (5)
        
        memory_integration.memory_manager.commit_journal = AsyncMock(return_value=True)
        
        snapshot = workspace.broadcast()
        await memory_integration.consolidate(snapshot)
        
        # Should NOT consolidate
        memory_integration.memory_manager.commit_journal.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_no_consolidation_low_activity(self, memory_integration, workspace):
        """Test that low activity doesn't trigger consolidation."""
        # Empty workspace
        memory_integration.cycles_since_consolidation = 50
        memory_integration.memory_manager.commit_journal = AsyncMock(return_value=True)
        
        snapshot = workspace.broadcast()
        await memory_integration.consolidate(snapshot)
        
        # Should NOT consolidate (low activity)
        memory_integration.memory_manager.commit_journal.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_periodic_consolidation(self, memory_integration, workspace):
        """Test periodic consolidation after many cycles."""
        # Add some content
        goal = Goal(
            type=GoalType.RESPOND_TO_USER,
            description="Active goal",
            priority=0.5
        )
        workspace.add_goal(goal)
        
        # Many cycles have passed
        memory_integration.cycles_since_consolidation = 150
        memory_integration.memory_manager.commit_journal = AsyncMock(return_value=True)
        
        snapshot = workspace.broadcast()
        await memory_integration.consolidate(snapshot)
        
        # Should consolidate periodically
        memory_integration.memory_manager.commit_journal.assert_called_once()


class TestMemoryEntryBuilding:
    """Test building memory entries from workspace."""
    
    def test_build_memory_entry(self, memory_integration, workspace):
        """Test creating memory entry from workspace state."""
        # Set up workspace
        goal = Goal(
            type=GoalType.RESPOND_TO_USER,
            description="Answer question about AI",
            priority=0.8,
            progress=0.5
        )
        workspace.add_goal(goal)
        
        percept = Percept(
            modality="text",
            raw="What is artificial intelligence?",
            complexity=10,
            metadata={"attention_score": 0.9}
        )
        workspace.active_percepts[percept.id] = percept
        
        workspace.emotional_state["arousal"] = 0.6
        workspace.emotional_state["valence"] = 0.4
        
        snapshot = workspace.broadcast()
        snapshot.metadata["emotion_label"] = "wonder"
        
        # Build memory entry
        entry = memory_integration._build_memory_entry(snapshot)
        
        # Verify entry structure
        assert isinstance(entry, JournalEntry)
        assert "AI" in entry.content or "artificial intelligence" in entry.content.lower()
        assert "Goal:" in entry.content
        assert "Feeling:" in entry.content
        assert entry.summary is not None
        assert len(entry.summary) <= 303  # Should be truncated
        assert "episodic" in entry.tags
        assert "workspace_consolidation" in entry.tags
        assert 1 <= entry.significance_score <= 10
        assert entry.metadata["num_goals"] == 1
        assert entry.metadata["num_percepts"] == 1
    
    def test_build_query_from_workspace(self, memory_integration, workspace):
        """Test building search query from workspace state."""
        # Add goals
        goal1 = Goal(
            type=GoalType.RESPOND_TO_USER,
            description="Explain quantum computing",
            priority=0.9
        )
        goal2 = Goal(
            type=GoalType.RETRIEVE_MEMORY,
            description="Find information about qubits",
            priority=0.7
        )
        workspace.add_goal(goal1)
        workspace.add_goal(goal2)
        
        # Add percept
        percept = Percept(
            modality="text",
            raw="Tell me about quantum entanglement",
            complexity=10,
            metadata={"attention_score": 0.85}
        )
        workspace.active_percepts[percept.id] = percept
        
        snapshot = workspace.broadcast()
        
        # Build query
        query = memory_integration._build_memory_query(snapshot)
        
        # Verify query content
        assert query is not None
        assert len(query) > 0
        assert "quantum" in query.lower()
        assert len(query) <= 500  # Should be limited


class TestShouldConsolidate:
    """Test consolidation decision logic."""
    
    def test_should_consolidate_high_arousal(self, memory_integration, workspace):
        """Test consolidation decision with high arousal."""
        workspace.emotional_state["arousal"] = 0.75
        memory_integration.cycles_since_consolidation = 10
        
        snapshot = workspace.broadcast()
        should = memory_integration._should_consolidate(snapshot)
        
        assert should is True
    
    def test_should_consolidate_extreme_valence(self, memory_integration, workspace):
        """Test consolidation decision with extreme valence."""
        workspace.emotional_state["valence"] = 0.65
        memory_integration.cycles_since_consolidation = 10
        
        snapshot = workspace.broadcast()
        should = memory_integration._should_consolidate(snapshot)
        
        assert should is True
    
    def test_should_not_consolidate_min_cycles(self, memory_integration, workspace):
        """Test that minimum cycles is enforced."""
        workspace.emotional_state["arousal"] = 0.9
        memory_integration.cycles_since_consolidation = 2
        
        snapshot = workspace.broadcast()
        should = memory_integration._should_consolidate(snapshot)
        
        assert should is False
    
    def test_should_not_consolidate_low_activity(self, memory_integration, workspace):
        """Test that low activity prevents consolidation."""
        memory_integration.cycles_since_consolidation = 30
        
        snapshot = workspace.broadcast()
        should = memory_integration._should_consolidate(snapshot)
        
        assert should is False


class TestIntegrationWithCognitiveCore:
    """Test integration with CognitiveCore."""
    
    @pytest.mark.asyncio
    async def test_memory_integration_initialization(self, tmp_path):
        """Test that CognitiveCore initializes memory integration."""
        from mind.cognitive_core.core import CognitiveCore
        
        config = {
            "memory": {
                "memory_config": {
                    "base_dir": str(tmp_path / "memories"),
                    "chroma_dir": str(tmp_path / "chroma"),
                }
            },
            "perception": {"mock_mode": True},
        }

        core = CognitiveCore(config=config)
        
        # Verify memory integration is initialized
        assert hasattr(core, "memory")
        assert core.memory is not None
        assert isinstance(core.memory, MemoryIntegration)
