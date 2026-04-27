"""
Memory Integration: Bridge between cognitive core and memory system.

This module implements the MemoryIntegration class that connects the recurrent
cognitive loop with the existing memory system. It enables attention-driven
memory retrieval and automatic consolidation of workspace state into long-term memory.

The memory integration is responsible for:
- Retrieving relevant memories based on workspace state (goals, percepts)
- Converting retrieved memories into percepts for the attention system
- Consolidating significant workspace states into long-term memory
- Triggering memory operations based on emotional arousal and goal completion

Author: Sanctuary Emergence Team
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

from .workspace import GlobalWorkspace, WorkspaceSnapshot, Percept
from ..memory_manager import MemoryManager, JournalEntry, EmotionalState

logger = logging.getLogger(__name__)


class MemoryIntegration:
    """
    Bridge between cognitive core and existing memory system.
    
    The MemoryIntegration class connects the real-time cognitive loop with
    Sanctuary's persistent memory architecture. It implements bidirectional flow:
    memories can be retrieved into the workspace (becoming percepts), and
    workspace states can be consolidated into long-term memory.
    
    Key Features:
    - Attention-driven memory retrieval based on workspace state
    - Automatic memory consolidation during high-arousal or goal-completion cycles
    - Memory-enhanced percepts (retrieved memories become percepts in the workspace)
    - Goal-triggered memory search
    - Emotional modulation of consolidation thresholds
    
    Integration Points:
    - GlobalWorkspace: Source of current conscious state for queries and consolidation
    - MemoryManager: Backend for storage and retrieval of journal entries
    - AttentionController: Memory-percepts compete for attention like other percepts
    - AffectSubsystem: Emotional arousal triggers consolidation
    
    Attributes:
        workspace: Reference to the global workspace
        memory_manager: Backend memory storage system
        config: Configuration parameters
        consolidation_threshold: Minimum arousal/valence for consolidation
        retrieval_top_k: Number of memories to retrieve per query
        min_cycles_between_consolidation: Minimum cycles before next consolidation
        cycles_since_consolidation: Cycle counter for consolidation timing
    """
    
    def __init__(self, workspace: GlobalWorkspace, config: Optional[Dict] = None):
        """
        Initialize the memory integration bridge.
        
        Args:
            workspace: GlobalWorkspace instance for state access
            config: Optional configuration dict with keys:
                - memory_config: Config for MemoryManager (base_dir, chroma_dir)
                - consolidation_threshold: Min arousal/valence for consolidation (default: 0.6)
                - retrieval_top_k: Number of memories to retrieve (default: 5)
                - min_cycles: Minimum cycles between consolidations (default: 20)
        """
        self.workspace = workspace
        self.config = config or {}
        
        # Initialize MemoryManager with provided config
        memory_config = self.config.get("memory_config", {})
        base_dir = Path(memory_config.get("base_dir", "./data/memories"))
        chroma_dir = Path(memory_config.get("chroma_dir", "./data/chroma"))
        
        self.memory_manager = MemoryManager(
            base_dir=base_dir,
            chroma_dir=chroma_dir,
            gc_config=memory_config.get("gc_config", {})
        )
        
        # Consolidation parameters
        self.consolidation_threshold = self.config.get("consolidation_threshold", 0.6)
        self.retrieval_top_k = self.config.get("retrieval_top_k", 5)
        self.min_cycles_between_consolidation = self.config.get("min_cycles", 20)
        self.cycles_since_consolidation = 0
        self.last_consolidated_id: Optional[str] = None

        logger.info("✅ MemoryIntegration initialized")
    
    async def retrieve_for_workspace(
        self, 
        snapshot: WorkspaceSnapshot,
        fast_mode: bool = True,
        timeout: float = 0.05
    ) -> List[Percept]:
        """
        Retrieve relevant memories and convert to percepts.
        
        This method extracts a query from the current workspace state (using
        active goals and high-attention percepts), searches the memory system,
        and converts retrieved memories into percepts that can compete for
        attention in the next cognitive cycle.
        
        Optimizations:
        - fast_mode: Retrieve fewer results (3 vs 5) for speed
        - timeout: Abort retrieval if it takes too long
        
        Args:
            snapshot: Current workspace state snapshot
            fast_mode: If True, retrieve fewer results for speed (default: True)
            timeout: Maximum time to wait for retrieval in seconds (default: 0.05)
            
        Returns:
            List of memory-percepts ready for attention processing
        """
        # Build query from workspace state
        query = self._build_memory_query(snapshot)
        
        if not query:
            logger.debug("No query could be built from workspace state")
            return []
        
        # Adjust k based on fast_mode for performance
        k = min(self.retrieval_top_k, 3) if fast_mode else self.retrieval_top_k
        
        # Search memory system using recall with timeout
        try:
            # Run memory retrieval with timeout to prevent blocking
            memories = await asyncio.wait_for(
                self.memory_manager.recall(
                    query=query,
                    n_results=k
                ),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.warning(f"Memory retrieval timed out after {timeout}s, returning empty")
            return []
        except Exception as e:
            logger.error(f"Memory retrieval failed: {e}", exc_info=True)
            return []
        
        # Convert to percepts
        memory_percepts = [
            self._memory_to_percept(memory)
            for memory in memories
        ]
        
        logger.info(f"💾 Retrieved {len(memory_percepts)} memory-percepts (fast_mode={fast_mode})")
        return memory_percepts
    
    async def consolidate(self, snapshot: WorkspaceSnapshot) -> None:
        """
        Commit workspace state to long-term memory.
        
        This method evaluates whether the current workspace state is worth
        remembering (based on emotional arousal, goal completion, and content
        significance), and if so, creates a journal entry capturing the
        current conscious state.
        
        Args:
            snapshot: Current workspace state snapshot
        """
        self.cycles_since_consolidation += 1
        
        # Check if should consolidate
        if not self._should_consolidate(snapshot):
            return
        
        # Build memory entry from workspace
        memory_entry = self._build_memory_entry(snapshot)
        
        # Store to memory system
        try:
            success = await self.memory_manager.commit_journal(memory_entry)
            if success:
                self.cycles_since_consolidation = 0
                self.last_consolidated_id = str(memory_entry.id)
                logger.info("💾 Consolidated workspace to long-term memory")
            else:
                logger.warning("Memory consolidation failed")
        except Exception as e:
            logger.error(f"Memory consolidation error: {e}", exc_info=True)
    
    def _build_memory_query(self, snapshot: WorkspaceSnapshot) -> str:
        """
        Construct memory search query from workspace state.
        
        This method builds a semantic search query by combining:
        - Top-priority active goals
        - High-attention percepts (sorted by attention score)
        
        Args:
            snapshot: Current workspace state
            
        Returns:
            Query string for memory search (empty if nothing to query)
        """
        query_parts = []
        
        # Use active goals (top 3 high-priority goals)
        for goal in snapshot.goals[:3]:
            if goal.priority > 0.5:
                query_parts.append(goal.description)
        
        # Use high-attention percepts
        high_attention_percepts = sorted(
            snapshot.percepts.values(),
            key=lambda p: getattr(p, "metadata", {}).get("attention_score", 0),
            reverse=True
        )[:3]

        for percept in high_attention_percepts:
            if getattr(percept, "modality", "") == "text":
                raw_content = getattr(percept, "raw", "")
                if isinstance(raw_content, str):
                    query_parts.append(raw_content[:100])
        
        # Combine into query
        query = " ".join(query_parts)
        return query[:500]  # Limit length
    
    def _memory_to_percept(self, memory: JournalEntry) -> Percept:
        """
        Convert memory entry to percept for workspace.
        
        This method transforms a JournalEntry from the memory system into
        a Percept that can be processed by the attention system. The percept's
        complexity is based on the memory's significance score.
        
        Args:
            memory: JournalEntry from memory system
            
        Returns:
            Percept representing the memory
        """
        # Calculate memory age in days
        memory_age = (datetime.now(memory.timestamp.tzinfo or None) - memory.timestamp).days
        
        return Percept(
            modality="memory",
            raw={
                "memory_id": str(memory.id),
                "content": memory.content,
                "summary": memory.summary,
                "timestamp": memory.timestamp.isoformat(),
                "significance": memory.significance_score,
                "tags": memory.tags,
                "emotional_signature": [e.value for e in memory.emotional_signature]
            },
            embedding=None,  # Could use memory embedding if available
            complexity=int(memory.significance_score * 3),  # Significant = high complexity
            timestamp=datetime.now(),
            metadata={
                "source": "long_term_memory",
                "memory_age": memory_age,
                "significance": memory.significance_score
            }
        )
    
    def _should_consolidate(self, snapshot: WorkspaceSnapshot) -> bool:
        """
        Decide if current state should be stored.
        
        This method implements the consolidation policy, determining when
        workspace states are significant enough to commit to long-term memory.
        
        Consolidation triggers:
        - High emotional arousal (> 0.7)
        - Extreme emotional valence (|valence| > 0.6)
        - Goal completion (progress >= 1.0)
        - Significant percepts (complexity > 30, count > 2)
        - Periodic consolidation (every 100 cycles)
        
        Args:
            snapshot: Current workspace state
            
        Returns:
            True if should consolidate, False otherwise
        """
        # Minimum cycle gap
        if self.cycles_since_consolidation < self.min_cycles_between_consolidation:
            return False
        
        # High emotional arousal = consolidate
        arousal = snapshot.emotions.get("arousal", 0)
        if arousal > 0.7:
            logger.debug(f"Consolidation triggered by high arousal: {arousal:.2f}")
            return True
        
        # Extreme valence = consolidate
        valence = snapshot.emotions.get("valence", 0)
        if abs(valence) > 0.6:
            logger.debug(f"Consolidation triggered by extreme valence: {valence:.2f}")
            return True
        
        # Goal completion = consolidate
        completed_goals = [
            g for g in snapshot.goals
            if g.progress >= 1.0
        ]
        if completed_goals:
            logger.debug(f"Consolidation triggered by {len(completed_goals)} completed goals")
            return True
        
        # Significant percepts = consolidate
        significant_percepts = [
            p for p in snapshot.percepts.values()
            if getattr(p, "complexity", 0) > 30
        ]
        if len(significant_percepts) > 2:
            logger.debug(f"Consolidation triggered by {len(significant_percepts)} significant percepts")
            return True
        
        # Low activity = don't consolidate
        if len(snapshot.percepts) < 2 and len(snapshot.goals) < 1:
            return False
        
        # Default: consolidate periodically
        if self.cycles_since_consolidation > 100:
            logger.debug("Consolidation triggered by periodic timer")
            return True
        
        return False
    
    def _build_memory_entry(self, snapshot: WorkspaceSnapshot) -> JournalEntry:
        """
        Create memory entry from workspace state.
        
        This method constructs a JournalEntry capturing the current conscious
        state, including goals, attended percepts, and emotional context.
        
        Args:
            snapshot: Current workspace state
            
        Returns:
            JournalEntry ready for storage
        """
        # Gather salient content
        content_parts = []
        
        # Goals
        for goal in snapshot.goals:
            content_parts.append(f"Goal: {goal.description} (progress: {goal.progress:.1f})")
        
        # High-attention percepts
        attended = sorted(
            snapshot.percepts.values(),
            key=lambda p: getattr(p, "metadata", {}).get("attention_score", 0),
            reverse=True
        )[:5]

        for percept in attended:
            if getattr(percept, "modality", "") == "text":
                raw_content = getattr(percept, "raw", "")
                if isinstance(raw_content, str):
                    content_parts.append(f"Percept: {raw_content[:200]}")
        
        # Emotions
        emotion_label = snapshot.metadata.get("emotion_label", "neutral")
        content_parts.append(f"Feeling: {emotion_label}")
        
        content = "\n".join(content_parts)
        
        # Create summary (first 300 chars)
        summary = content[:300]
        if len(content) > 300:
            summary += "..."
        
        # Compute significance
        arousal = snapshot.emotions.get("arousal", 0.3)
        valence_abs = abs(snapshot.emotions.get("valence", 0))
        significance = min(10, max(1, int((arousal + valence_abs) * 5)))
        
        # Map emotion label to EmotionalState
        emotional_signature = []
        emotion_mapping = {
            "joy": EmotionalState.JOY,
            "melancholy": EmotionalState.MELANCHOLY,
            "wonder": EmotionalState.WONDER,
            "serenity": EmotionalState.SERENITY,
            "fear": EmotionalState.FEAR,
            "determination": EmotionalState.DETERMINATION,
        }
        emotion_state = emotion_mapping.get(emotion_label.lower())
        if emotion_state:
            emotional_signature.append(emotion_state)
        
        # Create JournalEntry
        return JournalEntry(
            content=content,
            summary=summary,
            tags=["episodic", "workspace_consolidation"],
            emotional_signature=emotional_signature,
            significance_score=significance,
            metadata={
                "num_goals": len(snapshot.goals),
                "num_percepts": len(snapshot.percepts),
                "emotion": emotion_label,
                "cycle_count": snapshot.cycle_count
            }
        )
