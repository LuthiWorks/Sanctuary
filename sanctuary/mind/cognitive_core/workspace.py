"""
Global Workspace: The "conscious" working memory buffer.

This module implements the GlobalWorkspace class based on Global Workspace Theory,
which proposes that consciousness arises from a limited-capacity workspace that
broadcasts information to multiple specialized subsystems.

The GlobalWorkspace serves as:
- The "conscious" content at any given moment
- A bottleneck that creates selective attention
- A broadcast mechanism for system-wide coordination
- A unified representation of current goals, percepts, and emotions
"""

from __future__ import annotations

import copy
import uuid
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from enum import Enum

from pydantic import BaseModel, Field, ConfigDict

# Configure logging
logger = logging.getLogger(__name__)


class GoalType(str, Enum):
    """Types of goals the system can pursue."""
    RESPOND_TO_USER = "respond_to_user"
    COMMIT_MEMORY = "commit_memory"
    RETRIEVE_MEMORY = "retrieve_memory"
    INTROSPECT = "introspect"
    LEARN = "learn"
    CREATE = "create"
    MAINTAIN_VALUE = "maintain_value"
    SPEAK_AUTONOMOUS = "speak_autonomous"  # Unprompted speech initiated by Sanctuary


class Goal(BaseModel):
    """
    Represents a goal or intention in the workspace.
    
    Goals drive the cognitive system's behavior and provide top-down
    influence on attention and action selection.
    
    Attributes:
        id: Unique identifier for the goal
        type: Category of goal (respond, memory, introspect, etc.)
        description: Human-readable description of what the goal is
        priority: Importance of this goal (0.0-1.0, higher is more important)
        created_at: When this goal was created
        progress: How close to completion (0.0-1.0, 1.0 is complete)
        metadata: Additional goal-specific information
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: GoalType
    description: str
    priority: float = Field(ge=0.0, le=1.0, default=0.5)
    created_at: datetime = Field(default_factory=datetime.now)
    progress: float = Field(ge=0.0, le=1.0, default=0.0)
    deadline: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Percept(BaseModel):
    """
    Represents a perceptual input that has gained attention.
    
    Percepts are internal representations of external sensory input
    that have been selected by the attention system for conscious processing.
    
    Attributes:
        id: Unique identifier for the percept
        modality: Type of sensory input (text, image, audio, introspection)
        embedding: Optional vector representation for similarity comparison
        raw: Original data (string, dict, etc.)
        complexity: Cognitive cost in attention units
        timestamp: When the percept was created
        metadata: Additional percept-specific information
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    modality: str  # "text", "image", "audio", "introspection"
    embedding: Optional[List[float]] = None
    raw: Any
    complexity: int = 1
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Memory(BaseModel):
    """
    Represents a memory that has been retrieved into conscious awareness.
    
    Memories provide context from past experience and can influence
    current processing, goals, and actions.
    
    Attributes:
        id: Unique identifier for the memory
        content: The memory content (text, structured data, etc.)
        embedding: Optional vector representation for similarity
        timestamp: When the memory was originally created
        significance: Importance/relevance of this memory (0.0-1.0)
        tags: Category labels for the memory
        metadata: Additional memory-specific information
    """
    id: str
    content: str
    embedding: Optional[List[float]] = None
    timestamp: datetime
    significance: float = Field(ge=0.0, le=1.0)
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WorkspaceSnapshot(BaseModel):
    """
    Immutable snapshot of workspace state at a point in time.
    
    This frozen model allows subsystems to read the conscious state
    without risk of concurrent modification or unintended side effects.
    Broadcasting snapshots ensures clean information flow.
    
    Attributes:
        goals: Current active goals
        percepts: Currently attended perceptual inputs (keyed by ID)
        emotions: Current emotional state (valence, arousal, dominance, etc.)
        memories: Memories in conscious awareness
        timestamp: When this snapshot was taken
        cycle_count: Number of cognitive cycles processed
        metadata: Additional context information (e.g., recent_actions)
        temporal_context: Temporal awareness information (session time, time since events)
    """
    model_config = ConfigDict(frozen=True)  # Immutable
    
    goals: List[Goal]
    percepts: Dict[str, Any]
    emotions: Dict[str, float]
    memories: List[Any]
    timestamp: datetime
    cycle_count: int
    metadata: Dict[str, Any] = Field(default_factory=dict)
    temporal_context: Optional[Dict[str, Any]] = None


# Legacy dataclass kept for backward compatibility
from dataclasses import dataclass, field


@dataclass
class WorkspaceContent:
    """
    Legacy dataclass for backward compatibility.
    
    Note: New code should use Goal, Percept, Memory, and WorkspaceSnapshot
    Pydantic models instead for better validation and type safety.
    """
    goals: List[str] = field(default_factory=list)
    percepts: List[Dict[str, Any]] = field(default_factory=list)
    emotions: Dict[str, float] = field(default_factory=dict)
    memories: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class GlobalWorkspace:
    """
    The "conscious" working memory buffer with broadcast mechanism.

    The GlobalWorkspace is the central hub of the cognitive architecture, implementing
    Global Workspace Theory's core principle: consciousness emerges from a limited-capacity
    workspace that broadcasts unified information to multiple specialized subsystems.

    Key Responsibilities:
    - Maintain current conscious content (goals, percepts, emotions, memories)
    - Implement capacity limits to create selective attention bottleneck
    - Broadcast workspace updates to all registered subsystems
    - Ensure coherent integration of multimodal information
    - Track temporal continuity of conscious experience

    Integration Points:
    - AttentionController: Determines what information enters the workspace
    - PerceptionSubsystem: Provides candidate percepts for workspace inclusion
    - ActionSubsystem: Reads workspace to guide behavior selection
    - AffectSubsystem: Contributes emotional state to workspace content
    - SelfMonitor: Observes workspace state for meta-cognitive awareness
    - CognitiveCore: Orchestrates workspace updates in the main loop

    The workspace implements a "winner-take-all" dynamic where only the most
    salient information (as determined by AttentionController) gains access to
    the limited-capacity conscious buffer. This creates the selective nature of
    attention and conscious awareness.

    Broadcasting Mechanism:
    The workspace maintains a registry of subscriber callbacks that are invoked
    whenever workspace content changes. This allows all subsystems to stay
    synchronized with the current conscious state without tight coupling.

    Attributes:
        current_goals: Active goals the system is pursuing
        active_percepts: Currently attended sensory inputs (keyed by percept ID)
        emotional_state: Current emotional state (valence, arousal, dominance)
        attended_memories: Memories currently in conscious awareness
        timestamp: Last update time
        cycle_count: Number of cognitive cycles processed
    """

    def __init__(
        self,
        capacity: int = 7,
        persistence_dir: Optional[str] = None,
    ) -> None:
        """
        Initialize the global workspace.

        Args:
            capacity: Maximum number of items in workspace (default 7, based on
                Miller's "magical number" for working memory capacity).
            persistence_dir: Optional directory for saving/loading workspace history.
        """
        self.capacity = capacity
        self.persistence_dir = persistence_dir
        
        # Core state
        self.metadata: Dict[str, Any] = {}
        self.current_goals: List[Goal] = []
        self.active_percepts: Dict[str, Percept] = {}
        self.emotional_state: Dict[str, float] = {
            "valence": 0.0,  # positive/negative
            "arousal": 0.0,  # activation level
            "dominance": 0.0,  # sense of control
        }
        self.attended_memories: List[Memory] = []
        
        # Tracking
        self.timestamp: datetime = datetime.now()
        self.cycle_count: int = 0
        
        # Temporal context
        self.temporal_context: Optional[Dict[str, Any]] = None
        
        logger.info(f"GlobalWorkspace initialized with capacity={capacity}")

    def broadcast(self) -> WorkspaceSnapshot:
        """
        Returns an immutable snapshot of current workspace state.
        
        This method creates a frozen Pydantic model containing the current
        conscious state. Subsystems can read this snapshot without risk of
        concurrent modification.
        
        Returns:
            WorkspaceSnapshot: Immutable snapshot of workspace state
            
        Example:
            >>> workspace = GlobalWorkspace()
            >>> snapshot = workspace.broadcast()
            >>> # snapshot.goals is immutable - cannot be modified
        """
        # Deep-copy the nested containers so subsystems holding a snapshot
        # can't reach in and mutate Goal.priority, Percept.metadata, etc.,
        # and silently affect shared workspace state. The `frozen=True` on
        # WorkspaceSnapshot itself only prevents reassigning fields on the
        # snapshot — the list/dict contents still need explicit copying.
        # Audit 2026-05-10 flagged "Snapshot immutability is a lie."
        return WorkspaceSnapshot(
            goals=copy.deepcopy(self.current_goals),
            percepts=copy.deepcopy(self.active_percepts),
            emotions=self.emotional_state.copy(),
            memories=copy.deepcopy(self.attended_memories),
            timestamp=self.timestamp,
            cycle_count=self.cycle_count,
            metadata=copy.deepcopy(self.metadata),
            temporal_context=(
                copy.deepcopy(self.temporal_context)
                if self.temporal_context
                else None
            ),
        )

    def update(self, subsystem_outputs: List[Any]) -> None:
        """
        Integrates new information from subsystems into workspace.
        
        This method processes outputs from various cognitive subsystems
        and updates the workspace state accordingly. It handles:
        - New goals from planning/reasoning
        - New percepts from attention system
        - Emotional updates from affect subsystem
        - Memory updates from memory system
        
        Args:
            subsystem_outputs: List of outputs from cognitive subsystems
                Each output should be a dict with 'type' and relevant data
                
        Example:
            >>> outputs = [
            ...     {'type': 'goal', 'data': goal_instance},
            ...     {'type': 'emotion', 'data': {'valence': 0.5}},
            ... ]
            >>> workspace.update(outputs)
        """
        for output in subsystem_outputs:
            if isinstance(output, dict):
                output_type = output.get('type')
                data = output.get('data')
                
                if output_type == 'goal' and isinstance(data, Goal):
                    self.add_goal(data)
                elif output_type == 'percept' and isinstance(data, Percept):
                    self.active_percepts[data.id] = data
                elif output_type == 'emotion' and isinstance(data, dict):
                    self.emotional_state.update(data)
                elif output_type == 'memory' and isinstance(data, Memory):
                    self.attended_memories.append(data)
        
        # Update tracking
        self.cycle_count += 1
        self.timestamp = datetime.now()
        
        logger.debug(f"Workspace updated: cycle={self.cycle_count}, "
                    f"goals={len(self.current_goals)}, "
                    f"percepts={len(self.active_percepts)}")

    def add_percept(self, percept: Percept) -> None:
        """
        Adds a new percept to active_percepts.

        This method adds a percept to the workspace, using its ID as key.
        When at capacity, the oldest percept is evicted — GWT's limited-
        capacity bottleneck is the entire theoretical point of the model.

        Args:
            percept: The Percept instance to add
        """
        self.active_percepts[percept.id] = percept

        # Capacity enforcement (GWT bottleneck): evict oldest by timestamp
        # when over the limit. Audit 2026-05-10 flagged that capacity was
        # accepted but never enforced — making the workspace a dict instead
        # of the limited-capacity broadcast theory calls for.
        if len(self.active_percepts) > self.capacity:
            oldest_id = min(
                self.active_percepts,
                key=lambda pid: self.active_percepts[pid].timestamp,
            )
            evicted = self.active_percepts.pop(oldest_id)
            logger.debug(
                "Evicted percept %s (modality=%s) — workspace at capacity",
                evicted.id,
                evicted.modality,
            )

        logger.debug(f"Added percept: id={percept.id}, modality={percept.modality}")

    def add_goal(self, goal: Goal) -> None:
        """
        Adds a new goal to current_goals.

        This method adds a goal to the workspace, checking for duplicates
        and maintaining priority ordering if needed.

        Args:
            goal: The Goal instance to add

        Raises:
            ValueError: If goal validation fails

        Example:
            >>> goal = Goal(type=GoalType.RESPOND_TO_USER,
            ...             description="Answer user query")
            >>> workspace.add_goal(goal)
        """
        # Check for duplicate by ID
        if any(g.id == goal.id for g in self.current_goals):
            logger.warning(f"Goal {goal.id} already exists in workspace")
            return

        self.current_goals.append(goal)

        # Sort by priority (highest first)
        self.current_goals.sort(key=lambda g: g.priority, reverse=True)

        # Capacity enforcement: drop the lowest-priority tail when above
        # capacity. With the list already sorted high → low, this is the
        # cheapest eviction policy that preserves the most-important goals.
        if len(self.current_goals) > self.capacity:
            dropped = self.current_goals[self.capacity:]
            self.current_goals = self.current_goals[: self.capacity]
            for g in dropped:
                logger.debug(
                    "Evicted goal %s (priority=%.2f) — workspace at capacity",
                    g.id,
                    g.priority,
                )
        
        logger.info(f"Added goal: type={goal.type.value}, priority={goal.priority}")

    def update_goal_priority(self, goal_id: str, new_priority: float) -> bool:
        """
        Update the priority of an existing goal.

        Args:
            goal_id: ID of the goal to update
            new_priority: New priority value (clamped to 0.0-1.0)

        Returns:
            True if goal was found and updated, False otherwise
        """
        new_priority = max(0.0, min(1.0, new_priority))
        for goal in self.current_goals:
            if goal.id == goal_id:
                goal.priority = new_priority
                # Re-sort by priority
                self.current_goals.sort(key=lambda g: g.priority, reverse=True)
                return True
        return False

    def remove_completed_goals(
        self, completion_threshold: float = 1.0
    ) -> List[str]:
        """Remove goals whose progress has reached the completion threshold.

        Without this sweep, completed goals (especially RESPOND_TO_USER)
        accumulate in current_goals and generate redundant downstream
        actions — audit 2026-05-10 flagged this as a real cycle-level bug.
        The cycle executor should call this after each cycle once it's
        recorded progress on the active goals.

        Args:
            completion_threshold: Goals with `progress >= threshold` are
                removed. Default 1.0 (only fully-completed goals removed).

        Returns:
            List of removed goal IDs (for logging / metrics).
        """
        completed = [
            g for g in self.current_goals if g.progress >= completion_threshold
        ]
        if not completed:
            return []
        completed_ids = [g.id for g in completed]
        self.current_goals = [
            g for g in self.current_goals
            if g.progress < completion_threshold
        ]
        logger.info(
            "Removed %d completed goal(s): %s", len(completed_ids), completed_ids
        )
        return completed_ids

    def remove_goal(self, goal_id: str) -> None:
        """
        Removes completed or abandoned goals.
        
        Args:
            goal_id: ID of the goal to remove
            
        Example:
            >>> workspace.remove_goal("abc-123-def")
        """
        initial_count = len(self.current_goals)
        self.current_goals = [g for g in self.current_goals if g.id != goal_id]
        
        if len(self.current_goals) < initial_count:
            logger.info(f"Removed goal: id={goal_id}")
        else:
            logger.warning(f"Goal {goal_id} not found in workspace")

    def clear(self) -> None:
        """
        Resets workspace to initial state.
        
        This method is useful for testing and for clearing state
        between major context shifts.
        
        Example:
            >>> workspace.clear()
            >>> assert len(workspace.current_goals) == 0
        """
        self.current_goals.clear()
        self.active_percepts.clear()
        self.emotional_state = {
            "valence": 0.0,
            "arousal": 0.0,
            "dominance": 0.0,
        }
        self.attended_memories.clear()
        self.metadata = {}
        self.timestamp = datetime.now()
        self.cycle_count = 0
        self.temporal_context = None

        logger.info("Workspace cleared")
    
    def set_temporal_context(self, temporal_context: Dict[str, Any]) -> None:
        """
        Update temporal context for this cycle.
        
        Args:
            temporal_context: Temporal information dict from TemporalGrounding
        """
        self.temporal_context = temporal_context

    def to_dict(self) -> dict:
        """
        Serializes entire state to JSON-compatible dict.
        
        This method converts the workspace to a dictionary that can
        be saved to disk or transmitted over a network.
        
        Returns:
            dict: JSON-compatible representation of workspace state
            
        Example:
            >>> data = workspace.to_dict()
            >>> import json
            >>> json.dumps(data)  # Can be serialized
        """
        return {
            "current_goals": [g.model_dump(mode='json') for g in self.current_goals],
            "active_percepts": {k: v.model_dump(mode='json') for k, v in self.active_percepts.items()},
            "emotional_state": self.emotional_state,
            "attended_memories": [m.model_dump(mode='json') for m in self.attended_memories],
            "timestamp": self.timestamp.isoformat(),
            "cycle_count": self.cycle_count,
            "capacity": self.capacity,
        }

    @classmethod
    def from_dict(cls, data: dict) -> GlobalWorkspace:
        """
        Restores workspace from serialized state.
        
        This class method creates a new GlobalWorkspace instance from
        a dictionary created by to_dict().
        
        Args:
            data: Dictionary containing workspace state
            
        Returns:
            GlobalWorkspace: Restored workspace instance
            
        Raises:
            ValueError: If data format is invalid
            
        Example:
            >>> data = workspace.to_dict()
            >>> restored = GlobalWorkspace.from_dict(data)
        """
        workspace = cls(capacity=data.get("capacity", 7))
        
        # Restore goals
        workspace.current_goals = [Goal(**g) for g in data.get("current_goals", [])]
        
        # Restore percepts
        percepts_data = data.get("active_percepts", {})
        workspace.active_percepts = {k: Percept(**v) for k, v in percepts_data.items()}
        
        # Restore emotional state
        workspace.emotional_state = data.get("emotional_state", {
            "valence": 0.0,
            "arousal": 0.0,
            "dominance": 0.0,
        })
        
        # Restore memories
        workspace.attended_memories = [Memory(**m) for m in data.get("attended_memories", [])]
        
        # Restore tracking
        timestamp_str = data.get("timestamp")
        if timestamp_str:
            workspace.timestamp = datetime.fromisoformat(timestamp_str)
        workspace.cycle_count = data.get("cycle_count", 0)
        
        logger.info(f"Workspace restored from dict: cycle={workspace.cycle_count}")
        return workspace
