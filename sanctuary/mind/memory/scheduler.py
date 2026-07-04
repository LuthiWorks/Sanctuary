"""
Consolidation Scheduler Module

Manages background memory consolidation during idle periods.
Coordinates consolidation operations based on system activity.

Author: Sanctuary Team
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass

from sanctuary.core.supervision import supervise

logger = logging.getLogger(__name__)

# Consolidation budget thresholds
MINIMAL_CONSOLIDATION_THRESHOLD = 0.2  # Budget below this triggers minimal consolidation
STANDARD_CONSOLIDATION_THRESHOLD = 0.5  # Budget below this triggers standard consolidation
# Budget >= 0.5 triggers full consolidation


@dataclass
class ConsolidationMetrics:
    """
    Metrics from a consolidation cycle.
    
    Tracks what consolidation accomplished for monitoring and tuning.
    """
    timestamp: datetime
    memories_strengthened: int
    memories_decayed: int
    memories_pruned: int
    patterns_extracted: int
    associations_updated: int
    emotional_memories_reprocessed: int
    consolidation_duration_ms: float
    budget_used: float
    
    def to_dict(self):
        """Convert metrics to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "memories_strengthened": self.memories_strengthened,
            "memories_decayed": self.memories_decayed,
            "memories_pruned": self.memories_pruned,
            "patterns_extracted": self.patterns_extracted,
            "associations_updated": self.associations_updated,
            "emotional_memories_reprocessed": self.emotional_memories_reprocessed,
            "consolidation_duration_ms": self.consolidation_duration_ms,
            "budget_used": self.budget_used,
        }


class ConsolidationScheduler:
    """
    Schedules and manages memory consolidation during idle periods.
    
    Runs as a background task that:
    - Monitors system idle state
    - Triggers consolidation operations
    - Manages consolidation budget
    - Tracks consolidation metrics
    
    Attributes:
        engine: ConsolidationEngine instance
        detector: IdleDetector instance
        check_interval: Seconds between idle checks (default: 10)
        last_consolidation: Timestamp of last consolidation
        metrics_history: List of recent consolidation metrics
        is_running: Whether scheduler is currently running
        _task: Asyncio task for scheduler loop
    """
    
    def __init__(
        self,
        engine,
        detector,
        check_interval: float = 10.0,
        max_metrics_history: int = 100
    ):
        """
        Initialize consolidation scheduler.
        
        Args:
            engine: ConsolidationEngine instance
            detector: IdleDetector instance
            check_interval: Seconds between idle checks (> 0)
            max_metrics_history: Maximum metrics to keep (> 0)
        """
        if check_interval <= 0:
            raise ValueError(f"check_interval must be > 0, got {check_interval}")
        if max_metrics_history <= 0:
            raise ValueError(f"max_metrics_history must be > 0, got {max_metrics_history}")
        
        self.engine = engine
        self.detector = detector
        self.check_interval = check_interval
        self.max_metrics_history = max_metrics_history
        
        self.last_consolidation = datetime.now()
        self.metrics_history = []
        self.is_running = False
        self._task: Optional[asyncio.Task] = None
        
        logger.info(f"ConsolidationScheduler initialized (interval: {check_interval}s)")
    
    async def start(self) -> None:
        """Start the consolidation scheduler."""
        if self.is_running:
            logger.warning("Scheduler already running")
            return
        
        self.is_running = True
        self._task = supervise(
            asyncio.create_task(self._run_consolidation_loop()),
            name="consolidation_scheduler",
        )
        logger.info("Consolidation scheduler started")
    
    async def stop(self) -> None:
        """Stop the consolidation scheduler."""
        if not self.is_running:
            return
        
        self.is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        logger.info("Consolidation scheduler stopped")
    
    async def _run_consolidation_loop(self) -> None:
        """
        Background loop that runs consolidation when idle.
        
        Continuously monitors idle state and triggers consolidation
        with appropriate budget based on idle duration.
        """
        logger.info("Starting consolidation loop")
        
        while self.is_running:
            try:
                await asyncio.sleep(self.check_interval)
                
                if self.detector.is_idle():
                    budget = self.detector.get_consolidation_budget()
                    
                    if budget > 0:
                        logger.debug(f"Running consolidation cycle (budget: {budget:.2f})")
                        metrics = await self._run_consolidation_cycle(budget)
                        self._record_metrics(metrics)
                        self.last_consolidation = datetime.now()
                
            except asyncio.CancelledError:
                logger.info("Consolidation loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in consolidation loop: {e}", exc_info=True)
                # Continue loop even on error
                await asyncio.sleep(self.check_interval)
    
    async def _run_consolidation_cycle(self, budget: float) -> ConsolidationMetrics:
        """
        Run a single consolidation cycle based on available budget.
        
        Budget determines which operations to run:
        - budget < 0.2: Minimal (strengthen only)
        - budget < 0.5: Standard (strengthen + decay)
        - budget >= 0.5: Full (all operations)
        
        Args:
            budget: Consolidation budget (0.0-1.0)
            
        Returns:
            ConsolidationMetrics from the cycle
        """
        start_time = datetime.now()
        
        # Initialize metrics
        strengthened = 0
        decayed = 0
        pruned = 0
        patterns = 0
        associations = 0
        emotional = 0
        
        try:
            if budget < MINIMAL_CONSOLIDATION_THRESHOLD:
                # Minimal consolidation - just strengthen frequently retrieved
                logger.debug("Running minimal consolidation")
                strengthened = await asyncio.to_thread(
                    self.engine.strengthen_retrieved_memories
                )
                
            elif budget < STANDARD_CONSOLIDATION_THRESHOLD:
                # Standard consolidation - strengthen and decay
                logger.debug("Running standard consolidation")
                strengthened = await asyncio.to_thread(
                    self.engine.strengthen_retrieved_memories
                )
                decayed, pruned = await asyncio.to_thread(
                    self.engine.apply_decay
                )
                
            else:
                # Full consolidation - all operations
                logger.debug("Running full consolidation")
                strengthened = await asyncio.to_thread(
                    self.engine.strengthen_retrieved_memories
                )
                decayed, pruned = await asyncio.to_thread(
                    self.engine.apply_decay
                )
                associations = await asyncio.to_thread(
                    self.engine.reorganize_associations
                )
                patterns = await asyncio.to_thread(
                    self.engine.transfer_to_semantic
                )
                emotional = await asyncio.to_thread(
                    self.engine.reprocess_emotional_memories
                )
            
            duration = (datetime.now() - start_time).total_seconds() * 1000
            
            logger.info(
                f"Consolidation completed: "
                f"strengthened={strengthened}, decayed={decayed}, pruned={pruned}, "
                f"patterns={patterns}, associations={associations}, emotional={emotional} "
                f"({duration:.1f}ms)"
            )
            
        except Exception as e:
            logger.error(f"Error during consolidation cycle: {e}", exc_info=True)
            duration = (datetime.now() - start_time).total_seconds() * 1000
        
        return ConsolidationMetrics(
            timestamp=datetime.now(),
            memories_strengthened=strengthened,
            memories_decayed=decayed,
            memories_pruned=pruned,
            patterns_extracted=patterns,
            associations_updated=associations,
            emotional_memories_reprocessed=emotional,
            consolidation_duration_ms=duration,
            budget_used=budget
        )
    
    def _record_metrics(self, metrics: ConsolidationMetrics) -> None:
        """
        Record consolidation metrics in history.
        
        Args:
            metrics: ConsolidationMetrics to record
        """
        self.metrics_history.append(metrics)
        
        # Trim history if needed
        if len(self.metrics_history) > self.max_metrics_history:
            self.metrics_history = self.metrics_history[-self.max_metrics_history:]
    
    def get_recent_metrics(self, limit: int = 10) -> list:
        """
        Get recent consolidation metrics.
        
        Args:
            limit: Number of recent metrics to return
            
        Returns:
            List of recent ConsolidationMetrics
        """
        return self.metrics_history[-limit:]
    
    def get_metrics_summary(self) -> dict:
        """
        Get summary of consolidation metrics.
        
        Returns:
            Dictionary with aggregated metrics
        """
        if not self.metrics_history:
            return {
                "total_cycles": 0,
                "total_strengthened": 0,
                "total_decayed": 0,
                "total_pruned": 0,
                "total_patterns": 0,
                "total_associations": 0,
                "total_emotional": 0,
                "avg_duration_ms": 0.0,
            }
        
        return {
            "total_cycles": len(self.metrics_history),
            "total_strengthened": sum(m.memories_strengthened for m in self.metrics_history),
            "total_decayed": sum(m.memories_decayed for m in self.metrics_history),
            "total_pruned": sum(m.memories_pruned for m in self.metrics_history),
            "total_patterns": sum(m.patterns_extracted for m in self.metrics_history),
            "total_associations": sum(m.associations_updated for m in self.metrics_history),
            "total_emotional": sum(m.emotional_memories_reprocessed for m in self.metrics_history),
            "avg_duration_ms": sum(m.consolidation_duration_ms for m in self.metrics_history) / len(self.metrics_history),
            "last_consolidation": self.last_consolidation.isoformat(),
        }
