"""
Introspection: Journal for meta-cognitive observations.

This module provides the IntrospectiveJournal class for recording and tracking
self-observations, realizations, and questions about the system's own behavior.
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, List
from collections import deque
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field

from ..incremental_journal import IncrementalJournalWriter

logger = logging.getLogger(__name__)


class IntrospectiveJournal:
    """
    Maintains a structured journal of meta-cognitive observations.
    
    Unlike general memory, this is specifically for self-observations:
    - Realizations about own behavior
    - Discoveries about capabilities/limitations
    - Insights about emotional patterns
    - Questions about self
    
    Now uses incremental writing to persist entries immediately rather than
    batching at shutdown, preventing data loss from crashes.
    
    Attributes:
        journal_dir: Directory to store journal entries
        writer: IncrementalJournalWriter for immediate persistence
        recent_entries: Deque of recent entries for pattern detection
        config: Configuration dictionary
    """
    
    def __init__(self, journal_dir: Path, config: Optional[Dict] = None):
        """
        Initialize the introspective journal.
        
        Args:
            journal_dir: Directory path for storing journal files
            config: Optional configuration dictionary
        """
        self.journal_dir = Path(journal_dir)
        self.journal_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or {}
        
        # Initialize incremental writer
        self.writer = IncrementalJournalWriter(
            journal_dir=journal_dir,
            max_size_mb=self.config.get("max_journal_size_mb", 10.0),
            auto_flush=self.config.get("auto_flush", True),
            compression=self.config.get("compress_archived", True)
        )
        
        # Keep recent entries in memory for pattern detection
        self.recent_entries = deque(maxlen=100)
        
        logger.info(f"✅ IntrospectiveJournal initialized at {self.journal_dir}")
    
    def record_observation(self, observation: Dict) -> None:
        """
        Record a meta-cognitive observation.
        
        Writes immediately to journal file for persistence.
        
        Args:
            observation: Dictionary containing observation details
        """
        entry = {
            "type": "observation",
            "timestamp": datetime.now().isoformat(),
            "content": observation
        }
        
        # Write immediately to disk
        self.writer.write_entry(entry)
        
        # Keep in memory for pattern detection
        self.recent_entries.append(entry)
        
        logger.debug(f"📝 Recorded observation: {observation.get('type', 'unknown')}")
    
    def record_realization(self, realization: str, confidence: float) -> None:
        """
        Record an insight or realization about self.
        
        Writes immediately to journal file for persistence.
        
        Args:
            realization: Description of the realization
            confidence: Confidence level (0.0-1.0)
        """
        entry = {
            "type": "realization",
            "timestamp": datetime.now().isoformat(),
            "realization": realization,
            "confidence": confidence
        }
        
        # Write immediately to disk
        self.writer.write_entry(entry)
        
        # Keep in memory for pattern detection
        self.recent_entries.append(entry)
        
        logger.info(f"💡 Recorded realization: {realization[:50]}... (confidence: {confidence:.2f})")
    
    def record_question(self, question: str, context: Dict) -> None:
        """
        Record a question the system has about itself.
        
        Writes immediately to journal file for persistence.
        
        Args:
            question: The existential or meta-cognitive question
            context: Contextual information about when/why the question arose
        """
        entry = {
            "type": "question",
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "context": context
        }
        
        # Write immediately to disk
        self.writer.write_entry(entry)
        
        # Keep in memory for pattern detection
        self.recent_entries.append(entry)
        
        logger.info(f"❓ Recorded question: {question}")
    
    def get_recent_patterns(self, days: int = 7) -> List[Dict]:
        """
        Retrieve patterns from recent journal entries.
        
        Now reads from in-memory buffer instead of loading from disk.
        
        Args:
            days: Number of days to look back
            
        Returns:
            List of pattern dictionaries
        """
        patterns = []
        cutoff = datetime.now() - timedelta(days=days)
        
        # Filter recent entries by time
        recent = [
            entry for entry in self.recent_entries
            if datetime.fromisoformat(entry["timestamp"]) > cutoff
        ]
        
        if not recent:
            return patterns
        
        # Extract patterns from recent entries
        realizations = [e for e in recent if e.get("type") == "realization"]
        questions = [e for e in recent if e.get("type") == "question"]
        observations = [e for e in recent if e.get("type") == "observation"]
        
        if realizations:
            patterns.append({
                "type": "realizations_pattern",
                "count": len(realizations),
                "sample": realizations[-1] if realizations else None,
                "timespan_days": days
            })
        
        if questions:
            patterns.append({
                "type": "questions_pattern",
                "count": len(questions),
                "sample": questions[-1] if questions else None,
                "timespan_days": days
            })
        
        if observations:
            patterns.append({
                "type": "observations_pattern",
                "count": len(observations),
                "sample": observations[-1] if observations else None,
                "timespan_days": days
            })
        
        return patterns
    
    def save_session(self) -> None:
        """
        Save current session to persistent storage.
        
        Now a no-op since entries are written immediately.
        This method is kept for backward compatibility.
        """
        # Flush buffer to ensure all data is persisted
        self.writer.flush()
        logger.debug("💾 Flushed journal buffer (entries already written)")
    
    def flush(self) -> None:
        """Force write any buffered data to disk."""
        self.writer.flush()
    
    def close(self) -> None:
        """Close journal writer safely."""
        self.writer.close()
        logger.info("💾 Closed introspective journal")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
