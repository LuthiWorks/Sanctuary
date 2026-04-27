"""
Episodic Memory Module

Manages autobiographical episodes with temporal and contextual indexing.
Handles the "what, when, where" of experiential memories.

Author: Sanctuary Team
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

import chromadb.errors

logger = logging.getLogger(__name__)

# Storage operations against ChromaDB can fail for transport/state reasons;
# JSON encoding/decoding can fail on malformed data; file I/O can fail for
# the usual reasons. These are the legitimate "operation failed" cases that
# should be logged and either swallowed (per-entry skip) or re-raised as
# RuntimeError. Programming errors (AttributeError, etc.) propagate.
_STORAGE_OP_ERRORS = (
    chromadb.errors.ChromaError,
    json.JSONDecodeError,
    OSError,
    ValueError,
    TypeError,
    KeyError,
)


class EpisodicMemory:
    """
    Manages autobiographical memory (events, interactions).
    
    Responsibilities:
    - Store experiential memories
    - Temporal indexing (when did this happen)
    - Context binding (where, who, what)
    - Load journal entries
    """
    
    def __init__(self, storage, encoder, data_dir: Optional[Path] = None):
        """
        Initialize episodic memory manager.
        
        Args:
            storage: MemoryStorage instance
            encoder: MemoryEncoder instance
            data_dir: Optional data directory for loading journal files
        """
        self.storage = storage
        self.encoder = encoder
        self.data_dir = data_dir
    
    def store_experience(self, experience: Dict[str, Any]) -> None:
        """
        Store a new experience in episodic memory.

        Args:
            experience: Experience data dictionary
        """
        try:
            # Encode the experience
            document, metadata, doc_id = self.encoder.encode_experience(experience)

            # Store in episodic memory collection
            self.storage.add_episodic(document, metadata, doc_id)

            # Update mind file
            experience_data = json.loads(document)
            self.storage.update_mind_file(experience_data)

            logger.info("Experience stored successfully")

        except _STORAGE_OP_ERRORS as e:
            logger.error(f"Failed to store experience: {e}", exc_info=True)
            raise RuntimeError(f"Experience storage failed: {e}") from e

    def update_experience(self, experience_data: Dict[str, Any], doc_id: str) -> bool:
        """
        Update an existing experience in episodic memory.

        Args:
            experience_data: Updated experience data
            doc_id: Document ID of the experience to update

        Returns:
            True if successful, False otherwise
        """
        from datetime import datetime

        try:
            document = json.dumps(experience_data)
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "type": "experience",
            }
            self.storage.update_episodic(document, metadata, doc_id)
            logger.info(f"Experience {doc_id} updated")
            return True

        except _STORAGE_OP_ERRORS as e:
            logger.error(f"Failed to update experience: {e}")
            return False
    
    def load_journal_entries(self, limit: Optional[int] = None) -> int:
        """
        Load journal entries from data/journal/*.json into episodic memory.
        
        Args:
            limit: Optional limit on number of journals to load (most recent first)
            
        Returns:
            Number of journal entries loaded
        """
        if not self.data_dir:
            logger.warning("No data directory specified, cannot load journals")
            return 0
        
        try:
            journal_dir = self.data_dir / "journal"
            if not journal_dir.exists():
                raise FileNotFoundError(f"Journal directory not found: {journal_dir}")
            
            # Get all journal files (excluding index and manifest)
            journal_files = sorted(
                [f for f in journal_dir.glob("2025-*.json")],
                reverse=True  # Most recent first
            )
            
            if limit:
                journal_files = journal_files[:limit]
            
            logger.info(f"Loading {len(journal_files)} journal files...")
            entries_loaded = 0
            
            for journal_file in journal_files:
                try:
                    with open(journal_file, 'r', encoding='utf-8') as f:
                        journal_data = json.load(f)
                    
                    # Journal files are arrays of entries
                    if isinstance(journal_data, list):
                        for entry in journal_data:
                            if "journal_entry" in entry:
                                entry_data = entry["journal_entry"]
                                
                                # Encode the journal entry
                                document, metadata, doc_id = self.encoder.encode_journal_entry(
                                    entry_data,
                                    date=journal_file.stem,
                                    source_file=journal_file.name
                                )
                                
                                # Check if already exists
                                try:
                                    existing = self.storage.get_episodic([doc_id])
                                    if not existing['ids']:
                                        self.storage.add_episodic(document, metadata, doc_id)
                                        entries_loaded += 1
                                except _STORAGE_OP_ERRORS:
                                    # If get fails, try to add
                                    try:
                                        self.storage.add_episodic(document, metadata, doc_id)
                                        entries_loaded += 1
                                    except _STORAGE_OP_ERRORS as add_err:
                                        if "already exists" not in str(add_err).lower():
                                            logger.error(f"Failed to add journal entry: {add_err}")

                except _STORAGE_OP_ERRORS as e:
                    logger.error(f"Failed to load journal {journal_file.name}: {e}")
                    continue

            logger.info(f"Successfully loaded {entries_loaded} journal entries into episodic memory")
            return entries_loaded

        except _STORAGE_OP_ERRORS as e:
            logger.error(f"Failed to load journal entries: {e}", exc_info=True)
            raise RuntimeError(f"Journal loading failed: {e}") from e
