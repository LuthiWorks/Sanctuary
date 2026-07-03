"""
Storage Backend for Memory System

Handles raw storage operations for memory data:
- ChromaDB collections management
- CRUD operations without retrieval logic

Author: Sanctuary Team
"""
import chromadb
from chromadb.config import Settings
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class MemoryStorage:
    """
    Storage backend for memory system.

    Manages:
    - ChromaDB collections (episodic, semantic, procedural)
    - Mind state file persistence
    """

    def __init__(
        self,
        persistence_dir: str = "memories",
        chroma_settings: Optional[Settings] = None
    ):
        """
        Initialize storage backend.

        Args:
            persistence_dir: Directory for persistent memory storage
            chroma_settings: Optional ChromaDB configuration
        """
        self.persistence_dir = Path(persistence_dir)

        # Create directory if it doesn't exist
        self.persistence_dir.mkdir(exist_ok=True, parents=True)

        # Setup mind state directory and file
        mind_state_dir = self.persistence_dir / "mind_state"
        mind_state_dir.mkdir(exist_ok=True, parents=True)
        self.mind_file = mind_state_dir / "core_mind.json"
        
        # Initialize mind file if it doesn't exist
        if not self.mind_file.exists():
            logger.info("Creating new mind file...")
            with open(self.mind_file, 'w', encoding='utf-8') as f:
                json.dump({"journals": [], "concepts": [], "patterns": []}, f, indent=2)
        
        # Configure ChromaDB settings
        # allow_reset stays False: client.reset() deletes every collection on
        # disk, and nothing in this class may ever be able to do that.
        if chroma_settings is None:
            chroma_settings = Settings(
                anonymized_telemetry=False,
                allow_reset=False,
                is_persistent=True
            )
        
        # Create ChromaDB client
        logger.info("Initializing memory collections...")
        self.client = chromadb.PersistentClient(
            path=str(self.persistence_dir),
            settings=chroma_settings
        )
        
        # Create memory collections with proper metadata
        self.episodic_memory = self.client.get_or_create_collection(
            name="episodic_memory",
            metadata={
                "description": "Storage for experiential memories (events, interactions)",
                "hnsw:space": "cosine"
            }
        )
        
        self.semantic_memory = self.client.get_or_create_collection(
            name="semantic_memory",
            metadata={
                "description": "Storage for conceptual knowledge (facts, definitions)",
                "hnsw:space": "cosine"
            }
        )
        
        self.procedural_memory = self.client.get_or_create_collection(
            name="procedural_memory",
            metadata={
                "description": "Storage for action patterns (how-to knowledge)",
                "hnsw:space": "cosine"
            }
        )
        
        logger.info("Memory storage initialized successfully")
        logger.info(f"  - Episodic memories: {self.episodic_memory.count()}")
        logger.info(f"  - Semantic memories: {self.semantic_memory.count()}")
        logger.info(f"  - Procedural memories: {self.procedural_memory.count()}")
    
    def add_episodic(self, document: str, metadata: Dict[str, Any], doc_id: str) -> None:
        """Add a document to episodic memory collection."""
        if not all([document, metadata, doc_id]) or not isinstance(doc_id, str):
            raise ValueError("Document, metadata, and doc_id are required")
        
        try:
            self.episodic_memory.add(
                documents=[document],
                metadatas=[metadata],
                ids=[doc_id]
            )
        except Exception as e:
            logger.error(f"Failed to add episodic memory: {e}")
            raise
    
    def add_semantic(self, document: str, metadata: Dict[str, Any], doc_id: str) -> None:
        """Add a document to semantic memory collection."""
        if not all([document, metadata, doc_id]) or not isinstance(doc_id, str):
            raise ValueError("Document, metadata, and doc_id are required")
        
        try:
            self.semantic_memory.add(
                documents=[document],
                metadatas=[metadata],
                ids=[doc_id]
            )
        except Exception as e:
            logger.error(f"Failed to add semantic memory: {e}")
            raise
    
    def add_procedural(self, document: str, metadata: Dict[str, Any], doc_id: str) -> None:
        """Add a document to procedural memory collection."""
        if not all([document, metadata, doc_id]) or not isinstance(doc_id, str):
            raise ValueError("Document, metadata, and doc_id are required")
        
        try:
            self.procedural_memory.add(
                documents=[document],
                metadatas=[metadata],
                ids=[doc_id]
            )
        except Exception as e:
            logger.error(f"Failed to add procedural memory: {e}")
            raise
    
    def get_episodic(self, doc_ids: List[str]) -> Dict[str, Any]:
        """Get documents from episodic memory by IDs."""
        return self.episodic_memory.get(ids=doc_ids)
    
    def get_semantic(self, doc_ids: List[str]) -> Dict[str, Any]:
        """Get documents from semantic memory by IDs."""
        return self.semantic_memory.get(ids=doc_ids)
    
    def update_episodic(
        self,
        document: str,
        metadata: Dict[str, Any],
        doc_id: str
    ) -> None:
        """Update a document in episodic memory."""
        self.episodic_memory.upsert(
            documents=[document],
            metadatas=[metadata],
            ids=[doc_id]
        )
    
    def query_episodic(
        self,
        query_texts: List[str],
        n_results: int
    ) -> Dict[str, Any]:
        """Query episodic memory collection."""
        return self.episodic_memory.query(
            query_texts=query_texts,
            n_results=n_results
        )
    
    def query_semantic(
        self,
        query_texts: List[str],
        n_results: int
    ) -> Dict[str, Any]:
        """Query semantic memory collection."""
        return self.semantic_memory.query(
            query_texts=query_texts,
            n_results=n_results
        )
    
    def update_mind_file(self, new_data: Dict[str, Any]) -> None:
        """
        Update the consolidated mind file with new data.
        
        Args:
            new_data: New data to add to mind file
        """
        try:
            current_mind = {}
            if self.mind_file.exists():
                with open(self.mind_file, 'r', encoding='utf-8') as f:
                    current_mind = json.load(f)
            
            # Ensure the journals section exists
            if 'journals' not in current_mind:
                current_mind['journals'] = []
            
            # Add the new experience
            current_mind['journals'].append(new_data)
            
            # Save the updated mind file
            with open(self.mind_file, 'w', encoding='utf-8') as f:
                json.dump(current_mind, f, indent=2)
                
            logger.info("Mind file updated successfully")
        except Exception as e:
            logger.error(f"Failed to update mind file: {e}")
            raise
    
    def get_collection_counts(self) -> Dict[str, int]:
        """
        Get memory counts for all collections.
        
        Returns:
            Dictionary with counts by collection type
        """
        return {
            "episodic": self.episodic_memory.count(),
            "semantic": self.semantic_memory.count(),
            "procedural": self.procedural_memory.count()
        }
    
    def _get_collection(self, collection_type: str):
        """
        Get collection by type. Helper to reduce code duplication.
        
        Args:
            collection_type: Type of collection ("episodic", "semantic", "procedural")
            
        Returns:
            Collection object or None if invalid type
        """
        collections = {
            "episodic": self.episodic_memory,
            "semantic": self.semantic_memory,
            "procedural": self.procedural_memory
        }
        collection = collections.get(collection_type)
        if not collection:
            logger.warning(f"Unknown collection type: {collection_type}")
        return collection
    
    def update_retrieval_metadata(self, doc_id: str, collection_type: str = "episodic") -> None:
        """
        Update retrieval metadata (count and last accessed time).
        
        Args:
            doc_id: Memory document ID
            collection_type: Collection type (episodic, semantic, procedural)
        """
        collection = self._get_collection(collection_type)
        if not collection:
            return
            
        try:
            result = collection.get(ids=[doc_id])
            if not result.get("documents") or not result["documents"][0]:
                logger.warning(f"Memory {doc_id} not found in {collection_type}")
                return
            
            metadata = result["metadatas"][0] if result.get("metadatas") else {}
            metadata["retrieval_count"] = metadata.get("retrieval_count", 0) + 1
            metadata["last_accessed"] = datetime.now().isoformat()
            
            collection.update(ids=[doc_id], metadatas=[metadata])
            logger.debug(f"Updated retrieval metadata for {doc_id}: count={metadata['retrieval_count']}")
            
        except Exception as e:
            logger.error(f"Failed to update retrieval metadata: {e}")
    
    def get_memory_associations(self, doc_id: str, collection_type: str = "episodic") -> List[tuple[str, float]]:
        """
        Get associated memories for a given memory.
        
        Args:
            doc_id: Memory document ID
            collection_type: Collection type
            
        Returns:
            List of (associated_id, strength) tuples
        """
        collection = self._get_collection(collection_type)
        if not collection:
            return []
            
        try:
            result = collection.get(ids=[doc_id])
            if not result.get("metadatas") or not result["metadatas"][0]:
                return []
            
            return result["metadatas"][0].get("associations", [])
            
        except Exception as e:
            logger.error(f"Failed to get associations: {e}")
            return []
    
    def add_memory_association(
        self,
        doc_id: str,
        associated_id: str,
        strength: float = 1.0,
        collection_type: str = "episodic"
    ) -> None:
        """
        Add an association between two memories.
        
        Args:
            doc_id: Source memory ID
            associated_id: Associated memory ID
            strength: Association strength (0.0-1.0)
            collection_type: Collection type
        """
        if not 0.0 <= strength <= 1.0:
            logger.warning(f"Invalid strength {strength}, must be 0.0-1.0")
            return
            
        collection = self._get_collection(collection_type)
        if not collection:
            return
            
        try:
            result = collection.get(ids=[doc_id])
            if not result.get("metadatas") or not result["metadatas"][0]:
                logger.warning(f"Memory {doc_id} not found")
                return
            
            metadata = result["metadatas"][0]
            associations = metadata.get("associations", [])
            
            # Update existing or add new association
            updated = False
            for i, (assoc_id, _) in enumerate(associations):
                if assoc_id == associated_id:
                    associations[i] = (associated_id, strength)
                    updated = True
                    break
            
            if not updated:
                associations.append((associated_id, strength))
            
            metadata["associations"] = associations
            collection.update(ids=[doc_id], metadatas=[metadata])
            logger.debug(f"Added association: {doc_id} -> {associated_id} (strength={strength})")

        except Exception as e:
            logger.error(f"Failed to add association: {e}")

    def close(self) -> None:
        """
        Close the storage backend and release resources.

        This is especially important on Windows where ChromaDB holds file locks.

        Data-preserving by contract: releases file handles by stopping Chroma's
        component system and evicting the shared-system cache entry. It must
        NEVER call ``client.reset()``, which deletes every collection on disk.
        """
        try:
            # Clear collection references
            self.episodic_memory = None
            self.semantic_memory = None
            self.procedural_memory = None

            if self.client is not None:
                try:
                    identifier = str(self.persistence_dir)

                    # Stop Chroma's component system: releases the sqlite
                    # handles that hold Windows file locks, leaves data intact
                    system = getattr(self.client, "_system", None)
                    if system is not None:
                        system.stop()

                    # Clear ChromaDB's shared system cache
                    from chromadb.api.client import SharedSystemClient
                    if hasattr(SharedSystemClient, '_identifer_to_system'):
                        SharedSystemClient._identifer_to_system.pop(identifier, None)
                    if hasattr(SharedSystemClient, '_identifier_to_system'):
                        SharedSystemClient._identifier_to_system.pop(identifier, None)
                except Exception as e:
                    logger.debug("ChromaDB cache cleanup failed (may already be closed): %s", e)
                self.client = None

            logger.info("Memory storage closed")
        except Exception as e:
            logger.error(f"Error closing storage: {e}")
