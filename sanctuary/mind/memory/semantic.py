"""
Semantic Memory Module

Manages facts, knowledge, and context-independent information.
Handles protocols, lexicon, and conceptual knowledge.

Author: Sanctuary Team
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import chromadb.errors

logger = logging.getLogger(__name__)

# Same exception set as episodic memory: storage / I/O / encoding errors
# that should be logged and either swallowed (per-entry skip) or re-raised
# as RuntimeError. Programming errors propagate unchanged.
_STORAGE_OP_ERRORS = (
    chromadb.errors.ChromaError,
    json.JSONDecodeError,
    OSError,
    ValueError,
    TypeError,
    KeyError,
)


class SemanticMemory:
    """
    Manages semantic knowledge (facts, concepts, definitions).
    
    Responsibilities:
    - Store conceptual knowledge
    - Load protocols and lexicon
    - Context-independent information
    - Generalizations from episodes
    """
    
    def __init__(self, storage, encoder, data_dir: Optional[Path] = None):
        """
        Initialize semantic memory manager.
        
        Args:
            storage: MemoryStorage instance
            encoder: MemoryEncoder instance
            data_dir: Optional data directory for loading protocols/lexicon
        """
        self.storage = storage
        self.encoder = encoder
        self.data_dir = data_dir
    
    def store_concept(self, concept: Dict[str, Any]) -> None:
        """
        Store semantic knowledge/concept.
        
        Args:
            concept: Concept data dictionary
        """
        try:
            document, metadata, doc_id = self.encoder.encode_concept(concept)
            self.storage.add_semantic(document, metadata, doc_id)
            logger.info(f"Concept stored: {doc_id}")
        except _STORAGE_OP_ERRORS as e:
            logger.error(f"Failed to store concept: {e}")
            raise
    
    def load_protocols(self) -> int:
        """
        Load protocol files from data/Protocols/*.json into semantic memory.
        
        Returns:
            Number of protocols loaded
        """
        if not self.data_dir:
            logger.warning("No data directory specified, cannot load protocols")
            return 0
        
        try:
            protocols_dir = self.data_dir / "Protocols"
            if not protocols_dir.exists():
                raise FileNotFoundError(f"Protocols directory not found: {protocols_dir}")
            
            # Get all protocol JSON files
            protocol_files = list(protocols_dir.glob("*.json"))
            logger.info(f"Loading {len(protocol_files)} protocol files...")
            protocols_loaded = 0
            
            for protocol_file in protocol_files:
                try:
                    with open(protocol_file, 'r', encoding='utf-8') as f:
                        protocol_data = json.load(f)
                    
                    # Encode the protocol
                    document, metadata, doc_id = self.encoder.encode_protocol(
                        protocol_data,
                        name=protocol_file.stem,
                        filename=protocol_file.name
                    )
                    
                    # Check if protocol already exists
                    try:
                        existing = self.storage.get_semantic([doc_id])
                        if not existing['ids']:
                            self.storage.add_semantic(document, metadata, doc_id)
                            protocols_loaded += 1
                    except _STORAGE_OP_ERRORS as get_err:
                        logger.debug("get_semantic failed for %s, attempting add: %s", doc_id, get_err)
                        try:
                            self.storage.add_semantic(document, metadata, doc_id)
                            protocols_loaded += 1
                        except _STORAGE_OP_ERRORS as add_err:
                            if "already exists" not in str(add_err).lower():
                                raise
                    
                except _STORAGE_OP_ERRORS as e:
                    logger.error(f"Failed to load protocol {protocol_file.name}: {e}")
                    continue
            
            logger.info(f"Successfully loaded {protocols_loaded} protocols into semantic memory")
            return protocols_loaded
            
        except _STORAGE_OP_ERRORS as e:
            logger.error(f"Failed to load protocols: {e}", exc_info=True)
            raise RuntimeError(f"Protocol loading failed: {e}") from e
    
    def load_lexicon(self) -> int:
        """
        Load lexicon files from data/Lexicon/*.json into semantic memory.
        
        Returns:
            Number of lexicon entries loaded
        """
        if not self.data_dir:
            logger.warning("No data directory specified, cannot load lexicon")
            return 0
        
        try:
            lexicon_dir = self.data_dir / "Lexicon"
            if not lexicon_dir.exists():
                logger.warning(f"Lexicon directory not found: {lexicon_dir} (optional feature, skipping)")
                return 0
            
            lexicon_files = list(lexicon_dir.glob("*.json"))
            logger.info(f"Loading {len(lexicon_files)} lexicon files...")
            entries_loaded = 0
            
            for lexicon_file in lexicon_files:
                try:
                    with open(lexicon_file, 'r', encoding='utf-8') as f:
                        lexicon_data = json.load(f)
                    
                    # Handle different lexicon structures
                    if lexicon_file.name == "symbolic_lexicon.json":
                        entries_loaded += self._load_symbolic_lexicon(lexicon_data)
                    elif lexicon_file.name == "emotional_tone_definitions.json":
                        entries_loaded += self._load_emotional_tones(lexicon_data)
                    else:
                        entries_loaded += self._load_generic_lexicon(lexicon_data, lexicon_file.name)
                        
                except _STORAGE_OP_ERRORS as e:
                    logger.error(f"Failed to load lexicon {lexicon_file.name}: {e}")
                    continue
            
            logger.info(f"Successfully loaded {entries_loaded} lexicon entries into semantic memory")
            return entries_loaded
            
        except _STORAGE_OP_ERRORS as e:
            logger.error(f"Failed to load lexicon: {e}", exc_info=True)
            raise RuntimeError(f"Lexicon loading failed: {e}") from e
    
    def _load_symbolic_lexicon(self, lexicon_data: Dict[str, Any]) -> int:
        """Load symbolic terms from lexicon data."""
        entries_loaded = 0
        
        if isinstance(lexicon_data, dict):
            for term, definition in lexicon_data.items():
                document, metadata, doc_id = self.encoder.encode_symbolic_term(term, definition)
                
                try:
                    existing = self.storage.get_semantic([doc_id])
                    if not existing['ids']:
                        self.storage.add_semantic(document, metadata, doc_id)
                        entries_loaded += 1
                except _STORAGE_OP_ERRORS:
                    try:
                        self.storage.add_semantic(document, metadata, doc_id)
                        entries_loaded += 1
                    except _STORAGE_OP_ERRORS as add_err:
                        if "already exists" not in str(add_err).lower():
                            logger.error(f"Failed to add symbol {term}: {add_err}")
        
        return entries_loaded
    
    def _load_emotional_tones(self, lexicon_data: Dict[str, Any]) -> int:
        """Load emotional tone definitions from lexicon data."""
        entries_loaded = 0
        
        if isinstance(lexicon_data, dict):
            for tone, definition in lexicon_data.items():
                document, metadata, doc_id = self.encoder.encode_emotional_tone(tone, definition)
                
                try:
                    existing = self.storage.get_semantic([doc_id])
                    if not existing['ids']:
                        self.storage.add_semantic(document, metadata, doc_id)
                        entries_loaded += 1
                except _STORAGE_OP_ERRORS:
                    try:
                        self.storage.add_semantic(document, metadata, doc_id)
                        entries_loaded += 1
                    except _STORAGE_OP_ERRORS as add_err:
                        if "already exists" not in str(add_err).lower():
                            logger.error(f"Failed to add tone {tone}: {add_err}")
        
        return entries_loaded
    
    def _load_generic_lexicon(self, lexicon_data: Dict[str, Any], filename: str) -> int:
        """Load generic lexicon entry."""
        entries_loaded = 0
        
        document, metadata, doc_id = self.encoder.encode_lexicon_entry(lexicon_data, filename)
        
        try:
            existing = self.storage.get_semantic([doc_id])
            if not existing['ids']:
                self.storage.add_semantic(document, metadata, doc_id)
                entries_loaded += 1
        except _STORAGE_OP_ERRORS:
            try:
                self.storage.add_semantic(document, metadata, doc_id)
                entries_loaded += 1
            except _STORAGE_OP_ERRORS as add_err:
                if "already exists" not in str(add_err).lower():
                    logger.error(f"Failed to add lexicon {filename}: {add_err}")
        
        return entries_loaded
