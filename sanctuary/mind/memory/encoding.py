"""
Memory Encoding Module

Transforms raw experiences into storable memory representations.
Handles embedding generation and information structuring.

Author: Sanctuary Team
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class MemoryEncoder:
    """
    Encodes raw experiences into memory representations.
    
    Responsibilities:
    - Transform experiences into storable format
    - Generate embeddings for similarity matching
    - Structure information for storage
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize memory encoder.
        
        Args:
            data_dir: Optional data directory for loading static data
        """
        self.data_dir = data_dir
    
    def encode_experience(
        self,
        experience: Dict[str, Any],
    ) -> tuple[str, Dict[str, Any], str]:
        """
        Encode an experience for storage in episodic memory.

        Args:
            experience: Raw experience data

        Returns:
            Tuple of (document, metadata, doc_id)
        """
        timestamp = experience.get("timestamp", datetime.now().isoformat())

        experience_data = {
            **experience,
            "timestamp": timestamp,
            "type": "experience",
            "memory_type": "episodic"
        }

        document = json.dumps(experience_data)
        metadata = {
            "timestamp": timestamp,
            "type": "experience"
        }
        doc_id = f"exp_{timestamp}"

        return document, metadata, doc_id
    
    def encode_journal_entry(
        self,
        entry_data: Dict[str, Any],
        date: str,
        source_file: str
    ) -> tuple[str, Dict[str, Any], str]:
        """
        Encode a journal entry for episodic memory.
        
        Args:
            entry_data: Journal entry data
            date: Entry date (e.g., "2025-07-17")
            source_file: Source file name
            
        Returns:
            Tuple of (document, metadata, doc_id)
        """
        memory = {
            "type": "journal_entry",
            "date": date,
            "timestamp": entry_data.get("timestamp"),
            "description": entry_data.get("description", ""),
            "sanctuary_reflection": entry_data.get("sanctuary_reflection", ""),
            "emotional_tone": entry_data.get("emotional_tone", []),
            "tags": entry_data.get("tags", []),
            "key_insights": entry_data.get("key_insights", []),
            "source_file": source_file
        }
        
        document = json.dumps(memory)
        metadata = {
            "type": "journal_entry",
            "date": memory["date"],
            "timestamp": memory["timestamp"],
            "source": "journal_file"
        }
        doc_id = f"journal_{memory['date']}_{entry_data.get('timestamp')}"
        
        return document, metadata, doc_id
    
    def encode_protocol(
        self,
        protocol_data: Dict[str, Any],
        name: str,
        filename: str
    ) -> tuple[str, Dict[str, Any], str]:
        """
        Encode a protocol for semantic memory.
        
        Args:
            protocol_data: Protocol data
            name: Protocol name
            filename: Source filename
            
        Returns:
            Tuple of (document, metadata, doc_id)
        """
        memory = {
            "type": "protocol",
            "name": name,
            "filename": filename,
            "content": protocol_data,
            "description": protocol_data.get("description", ""),
            "purpose": protocol_data.get("purpose", ""),
            "full_text": json.dumps(protocol_data, indent=2)
        }
        
        document = memory["full_text"]
        metadata = {
            "type": "protocol",
            "name": memory["name"],
            "source": "protocol_file"
        }
        doc_id = f"protocol_{memory['name']}"
        
        return document, metadata, doc_id
    
    def encode_symbolic_term(
        self,
        term: str,
        definition: str
    ) -> tuple[str, Dict[str, Any], str]:
        """
        Encode a symbolic lexicon term for semantic memory.
        
        Args:
            term: Symbolic term
            definition: Term definition
            
        Returns:
            Tuple of (document, metadata, doc_id)
        """
        memory = {
            "type": "symbolic_term",
            "term": term,
            "definition": definition,
            "source": "symbolic_lexicon"
        }
        
        document = json.dumps(memory)
        metadata = {
            "type": "symbolic_term",
            "term": term,
            "source": "lexicon_file"
        }
        doc_id = f"symbol_{term}"
        
        return document, metadata, doc_id
    
    def encode_emotional_tone(
        self,
        tone: str,
        definition: str
    ) -> tuple[str, Dict[str, Any], str]:
        """
        Encode an emotional tone definition for semantic memory.
        
        Args:
            tone: Emotional tone name
            definition: Tone definition
            
        Returns:
            Tuple of (document, metadata, doc_id)
        """
        memory = {
            "type": "emotional_tone",
            "tone": tone,
            "definition": definition,
            "source": "emotional_tone_definitions"
        }
        
        document = json.dumps(memory)
        metadata = {
            "type": "emotional_tone",
            "tone": tone,
            "source": "lexicon_file"
        }
        doc_id = f"tone_{tone}"
        
        return document, metadata, doc_id
    
    def encode_lexicon_entry(
        self,
        lexicon_data: Dict[str, Any],
        filename: str
    ) -> tuple[str, Dict[str, Any], str]:
        """
        Encode a generic lexicon entry for semantic memory.
        
        Args:
            lexicon_data: Lexicon data
            filename: Source filename
            
        Returns:
            Tuple of (document, metadata, doc_id)
        """
        memory = {
            "type": "lexicon",
            "filename": filename,
            "content": lexicon_data,
            "full_text": json.dumps(lexicon_data, indent=2)
        }
        
        document = memory["full_text"]
        metadata = {
            "type": "lexicon",
            "filename": filename,
            "source": "lexicon_file"
        }
        doc_id = f"lexicon_{Path(filename).stem}"
        
        return document, metadata, doc_id
    
    def encode_concept(
        self,
        concept: Dict[str, Any]
    ) -> tuple[str, Dict[str, Any], str]:
        """
        Encode a semantic concept.
        
        Args:
            concept: Concept data
            
        Returns:
            Tuple of (document, metadata, doc_id)
        """
        timestamp = datetime.now().isoformat()
        
        document = json.dumps(concept)
        metadata = {
            "timestamp": timestamp,
            "type": "concept"
        }
        doc_id = f"concept_{timestamp}"
        
        return document, metadata, doc_id
