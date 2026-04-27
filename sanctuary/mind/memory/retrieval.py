"""
Memory Retrieval Module

Cue-based memory retrieval with similarity matching.
Supports both RAG-based and direct ChromaDB queries.
Implements cue-dependent retrieval with spreading activation.

Author: Sanctuary Team
"""
import json
import logging
import math
from datetime import datetime
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class MemoryRetriever:
    """
    Handles memory retrieval with similarity matching.
    
    Responsibilities:
    - Cue-based memory retrieval
    - Similarity matching using embeddings
    - Retrieval context management
    - Cue-dependent retrieval (optional)
    """
    
    def __init__(self, storage, vector_db, emotional_weighting=None):
        """
        Initialize memory retriever.
        
        Args:
            storage: MemoryStorage instance
            vector_db: MindVectorDB instance for RAG
            emotional_weighting: Optional EmotionalWeighting instance for cue-dependent retrieval
        """
        self.storage = storage
        self.vector_db = vector_db
        self.emotional_weighting = emotional_weighting
        
        # Initialize cue-dependent retrieval if emotional weighting is available
        self.cue_dependent = None
        if emotional_weighting:
            self.cue_dependent = CueDependentRetrieval(
                storage=storage,
                emotional_weighting=emotional_weighting
            )
    
    def retrieve_memories(
        self,
        query: str,
        k: int = 5,
        use_rag: bool = True
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant memories based on a query."""
        if not query or not isinstance(query, str) or not query.strip():
            logger.warning("Empty or invalid query provided")
            return []
        
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        
        try:
            memories = self._retrieve_with_rag(query, k) if use_rag else self._retrieve_direct(query, k)
            memories.sort(key=self._sort_key, reverse=True)
            
            verified_count = sum(1 for m in memories if m.get('verification', {}).get('status') == 'verified')
            logger.info(f"Retrieved {len(memories[:k])} memories (verified: {verified_count})")
            return memories[:k]
            
        except Exception as e:
            raise RuntimeError(f"Memory retrieval failed: {e}") from e
    
    def retrieve_with_cues(
        self,
        workspace_state: Dict[str, Any],
        limit: int = 5,
        use_spreading: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Retrieve memories using cue-dependent retrieval from workspace state.
        
        This method implements cognitively realistic retrieval where:
        - Current workspace state provides retrieval cues
        - Memories are activated based on cue similarity, recency, and emotional congruence
        - Activation spreads to associated memories
        - Similar memories compete for retrieval
        - Retrieved memories are strengthened
        
        Args:
            workspace_state: Current workspace state (goals, emotions, percepts, memories)
            limit: Maximum number of memories to retrieve
            use_spreading: Whether to use spreading activation
            
        Returns:
            List of retrieved memories with activation scores
        """
        if not self.cue_dependent:
            logger.warning("Cue-dependent retrieval not available (no emotional weighting)")
            # Fall back to simple query-based retrieval
            cue_text = self._extract_cue_text(workspace_state)
            return self.retrieve_memories(cue_text, k=limit, use_rag=False)
        
        return self.cue_dependent.retrieve(
            workspace_state=workspace_state,
            limit=limit,
            use_spreading=use_spreading
        )
    
    def _extract_cue_text(self, workspace_state: Dict[str, Any]) -> str:
        """Extract text from workspace state for fallback retrieval."""
        cues = []
        
        goals = workspace_state.get("goals", [])
        for goal in goals:
            if isinstance(goal, dict):
                cues.append(goal.get("description", ""))
        
        percepts = workspace_state.get("percepts", {})
        if isinstance(percepts, dict):
            for p_data in percepts.values():
                if isinstance(p_data, dict):
                    cues.append(str(p_data.get("raw", "")))
        
        return " ".join(cues) if cues else "current context"
    
    def _retrieve_with_rag(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Retrieve memories using RAG system for deep semantic search."""
        logger.debug(f"RAG retrieval for: {query[:50]}...")
        memories = []
        
        try:
            retriever = self.vector_db.as_retriever({"k": k})
            docs = retriever.get_relevant_documents(query)
        except Exception as e:
            logger.error(f"RAG retrieval failed: {e}")
            return memories
        
        for doc in docs:
            try:
                content = json.loads(doc.page_content) if isinstance(doc.page_content, str) else doc.page_content
                memories.append(content)
            except (json.JSONDecodeError, AttributeError, KeyError) as e:
                logger.warning(f"Failed to parse memory: {e}")
                continue
        
        return memories
    
    def _retrieve_direct(self, query: str, k: int) -> List[Dict[str, Any]]:
        """
        Retrieve memories using direct ChromaDB query.
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of memory dictionaries
        """
        logger.debug(f"Direct ChromaDB retrieval for query: {query[:50]}...")
        memories = []
        
        # Query episodic memory
        episodic_count = self.storage.episodic_memory.count()
        episodic_results = self.storage.query_episodic(
            query_texts=[query],
            n_results=min(k, episodic_count) if episodic_count > 0 else 1
        )
        
        # Query semantic memory
        semantic_count = self.storage.semantic_memory.count()
        semantic_results = self.storage.query_semantic(
            query_texts=[query],
            n_results=min(k, semantic_count) if semantic_count > 0 else 1
        )
        
        # Process episodic memories
        memories.extend(self._process_episodic_results(episodic_results))
        
        # Process semantic memories
        memories.extend(self._process_semantic_results(semantic_results))
        
        return memories
    
    def _process_episodic_results(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process episodic memory query results."""
        memories = []
        
        if results["documents"] and results["documents"][0]:
            for result, metadata in zip(results["documents"][0], results["metadatas"][0]):
                try:
                    memory_data = json.loads(result) if isinstance(result, str) else result
                    memories.append(memory_data)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse episodic memory: {e}")
                    continue

        return memories
    
    def _process_semantic_results(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process semantic memory query results."""
        memories = []
        
        if results["documents"] and results["documents"][0]:
            for result, metadata in zip(results["documents"][0], results["metadatas"][0]):
                try:
                    memory_data = json.loads(result) if isinstance(result, str) else result
                    memory_data["verification"] = {"status": "semantic"}
                    memories.append(memory_data)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse semantic memory: {e}")
                    continue
        
        return memories
    
    def _sort_key(self, memory: Dict[str, Any]) -> tuple:
        """
        Generate sort key for memory based on verification status and timestamp.
        
        Args:
            memory: Memory dictionary
            
        Returns:
            Tuple for sorting (status_priority, timestamp)
        """
        verification = memory.get("verification", {})
        status_priority = {
            "verified": 0,
            "legacy": 1,
            "semantic": 2,
            "verification_failed": 3
        }
        return (
            status_priority.get(verification.get("status", "verification_failed"), 4),
            verification.get("verified_at", memory.get("timestamp", ""))
        )


class CueDependentRetrieval:
    """
    Implements cue-dependent memory retrieval based on cognitive science principles.
    
    Memory retrieval is cue-dependent - what you remember depends on the cues
    present in your current context. This class implements:
    - Retrieval based on workspace state as cues
    - Spreading activation to associated memories
    - Emotional state biasing retrieval
    - Retrieval competition (interference between similar memories)
    - Retrieval strengthening (use it or lose it)
    
    Attributes:
        storage: MemoryStorage instance for accessing memory data
        emotional_weighting: EmotionalWeighting for emotional congruence
        retrieval_threshold: Minimum activation for retrieval (default: 0.3)
        inhibition_strength: How much similar memories inhibit each other (default: 0.4)
        strengthening_factor: How much retrieval boosts activation (default: 0.05)
        spread_factor: How much activation spreads to associates (default: 0.3)
    """
    
    # Activation weight constants for clarity and easy tuning
    SIMILARITY_WEIGHT = 0.5
    RECENCY_WEIGHT = 0.2
    EMOTIONAL_WEIGHT = 0.3
    
    # Default parameters
    DEFAULT_RETRIEVAL_THRESHOLD = 0.3
    DEFAULT_RECENCY_FALLBACK = 0.3
    RECENCY_DECAY_RATE = 0.01  # λ in exponential decay (half-life ~69 hours)
    
    def __init__(
        self,
        storage,
        emotional_weighting,
        retrieval_threshold: float = DEFAULT_RETRIEVAL_THRESHOLD,
        inhibition_strength: float = 0.4,
        strengthening_factor: float = 0.05,
        spread_factor: float = 0.3
    ):
        """
        Initialize cue-dependent retrieval system.
        
        Args:
            storage: MemoryStorage instance
            emotional_weighting: EmotionalWeighting instance
            retrieval_threshold: Minimum activation for retrieval
            inhibition_strength: Inhibition between similar memories
            strengthening_factor: Retrieval strengthening amount
            spread_factor: Spreading activation factor
        """
        self.storage = storage
        self.emotional_weighting = emotional_weighting
        self.retrieval_threshold = retrieval_threshold
        self.inhibition_strength = inhibition_strength
        self.strengthening_factor = strengthening_factor
        self.spread_factor = spread_factor
        
        # Metrics tracking
        self.metrics = {
            "total_retrievals": 0,
            "avg_cue_similarity": 0.0,
            "spreading_activations": 0,
            "interference_events": 0,
            "strengthening_events": 0
        }
    
    def retrieve(
        self,
        workspace_state: Dict[str, Any],
        limit: int = 5,
        use_spreading: bool = True,
        spread_iterations: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Retrieve memories activated by current workspace cues.
        
        Args:
            workspace_state: Current workspace state with goals, emotions, percepts, memories
            limit: Maximum number of memories to retrieve
            use_spreading: Whether to use spreading activation
            spread_iterations: Number of spreading iterations
            
        Returns:
            List of retrieved memory dictionaries with activation scores
        """
        # Input validation
        if not workspace_state or not isinstance(workspace_state, dict):
            logger.warning("Invalid workspace state provided")
            return []
        
        if limit <= 0:
            logger.warning(f"Invalid limit {limit}, using default of 5")
            limit = 5
        
        logger.debug("Cue-dependent retrieval with workspace state")
        
        # Extract cues and get candidates
        cue_embeddings = self._encode_cues(workspace_state)
        candidates = self._get_candidates(cue_embeddings)
        
        if not candidates:
            logger.debug("No candidate memories found")
            return []
        
        # Compute initial activations
        current_emotional_state = workspace_state.get("emotions", {})
        activations = self._compute_activations(candidates, current_emotional_state)
        
        # Spreading activation
        if use_spreading:
            activations = self._spread_activation(
                activations,
                self.spread_factor,
                spread_iterations
            )
        
        # Competitive retrieval with interference
        retrieved = self._competitive_retrieval(activations, candidates, limit)
        
        # Strengthen retrieved memories
        self._strengthen_retrieved(retrieved)
        
        # Update metrics
        self.metrics["total_retrievals"] += 1
        
        logger.info(f"Retrieved {len(retrieved)} memories via cue-dependent retrieval")
        return retrieved
    
    def _compute_activations(
        self,
        candidates: Dict[str, Dict[str, Any]],
        current_emotional_state: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Compute activation scores for candidate memories.
        
        Args:
            candidates: Candidate memories
            current_emotional_state: Current emotional state (PAD values)
            
        Returns:
            Dictionary mapping memory_id to activation score
        """
        activations = {}
        cue_similarities = []
        
        for memory_id, memory_data in candidates.items():
            similarity = memory_data.get("similarity", 0.5)
            cue_similarities.append(similarity)
            
            recency = self._recency_weight(memory_data.get("metadata", {}))
            
            memory_emotional_state = memory_data.get("metadata", {}).get("emotional_state")
            emotional_match = self.emotional_weighting.emotional_congruence_pad(
                current_emotional_state,
                memory_emotional_state
            )
            
            # Combined activation using class constants
            activations[memory_id] = (
                similarity * self.SIMILARITY_WEIGHT +
                recency * self.RECENCY_WEIGHT +
                emotional_match * self.EMOTIONAL_WEIGHT
            )
        
        # Update metrics
        if cue_similarities:
            self.metrics["avg_cue_similarity"] = sum(cue_similarities) / len(cue_similarities)
        
        return activations
    
    def _encode_cues(self, workspace_state: Dict[str, Any]) -> str:
        """
        Extract and encode retrieval cues from workspace state.
        
        Cues include:
        - Current goals (what we're trying to do)
        - Active percepts (what we're perceiving)
        - Emotional state (how we're feeling)
        - Recent memories in workspace (what we're thinking about)
        
        Args:
            workspace_state: Workspace state dictionary
            
        Returns:
            Combined cue text for embedding
        """
        cues = []
        
        # Extract goal descriptions
        goals = workspace_state.get("goals", [])
        for goal in goals:
            if isinstance(goal, dict):
                desc = goal.get("description", "")
            else:
                desc = str(goal)
            if desc:
                cues.append(desc)
        
        # Extract percept content
        percepts = workspace_state.get("percepts", {})
        if isinstance(percepts, dict):
            for percept_id, percept_data in percepts.items():
                if isinstance(percept_data, dict):
                    raw = percept_data.get("raw", "")
                    cues.append(str(raw))
        elif isinstance(percepts, list):
            for percept in percepts:
                if isinstance(percept, dict):
                    raw = percept.get("raw", "")
                    cues.append(str(raw))
        
        # Extract memory content already in workspace
        memories = workspace_state.get("memories", [])
        for memory in memories:
            if isinstance(memory, dict):
                content = memory.get("content", "")
                cues.append(str(content))
        
        # Combine cues
        combined_cues = " ".join(cues) if cues else "current context"
        return combined_cues
    
    def _get_candidates(self, cue_text: str, max_candidates: int = 50) -> Dict[str, Dict[str, Any]]:
        """
        Get candidate memories using cue embedding similarity.
        
        Args:
            cue_text: Text representation of cues
            max_candidates: Maximum number of candidates to retrieve
            
        Returns:
            Dictionary mapping memory_id to memory data with similarity score
        """
        candidates = {}
        
        try:
            # Query episodic memories
            candidates.update(
                self._query_collection_candidates(
                    "episodic",
                    cue_text,
                    max_candidates
                )
            )
            
            # Query semantic memories if space remains
            if len(candidates) < max_candidates:
                remaining = max_candidates - len(candidates)
                candidates.update(
                    self._query_collection_candidates(
                        "semantic",
                        cue_text,
                        remaining
                    )
                )
        
        except Exception as e:
            logger.error(f"Failed to get candidate memories: {e}")
        
        return candidates
    
    def _query_collection_candidates(
        self,
        collection_type: str,
        cue_text: str,
        max_results: int
    ) -> Dict[str, Dict[str, Any]]:
        """
        Query a specific collection for candidate memories.
        
        Args:
            collection_type: "episodic" or "semantic"
            cue_text: Query text
            max_results: Maximum results to return
            
        Returns:
            Dictionary of candidates from this collection
        """
        candidates = {}
        
        # Get collection and count
        if collection_type == "episodic":
            collection = self.storage.episodic_memory
            query_method = self.storage.query_episodic
        else:  # semantic
            collection = self.storage.semantic_memory
            query_method = self.storage.query_semantic
        
        count = collection.count()
        if count == 0:
            return candidates
        
        # Query collection
        results = query_method(
            query_texts=[cue_text],
            n_results=min(max_results, count)
        )
        
        # Process results
        if not results.get("documents") or not results["documents"][0]:
            return candidates
        
        for i, (doc, metadata, distance) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results.get("distances", [[0.0] * len(results["documents"][0])])[0]
        )):
            try:
                memory_data = json.loads(doc) if isinstance(doc, str) else doc
                memory_id = results["ids"][0][i]
                similarity = 1.0 - distance  # Convert distance to similarity
                
                candidates[memory_id] = {
                    "data": memory_data,
                    "metadata": metadata,
                    "similarity": similarity,
                    "collection": collection_type
                }
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to parse candidate memory: {e}")
                continue
        
        return candidates
    
    def _recency_weight(self, metadata: Dict[str, Any]) -> float:
        """
        Calculate recency weight with exponential decay.
        
        More recent memories are easier to retrieve.
        
        Args:
            metadata: Memory metadata containing timestamp or last_accessed
            
        Returns:
            Recency weight (0.0-1.0)
        """
        last_accessed_str = metadata.get("last_accessed") or metadata.get("timestamp")
        
        if not last_accessed_str:
            return self.DEFAULT_RECENCY_FALLBACK
        
        try:
            last_accessed = datetime.fromisoformat(last_accessed_str)
            age_hours = (datetime.now() - last_accessed).total_seconds() / 3600.0
            
            # Exponential decay: weight = e^(-λ * age)
            return math.exp(-self.RECENCY_DECAY_RATE * age_hours)
            
        except (ValueError, TypeError) as e:
            logger.debug(f"Failed to parse timestamp: {e}")
            return self.DEFAULT_RECENCY_FALLBACK
    
    def _spread_activation(
        self,
        initial_activations: Dict[str, float],
        spread_factor: float = 0.3,
        iterations: int = 2
    ) -> Dict[str, float]:
        """
        Activation spreads to associated memories.
        
        Implements spreading activation: highly activated memories
        spread some activation to their associated memories.
        
        Args:
            initial_activations: Initial activation values
            spread_factor: How much activation spreads (0.0-1.0)
            iterations: Number of spreading iterations
            
        Returns:
            Updated activation values
        """
        activations = initial_activations.copy()
        
        for iteration in range(iterations):
            new_activations = activations.copy()
            
            for memory_id, activation in activations.items():
                if activation < self.retrieval_threshold:
                    continue  # Don't spread from weak activations
                
                # Get associations for this memory
                # For now, use metadata-based associations
                # In future, could also use semantic similarity
                associations = self.storage.get_memory_associations(
                    memory_id,
                    collection_type="episodic"
                )
                
                if not associations:
                    continue
                
                # Spread activation to associates
                for assoc_id, strength in associations:
                    spread = activation * strength * spread_factor
                    new_activations[assoc_id] = max(
                        new_activations.get(assoc_id, 0.0),
                        new_activations.get(assoc_id, 0.0) + spread
                    )
                    self.metrics["spreading_activations"] += 1
            
            activations = new_activations
        
        return activations
    
    def _competitive_retrieval(
        self,
        activations: Dict[str, float],
        candidates: Dict[str, Dict[str, Any]],
        limit: int
    ) -> List[Dict[str, Any]]:
        """
        Similar memories compete - similar memories inhibit each other.
        
        Implements retrieval competition where similar memories
        interfere with each other, simulating the competition for
        limited retrieval slots.
        
        Args:
            activations: Activation values for each memory
            candidates: Candidate memory data
            limit: Maximum number of memories to retrieve
            
        Returns:
            List of retrieved memories with activation scores
        """
        retrieved = []
        remaining = dict(activations)
        
        while len(retrieved) < limit and remaining:
            # Get highest activation
            best_id = max(remaining, key=remaining.get)
            best_activation = remaining[best_id]
            
            if best_activation < self.retrieval_threshold:
                break  # No more memories above threshold
            
            # Get memory data
            if best_id not in candidates:
                del remaining[best_id]
                continue
            
            best_memory = candidates[best_id]["data"].copy()
            best_memory["activation"] = best_activation
            best_memory["memory_id"] = best_id
            best_memory["collection"] = candidates[best_id].get("collection", "episodic")
            
            retrieved.append(best_memory)
            del remaining[best_id]
            
            # Inhibit similar memories (interference)
            self._apply_interference(remaining, candidates, best_id)
        
        return retrieved
    
    def _apply_interference(
        self,
        remaining: Dict[str, float],
        candidates: Dict[str, Dict[str, Any]],
        retrieved_id: str
    ) -> None:
        """
        Apply interference from a retrieved memory to remaining candidates.
        
        Args:
            remaining: Remaining activation values (modified in place)
            candidates: Candidate memory data
            retrieved_id: ID of the just-retrieved memory
        """
        retrieved_similarity = candidates[retrieved_id]["similarity"]
        
        for mem_id in list(remaining.keys()):
            if mem_id not in candidates:
                continue
            
            # Estimate similarity between memories based on cue similarity
            # Simple heuristic: if both have high similarity to cues, they're likely similar
            mem_similarity = candidates[mem_id]["similarity"]
            
            if retrieved_similarity > 0.7 and mem_similarity > 0.7:
                similarity = 0.8  # High similarity
            elif retrieved_similarity > 0.5 and mem_similarity > 0.5:
                similarity = 0.5  # Moderate similarity
            else:
                similarity = 0.2  # Low similarity
            
            # Apply inhibition
            inhibition = similarity * self.inhibition_strength
            remaining[mem_id] -= inhibition
            
            if inhibition > 0.1:  # Only count significant interference
                self.metrics["interference_events"] += 1
    
    def _strengthen_retrieved(self, memories: List[Dict[str, Any]]) -> None:
        """
        Retrieved memories get stronger (use it or lose it).
        
        Successfully retrieved memories become easier to retrieve
        in the future by updating their metadata.
        
        Args:
            memories: List of retrieved memories
        """
        for memory in memories:
            try:
                memory_id = memory.get("memory_id")
                collection = memory.get("collection", "episodic")
                
                if not memory_id:
                    continue
                
                # Update retrieval metadata (count and timestamp)
                self.storage.update_retrieval_metadata(memory_id, collection)
                
                self.metrics["strengthening_events"] += 1
                
            except Exception as e:
                logger.warning(f"Failed to strengthen memory: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get retrieval dynamics metrics.
        
        Returns:
            Dictionary of metrics tracking retrieval behavior
        """
        return self.metrics.copy()
