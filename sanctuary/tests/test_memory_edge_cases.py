"""
Comprehensive edge case tests for memory modules.

Tests unusual inputs, error conditions, and boundary cases to ensure robustness.
"""
import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock

from mind.memory.emotional_weighting import EmotionalWeighting
from mind.memory.working import WorkingMemory


class TestEmotionalWeightingEdgeCases:
    """Test edge cases for emotional weighting."""
    
    def setup_method(self):
        self.weighting = EmotionalWeighting()
    
    def test_empty_memory(self):
        """Test with empty memory dict."""
        assert self.weighting.calculate_salience({}) == 0.5
    
    def test_none_memory(self):
        """Test with None instead of dict."""
        assert self.weighting.calculate_salience(None) == 0.5
    
    def test_missing_emotional_tone(self):
        """Test memory without emotional_tone field."""
        memory = {"description": "test"}
        assert self.weighting.calculate_salience(memory) == 0.5
    
    def test_empty_emotional_tone_list(self):
        """Test with empty emotional tone list."""
        memory = {"emotional_tone": []}
        assert self.weighting.calculate_salience(memory) == 0.5
    
    def test_invalid_emotional_tone_types(self):
        """Test with non-string emotional tones."""
        memory = {"emotional_tone": [None, 123, {}, [], ""]}
        assert self.weighting.calculate_salience(memory) == 0.5
    
    def test_mixed_valid_invalid_tones(self):
        """Test with mix of valid and invalid tones."""
        memory = {"emotional_tone": ["joy", None, "fear", 123, ""]}
        salience = self.weighting.calculate_salience(memory)
        # Should only consider "joy" (0.8) and "fear" (1.0) = avg 0.9
        assert 0.85 <= salience <= 0.95
    
    def test_unknown_emotional_tones(self):
        """Test with unrecognized emotional tones."""
        memory = {"emotional_tone": ["nonexistent_emotion", "fake_tone"]}
        assert self.weighting.calculate_salience(memory) == 0.5  # Default weight
    
    def test_case_insensitive_tones(self):
        """Test that emotional tones are case-insensitive."""
        memory1 = {"emotional_tone": ["JOY", "FEAR"]}
        memory2 = {"emotional_tone": ["joy", "fear"]}
        assert self.weighting.calculate_salience(memory1) == self.weighting.calculate_salience(memory2)
    
    def test_prioritize_with_invalid_threshold(self):
        """Test prioritization with invalid threshold values."""
        memory = {"emotional_tone": ["joy"]}
        assert self.weighting.should_prioritize_storage(memory, -0.5) == False
        assert self.weighting.should_prioritize_storage(memory, 1.5) == False
        assert self.weighting.should_prioritize_storage(memory, "invalid") == False
    
    def test_prioritize_with_none_memory(self):
        """Test prioritization with None memory."""
        assert self.weighting.should_prioritize_storage(None) == False
    
    def test_weight_retrieval_empty_memories(self):
        """Test weighting with empty memories list."""
        result = self.weighting.weight_retrieval_results([], ["joy"])
        assert result == []
    
    def test_weight_retrieval_none_state(self):
        """Test weighting with None emotional state."""
        memories = [{"emotional_tone": ["joy"]}]
        result = self.weighting.weight_retrieval_results(memories, None)
        assert result == memories
    
    def test_weight_retrieval_empty_state(self):
        """Test weighting with empty emotional state."""
        memories = [{"emotional_tone": ["joy"]}]
        result = self.weighting.weight_retrieval_results(memories, [])
        assert result == memories
    
    def test_weight_retrieval_invalid_state(self):
        """Test weighting with invalid emotional state."""
        memories = [{"emotional_tone": ["joy"]}]
        result = self.weighting.weight_retrieval_results(memories, [None, 123, ""])
        assert result == memories
    
    def test_weight_retrieval_invalid_memory_structure(self):
        """Test weighting with invalid memory structures."""
        memories = [None, "string", 123, {"emotional_tone": ["joy"]}]
        result = self.weighting.weight_retrieval_results(memories, ["joy"])
        # Should handle invalid memories gracefully
        assert isinstance(result, list)


class TestWorkingMemoryEdgeCases:
    """Test edge cases for working memory."""
    
    def setup_method(self):
        self.working = WorkingMemory()
    
    def test_update_with_empty_key(self):
        """Test update with empty string key."""
        self.working.update("", "value")
        assert self.working.get("") is None
    
    def test_update_with_none_key(self):
        """Test update with None key."""
        self.working.update(None, "value")
        assert self.working.get(None) is None
    
    def test_update_with_invalid_key_type(self):
        """Test update with non-string key."""
        self.working.update(123, "value")
        assert self.working.size() == 0
    
    def test_update_with_negative_ttl(self):
        """Test update with negative TTL."""
        self.working.update("key", "value", ttl_seconds=-100)
        # Should ignore invalid TTL
        value = self.working.get("key")
        assert value == "value"
    
    def test_update_with_invalid_ttl_type(self):
        """Test update with non-integer TTL."""
        self.working.update("key", "value", ttl_seconds="invalid")
        value = self.working.get("key")
        assert value == "value"
    
    def test_get_with_empty_key(self):
        """Test get with empty key."""
        assert self.working.get("") is None
    
    def test_get_with_none_key(self):
        """Test get with None key."""
        assert self.working.get(None) is None
    
    def test_get_with_invalid_key_type(self):
        """Test get with non-string key."""
        assert self.working.get(123) is None
    
    def test_get_context_with_zero_items(self):
        """Test get_context with zero max_items."""
        self.working.update("key", "value")
        assert self.working.get_context(0) == []
    
    def test_get_context_with_negative_items(self):
        """Test get_context with negative max_items."""
        self.working.update("key", "value")
        assert self.working.get_context(-5) == []
    
    def test_get_context_with_corrupted_entries(self):
        """Test get_context with corrupted memory entries."""
        # Manually corrupt memory
        self.working.memory["valid"] = {"value": "test", "created_at": datetime.now().isoformat()}
        self.working.memory["invalid1"] = {"value": "test"}  # Missing created_at
        self.working.memory["invalid2"] = {"created_at": datetime.now().isoformat()}  # Missing value
        self.working.memory["invalid3"] = "not_a_dict"
        
        context = self.working.get_context(10)
        # Should only return valid entry
        assert len(context) == 1
        assert context[0]["key"] == "valid"
    
    def test_multiple_updates_same_key(self):
        """Test updating the same key multiple times."""
        self.working.update("key", "value1")
        self.working.update("key", "value2")
        self.working.update("key", "value3")
        assert self.working.get("key") == "value3"
        assert self.working.size() == 1
    
    def test_clear_on_empty_memory(self):
        """Test clear on empty memory."""
        self.working.clear()  # Should not raise error
        assert self.working.size() == 0


class TestMemoryRetrieverEdgeCases:
    """Test edge cases for memory retrieval (mocked)."""
    
    def test_retrieve_with_empty_query(self):
        """Test retrieval with empty query."""
        from mind.memory.retrieval import MemoryRetriever
        
        storage = Mock()
        vector_db = Mock()
        retriever = MemoryRetriever(storage, vector_db)
        
        result = retriever.retrieve_memories("")
        assert result == []
    
    def test_retrieve_with_none_query(self):
        """Test retrieval with None query."""
        from mind.memory.retrieval import MemoryRetriever
        
        storage = Mock()
        vector_db = Mock()
        retriever = MemoryRetriever(storage, vector_db)
        
        result = retriever.retrieve_memories(None)
        assert result == []
    
    def test_retrieve_with_whitespace_query(self):
        """Test retrieval with whitespace-only query."""
        from mind.memory.retrieval import MemoryRetriever
        
        storage = Mock()
        vector_db = Mock()
        retriever = MemoryRetriever(storage, vector_db)
        
        result = retriever.retrieve_memories("   ")
        assert result == []
    
    def test_retrieve_with_zero_k(self):
        """Test retrieval with k=0."""
        from mind.memory.retrieval import MemoryRetriever
        
        storage = Mock()
        vector_db = Mock()
        retriever = MemoryRetriever(storage, vector_db)
        
        # Should raise on invalid k
        with pytest.raises(ValueError, match="k must be positive"):
            retriever.retrieve_memories("test", k=0)

    def test_retrieve_with_negative_k(self):
        """Test retrieval with negative k raises ValueError."""
        from mind.memory.retrieval import MemoryRetriever

        storage = Mock()
        vector_db = Mock()
        retriever = MemoryRetriever(storage, vector_db)

        with pytest.raises(ValueError, match="k must be positive"):
            retriever.retrieve_memories("test", k=-5)


if __name__ == "__main__":
    print("Running edge case tests...")
    
    # Run emotional weighting tests
    print("\n=== Emotional Weighting Edge Cases ===")
    test_ew = TestEmotionalWeightingEdgeCases()
    for method_name in dir(test_ew):
        if method_name.startswith("test_"):
            test_ew.setup_method()
            try:
                method = getattr(test_ew, method_name)
                method()
                print(f"✓ {method_name}")
            except AssertionError as e:
                print(f"✗ {method_name}: {e}")
            except Exception as e:
                print(f"✗ {method_name}: Unexpected error: {e}")
    
    # Run working memory tests
    print("\n=== Working Memory Edge Cases ===")
    test_wm = TestWorkingMemoryEdgeCases()
    for method_name in dir(test_wm):
        if method_name.startswith("test_"):
            test_wm.setup_method()
            try:
                method = getattr(test_wm, method_name)
                method()
                print(f"✓ {method_name}")
            except AssertionError as e:
                print(f"✗ {method_name}: {e}")
            except Exception as e:
                print(f"✗ {method_name}: Unexpected error: {e}")
    
    # Run retriever tests
    print("\n=== Memory Retriever Edge Cases ===")
    test_mr = TestMemoryRetrieverEdgeCases()
    for method_name in dir(test_mr):
        if method_name.startswith("test_"):
            try:
                method = getattr(test_mr, method_name)
                method()
                print(f"✓ {method_name}")
            except AssertionError as e:
                print(f"✗ {method_name}: {e}")
            except Exception as e:
                print(f"✗ {method_name}: Unexpected error: {e}")
    
    print("\n✓ All edge case tests completed")
