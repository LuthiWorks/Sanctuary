"""
Scenario-based integration tests.

Tests specific use cases and interaction patterns.
"""

import pytest

from mind import SanctuaryAPI


@pytest.mark.integration
class TestConversationScenarios:
    """Test realistic conversation scenarios."""
    
    @pytest.mark.asyncio
    async def test_greeting_and_introduction(self):
        """Test greeting and self-introduction scenario."""
        api = SanctuaryAPI(config={"cognitive_core": {"perception": {"mock_mode": True}}})
        await api.start()
        
        try:
            # Greeting
            turn1 = await api.chat("Hi there!")
            assert len(turn1.system_response) > 0
            
            # Introduction
            turn2 = await api.chat("My name is Bob. What's your name?")
            assert len(turn2.system_response) > 0  # Mock the model won't know its name
            
            # Follow-up
            turn3 = await api.chat("Nice to meet you!")
            assert len(turn3.system_response) > 0
            
        finally:
            await api.stop()
    
    @pytest.mark.asyncio
    async def test_emotional_support_scenario(self):
        """Test providing emotional support."""
        api = SanctuaryAPI(config={"cognitive_core": {"perception": {"mock_mode": True}}})
        await api.start()
        
        try:
            # User expresses distress
            turn1 = await api.chat("I'm feeling really stressed about work.")
            assert len(turn1.system_response) > 0
            
            # Follow-up
            turn2 = await api.chat("Everything feels overwhelming.")
            assert len(turn2.system_response) > 0
            
            # Should show empathy/support
            
        finally:
            await api.stop()
    
    @pytest.mark.asyncio
    async def test_introspective_conversation(self):
        """Test conversation about self-awareness."""
        api = SanctuaryAPI(config={"cognitive_core": {"perception": {"mock_mode": True}}})
        await api.start()
        
        try:
            # Ask about self-awareness
            turn1 = await api.chat("Do you have feelings?")
            assert len(turn1.system_response) > 0
            
            # Ask about experience
            turn2 = await api.chat("What's it like to be you?")
            assert len(turn2.system_response) > 0
            
        finally:
            await api.stop()
