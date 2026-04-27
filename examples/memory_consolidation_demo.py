"""
Memory Consolidation Integration Example

Demonstrates memory consolidation with idle detection and scheduling.
"""
import asyncio
from datetime import datetime
from mind.memory import (
    MemoryStorage, MemoryEncoder, MemoryConsolidator,
    EpisodicMemory, IdleDetector, ConsolidationScheduler,
)


async def main():
    """Example integration."""
    
    print("Initializing memory system...")
    storage = MemoryStorage(persistence_dir="memories")
    encoder = MemoryEncoder()
    
    print("Initializing consolidation system...")
    consolidator = MemoryConsolidator(
        storage=storage, encoder=encoder,
        strengthening_factor=0.1, decay_rate=0.95,
        deletion_threshold=0.1, pattern_threshold=3
    )
    
    idle_detector = IdleDetector(idle_threshold_seconds=30.0)
    scheduler = ConsolidationScheduler(consolidator, idle_detector, check_interval=10.0)
    
    print("Starting background consolidation...")
    await scheduler.start()
    
    try:
        episodic = EpisodicMemory(storage, encoder)
        
        print("\nStoring experiences...")
        for i in range(5):
            experience = {
                "description": f"Experience {i}",
                "timestamp": datetime.now().isoformat()
            }
            episodic.store_experience(experience)
            idle_detector.record_activity()
            await asyncio.sleep(0.5)
        
        print("\nSimulating retrievals...")
        for _ in range(3):
            consolidator.record_retrieval("exp_0", session_id="session_1")
            consolidator.record_retrieval("exp_1", session_id="session_1")
            idle_detector.record_activity()
            await asyncio.sleep(0.5)
        
        print("\nSystem going idle... consolidation should run soon...")
        await asyncio.sleep(35)  # Wait for idle + consolidation
        
        # Check consolidation metrics
        print("\nConsolidation Metrics:")
        summary = scheduler.get_metrics_summary()
        for key in ['total_cycles', 'total_strengthened', 'total_decayed', 
                    'total_pruned', 'total_patterns', 'total_associations']:
            print(f"  {key}: {summary[key]}")
        
        # Get recent metrics
        recent = scheduler.get_recent_metrics(limit=1)
        if recent:
            last = recent[-1]
            print(f"\nLast consolidation: {last.timestamp}, "
                  f"budget: {last.budget_used:.2f}, "
                  f"duration: {last.consolidation_duration_ms:.1f}ms")
        
        # Check associations
        print("\nMemory Associations:")
        associated = consolidator.get_associated_memories("exp_0", threshold=0.05)
        for mem_id, strength in associated[:3]:
            print(f"  {mem_id}: {strength:.3f}")
        
    finally:
        print("\nStopping...")
        await scheduler.stop()


if __name__ == "__main__":
    asyncio.run(main())
