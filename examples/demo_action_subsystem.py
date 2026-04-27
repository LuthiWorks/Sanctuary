"""
Integration test demonstrating ActionSubsystem in action.

This script creates a simple cognitive cycle with goals and shows
how the ActionSubsystem makes decisions based on workspace state.
"""

import asyncio
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(name)s - %(message)s')

# NOTE: This demo predates the emergence_core -> sanctuary/mind migration
# and the legacy import paths below no longer exist. Kept for historical
# reference; will not run as-is. Update imports to current sanctuary.mind.*
# paths before running.
from mind.cognitive_core.workspace import GlobalWorkspace, Goal, GoalType, Percept, WorkspaceSnapshot  # noqa: E402
from mind.cognitive_core.action import ActionSubsystem, Action, ActionType  # noqa: E402


def demo_action_decision():
    """Demonstrate action decision making with various scenarios."""
    
    print("=" * 70)
    print("ActionSubsystem Integration Demo")
    print("=" * 70)
    
    # Initialize subsystem
    print("\n1. Initializing ActionSubsystem...")
    subsystem = ActionSubsystem()
    stats = subsystem.get_stats()
    print(f"   - Protocol constraints loaded: {len(subsystem.protocol_constraints)}")
    print(f"   - Tool registry: {len(subsystem.tool_registry)} tools")
    print(f"   - Initial stats: {stats}")
    
    # Scenario 1: User request (high priority)
    print("\n2. Scenario 1: User wants a response")
    goal1 = Goal(
        type=GoalType.RESPOND_TO_USER,
        description="Answer the user's question about consciousness",
        priority=0.9
    )
    
    snapshot1 = WorkspaceSnapshot(
        goals=[goal1],
        percepts={},
        emotions={"valence": 0.0, "arousal": 0.5},
        memories=[],
        timestamp=datetime.now(),
        cycle_count=1
    )
    
    actions1 = subsystem.decide(snapshot1)
    print(f"   - Generated {len(actions1)} action(s):")
    for i, action in enumerate(actions1, 1):
        print(f"     {i}. {action.type.value} (priority={action.priority:.2f}): {action.reason}")
    
    # Scenario 2: High arousal amplifies urgency
    print("\n3. Scenario 2: User request + HIGH arousal (urgent!)")
    snapshot2 = WorkspaceSnapshot(
        goals=[goal1],
        percepts={},
        emotions={"valence": 0.0, "arousal": 0.9},  # High arousal!
        memories=[],
        timestamp=datetime.now(),
        cycle_count=2
    )
    
    actions2 = subsystem.decide(snapshot2)
    print(f"   - Generated {len(actions2)} action(s):")
    for i, action in enumerate(actions2, 1):
        print(f"     {i}. {action.type.value} (priority={action.priority:.2f}): {action.reason}")
    
    # Scenario 3: Multiple competing goals
    print("\n4. Scenario 3: Multiple competing goals")
    goals = [
        Goal(type=GoalType.RESPOND_TO_USER, description="Answer Q1", priority=0.9),
        Goal(type=GoalType.RETRIEVE_MEMORY, description="Find relevant info", priority=0.7),
        Goal(type=GoalType.COMMIT_MEMORY, description="Save experience", priority=0.6),
        Goal(type=GoalType.INTROSPECT, description="Reflect on state", priority=0.4),
    ]
    
    snapshot3 = WorkspaceSnapshot(
        goals=goals,
        percepts={},
        emotions={"valence": 0.3, "arousal": 0.6},
        memories=[],
        timestamp=datetime.now(),
        cycle_count=3
    )
    
    actions3 = subsystem.decide(snapshot3)
    print(f"   - Generated {len(actions3)} action(s) (top 3 selected):")
    for i, action in enumerate(actions3, 1):
        print(f"     {i}. {action.type.value} (priority={action.priority:.2f}): {action.reason}")
    
    # Scenario 4: Negative emotion triggers introspection
    print("\n5. Scenario 4: Negative emotion state")
    snapshot4 = WorkspaceSnapshot(
        goals=[],
        percepts={},
        emotions={"valence": -0.6, "arousal": 0.4},  # Negative valence
        memories=[],
        timestamp=datetime.now(),
        cycle_count=4
    )
    
    actions4 = subsystem.decide(snapshot4)
    print(f"   - Generated {len(actions4)} action(s):")
    for i, action in enumerate(actions4, 1):
        print(f"     {i}. {action.type.value} (priority={action.priority:.2f}): {action.reason}")
    
    # Scenario 5: Introspection percept
    print("\n6. Scenario 5: Meta-cognitive percept")
    percept_data = {
        "id": "p1",
        "modality": "introspection",
        "raw": "Noticing patterns in my responses",
        "timestamp": datetime.now().isoformat()
    }
    
    snapshot5 = WorkspaceSnapshot(
        goals=[],
        percepts={"p1": percept_data},
        emotions={"valence": 0.0, "arousal": 0.3},
        memories=[],
        timestamp=datetime.now(),
        cycle_count=5
    )
    
    actions5 = subsystem.decide(snapshot5)
    print(f"   - Generated {len(actions5)} action(s):")
    for i, action in enumerate(actions5, 1):
        print(f"     {i}. {action.type.value} (priority={action.priority:.2f}): {action.reason}")
    
    # Scenario 6: Nothing urgent (WAIT action)
    print("\n7. Scenario 6: No urgent goals or stimuli")
    snapshot6 = WorkspaceSnapshot(
        goals=[],
        percepts={},
        emotions={"valence": 0.0, "arousal": 0.1},
        memories=[],
        timestamp=datetime.now(),
        cycle_count=6
    )
    
    actions6 = subsystem.decide(snapshot6)
    print(f"   - Generated {len(actions6)} action(s):")
    for i, action in enumerate(actions6, 1):
        print(f"     {i}. {action.type.value} (priority={action.priority:.2f}): {action.reason}")
    
    # Final statistics
    print("\n8. Final Statistics")
    final_stats = subsystem.get_stats()
    print(f"   - Total actions selected: {final_stats['total_actions']}")
    print(f"   - Actions blocked by protocols: {final_stats['blocked_actions']}")
    print(f"   - Action counts by type:")
    for action_type, count in final_stats['action_counts'].items():
        print(f"     - {action_type}: {count}")
    print(f"   - History size: {final_stats['history_size']}")
    
    # Demonstrate protocol constraint
    print("\n9. Demonstrating Protocol Constraint")
    from mind.identity.loader import ActionConstraint
    
    # Add a constraint that blocks WAIT actions
    constraint = ActionConstraint(
        rule="Never wait when there's work to be done",
        priority=1.0,
        test_fn=lambda action: action.type == ActionType.WAIT,
        source="demo"
    )
    subsystem.add_constraint(constraint)
    print(f"   - Added constraint: '{constraint.rule}'")
    
    # Try to generate WAIT action with empty workspace
    actions_blocked = subsystem.decide(snapshot6)
    print(f"   - Attempted to generate actions with empty workspace")
    print(f"   - Result: {len(actions_blocked)} action(s) generated")
    if actions_blocked:
        for i, action in enumerate(actions_blocked, 1):
            print(f"     {i}. {action.type.value} (WAIT was blocked!)")
    
    blocked_count = subsystem.get_stats()['blocked_actions']
    print(f"   - Total blocked actions: {blocked_count}")
    
    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)


async def demo_tool_execution():
    """Demonstrate tool registration and execution."""
    
    print("\n" + "=" * 70)
    print("Tool Execution Demo")
    print("=" * 70)
    
    subsystem = ActionSubsystem()
    
    # Register a test tool
    async def weather_tool(params):
        location = params.get("location", "unknown")
        return f"Weather in {location}: Sunny, 72°F"
    
    subsystem.register_tool("get_weather", weather_tool, "Get weather for a location")
    print("\n1. Registered tool: get_weather")
    
    # Create and execute a TOOL_CALL action
    action = Action(
        type=ActionType.TOOL_CALL,
        parameters={"tool": "get_weather", "location": "San Francisco"},
        priority=0.8,
        reason="User asked for weather"
    )
    
    print(f"2. Executing action: {action.type.value}")
    result = await subsystem.execute_action(action)
    print(f"3. Result: {result}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    # Run demos
    demo_action_decision()
    asyncio.run(demo_tool_execution())
    
    print("\n✅ All demos completed successfully!")
