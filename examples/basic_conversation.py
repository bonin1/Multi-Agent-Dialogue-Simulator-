"""
Example: Basic conversation between 3 agents
"""
import sys
from pathlib import Path
import asyncio

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulation.dialogue_manager import DialogueManager


async def basic_conversation_example():
    """Run a basic conversation example"""
    print("🚀 Basic Conversation Example")
    print("=" * 50)
    
    # Create dialogue manager with 3 agents
    manager = DialogueManager(
        num_agents=3,
        scenario="team_building"
    )
    
    print("Agents:")
    for agent in manager.agents.values():
        print(f"  • {agent.name} ({agent.role})")
    print()
    
    # Run conversation
    print("Starting conversation...\n")
    
    results = await manager.simulate_conversation(max_turns=10)
    
    # Display results
    for turn in results:
        print(f"{turn['speaker']}: {turn['content']}")
        print()
    
    # Show summary
    summary = manager.get_conversation_summary()
    print("Summary:")
    print(f"  Total turns: {summary.get('total_turns', 0)}")
    print(f"  Total words: {summary.get('total_words', 0)}")


if __name__ == "__main__":
    asyncio.run(basic_conversation_example())
