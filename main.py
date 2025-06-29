"""
Multi-Agent Dialogue Simulator - Main Entry Point

A sophisticated multi-agent system where AI agents with distinct personalities, roles, and memory 
engage in dynamic conversations with emotional intelligence and long-term reflection capabilities.
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
import json

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.simulation.dialogue_manager import DialogueManager
from src.agents.personality import PersonalityGenerator
from src.utils.model_manager import get_model_manager
from src.config.settings import get_config, LOGGING_CONFIG


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, LOGGING_CONFIG["level"]),
        format=LOGGING_CONFIG["format"],
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOGGING_CONFIG["log_file"]) if LOGGING_CONFIG["file_handler"] else logging.NullHandler()
        ]
    )


def print_banner():
    """Print application banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🤖 Multi-Agent Dialogue Simulator 🤖                      ║
║                                                                              ║
║  Autonomous AI agents with personalities, memory, and emotional intelligence ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_available_scenarios():
    """Print available scenarios"""
    from src.simulation.scenarios import ScenarioManager
    
    scenario_manager = ScenarioManager()
    scenarios = scenario_manager.list_scenarios()
    
    print("\n📋 Available Scenarios:")
    print("=" * 50)
    
    for scenario_name in scenarios:
        details = scenario_manager.get_scenario_details(scenario_name)
        tension = "🔥" * int(details.get("tension_level", 0.5) * 5)
        print(f"  • {scenario_name:<20} - {details.get('description', 'No description')}")
        print(f"    Tension: {tension} | Complexity: {details.get('complexity', 'Unknown')}")
        print(f"    Duration: {details.get('duration_estimate', 'Unknown')}")
        print()


def print_agent_roles():
    """Print available agent roles"""
    from src.config.settings import AGENT_CONFIG
    
    print("\n👥 Available Agent Roles:")
    print("=" * 50)
    
    for role, config in AGENT_CONFIG["default_personalities"].items():
        traits = ", ".join(config.get("traits", []))
        background = config.get("background", "No background")
        
        print(f"  • {role.title():<12} - {background}")
        print(f"    Traits: {traits}")
        print()


async def run_interactive_mode():
    """Run interactive configuration mode"""
    print("\n🎮 Interactive Mode")
    print("=" * 50)
    
    # Get scenario choice
    print_available_scenarios()
    scenario = input("Enter scenario name (or press Enter for 'team_building'): ").strip()
    if not scenario:
        scenario = "team_building"
    
    # Get number of agents
    while True:
        try:
            num_agents = input("Number of agents (2-5, default 3): ").strip()
            if not num_agents:
                num_agents = 3
            else:
                num_agents = int(num_agents)
            
            if 2 <= num_agents <= 5:
                break
            else:
                print("Please enter a number between 2 and 5.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Get max turns
    while True:
        try:
            max_turns = input("Maximum conversation turns (default 20): ").strip()
            if not max_turns:
                max_turns = 20
            else:
                max_turns = int(max_turns)
            
            if max_turns > 0:
                break
            else:
                print("Please enter a positive number.")
        except ValueError:
            print("Please enter a valid number.")
    
    return {
        "scenario": scenario,
        "num_agents": num_agents,
        "max_turns": max_turns
    }


async def run_simulation(
    scenario: str = "team_building",
    num_agents: int = 3,
    max_turns: int = 20,
    output_file: Optional[str] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """Run the main simulation"""
    
    logger = logging.getLogger("MainSimulation")
    
    try:
        # Initialize dialogue manager
        logger.info(f"Initializing simulation with {num_agents} agents for scenario: {scenario}")
        
        dialogue_manager = DialogueManager(
            num_agents=num_agents,
            scenario=scenario
        )
        
        # Initialize model
        print("🔄 Loading AI model (this may take a few minutes on first run)...")
        success = await dialogue_manager.initialize_model()
        
        if not success:
            print("⚠️  Warning: AI model failed to load. Using fallback response generation.")
            logger.warning("Model loading failed, using fallback responses")
        else:
            print("✅ AI model loaded successfully!")
        
        # Print simulation info
        print(f"\n🚀 Starting simulation:")
        print(f"   Scenario: {scenario}")
        print(f"   Agents: {num_agents}")
        print(f"   Max turns: {max_turns}")
        print(f"   Verbose: {verbose}")
        print()
        
        # List agents
        print("👥 Participating Agents:")
        for agent in dialogue_manager.agents.values():
            personality_desc = agent.personality.get_trait_description()
            print(f"   • {agent.name} ({agent.role}) - {personality_desc[:100]}...")
        print()
        
        # Start conversation
        print("💬 Starting conversation...")
        print("=" * 80)
        
        conversation_results = await dialogue_manager.simulate_conversation(
            max_turns=max_turns
        )
        
        # Display conversation
        for turn in conversation_results:
            timestamp = turn["timestamp"].strftime("%H:%M:%S") if hasattr(turn["timestamp"], "strftime") else str(turn["timestamp"])
            speaker = turn["speaker"]
            content = turn["content"]
            
            if verbose and "role" in turn:
                role_info = f" ({turn['role']})"
            else:
                role_info = ""
            
            print(f"[{timestamp}] {speaker}{role_info}: {content}")
            print()
        
        print("=" * 80)
        print("✅ Conversation completed!")
        
        # Get summary
        summary = dialogue_manager.get_conversation_summary()
        
        print(f"\n📊 Conversation Summary:")
        print(f"   Total turns: {summary.get('total_turns', 0)}")
        print(f"   Duration: {summary.get('duration_minutes', 0):.1f} minutes")
        print(f"   Total words: {summary.get('total_words', 0)}")
        print(f"   Emotional moments: {len(summary.get('emotional_moments', []))}")
        
        # Agent participation
        if summary.get('agent_participation'):
            print(f"\n   Agent Participation:")
            for agent, stats in summary['agent_participation'].items():
                print(f"     • {agent}: {stats['turns']} turns, {stats['words']} words")
        
        # Relationship changes
        if summary.get('final_relationships'):
            print(f"\n   Final Relationships:")
            for agent, rel_count in summary['final_relationships'].items():
                print(f"     • {agent}: {rel_count} relationships formed")
        
        # Export conversation if requested
        if output_file:
            print(f"\n💾 Exporting conversation to {output_file}...")
            
            if output_file.endswith('.json'):
                export_data = dialogue_manager.export_conversation('json')
            else:
                export_data = dialogue_manager.export_conversation('text')
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(export_data)
            
            print(f"✅ Conversation exported successfully!")
        
        # Return results
        return {
            "conversation": conversation_results,
            "summary": summary,
            "agents": {agent.name: agent.get_agent_state() for agent in dialogue_manager.agents.values()},
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        print(f"❌ Simulation failed: {e}")
        return {"success": False, "error": str(e)}


async def check_system_requirements():
    """Check system requirements and model availability"""
    print("🔍 Checking system requirements...")
    
    # Check model manager
    try:
        model_manager = get_model_manager()
        availability = model_manager.check_model_availability()
        
        print(f"   Storage available: {availability['local_storage_gb']:.1f} GB")
        print(f"   Model size needed: ~{availability['estimated_model_size_gb']} GB")
        
        if availability['local_storage_gb'] < availability['estimated_model_size_gb']:
            print("   ⚠️  Warning: Low disk space for model download")
        else:
            print("   ✅ Sufficient storage available")
        
        if availability['model_cached']:
            print("   ✅ Model already downloaded")
        else:
            print("   📥 Model will be downloaded on first use")
        
    except Exception as e:
        print(f"   ❌ Model check failed: {e}")
    
    # Check dependencies
    try:
        import torch
        print(f"   ✅ PyTorch available: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"   ✅ CUDA available: {torch.cuda.get_device_name()}")
        else:
            print("   ⚠️  CUDA not available (will use CPU - slower)")
        
    except ImportError:
        print("   ❌ PyTorch not available")
    
    try:
        import transformers
        print(f"   ✅ Transformers available: {transformers.__version__}")
    except ImportError:
        print("   ❌ Transformers not available")
    
    try:
        import chromadb
        print("   ✅ ChromaDB available for memory storage")
    except ImportError:
        print("   ⚠️  ChromaDB not available (will use in-memory storage)")
    
    print()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Dialogue Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # Interactive mode
  python main.py --scenario political_negotiation  # Quick start with scenario
  python main.py --agents 4 --turns 30             # Custom configuration
  python main.py --list-scenarios                  # List available scenarios
  python main.py --check-requirements              # Check system requirements
        """
    )
    
    parser.add_argument("--scenario", "-s", 
                       help="Scenario to run (use --list-scenarios to see options)")
    parser.add_argument("--agents", "-a", type=int, choices=range(2, 6), 
                       help="Number of agents (2-5)")
    parser.add_argument("--turns", "-t", type=int, 
                       help="Maximum conversation turns")
    parser.add_argument("--output", "-o", 
                       help="Output file for conversation export (.json or .txt)")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Verbose output with additional details")
    parser.add_argument("--interactive", "-i", action="store_true", 
                       help="Force interactive mode")
    parser.add_argument("--list-scenarios", action="store_true", 
                       help="List available scenarios and exit")
    parser.add_argument("--list-roles", action="store_true", 
                       help="List available agent roles and exit")
    parser.add_argument("--check-requirements", action="store_true", 
                       help="Check system requirements and exit")
    parser.add_argument("--debug", action="store_true", 
                       help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Setup logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    setup_logging()
    
    # Print banner
    print_banner()
    
    # Handle list commands
    if args.list_scenarios:
        print_available_scenarios()
        return
    
    if args.list_roles:
        print_agent_roles()
        return
    
    if args.check_requirements:
        asyncio.run(check_system_requirements())
        return
    
    # Determine run mode
    if args.interactive or (not args.scenario and not args.agents and not args.turns):
        # Interactive mode
        config = asyncio.run(run_interactive_mode())
        scenario = config["scenario"]
        num_agents = config["num_agents"]
        max_turns = config["max_turns"]
    else:
        # Direct mode with arguments
        scenario = args.scenario or "team_building"
        num_agents = args.agents or 3
        max_turns = args.turns or 20
    
    # Run the simulation
    try:
        results = asyncio.run(run_simulation(
            scenario=scenario,
            num_agents=num_agents,
            max_turns=max_turns,
            output_file=args.output,
            verbose=args.verbose
        ))
        
        if results["success"]:
            print("\n🎉 Simulation completed successfully!")
            sys.exit(0)
        else:
            print(f"\n❌ Simulation failed: {results.get('error', 'Unknown error')}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏸️  Simulation interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
