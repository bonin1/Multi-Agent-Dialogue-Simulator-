#!/usr/bin/env python3
"""
Example script demonstrating the Multi-Agent Dialogue Simulator
This shows how to use the system programmatically without the UI
"""

import logging
import json
from datetime import datetime
import time

# Configure logging
logging.basicConfig(level=logging.INFO)

def run_example_simulation():
    """Run a simple example simulation"""
    try:
        # Import modules
        from models.model_manager import ModelManager
        from agents.agent import Agent
        from models.agent_models import AgentRole, PersonalityTrait
        from scenarios.scenario_manager import ScenarioManager, AGENT_CONFIGS
        
        print("🤖 Multi-Agent Dialogue Simulator - Example")
        print("=" * 50)
        
        # Initialize model manager
        print("\n📦 Loading AI model...")
        model_manager = ModelManager()
        print("✅ Model loaded successfully!")
        
        # Initialize scenario manager
        scenario_manager = ScenarioManager()
        scenario_manager.set_scenario("Team Building")
        
        # Create agents
        print("\n🎭 Creating agents...")
        agents = {}
        
        # Create Dr. Sarah Chen
        sarah_config = AGENT_CONFIGS["Dr. Sarah Chen"]
        sarah = Agent(
            name="Dr. Sarah Chen",
            role=AgentRole(sarah_config['role']),
            personality=PersonalityTrait(**sarah_config['personality']),
            model_manager=model_manager,
            background_story=sarah_config['background']
        )
        sarah.state.goals = sarah_config['goals']
        agents["Dr. Sarah Chen"] = sarah
        
        # Create Marcus Steel
        marcus_config = AGENT_CONFIGS["Marcus Steel"]
        marcus = Agent(
            name="Marcus Steel",
            role=AgentRole(marcus_config['role']),
            personality=PersonalityTrait(**marcus_config['personality']),
            model_manager=model_manager,
            background_story=marcus_config['background']
        )
        marcus.state.goals = marcus_config['goals']
        agents["Marcus Steel"] = marcus
        
        # Create Maya Rodriguez
        maya_config = AGENT_CONFIGS["Maya Rodriguez"]
        maya = Agent(
            name="Maya Rodriguez",
            role=AgentRole(maya_config['role']),
            personality=PersonalityTrait(**maya_config['personality']),
            model_manager=model_manager,
            background_story=maya_config['background']
        )
        maya.state.goals = maya_config['goals']
        agents["Maya Rodriguez"] = maya
        
        print(f"✅ Created {len(agents)} agents: {', '.join(agents.keys())}")
        
        # Start simulation
        print("\n🚀 Starting simulation...")
        print("Scenario: Team Building")
        print("Context: A diverse team has been assembled to tackle a crisis situation")
        print("\n" + "=" * 50)
        
        conversation_history = []
        
        # Initial prompt
        initial_prompt = "We've been brought together to handle this crisis. Let's start by understanding what each of us brings to the table."
        print(f"\n[SYSTEM]: {initial_prompt}")
        conversation_history.append({
            'speaker': 'System',
            'message': initial_prompt,
            'timestamp': datetime.now(),
            'turn': 0
        })
        
        # Process initial message for all agents
        for agent in agents.values():
            agent.process_message(initial_prompt, "System", {"turn": 0})
        
        # Simulate conversation turns - CONTINUOUS until Ctrl+C
        agent_names = list(agents.keys())
        turn = 0
        last_speaker = None
        
        # Dynamic event prompts to inject conflict and urgency
        pressure_prompts = [
            "Someone just received a suspicious message. Who can we really trust?",
            "Time is running out. We need to make a decision NOW.",
            "I noticed someone acting strangely during our last meeting...",
            "The situation has changed. Our original plan may be compromised.",
            "I have information that suggests one of us isn't who they claim to be.",
            "We're being watched. Someone here might be feeding information to the enemy.",
            "I found evidence that contradicts what someone told us earlier.",
            "The mission parameters have shifted. Not everyone here has the same objectives.",
            "Security has been breached. We need to identify the leak immediately.",
            "I overheard something that makes me question our current approach."
        ]
        
        print("\n🚀 Starting CONTINUOUS simulation...")
        print("💡 Press Ctrl+C to stop the simulation")
        print("🎯 Agents will actively pursue their goals and challenge each other")
        print("\n" + "=" * 50)
        
        try:
            while True:
                turn += 1
                print(f"\n--- Turn {turn} ---")
                
                # Choose speaker (avoid same speaker twice in a row)
                available_speakers = [name for name in agent_names if name != last_speaker]
                if not available_speakers:
                    available_speakers = agent_names
                
                speaker_name = available_speakers[turn % len(available_speakers)]
                speaker = agents[speaker_name]
                last_speaker = speaker_name
                
                print(f"🤖 {speaker_name} is thinking...")
                
                # Enhanced context with pressure and conflict encouragement
                context = {
                    'recent_messages': conversation_history[-5:],
                    'scenario_phase': scenario_manager.get_current_context().get('current_phase', ''),
                    'turn_count': turn,
                    'total_agents': len(agents),
                    'scenario_urgency': 'high',
                    'conflict_encouragement': True
                }
                
                # Inject pressure prompts periodically to prevent agreement loops
                if turn % 4 == 0 and turn > 1:
                    pressure_prompt = pressure_prompts[(turn // 4 - 1) % len(pressure_prompts)]
                    print(f"⚡ [SYSTEM PRESSURE]: {pressure_prompt}")
                    conversation_history.append({
                        'speaker': 'System',
                        'message': f"[PRESSURE] {pressure_prompt}",
                        'timestamp': datetime.now(),
                        'turn': turn,
                        'type': 'pressure'
                    })
                    
                    # All agents process the pressure prompt
                    for agent in agents.values():
                        agent.process_message(pressure_prompt, "System", {"turn": turn, "type": "pressure"})
                
                start_time = time.time()
                response = speaker.generate_response(context)
                response_time = time.time() - start_time
                
                print(f"[{speaker_name}]: {response}")
                print(f"   (Response time: {response_time:.1f}s)")
                
                # Add to conversation history
                conversation_history.append({
                    'speaker': speaker_name,
                    'message': response,
                    'timestamp': datetime.now(),
                    'turn': turn
                })
                
                # Process message for all other agents
                for name, agent in agents.items():
                    if name != speaker_name:
                        agent.process_message(response, speaker_name, {"turn": turn})
                
                # Show emotional states and relationships
                emotions = {}
                suspicions = {}
                for name, agent in agents.items():
                    emotion = agent.state.emotional_state.primary_emotion.value
                    intensity = agent.state.emotional_state.intensity
                    emotions[name] = f"{emotion} ({intensity:.1f})"
                    
                    # Show who they're most suspicious of
                    if agent.state.relationships:
                        most_suspicious = min(agent.state.relationships.items(), 
                                            key=lambda x: x[1])
                        suspicions[name] = f"suspicious of {most_suspicious[0]} ({most_suspicious[1]:.1f})"
                
                print(f"   Emotions: {emotions}")
                if suspicions:
                    print(f"   Suspicions: {suspicions}")
                
                # Show memory activity
                memory_counts = {}
                for name, agent in agents.items():
                    memory_count = len(agent.memory_system.get_recent_conversations(n_recent=10))
                    memory_counts[name] = memory_count
                print(f"   Memory activity: {memory_counts}")
                
                # Auto-save progress every 10 turns
                if turn % 10 == 0:
                    export_data = {
                        'simulation_info': {
                            'scenario': 'Team Building (Continuous)',
                            'agents': list(agents.keys()),
                            'current_turn': turn,
                            'timestamp': datetime.now().isoformat()
                        },
                        'conversation_history': [
                            {
                                'speaker': entry['speaker'],
                                'message': entry['message'],
                                'timestamp': entry['timestamp'].isoformat(),
                                'turn': entry['turn']
                            }
                            for entry in conversation_history[-20:]  # Last 20 messages
                        ]
                    }
                    
                    filename = f"continuous_sim_autosave_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    with open(filename, 'w') as f:
                        json.dump(export_data, f, indent=2, default=str)
                    print(f"   💾 Auto-saved to {filename}")
                
                # Brief pause for readability
                time.sleep(2)
                
        except KeyboardInterrupt:
            print(f"\n\n🛑 Simulation stopped by user after {turn} turns")
        
        # Show final analysis
        print("\n" + "=" * 50)
        print("📊 SIMULATION COMPLETE - ANALYSIS")
        print("=" * 50)
        
        for name, agent in agents.items():
            summary = agent.get_agent_summary()
            print(f"\n🤖 {name}:")
            print(f"   Role: {summary['role']}")
            print(f"   Final Emotion: {summary['emotional_state']['emotion']} ({summary['emotional_state']['intensity']:.1f})")
            print(f"   Relationships: {summary['relationships']}")
            if summary['recent_reflections']:
                print(f"   Latest Reflection: {summary['recent_reflections'][-1]}")
            print(f"   Memory Stats: {summary['memory_stats']}")
        
        # Export conversation
        export_data = {
            'simulation_info': {
                'scenario': 'Team Building',
                'agents': list(agents.keys()),
                'turns': turn,
                'timestamp': datetime.now().isoformat()
            },
            'conversation_history': [
                {
                    'speaker': entry['speaker'],
                    'message': entry['message'],
                    'timestamp': entry['timestamp'].isoformat(),
                    'turn': entry['turn']
                }
                for entry in conversation_history
            ],
            'agent_summaries': {
                name: agent.get_agent_summary()
                for name, agent in agents.items()
            }
        }
        
        with open(f"example_simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print("\n✅ Simulation data exported to JSON file")
        print("\n🎉 Example simulation completed successfully!")
        print("To run the full interactive version, use: streamlit run main.py")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install dependencies first: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error during simulation: {e}")
        logging.exception("Detailed error:")

if __name__ == "__main__":
    run_example_simulation()
