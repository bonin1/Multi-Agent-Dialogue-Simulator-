#!/usr/bin/env python3
"""
Continuous Multi-Agent Dialogue Simulator
This runs an ongoing simulation with dynamic events and interventions
"""

import logging
import json
from datetime import datetime
import time
import random
import signal
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)

# Global variable to control simulation
running = True

def signal_handler(sig, frame):
    global running
    print('\n🛑 Stopping simulation...')
    running = False

def add_dynamic_events(turn_count: int, scenario_name: str) -> str:
    """Add dynamic events to keep the conversation interesting"""
    
    crisis_events = [
        "🚨 BREAKING: New intelligence suggests there's a mole in our organization!",
        "⚠️ ALERT: Security cameras just detected suspicious activity in sector 7.",
        "📱 URGENT: We just received an encrypted message from an unknown source.",
        "🔥 CRISIS: The situation has escalated - we have 30 minutes to decide.",
        "💰 REVELATION: Someone just offered us a significant bribe to change our approach.",
        "🕵️ DISCOVERY: We found classified documents that contradict what we've been told.",
        "⏰ TIME PRESSURE: The deadline has been moved up - we need to act NOW.",
        "🤝 OPPORTUNITY: A potential ally has reached out with valuable information.",
        "🎭 DECEPTION: Someone in the room is not who they claim to be.",
        "💣 THREAT: We just learned that failure could have catastrophic consequences."
    ]
    
    team_events = [
        "🎯 CHALLENGE: The project requirements just changed dramatically.",
        "💡 BREAKTHROUGH: New technology could revolutionize our approach.",
        "🔧 SETBACK: Our current solution has a critical flaw that needs fixing.",
        "📊 DATA: New research data suggests we should reconsider our strategy.",
        "🤔 DILEMMA: We have to choose between two equally important priorities.",
        "🚀 OPPORTUNITY: We could expand the project scope for greater impact.",
        "⚖️ ETHICS: Someone raised concerns about the ethical implications.",
        "🎲 RISK: There's a 50% chance our approach could backfire spectacularly.",
        "👥 TEAM DYNAMICS: There's growing tension between different approaches.",
        "🔄 PIVOT: Market conditions require us to completely change direction."
    ]
    
    negotiation_events = [
        "📋 PROPOSAL: The other side just made a surprising counter-offer.",
        "🎭 REVELATION: We discovered hidden motives behind their position.",
        "⚡ LEVERAGE: We just gained significant bargaining power.",
        "🤝 ALLIANCE: A third party wants to join the negotiations.",
        "💔 BREAKDOWN: Trust between parties is deteriorating rapidly.",
        "🎯 ULTIMATUM: They've given us 24 hours to accept their terms.",
        "🔍 INVESTIGATION: New evidence changes everything we thought we knew.",
        "💰 STAKES: The financial implications are much higher than expected.",
        "🗳️ PRESSURE: Public opinion is strongly influencing the negotiations.",
        "⚖️ LEGAL: Lawyers are now involved, complicating everything."
    ]
    
    # Choose events based on scenario
    if "espionage" in scenario_name.lower() or "spy" in scenario_name.lower():
        events = crisis_events
    elif "team" in scenario_name.lower() or "innovation" in scenario_name.lower():
        events = team_events
    elif "negotiation" in scenario_name.lower() or "political" in scenario_name.lower():
        events = negotiation_events
    else:
        events = crisis_events + team_events + negotiation_events
    
    # Add events every 3-7 turns with some randomness
    if turn_count % random.randint(3, 7) == 0:
        return random.choice(events)
    
    return ""

def add_pressure_prompts(turn_count: int) -> str:
    """Add pressure and conflict to prevent agreement loops"""
    
    pressure_prompts = [
        "Someone needs to take a strong position - enough agreeing!",
        "What are you NOT telling the others? What's your hidden agenda?",
        "Time is running out - make a decision NOW!",
        "Someone here is lying - who do you suspect and why?",
        "What would you do if you were in charge? Be specific!",
        "What's the worst case scenario if this fails?",
        "Who benefits most from the current situation?",
        "What are you willing to sacrifice to achieve your goals?",
        "If you had to choose sides right now, who would you choose?",
        "What information are you holding back from the group?"
    ]
    
    # Add pressure every 4-6 turns
    if turn_count % random.randint(4, 6) == 0:
        return f"🔥 PRESSURE: {random.choice(pressure_prompts)}"
    
    return ""

def choose_next_speaker(agent_names: list, turn_count: int, last_speakers: list) -> str:
    """Choose next speaker more dynamically to avoid patterns"""
    
    # Avoid the same person speaking twice in a row
    available_speakers = [name for name in agent_names if name not in last_speakers[-1:]]
    
    if not available_speakers:
        available_speakers = agent_names
    
    # Add some randomness but prefer speakers who haven't spoken recently
    recent_speakers = last_speakers[-3:] if len(last_speakers) >= 3 else last_speakers
    preferred_speakers = [name for name in available_speakers if name not in recent_speakers]
    
    if preferred_speakers and random.random() > 0.3:  # 70% chance to prefer less recent speakers
        return random.choice(preferred_speakers)
    else:
        return random.choice(available_speakers)

def run_continuous_simulation():
    """Run a continuous simulation that keeps going until stopped"""
    global running
    
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Import modules
        from models.model_manager import ModelManager
        from agents.agent import Agent
        from models.agent_models import AgentRole, PersonalityTrait
        from scenarios.scenario_manager import ScenarioManager, AGENT_CONFIGS
        
        print("🤖 Continuous Multi-Agent Dialogue Simulator")
        print("=" * 60)
        print("Press Ctrl+C to stop the simulation at any time")
        print("=" * 60)
        
        # Initialize model manager
        print("\n📦 Loading AI model...")
        model_manager = ModelManager()
        print("✅ Model loaded successfully!")
        
        # Initialize scenario manager
        scenario_manager = ScenarioManager()
        scenario_manager.set_scenario("Corporate Espionage") 
        
        # Create agents
        print("\n🎭 Creating agents...")
        agents = {}
        
        # Create Agent X (Spy)
        spy_config = AGENT_CONFIGS["Agent X"]
        spy = Agent(
            name="Agent X",
            role=AgentRole(spy_config['role']),
            personality=PersonalityTrait(**spy_config['personality']),
            model_manager=model_manager,
            background_story=spy_config['background']
        )
        spy.state.goals = spy_config['goals']
        agents["Agent X"] = spy
        
        # Create Maya Rodriguez (Rebel)
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
        
        # Create Marcus Steel (Engineer)
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
        
        print(f"✅ Created {len(agents)} agents: {', '.join(agents.keys())}")
        
        # Start simulation
        scenario_context = scenario_manager.get_current_context()
        print(f"\n🚀 Starting continuous simulation...")
        print(f"📋 Scenario: {scenario_context.get('scenario_description', 'Unknown')}")
        print(f"🎯 Context: {scenario_context.get('scenario_context', 'Unknown')}")
        print("\n" + "=" * 60)
        
        conversation_history = []
        last_speakers = []
        
        # Initial prompt - more dramatic
        initial_prompt = scenario_context.get('initial_prompt', 'The situation is critical and trust is scarce.')
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
        
        # Main simulation loop
        agent_names = list(agents.keys())
        turn_count = 0
        
        while running:
            turn_count += 1
            print(f"\n--- Turn {turn_count} ---")
            
            # Add dynamic events
            event = add_dynamic_events(turn_count, "Corporate Espionage")
            if event:
                print(f"\n🎬 EVENT: {event}")
                conversation_history.append({
                    'speaker': 'System',
                    'message': event,
                    'timestamp': datetime.now(),
                    'turn': turn_count
                })
                # Process event for all agents
                for agent in agents.values():
                    agent.process_message(event, "System", {"turn": turn_count, "event": True})
            
            # Add pressure prompts
            pressure = add_pressure_prompts(turn_count)
            if pressure:
                print(f"\n{pressure}")
                conversation_history.append({
                    'speaker': 'Moderator',
                    'message': pressure,
                    'timestamp': datetime.now(),
                    'turn': turn_count
                })
                # Process pressure for all agents
                for agent in agents.values():
                    agent.process_message(pressure, "Moderator", {"turn": turn_count, "pressure": True})
            
            # Choose speaker dynamically
            speaker_name = choose_next_speaker(agent_names, turn_count, last_speakers)
            speaker = agents[speaker_name]
            last_speakers.append(speaker_name)
            
            print(f"🤖 {speaker_name} is thinking...")
            
            # Generate response with enhanced context
            context = {
                'recent_messages': conversation_history[-5:],  # More context
                'scenario_phase': scenario_manager.get_current_context().get('current_phase', ''),
                'turn_count': turn_count,
                'total_agents': len(agents),
                'speaker_history': last_speakers[-5:],
                'high_stakes': True,
                'conflict_mode': turn_count > 5  # Increase conflict after initial turns
            }
            
            start_time = time.time()
            response = speaker.generate_response(context)
            response_time = time.time() - start_time
            
            # Enhance the response display
            emotion_icon = {
                "happy": "😊", "confident": "😎", "neutral": "😐",
                "frustrated": "😤", "angry": "😠", "sad": "😢",
                "anxious": "😰", "suspicious": "🤨", "excited": "🤩"
            }.get(speaker.state.emotional_state.primary_emotion.value, "🤖")
            
            print(f"{emotion_icon} [{datetime.now().strftime('%H:%M:%S')}] {speaker_name}: {response}")
            print(f"   ⏱️ Response time: {response_time:.1f}s")
            
            # Add to conversation history
            conversation_history.append({
                'speaker': speaker_name,
                'message': response,
                'timestamp': datetime.now(),
                'turn': turn_count
            })
            
            # Process message for all other agents
            for name, agent in agents.items():
                if name != speaker_name:
                    agent.process_message(response, speaker_name, {"turn": turn_count})
            
            # Show emotional states and relationships
            emotions = {}
            trust_levels = {}
            for name, agent in agents.items():
                emotion = agent.state.emotional_state.primary_emotion.value
                intensity = agent.state.emotional_state.intensity
                emotions[name] = f"{emotion} ({intensity:.1f})"
                
                # Show trust towards others
                trust_summary = []
                for other_name, trust in agent.state.relationships.items():
                    if trust < 0.3:
                        trust_summary.append(f"distrusts {other_name}")
                    elif trust > 0.7:
                        trust_summary.append(f"trusts {other_name}")
                
                if trust_summary:
                    trust_levels[name] = ", ".join(trust_summary)
            
            print(f"   😄 Emotions: {emotions}")
            if trust_levels:
                print(f"   🤝 Trust: {trust_levels}")
            
            # Auto-save every 10 turns
            if turn_count % 10 == 0:
                export_data = {
                    'simulation_info': {
                        'scenario': 'Corporate Espionage',
                        'agents': list(agents.keys()),
                        'turns': turn_count,
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
                
                filename = f"continuous_simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(filename, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
                print(f"   💾 Auto-saved to {filename}")
            
            # Brief pause for readability
            time.sleep(2)
            
            # Keep only recent conversation history to prevent memory issues
            if len(conversation_history) > 100:
                conversation_history = conversation_history[-50:]
        
        # Final save when simulation ends
        print("\n" + "=" * 60)
        print("💾 SAVING FINAL SIMULATION DATA")
        print("=" * 60)
        
        final_export = {
            'simulation_info': {
                'scenario': 'Corporate Espionage',
                'agents': list(agents.keys()),
                'total_turns': turn_count,
                'timestamp': datetime.now().isoformat(),
                'status': 'manually_stopped'
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
            'final_agent_summaries': {
                name: agent.get_agent_summary()
                for name, agent in agents.items()
            }
        }
        
        final_filename = f"final_simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(final_filename, 'w') as f:
            json.dump(final_export, f, indent=2, default=str)
        
        print(f"✅ Final simulation data saved to {final_filename}")
        print(f"🎯 Completed {turn_count} turns of conversation")
        print("🎉 Simulation ended successfully!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install dependencies first: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error during simulation: {e}")
        logging.exception("Detailed error:")

if __name__ == "__main__":
    run_continuous_simulation()
