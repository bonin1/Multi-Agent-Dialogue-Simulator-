import streamlit as st
import logging
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import traceback
import time
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)

# Import custom modules (with fallback handling)
try:
    from models.model_manager import ModelManager
    from agents.agent import Agent
    from models.agent_models import AgentRole, PersonalityTrait, EmotionalState, EmotionType
    from scenarios.scenario_manager import ScenarioManager, AGENT_CONFIGS, SCENARIOS
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.error("Please install required dependencies: pip install -r requirements.txt")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Multi-Agent Dialogue Simulator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'agents' not in st.session_state:
    st.session_state.agents = {}
if 'model_manager' not in st.session_state:
    st.session_state.model_manager = None
if 'scenario_manager' not in st.session_state:
    st.session_state.scenario_manager = ScenarioManager()
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'simulation_running' not in st.session_state:
    st.session_state.simulation_running = False
if 'turn_count' not in st.session_state:
    st.session_state.turn_count = 0

def load_model():
    """Load the language model"""
    try:
        if st.session_state.model_manager is None:
            with st.spinner("Loading AI model... This may take a few minutes."):
                st.session_state.model_manager = ModelManager()
        return True
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return False

def create_agent(name: str, config: Dict[str, Any]) -> Optional[Agent]:
    """Create an agent from configuration"""
    try:
        role = AgentRole(config['role'])
        personality = PersonalityTrait(**config['personality'])
        background = config.get('background', '')
        
        agent = Agent(
            name=name,
            role=role,
            personality=personality,
            model_manager=st.session_state.model_manager,
            background_story=background
        )
        
        # Set goals
        agent.state.goals = config.get('goals', [])
        
        return agent
    except Exception as e:
        st.error(f"Error creating agent {name}: {e}")
        return None

def display_agent_card(agent: Agent):
    """Display an agent information card"""
    summary = agent.get_agent_summary()
    
    with st.container():
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader(f"🤖 {summary['name']}")
            st.write(f"**Role:** {summary['role'].title()}")
            
            # Emotional state
            emotion_color = {
                "happy": "🟢", "confident": "🔵", "neutral": "⚪",
                "frustrated": "🟡", "angry": "🔴", "sad": "🟣",
                "anxious": "🟠", "suspicious": "🟤", "excited": "💚"
            }
            emotion_icon = emotion_color.get(summary['emotional_state']['emotion'], "⚪")
            st.write(f"**Emotion:** {emotion_icon} {summary['emotional_state']['emotion'].title()}")
            st.write(f"**Intensity:** {summary['emotional_state']['intensity']:.1f}")
        
        with col2:
            # Personality radar chart
            personality = summary['personality']
            fig = go.Figure()
            
            categories = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
            values = [personality[k] for k in ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']]
            values += values[:1]  # Complete the circle
            categories += categories[:1]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=summary['name']
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1])
                ),
                showlegend=False,
                height=200,
                margin=dict(l=20, r=20, t=20, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Memory stats
        memory_stats = summary['memory_stats']
        if any(memory_stats.values()):
            st.write("**Memory:**")
            cols = st.columns(4)
            stats_items = list(memory_stats.items())
            for i, (mem_type, count) in enumerate(stats_items):
                with cols[i % 4]:
                    st.metric(mem_type.title(), count)
        
        # Recent reflections
        if summary['recent_reflections']:
            with st.expander("Recent Reflections"):
                for reflection in summary['recent_reflections']:
                    st.write(f"💭 {reflection}")
        
        # Relationships
        if summary['relationships']:
            with st.expander("Relationships"):
                for agent_name, trust_level in summary['relationships'].items():
                    trust_color = "🔴" if trust_level < 0.3 else "🟡" if trust_level < 0.7 else "🟢"
                    st.write(f"{trust_color} {agent_name}: {trust_level:.2f}")

def main():
    st.title("🤖 Autonomous Multi-Agent Dialogue Simulator")
    st.markdown("*Simulate complex multi-agent conversations with AI entities that have memory, emotions, and distinct personalities.*")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Model loading
        if st.button("Load AI Model", type="primary"):
            if load_model():
                st.success("Model loaded successfully!")
            else:
                st.error("Failed to load model")
        
        if st.session_state.model_manager:
            st.success("✅ Model Ready")
            
            # Model info
            with st.expander("Model Information"):
                model_info = st.session_state.model_manager.get_model_info()
                st.json(model_info)
        else:
            st.warning("⚠️ Model not loaded")
        
        st.divider()
        
        # Scenario selection
        st.subheader("Scenario Setup")
        scenario_names = list(SCENARIOS.keys())
        selected_scenario = st.selectbox("Choose Scenario", scenario_names)
        
        if st.button("Set Scenario"):
            st.session_state.scenario_manager.set_scenario(selected_scenario)
            st.success(f"Scenario set: {selected_scenario}")
        
        # Display current scenario
        current_context = st.session_state.scenario_manager.get_current_context()
        if current_context:
            with st.expander("Current Scenario", expanded=True):
                st.write(f"**Phase {current_context['phase_number']}/{current_context['total_phases']}:** {current_context['current_phase']}")
                st.write(current_context['scenario_description'])
        
        st.divider()
        
        # Agent selection
        st.subheader("Agent Selection")
        available_agents = list(AGENT_CONFIGS.keys())
        
        if selected_scenario and selected_scenario in SCENARIOS:
            suggested_agents = SCENARIOS[selected_scenario].get('suggested_agents', [])
            st.write("**Suggested agents:**")
            for agent in suggested_agents:
                st.write(f"• {agent}")
        
        selected_agents = st.multiselect(
            "Select Agents (2-5 recommended)",
            available_agents,
            default=SCENARIOS[selected_scenario].get('suggested_agents', [])[:3] if selected_scenario else []
        )
        
        if st.button("Create Agents") and st.session_state.model_manager:
            st.session_state.agents = {}
            for agent_name in selected_agents:
                agent = create_agent(agent_name, AGENT_CONFIGS[agent_name])
                if agent:
                    st.session_state.agents[agent_name] = agent
            st.success(f"Created {len(st.session_state.agents)} agents")
        
        st.divider()
        
        # Simulation controls
        st.subheader("Simulation Controls")
        
        # Check if we have agents and either a scenario is set or we have a current context
        can_start_simulation = (
            st.session_state.agents and 
            len(st.session_state.agents) > 0 and
            st.session_state.model_manager is not None
        )
        
        current_context = st.session_state.scenario_manager.get_current_context()
        
        if can_start_simulation:
            if not st.session_state.simulation_running:
                # Show scenario info if available
                if current_context:
                    st.info(f"🎭 **Scenario**: {current_context.get('scenario_description', 'Custom scenario')}")
                else:
                    st.warning("⚠️ No scenario selected. You can still start a free-form conversation.")
                
                if st.button("Start Simulation", type="primary"):
                    st.session_state.simulation_running = True
                    st.session_state.turn_count = 0
                    # Add initial prompt
                    if current_context:
                        initial_prompt = current_context.get('initial_prompt', 'Let\'s begin our discussion.')
                    else:
                        initial_prompt = 'Let\'s begin our discussion.'
                    st.session_state.conversation_history.append({
                        'speaker': 'System',
                        'message': initial_prompt,
                        'timestamp': datetime.now(),
                        'turn': 0
                    })
                    st.rerun()
            else:
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Next Turn"):
                        simulate_turn()
                        st.rerun()
                with col2:
                    if st.button("Stop"):
                        st.session_state.simulation_running = False
                        st.rerun()
                
                # Only show phase advance if we have a scenario
                if current_context and current_context.get('current_phase'):
                    if st.button("Advance Phase"):
                        st.session_state.scenario_manager.advance_phase()
                        updated_context = st.session_state.scenario_manager.get_current_context()
                        phase_prompt = st.session_state.scenario_manager.get_phase_transition_prompt()
                        st.session_state.conversation_history.append({
                            'speaker': 'System',
                            'message': f"Phase Transition: {phase_prompt}",
                            'timestamp': datetime.now(),
                            'turn': st.session_state.turn_count
                        })
                        st.rerun()
        else:
            if not st.session_state.agents:
                st.info("👈 Please create agents first to start simulation.")
            elif not st.session_state.model_manager:
                st.warning("⚠️ Please load the AI model first.")
            
            # Debug information
            with st.expander("Debug Info"):
                st.write(f"Agents created: {len(st.session_state.agents) if st.session_state.agents else 0}")
                st.write(f"Model loaded: {st.session_state.model_manager is not None}")
                st.write(f"Current context: {bool(current_context)}")
                if current_context:
                    st.json(current_context)
        
        # Manual intervention
        if st.session_state.simulation_running:
            st.subheader("Manual Intervention")
            user_input = st.text_area("Add context or intervention:")
            if st.button("Send") and user_input:
                st.session_state.conversation_history.append({
                    'speaker': 'Moderator',
                    'message': user_input,
                    'timestamp': datetime.now(),
                    'turn': st.session_state.turn_count
                })
                st.rerun()
    
    # Main content area
    if not st.session_state.agents:
        st.info("👈 Please configure and create agents in the sidebar to begin.")
        
        # Show available scenarios and agents
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Available Scenarios")
            for name, scenario in SCENARIOS.items():
                with st.expander(name):
                    st.write(scenario['description'])
                    st.write(f"**Context:** {scenario['context']}")
                    st.write(f"**Phases:** {', '.join(scenario['phases'])}")
        
        with col2:
            st.subheader("Available Agents")
            for name, config in AGENT_CONFIGS.items():
                with st.expander(name):
                    st.write(f"**Role:** {config['role'].title()}")
                    st.write(f"**Background:** {config['background']}")
                    st.write(f"**Goals:** {', '.join(config['goals'])}")
    
    else:
        # Display agents
        st.header("Active Agents")
        
        agent_cols = st.columns(min(len(st.session_state.agents), 3))
        for i, (name, agent) in enumerate(st.session_state.agents.items()):
            with agent_cols[i % 3]:
                display_agent_card(agent)
        
        st.divider()
        
        # Conversation display
        st.header("Conversation")
        
        # Conversation history
        conversation_container = st.container()
        
        with conversation_container:
            if st.session_state.conversation_history:
                for entry in st.session_state.conversation_history[-20:]:  # Show last 20 messages
                    timestamp = entry['timestamp'].strftime("%H:%M:%S")
                    speaker = entry['speaker']
                    message = entry['message']
                    
                    if speaker == 'System':
                        st.info(f"**[{timestamp}] {speaker}:** {message}")
                    elif speaker == 'Moderator':
                        st.warning(f"**[{timestamp}] {speaker}:** {message}")
                    else:
                        # Get agent's current emotion for styling
                        if speaker in st.session_state.agents:
                            agent = st.session_state.agents[speaker]
                            emotion = agent.state.emotional_state.primary_emotion.value
                            emotion_icons = {
                                "happy": "😊", "confident": "😎", "neutral": "😐",
                                "frustrated": "😤", "angry": "😠", "sad": "😢",
                                "anxious": "😰", "suspicious": "🤨", "excited": "🤩"
                            }
                            icon = emotion_icons.get(emotion, "🤖")
                            st.markdown(f"**[{timestamp}] {icon} {speaker}:** {message}")
                        else:
                            st.markdown(f"**[{timestamp}] 🤖 {speaker}:** {message}")
            else:
                st.info("Conversation will appear here once the simulation starts.")
        
        # Analytics
        if st.session_state.conversation_history:
            st.header("Analytics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Participation chart
                speaker_counts = {}
                for entry in st.session_state.conversation_history:
                    speaker = entry['speaker']
                    if speaker not in ['System', 'Moderator']:
                        speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
                
                if speaker_counts:
                    fig = px.bar(
                        x=list(speaker_counts.keys()),
                        y=list(speaker_counts.values()),
                        title="Participation Count"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Emotional state timeline
                if st.session_state.agents:
                    emotions_data = []
                    for name, agent in st.session_state.agents.items():
                        emotions_data.append({
                            'Agent': name,
                            'Emotion': agent.state.emotional_state.primary_emotion.value,
                            'Intensity': agent.state.emotional_state.intensity
                        })
                    
                    if emotions_data:
                        df = pd.DataFrame(emotions_data)
                        fig = px.bar(
                            df,
                            x='Agent',
                            y='Intensity',
                            color='Emotion',
                            title="Current Emotional States"
                        )
                        st.plotly_chart(fig, use_container_width=True)

def simulate_turn():
    """Simulate one turn of conversation"""
    if not st.session_state.agents:
        return
    
    st.session_state.turn_count += 1
    
    # Choose next speaker (can be improved with more sophisticated logic)
    agent_names = list(st.session_state.agents.keys())
    current_speaker_name = agent_names[st.session_state.turn_count % len(agent_names)]
    current_speaker = st.session_state.agents[current_speaker_name]
    
    # Process recent messages for all agents
    recent_messages = st.session_state.conversation_history[-5:]
    for message_entry in recent_messages:
        for agent in st.session_state.agents.values():
            agent.process_message(
                message_entry['message'],
                message_entry['speaker'],
                context={'turn': message_entry.get('turn', 0)}
            )
    
    # Generate response from current speaker
    context = {
        'recent_messages': recent_messages,
        'scenario_phase': st.session_state.scenario_manager.get_current_context().get('current_phase', ''),
        'agent_emotions': {name: agent.state.emotional_state.primary_emotion.value 
                          for name, agent in st.session_state.agents.items()},
        'turn_count': st.session_state.turn_count
    }
    
    try:
        response = current_speaker.generate_response(context)
        
        # Add to conversation history
        st.session_state.conversation_history.append({
            'speaker': current_speaker_name,
            'message': response,
            'timestamp': datetime.now(),
            'turn': st.session_state.turn_count
        })
        
    except Exception as e:
        st.error(f"Error generating response from {current_speaker_name}: {e}")
        st.error(traceback.format_exc())

if __name__ == "__main__":
    main()
