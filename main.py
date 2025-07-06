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
    from ui.agent_creator_ui import show_agent_creator_ui
    from agents.agent_creator import AgentCreator
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
if 'continuous_mode' not in st.session_state:
    st.session_state.continuous_mode = False
if 'turn_count' not in st.session_state:
    st.session_state.turn_count = 0
if 'agent_creator' not in st.session_state:
    st.session_state.agent_creator = AgentCreator()
if 'auto_turn_interval' not in st.session_state:
    st.session_state.auto_turn_interval = 3
if 'auto_turn_interval' not in st.session_state:
    st.session_state.auto_turn_interval = 3

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
    
    # Main navigation
    tab1, tab2, tab3 = st.tabs(["💬 Simulation", "🛠️ Create Agents", "📊 Analytics"])
    
    with tab1:
        show_simulation_interface()
    
    with tab2:
        show_agent_creator_interface()
    
    with tab3:
        show_analytics_interface()

def show_simulation_interface():
    """Show the main simulation interface"""
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
        
        # Get all available agents (predefined + custom)
        all_available_agents = get_all_available_agents()
        available_agent_names = list(all_available_agents.keys())
        
        # Separate predefined and custom agents for display
        predefined_agents = [name for name in available_agent_names if name in AGENT_CONFIGS]
        custom_agents = [name for name in available_agent_names if name not in AGENT_CONFIGS]
        
        if custom_agents:
            st.write(f"**Available agents:** {len(predefined_agents)} predefined + {len(custom_agents)} custom")
        else:
            st.write(f"**Available agents:** {len(predefined_agents)} predefined")
            st.info("💡 Create custom agents in the 'Create Agents' tab!")
        
        if selected_scenario and selected_scenario in SCENARIOS:
            suggested_agents = SCENARIOS[selected_scenario].get('suggested_agents', [])
            st.write("**Suggested agents for this scenario:**")
            for agent in suggested_agents:
                agent_type = "predefined" if agent in AGENT_CONFIGS else "custom" if agent in all_available_agents else "missing"
                if agent_type == "missing":
                    st.write(f"• ⚠️ {agent} (not available)")
                else:
                    st.write(f"• {agent} ({agent_type})")
        
        # Agent selection with better organization
        if custom_agents:
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Predefined Agents:**")
                selected_predefined = st.multiselect(
                    "Choose predefined agents",
                    predefined_agents,
                    default=[],
                    key="predefined_agents"
                )
            with col2:
                st.write("**Your Custom Agents:**")
                selected_custom = st.multiselect(
                    "Choose custom agents",
                    custom_agents,
                    default=[],
                    key="custom_agents"
                )
            selected_agents = selected_predefined + selected_custom
        else:
            selected_agents = st.multiselect(
                "Select Agents (2-5 recommended)",
                available_agent_names,
                default=SCENARIOS[selected_scenario].get('suggested_agents', [])[:3] if selected_scenario else []
            )
        
        # Show selected agents info
        if selected_agents:
            st.write(f"**Selected:** {len(selected_agents)} agents")
            for agent_name in selected_agents:
                agent_config = all_available_agents[agent_name]
                agent_type = "🎭 Custom" if 'custom_config' in agent_config else "🤖 Predefined"
                
                # Extract role properly for both custom and predefined agents
                if 'custom_config' in agent_config:
                    # Custom agent - get role from the CustomAgentConfig object
                    role = agent_config['custom_config'].role
                else:
                    # Predefined agent - get role from dictionary
                    role = agent_config.get('role', 'Unknown')
                
                st.write(f"  • {agent_type} {agent_name} ({role})")
        
        if st.button("Create Agents") and st.session_state.model_manager and selected_agents:
            st.session_state.agents = {}
            success_count = 0
            
            with st.spinner("Creating agents..."):
                for agent_name in selected_agents:
                    agent_config = all_available_agents[agent_name]
                    agent = create_agent_from_config(agent_name, agent_config)
                    if agent:
                        st.session_state.agents[agent_name] = agent
                        success_count += 1
                    else:
                        st.error(f"Failed to create agent: {agent_name}")
            
            if success_count > 0:
                st.success(f"✅ Created {success_count} agents successfully!")
            else:
                st.error("❌ No agents were created successfully")
        
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
                
                # Simulation mode selection
                st.subheader("Simulation Mode")
                mode = st.radio(
                    "Choose simulation mode:",
                    ["Manual Turn-by-Turn", "Continuous Auto-Play"],
                    help="Manual: Click 'Next Turn' for each response. Continuous: Agents talk automatically."
                )
                
                if mode == "Continuous Auto-Play":
                    st.session_state.continuous_mode = True
                    st.session_state.auto_turn_interval = st.slider(
                        "Seconds between turns:", 
                        min_value=1, 
                        max_value=10, 
                        value=3,
                        help="How long to wait between agent responses"
                    )
                else:
                    st.session_state.continuous_mode = False
                
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
                # Show current mode
                mode_text = "🔄 Continuous Mode" if st.session_state.continuous_mode else "👆 Manual Mode"
                st.info(f"**Active Mode:** {mode_text}")
                
                if st.session_state.continuous_mode:
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("⏸️ Pause", type="secondary"):
                            st.session_state.continuous_mode = False
                            st.rerun()
                    with col2:
                        if st.button("🛑 Stop Simulation", type="primary"):
                            st.session_state.simulation_running = False
                            st.session_state.continuous_mode = False
                            st.rerun()
                    
                    st.write(f"⏱️ Auto-turn every {st.session_state.auto_turn_interval} seconds")
                    
                else:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button("▶️ Next Turn"):
                            simulate_turn()
                            st.rerun()
                    with col2:
                        if st.button("🔄 Resume Auto", type="secondary"):
                            st.session_state.continuous_mode = True
                            st.rerun()
                    with col3:
                        if st.button("🛑 Stop"):
                            st.session_state.simulation_running = False
                            st.rerun()
                
                # Only show phase advance if we have a scenario
                if current_context and current_context.get('current_phase'):
                    if st.button("⏭️ Advance Phase"):
                        st.session_state.scenario_manager.advance_phase()
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

def show_agent_creator_interface():
    """Show the agent creator interface"""
    try:
        show_agent_creator_ui()
    except Exception as e:
        st.error(f"Error in agent creator: {e}")
        st.write("Fallback: Basic agent creator")
        show_basic_agent_creator()

def show_basic_agent_creator():
    """Basic agent creator as fallback"""
    st.header("🛠️ Custom Agent Creator")
    st.info("Create your own custom agents with unique personalities!")
    
    with st.form("basic_agent_creator"):
        col1, col2 = st.columns(2)
        
        with col1:
            agent_name = st.text_input("Agent Name", placeholder="e.g., Dr. Alex Smith")
            agent_role = st.selectbox("Role", ["custom", "doctor", "engineer", "scientist", "teacher", "journalist"])
            agent_description = st.text_area("Description", placeholder="Describe your agent...")
        
        with col2:
            st.subheader("Personality Traits")
            openness = st.slider("Openness", 0.0, 1.0, 0.5)
            conscientiousness = st.slider("Conscientiousness", 0.0, 1.0, 0.5)
            extraversion = st.slider("Extraversion", 0.0, 1.0, 0.5)
            agreeableness = st.slider("Agreeableness", 0.0, 1.0, 0.5)
            neuroticism = st.slider("Neuroticism", 0.0, 1.0, 0.5)
        
        background_story = st.text_area("Background Story", placeholder="Tell us about your agent's history...")
        goals = st.text_area("Goals (one per line)", placeholder="Help people\nSolve problems\nLearn new things")
        
        if st.form_submit_button("Create Agent"):
            if agent_name and agent_description:
                # Create basic agent config
                try:
                    if 'custom_agents' not in st.session_state:
                        st.session_state.custom_agents = {}
                    
                    config = {
                        'name': agent_name,
                        'role': agent_role,
                        'description': agent_description,
                        'background': background_story,
                        'personality': {
                            'openness': openness,
                            'conscientiousness': conscientiousness,
                            'extraversion': extraversion,
                            'agreeableness': agreeableness,
                            'neuroticism': neuroticism
                        },
                        'goals': [goal.strip() for goal in goals.split('\n') if goal.strip()]
                    }
                    
                    st.session_state.custom_agents[agent_name] = config
                    st.success(f"✅ Created agent: {agent_name}")
                    
                except Exception as e:
                    st.error(f"Error creating agent: {e}")
            else:
                st.error("Please fill in at least the name and description")
    
    # Show created agents
    if 'custom_agents' in st.session_state and st.session_state.custom_agents:
        st.subheader("📚 Your Custom Agents")
        for name, config in st.session_state.custom_agents.items():
            with st.expander(f"🤖 {name}"):
                st.write(f"**Role:** {config['role']}")
                st.write(f"**Description:** {config['description']}")
                if config.get('goals'):
                    st.write("**Goals:**")
                    for goal in config['goals']:
                        st.write(f"• {goal}")

def show_analytics_interface():
    """Show conversation analytics"""
    st.header("📊 Conversation Analytics")
    
    if not st.session_state.conversation_history:
        st.info("No conversation data available. Start a simulation to see analytics!")
        return
    
    # Basic analytics
    df = pd.DataFrame(st.session_state.conversation_history)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Messages", len(df))
    
    with col2:
        unique_speakers = df['speaker'].nunique()
        st.metric("Active Agents", unique_speakers)
    
    with col3:
        if len(df) > 0:
            avg_length = df['message'].str.len().mean()
            st.metric("Avg Message Length", f"{avg_length:.0f} chars")
    
    # Speaker distribution
    if len(df) > 0:
        speaker_counts = df['speaker'].value_counts()
        
        fig = px.bar(
            x=speaker_counts.index, 
            y=speaker_counts.values,
            labels={'x': 'Agent', 'y': 'Message Count'},
            title="Messages per Agent"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Recent conversation timeline
        if len(df) > 10:
            recent_df = df.tail(20)
            fig2 = px.timeline(
                recent_df, 
                x_start="timestamp", 
                x_end="timestamp",
                y="speaker",
                title="Recent Conversation Timeline"
            )
            st.plotly_chart(fig2, use_container_width=True)
    
    # Agent emotional states
    if st.session_state.agents:
        st.subheader("🎭 Current Agent States")
        
        for agent_name, agent in st.session_state.agents.items():
            with st.expander(f"🤖 {agent_name}"):
                summary = agent.get_agent_summary()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Emotion:** {summary['emotional_state']['emotion']}")
                    st.write(f"**Intensity:** {summary['emotional_state']['intensity']:.2f}")
                
                with col2:
                    if summary.get('recent_reflections'):
                        st.write("**Recent Thoughts:**")
                        for reflection in summary['recent_reflections']:
                            st.write(f"• {reflection}")

def get_all_available_agents():
    """Get all available agents (predefined + custom)"""
    all_agents = {}
    
    # Add predefined agents
    all_agents.update(AGENT_CONFIGS)
    
    # Add custom agents
    try:
        creator = st.session_state.agent_creator
        custom_agents = creator.list_saved_agents()
        
        for custom_agent in custom_agents:
            try:
                config = creator.load_agent_config(custom_agent['filename'])
                # Convert custom agent config to format compatible with create_agent
                agent_config = {
                    'role': config.role,
                    'personality': config.personality_traits,
                    'background': config.background_story,
                    'goals': config.goals,
                    'custom_config': config  # Store full config for custom agents
                }
                all_agents[config.name] = agent_config
            except Exception as e:
                st.warning(f"Could not load custom agent {custom_agent['name']}: {e}")
    except Exception as e:
        st.error(f"Error loading custom agents: {e}")
    
    return all_agents

def create_agent_from_config(name: str, config: Dict[str, Any]) -> Optional[Agent]:
    """Create an agent from either predefined or custom configuration"""
    try:
        # Check if this is a custom agent
        if 'custom_config' in config:
            custom_config = config['custom_config']
            return st.session_state.agent_creator.create_custom_agent(
                custom_config, 
                st.session_state.model_manager
            )
        else:
            # Use existing create_agent function for predefined agents
            return create_agent(name, config)
    except Exception as e:
        st.error(f"Error creating agent {name}: {e}")
        return None

if __name__ == "__main__":
    main()
    
    # Auto-refresh for continuous mode
    if (st.session_state.get('simulation_running', False) and 
        st.session_state.get('continuous_mode', False)):
        time.sleep(st.session_state.get('auto_turn_interval', 3))
        simulate_turn()
        st.rerun()