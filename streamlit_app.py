"""
Streamlit UI for the Multi-Agent Dialogue Simulator
"""
import streamlit as st
import asyncio
import sys
from pathlib import Path
import json
from datetime import datetime
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.simulation.dialogue_manager import DialogueManager
from src.simulation.scenarios import ScenarioManager
from src.agents.personality import PersonalityGenerator

st.set_page_config(
    page_title="Multi-Agent Dialogue Simulator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .agent-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    .message-box {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        margin-bottom: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .scenario-box {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def init_session_state():
    """Initialize session state variables"""
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'agents' not in st.session_state:
        st.session_state.agents = {}
    if 'dialogue_manager' not in st.session_state:
        st.session_state.dialogue_manager = None
    if 'scenario_info' not in st.session_state:
        st.session_state.scenario_info = None

def render_agent_card(agent):
    """Render an agent card with personality info"""
    with st.container():
        st.markdown(f"""
        <div class="agent-card">
            <h4>🤖 {agent.name}</h4>
            <p><strong>Role:</strong> {agent.role.title()}</p>
            <p><strong>Personality:</strong> {agent.personality.name if hasattr(agent.personality, 'name') else 'Generated'}</p>
        </div>
        """, unsafe_allow_html=True)

def render_message(message, sender_type="agent"):
    """Render a conversation message"""
    icon = "🤖" if sender_type == "agent" else "🔧"
    timestamp = message.get('timestamp', datetime.now()).strftime("%H:%M:%S")
    
    st.markdown(f"""
    <div class="message-box">
        <strong>{icon} {message['speaker']}</strong> <small style="color: #666;">({timestamp})</small>
        <p style="margin: 0.5rem 0 0 0;">{message['content']}</p>
    </div>
    """, unsafe_allow_html=True)

def render_scenario_info(scenario_info):
    """Render scenario information"""
    st.markdown(f"""
    <div class="scenario-box">
        <h4>📋 Scenario: {scenario_info['name']}</h4>
        <p><strong>Description:</strong> {scenario_info['description']}</p>
        <p><strong>Context:</strong> {scenario_info['context']}</p>
        <p><strong>Goals:</strong> {', '.join(scenario_info['goals'])}</p>
        <p><strong>Duration:</strong> {scenario_info.get('duration_estimate', 'N/A')}</p>
    </div>
    """, unsafe_allow_html=True)

async def run_conversation(dialogue_manager, max_turns):
    """Run the conversation asynchronously"""
    try:
        results = await dialogue_manager.simulate_conversation(max_turns=max_turns)
        return results, dialogue_manager.get_conversation_summary()
    except Exception as e:
        st.error(f"Error during conversation: {str(e)}")
        return [], {}

def main():
    """Main Streamlit application"""
    init_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Multi-Agent Dialogue Simulator</h1>
        <p>Create and observe conversations between AI agents with distinct personalities</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Scenario selection
        st.subheader("📋 Scenario")
        scenario_manager = ScenarioManager()
        available_scenarios = list(scenario_manager.scenarios.keys())
        selected_scenario = st.selectbox(
            "Choose a scenario:",
            available_scenarios,
            format_func=lambda x: scenario_manager.scenarios[x]['name']
        )
        
        # Agent configuration
        st.subheader("🤖 Agents")
        num_agents = st.slider("Number of agents:", 2, 5, 3)
        
        # Conversation settings
        st.subheader("💬 Conversation")
        max_turns = st.slider("Maximum turns:", 5, 50, 10)
        
        # Start conversation button
        start_conversation = st.button("🚀 Start Conversation", type="primary")
        
        # Clear conversation button
        if st.button("🗑️ Clear History"):
            st.session_state.conversation_history = []
            st.session_state.agents = {}
            st.session_state.dialogue_manager = None
            st.session_state.scenario_info = None
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Conversation")
        
        if start_conversation:
            # Create dialogue manager
            with st.spinner("Setting up agents and scenario..."):
                try:
                    dialogue_manager = DialogueManager(
                        num_agents=num_agents,
                        scenario=selected_scenario
                    )
                    
                    st.session_state.dialogue_manager = dialogue_manager
                    st.session_state.agents = dialogue_manager.agents
                    st.session_state.scenario_info = scenario_manager.scenarios[selected_scenario]
                    
                    st.success("✅ Agents created successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error creating agents: {str(e)}")
                    st.stop()
            
            # Run conversation
            with st.spinner("Running conversation..."):
                try:
                    # Run the async conversation
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    results, summary = loop.run_until_complete(
                        run_conversation(st.session_state.dialogue_manager, max_turns)
                    )
                    loop.close()
                    
                    st.session_state.conversation_history = results
                    
                    if results:
                        st.success(f"✅ Conversation completed! {len(results)} turns generated.")
                    else:
                        st.warning("⚠️ No conversation turns were generated.")
                        
                except Exception as e:
                    st.error(f"❌ Error running conversation: {str(e)}")
        
        # Display conversation history
        if st.session_state.conversation_history:
            st.subheader("📜 Conversation History")
            
            for message in st.session_state.conversation_history:
                sender_type = "system" if message['speaker'] == "System" else "agent"
                render_message(message, sender_type)
            
            # Conversation summary
            if st.session_state.dialogue_manager:
                summary = st.session_state.dialogue_manager.get_conversation_summary()
                st.subheader("📊 Summary")
                
                col1_summary, col2_summary, col3_summary = st.columns(3)
                with col1_summary:
                    st.metric("Total Turns", summary.get('total_turns', 0))
                with col2_summary:
                    st.metric("Total Words", summary.get('total_words', 0))
                with col3_summary:
                    avg_words = summary.get('total_words', 0) / max(summary.get('total_turns', 1), 1)
                    st.metric("Avg Words/Turn", f"{avg_words:.1f}")
                
                # Export conversation
                if st.button("📥 Download Conversation"):
                    conversation_data = {
                        "scenario": st.session_state.scenario_info,
                        "agents": {aid: {
                            "name": agent.name,
                            "role": agent.role,
                            "personality": agent.personality.model_dump() if hasattr(agent.personality, 'model_dump') else str(agent.personality)
                        } for aid, agent in st.session_state.agents.items()},
                        "conversation": st.session_state.conversation_history,
                        "summary": summary,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    st.download_button(
                        label="Download as JSON",
                        data=json.dumps(conversation_data, indent=2, default=str),
                        file_name=f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
        
        else:
            st.info("👆 Configure settings in the sidebar and click 'Start Conversation' to begin!")
    
    with col2:
        st.header("🎭 Participants")
        
        # Display scenario info
        if st.session_state.scenario_info:
            render_scenario_info(st.session_state.scenario_info)
        
        # Display agents
        if st.session_state.agents:
            st.subheader("🤖 Active Agents")
            for agent in st.session_state.agents.values():
                render_agent_card(agent)
        else:
            st.info("No agents created yet. Start a conversation to see them here!")
        
        # Model status
        st.subheader("🔧 System Status")
        try:
            from src.utils.model_manager import get_model_manager
            model_manager = get_model_manager()
            
            if model_manager.is_loaded:
                st.success("✅ Language Model: Loaded")
                st.info(f"Model: {model_manager.model_name}")
            else:
                st.warning("⚠️ Language Model: Not loaded")
        except Exception as e:
            st.error(f"❌ Model Manager Error: {str(e)}")

if __name__ == "__main__":
    main()
