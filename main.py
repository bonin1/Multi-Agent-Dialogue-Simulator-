import logging
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import traceback
import time

# Quiet "missing ScriptRunContext" when this file is executed with `python main.py`
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import plotly.graph_objects as go
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)

# Import custom modules (with fallback handling)
try:
    from models.model_manager import ModelManager
    from models.remote_llm import (
        AnthropicChatBackend,
        GroqChatBackend,
        OpenAIChatBackend,
        OpenRouterChatBackend,
    )
    from agents.agent import Agent
    from models.agent_models import AgentRole, PersonalityTrait, EmotionalState, EmotionType
    from scenarios.scenario_manager import ScenarioManager, AGENT_CONFIGS, SCENARIOS
    from ui.agent_creator_ui import show_agent_creator_ui
    from ui.sim_dashboard import (
        add_cost_from_last_response,
        append_emotion_timeline,
        init_extended_session_state,
        log_run,
        pick_next_speaker,
        render_accessibility_css,
        render_enhanced_analytics,
        render_lab_tab,
        render_library_tab,
        render_run_controls_sidebar,
        render_run_logs_drawer,
        render_settings_tab,
        render_transcript_tab,
    )
    from agents.agent_creator import AgentCreator
    from ui.research_ui import (
        render_research_tab,
        render_sidebar_agent_research,
        run_auto_research_for_start,
        get_research_brief,
        handle_moderator_research_command,
    )
    from ui.session_controls import render_reset_buttons
except ImportError as e:
    print(f"Error importing modules: {e}", flush=True)
    print("Install dependencies: pip install -r requirements.txt", flush=True)
    raise SystemExit(1)

if __name__ == "__main__":
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except ImportError:
        get_script_run_ctx = lambda: None  # type: ignore[misc, assignment]

    if get_script_run_ctx() is None:
        print(
            "\nThis app must be started with Streamlit (not `python main.py`):\n\n"
            "  streamlit run main.py\n",
            flush=True,
        )
        raise SystemExit(1)

# Page configuration
st.set_page_config(
    page_title="Multi-Agent Dialogue Simulator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

init_extended_session_state()
try:
    from ui.research_ui import init_research_session_state
    init_research_session_state()
except ImportError:
    pass
render_accessibility_css()

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

def load_model():
    """Load local HF model or configure a remote API backend (BYOK)."""
    import os

    try:
        try:
            secrets = getattr(st, "secrets", None)
            if secrets and "HUGGING_FACE_HUB_TOKEN" in secrets:
                os.environ["HUGGING_FACE_HUB_TOKEN"] = secrets["HUGGING_FACE_HUB_TOKEN"]
        except Exception:
            pass

        hf_ui = (st.session_state.get("huggingface_token") or "").strip()
        if hf_ui:
            os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_ui

        backend = st.session_state.get("llm_backend", "local")

        if backend == "openai":
            key = (st.session_state.get("openai_api_key") or os.environ.get("OPENAI_API_KEY") or "").strip()
            if not key:
                st.error("OpenAI API key required (Settings tab or OPENAI_API_KEY env).")
                return False
            with st.spinner("Connecting to OpenAI…"):
                st.session_state.model_manager = OpenAIChatBackend(
                    key, st.session_state.get("openai_model", "gpt-4o-mini")
                )
            log_run("Model backend: OpenAI")
            return True

        if backend == "anthropic":
            key = (st.session_state.get("anthropic_api_key") or os.environ.get("ANTHROPIC_API_KEY") or "").strip()
            if not key:
                st.error("Anthropic API key required (Settings tab or ANTHROPIC_API_KEY env).")
                return False
            with st.spinner("Connecting to Anthropic…"):
                st.session_state.model_manager = AnthropicChatBackend(
                    key, st.session_state.get("anthropic_model", "claude-3-5-haiku-20241022")
                )
            log_run("Model backend: Anthropic")
            return True

        if backend == "openrouter":
            key = (
                st.session_state.get("openrouter_api_key")
                or os.environ.get("OPENROUTER_API_KEY")
                or ""
            ).strip()
            if not key:
                st.error("OpenRouter API key required (Settings tab or OPENROUTER_API_KEY env).")
                return False
            with st.spinner("Connecting to OpenRouter…"):
                st.session_state.model_manager = OpenRouterChatBackend(
                    key,
                    st.session_state.get("openrouter_model", "openai/gpt-4o-mini"),
                    site_url=st.session_state.get("openrouter_site_url", ""),
                    site_name=st.session_state.get("openrouter_site_name", "Multi-Agent Dialogue Simulator"),
                )
            log_run("Model backend: OpenRouter")
            return True

        if backend == "groq":
            key = (st.session_state.get("groq_api_key") or os.environ.get("GROQ_API_KEY") or "").strip()
            if not key:
                st.error("Groq API key required (Settings tab or GROQ_API_KEY env).")
                return False
            with st.spinner("Connecting to Groq…"):
                st.session_state.model_manager = GroqChatBackend(
                    key, st.session_state.get("groq_model", "llama-3.3-70b-versatile")
                )
            log_run("Model backend: Groq")
            return True

        with st.spinner("Loading AI model… This may take a few minutes."):
            model_id = st.session_state.get("local_model_name", "teknium/OpenHermes-2.5-Mistral-7B")
            st.session_state.model_manager = ModelManager(model_id)
        log_run(f"Model backend: local HF ({model_id})")
        return True
    except Exception as e:
        st.error(f"Error loading model: {e}")
        log_run(str(e), "error")
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

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
        [
            "💬 Simulation",
            "🛠️ Create Agents",
            "📊 Analytics",
            "🌐 Live Research",
            "📜 Transcript",
            "🧪 Lab",
            "📚 Library",
            "⚙️ Settings",
        ]
    )

    with tab1:
        show_simulation_interface()

    with tab2:
        show_agent_creator_interface()

    with tab3:
        show_analytics_interface()

    with tab4:
        render_research_tab()

    with tab5:
        render_transcript_tab()

    with tab6:
        render_lab_tab(st.session_state.scenario_manager)

    with tab7:
        render_library_tab()

    with tab8:
        render_settings_tab()

def show_simulation_interface():
    """Show the main simulation interface"""
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Model loading
        if st.button("Load AI Model", type="primary"):
            st.session_state.model_manager = None
            if load_model():
                st.success("Model / API ready!")
            else:
                st.error("Failed to load model")
        
        if st.session_state.model_manager:
            st.success("✅ Model Ready")
            
            # Model info
            with st.expander("Model Information"):
                model_info = st.session_state.model_manager.get_model_info()
                st.json(model_info)

            last = getattr(st.session_state.model_manager, "last_stats", None) or {}
            if last:
                with st.expander("Last generation (technical)"):
                    st.write(
                        f"**Latency:** {last.get('latency_s', 0):.2f}s"
                        if last.get("latency_s") is not None
                        else "**Latency:** —"
                    )
                    if last.get("input_tokens") is not None:
                        st.write(f"**Tokens:** {last['input_tokens']} in / {last.get('output_tokens', 0)} out")
                    if last.get("error"):
                        st.error(last["error"])
        else:
            st.warning("⚠️ Model not loaded")
        
        rb = get_research_brief()
        if rb:
            st.caption(f"🌐 Research loaded: {rb.topic}")

        render_reset_buttons()

        st.divider()
        
        # Scenario selection
        st.subheader("Scenario Setup")
        NO_SCENARIO_LABEL = "— No scenario (free-form + auto research) —"
        scenario_names = [NO_SCENARIO_LABEL] + list(
            st.session_state.scenario_manager.get_available_scenarios().keys()
        )
        selected_scenario = st.selectbox("Choose Scenario", scenario_names)
        
        if st.button("Apply scenario"):
            if selected_scenario == NO_SCENARIO_LABEL:
                st.session_state.scenario_manager.clear_scenario()
                st.success("Free-form mode — no scripted scenario. Set a topic under Live Research or below.")
            else:
                st.session_state.scenario_manager.set_scenario(selected_scenario)
                st.success(f"Scenario set: {selected_scenario}")

        st.session_state.freeform_topic = st.text_input(
            "Free-form topic (no scenario)",
            value=st.session_state.get("freeform_topic", ""),
            help="Used for auto web research when starting without a scenario.",
        )
        
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
        
        all_scenarios = st.session_state.scenario_manager.get_available_scenarios()
        if selected_scenario != NO_SCENARIO_LABEL and selected_scenario in all_scenarios:
            suggested_agents = all_scenarios[selected_scenario].get('suggested_agents', [])
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
                default=all_scenarios[selected_scenario].get('suggested_agents', [])[:3] if selected_scenario in all_scenarios else []
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
                snapshots = {}
                for name, ag in st.session_state.agents.items():
                    summ = ag.get_agent_summary()
                    snapshots[name] = {
                        "personality": summ.get("personality"),
                        "role": summ.get("role"),
                        "goals": summ.get("goals", []),
                    }
                st.session_state.agent_persona_snapshots = snapshots
            else:
                st.error("❌ No agents were created successfully")
        
        if st.session_state.agents:
            st.divider()
            render_run_controls_sidebar()
            st.subheader("Turn-taking")
            st.session_state.next_speaker_mode = st.selectbox(
                "Next speaker mode",
                ["round_robin", "random", "manual", "reply_chain", "balance"],
                help="Who speaks next on each agent turn.",
            )
            if st.session_state.next_speaker_mode == "manual":
                st.session_state.manual_next_speaker = st.selectbox(
                    "Manual: who speaks",
                    list(st.session_state.agents.keys()),
                )
            st.session_state.muted_agents = st.multiselect(
                "Muted agents (excluded from rotation)",
                list(st.session_state.agents.keys()),
                default=[m for m in st.session_state.get("muted_agents", []) if m in st.session_state.agents],
            )
            solo_opts = ["— none —"] + list(st.session_state.agents.keys())
            cur_solo = st.session_state.get("solo_agent")
            solo_index = solo_opts.index(cur_solo) if cur_solo in solo_opts else 0
            st.session_state.solo_agent = st.selectbox("Solo mode (only this agent speaks)", solo_opts, index=solo_index)

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
                    st.info(
                        "🌐 **Free-form mode** — no scenario. On start, agents will "
                        "**research the web** using your free-form topic or Research tab input."
                    )
                
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
                    import random

                    st.session_state.emotion_timeline = []
                    if st.session_state.get("use_fixed_seed"):
                        seed_v = int(st.session_state.get("run_seed_value", 42))
                    else:
                        seed_v = random.randint(1, 2**31 - 1)
                    st.session_state.run_seed = seed_v
                    random.seed(seed_v)
                    log_run(
                        f"Simulation start seed={seed_v} label={st.session_state.get('run_label', '')!r}"
                    )
                    st.session_state.simulation_running = True
                    st.session_state.turn_count = 0
                    # Add initial prompt
                    if current_context:
                        initial_prompt = current_context.get('initial_prompt', 'Let\'s begin our discussion.')
                    else:
                        ft = (st.session_state.get("freeform_topic") or "").strip()
                        initial_prompt = (
                            f"Free-form discussion{f' about: {ft}' if ft else ''}. "
                            "Use the live research brief — react to real facts, stay concise."
                        )
                    st.session_state.conversation_history.append({
                        'speaker': 'System',
                        'message': initial_prompt,
                        'timestamp': datetime.now(),
                        'turn': 0
                    })
                    # Live research before debate
                    try:
                        need_research = (
                            not current_context
                            and st.session_state.get("auto_research_no_scenario", True)
                        ) or st.session_state.get("auto_research_on_start")
                        if need_research:
                            topic = (
                                st.session_state.get("research_topic")
                                or st.session_state.get("freeform_topic")
                                or ""
                            ).strip()
                            if not topic and not current_context and not st.session_state.get("research_brief"):
                                st.sidebar.warning(
                                    "No topic set — add **Free-form topic** or paste a URL in **Live Research**."
                                )
                            elif run_auto_research_for_start(
                                freeform_topic=topic, force=not current_context
                            ):
                                log_run(
                                    f"Auto-research on start: "
                                    f"{st.session_state.research_brief.topic[:80]}"
                                )
                    except Exception as ex:
                        log_run(f"Auto-research failed: {ex}", "error")
                        st.sidebar.error(f"Research failed: {ex}")
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
        
        if st.session_state.simulation_running:
            render_sidebar_agent_research()

        # Manual intervention
        if st.session_state.simulation_running:
            st.subheader("Manual Intervention")
            user_input = st.text_area("Add context or intervention:")
            if st.button("Send") and user_input:
                if handle_moderator_research_command(user_input):
                    st.session_state.conversation_history.append({
                        'speaker': 'Moderator',
                        'message': user_input,
                        'timestamp': datetime.now(),
                        'turn': st.session_state.turn_count
                    })
                    st.success("Agents are researching that topic…")
                else:
                    st.session_state.conversation_history.append({
                        'speaker': 'Moderator',
                        'message': user_input,
                        'timestamp': datetime.now(),
                        'turn': st.session_state.turn_count
                    })
                st.rerun()
        
        render_run_logs_drawer()
    
    # Main content area
    if not st.session_state.agents:
        st.info("👈 Please configure and create agents in the sidebar to begin.")
        
        # Show available scenarios and agents
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Available Scenarios")
            for name, scenario in st.session_state.scenario_manager.get_available_scenarios().items():
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
        
        rb = get_research_brief()
        if rb:
            title = rb.topic if not rb.topic.startswith("http") else (
                rb.article_hit.title if rb.article_hit else "Live article"
            )
            st.info(f"🌐 **Live research active:** {title}")

        # Conversation display
        st.header("Conversation")
        
        # Conversation history
        conversation_container = st.container()
        
        with conversation_container:
            if st.session_state.conversation_history:
                for entry in st.session_state.conversation_history[-20:]:  # Show last 20 messages
                    timestamp = entry['timestamp'].strftime("%H:%M:%S")
                    speaker = entry['speaker']
                    message = entry.get('message', '')
                    try:
                        from utils.response_cleaner import clean_dialogue_text
                        message = clean_dialogue_text(message)
                    except ImportError:
                        pass
                    
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

    agent_names = list(st.session_state.agents.keys())
    mode = st.session_state.get("next_speaker_mode", "round_robin")
    manual = st.session_state.get("manual_next_speaker")
    muted = st.session_state.get("muted_agents") or []
    solo_raw = st.session_state.get("solo_agent")
    solo = solo_raw if solo_raw and solo_raw != "— none —" else None

    current_speaker_name = pick_next_speaker(
        agent_names,
        st.session_state.turn_count,
        st.session_state.conversation_history,
        mode,
        manual,
        muted,
        solo,
    )
    current_speaker = st.session_state.agents[current_speaker_name]

    recent_messages = st.session_state.conversation_history[-5:]
    for message_entry in recent_messages:
        for agent in st.session_state.agents.values():
            agent.process_message(
                message_entry["message"],
                message_entry["speaker"],
                context={"turn": message_entry.get("turn", 0)},
            )

    last_msg = recent_messages[-1]["message"] if recent_messages else ""
    last_spk = recent_messages[-1]["speaker"] if recent_messages else ""

    ctx = st.session_state.scenario_manager.get_current_context() or {}
    context = {
        "recent_messages": recent_messages,
        "conversation_flow": True,
        "last_speaker": last_spk,
        "last_message": last_msg,
        "scenario_phase": ctx.get("current_phase", ""),
        "agent_emotions": {
            name: agent.state.emotional_state.primary_emotion.value
            for name, agent in st.session_state.agents.items()
        },
        "turn_count": st.session_state.turn_count,
        "prompt_lab_system": st.session_state.get("prompt_lab_system", ""),
        "prompt_lab_style": st.session_state.get("prompt_lab_style", ""),
    }
    brief = get_research_brief()
    if brief is not None:
        context["research_brief"] = brief.to_context_block(max_chars=1800)

    try:
        response = current_speaker.generate_response(context)

        entry = {
            "speaker": current_speaker_name,
            "message": response,
            "timestamp": datetime.now(),
            "turn": st.session_state.turn_count,
        }
        mm = st.session_state.model_manager
        if mm is not None and hasattr(mm, "last_stats"):
            stats = getattr(mm, "last_stats", {}) or {}
            if stats.get("latency_s") is not None:
                entry["latency_s"] = stats["latency_s"]
            if stats.get("input_tokens") is not None:
                entry["input_tokens"] = stats["input_tokens"]
            if stats.get("output_tokens") is not None:
                entry["output_tokens"] = stats["output_tokens"]
            if hasattr(mm, "get_model_info"):
                info = mm.get_model_info() or {}
                mid = info.get("model_name")
                if mid:
                    entry["model_id"] = mid

        st.session_state.conversation_history.append(entry)
        add_cost_from_last_response()
        append_emotion_timeline(st.session_state.turn_count, st.session_state.agents)

    except Exception as e:
        st.error(f"Error generating response from {current_speaker_name}: {e}")
        st.error(traceback.format_exc())
        log_run(f"generate_response: {e}", "error")

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
    
    st.metric("Approx. session API cost (USD)", f"{st.session_state.get('total_cost_usd_session', 0.0):.5f}")
    
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

    st.divider()
    render_enhanced_analytics()


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
        interval = float(st.session_state.get('auto_turn_interval', 3))
        if st.session_state.get('reduce_motion'):
            interval = max(interval, 5.0)
        time.sleep(interval)
        simulate_turn()
        st.rerun()