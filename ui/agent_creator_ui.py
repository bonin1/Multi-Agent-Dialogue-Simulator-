"""
Streamlit UI for Custom Agent Creation
User-friendly interface for creating custom agents
"""

import streamlit as st
import json
from typing import Dict, List, Any
from agents.agent_creator import AgentCreator, CustomAgentConfig
from datetime import datetime
import os

def show_agent_creator_ui():
    """Main UI for agent creation"""
    st.title("🤖 Custom Agent Creator")
    st.markdown("Create your own AI agents with unique personalities and behaviors!")
    
    # Initialize agent creator
    if 'agent_creator' not in st.session_state:
        st.session_state.agent_creator = AgentCreator()
    
    creator = st.session_state.agent_creator
    
    # Sidebar for navigation
    st.sidebar.title("Agent Creator")
    
    page = st.sidebar.selectbox(
        "Choose Action",
        ["Create New Agent", "Browse Templates", "Manage Saved Agents", "Import/Export"]
    )
    
    if page == "Create New Agent":
        show_agent_creation_wizard(creator)
    elif page == "Browse Templates":
        show_template_browser(creator)
    elif page == "Manage Saved Agents":
        show_saved_agents_manager(creator)
    elif page == "Import/Export":
        show_import_export_ui(creator)

def show_agent_creation_wizard(creator: AgentCreator):
    """Step-by-step agent creation wizard"""
    st.header("🎭 Agent Creation Wizard")
    
    # Initialize wizard state
    if 'wizard_step' not in st.session_state:
        st.session_state.wizard_step = 1
    if 'agent_config' not in st.session_state:
        st.session_state.agent_config = {}
    
    # Get wizard steps
    steps = creator.get_agent_creation_wizard_steps()
    current_step = st.session_state.wizard_step
    
    # Progress bar
    progress = current_step / len(steps)
    st.progress(progress)
    st.write(f"Step {current_step} of {len(steps)}")
    
    # Current step info
    step_info = steps[current_step - 1]
    st.subheader(f"📋 {step_info['title']}")
    st.write(step_info['description'])
    
    # Step content
    if current_step == 1:
        show_basic_info_step()
    elif current_step == 2:
        show_personality_step(creator)
    elif current_step == 3:
        show_background_step()
    elif current_step == 4:
        show_behavior_step()
    elif current_step == 5:
        show_psychology_step()
    elif current_step == 6:
        show_advanced_step()
    
    # Navigation buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if current_step > 1:
            if st.button("⬅️ Previous"):
                st.session_state.wizard_step -= 1
                st.rerun()
    
    with col2:
        if current_step < len(steps):
            if st.button("Next ➡️"):
                st.session_state.wizard_step += 1
                st.rerun()
    
    with col3:
        if current_step == len(steps):
            if st.button("🚀 Create Agent"):
                create_final_agent(creator)

def show_basic_info_step():
    """Step 1: Basic agent information"""
    config = st.session_state.agent_config
    
    config['name'] = st.text_input(
        "Agent Name",
        value=config.get('name', ''),
        placeholder="e.g., Dr. Emma Watson",
        help="Give your agent a memorable name"
    )
    
    config['description'] = st.text_area(
        "Description",
        value=config.get('description', ''),
        placeholder="A brief description of your agent's character and role...",
        height=100,
        help="Describe who your agent is and what they do"
    )
    
    role_options = [
        "custom", "doctor", "engineer", "spy", "rebel", "diplomat", 
        "scientist", "journalist", "teacher", "entrepreneur", "activist", 
        "academic", "influencer", "artist", "politician", "lawyer"
    ]
    
    config['role'] = st.selectbox(
        "Role",
        options=role_options,
        index=role_options.index(config.get('role', 'custom')),
        help="Choose a predefined role or select 'custom'"
    )
    
    if config['role'] == 'custom':
        config['custom_role'] = st.text_input(
            "Custom Role",
            value=config.get('custom_role', ''),
            placeholder="e.g., Space Explorer, Food Critic, etc."
        )

def show_personality_step(creator: AgentCreator):
    """Step 2: Personality configuration"""
    config = st.session_state.agent_config
    
    st.subheader("🧠 Personality Traits")
    
    # Personality template selector
    template_options = ["custom"] + list(creator.personality_templates.keys())
    selected_template = st.selectbox(
        "Personality Template",
        options=template_options,
        help="Choose a personality template or customize manually"
    )
    
    if selected_template != "custom":
        template = creator.personality_templates[selected_template]
        st.info(f"Using {selected_template} personality template")
        config['personality_traits'] = template.copy()
    
    # Personality sliders
    if 'personality_traits' not in config:
        config['personality_traits'] = {
            'openness': 0.5,
            'conscientiousness': 0.5,
            'extraversion': 0.5,
            'agreeableness': 0.5,
            'neuroticism': 0.5
        }
    
    col1, col2 = st.columns(2)
    
    with col1:
        config['personality_traits']['openness'] = st.slider(
            "Openness to Experience",
            0.0, 1.0, config['personality_traits']['openness'],
            help="How open to new ideas and experiences"
        )
        
        config['personality_traits']['conscientiousness'] = st.slider(
            "Conscientiousness",
            0.0, 1.0, config['personality_traits']['conscientiousness'],
            help="How organized and goal-oriented"
        )
        
        config['personality_traits']['extraversion'] = st.slider(
            "Extraversion",
            0.0, 1.0, config['personality_traits']['extraversion'],
            help="How outgoing and social"
        )
    
    with col2:
        config['personality_traits']['agreeableness'] = st.slider(
            "Agreeableness",
            0.0, 1.0, config['personality_traits']['agreeableness'],
            help="How cooperative and trusting"
        )
        
        config['personality_traits']['neuroticism'] = st.slider(
            "Neuroticism",
            0.0, 1.0, config['personality_traits']['neuroticism'],
            help="How prone to stress and anxiety"
        )

def show_background_step():
    """Step 3: Background and goals"""
    config = st.session_state.agent_config
    
    config['background_story'] = st.text_area(
        "Background Story",
        value=config.get('background_story', ''),
        placeholder="Tell your agent's story... Where did they come from? What experiences shaped them?",
        height=150,
        help="This background will influence how your agent thinks and responds"
    )
    
    # Goals section
    st.subheader("🎯 Goals")
    if 'goals' not in config:
        config['goals'] = []
    
    # Add new goal
    new_goal = st.text_input("Add a goal", placeholder="e.g., Help people solve problems")
    if st.button("Add Goal") and new_goal:
        config['goals'].append(new_goal)
        st.rerun()
    
    # Display and manage existing goals
    for i, goal in enumerate(config['goals']):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.write(f"• {goal}")
        with col2:
            if st.button("❌", key=f"remove_goal_{i}"):
                config['goals'].pop(i)
                st.rerun()
    
    # Motivations section
    st.subheader("💪 Motivations")
    if 'motivations' not in config:
        config['motivations'] = []
    
    motivations_text = st.text_area(
        "What drives your agent?",
        value='\n'.join(config.get('motivations', [])),
        placeholder="Enter each motivation on a new line...",
        help="What deeply motivates your agent to act?"
    )
    
    if motivations_text:
        config['motivations'] = [m.strip() for m in motivations_text.split('\n') if m.strip()]

def show_behavior_step():
    """Step 4: Behavior patterns"""
    config = st.session_state.agent_config
    
    # Speech patterns
    st.subheader("🗣️ Speech Patterns")
    speech_patterns_text = st.text_area(
        "How does your agent speak?",
        value='\n'.join(config.get('speech_patterns', [])),
        placeholder="e.g., 'Let me think about this...'\n'From my experience...'\n'That reminds me of...'",
        help="Enter typical phrases your agent uses, one per line"
    )
    
    if speech_patterns_text:
        config['speech_patterns'] = [p.strip() for p in speech_patterns_text.split('\n') if p.strip()]
    
    # Response style
    config['response_style'] = st.selectbox(
        "Response Style",
        options=["brief", "detailed", "emotional", "analytical", "casual", "formal"],
        index=["brief", "detailed", "emotional", "analytical", "casual", "formal"].index(
            config.get('response_style', 'brief')
        ),
        help="How does your agent typically respond?"
    )
    
    # Quirks
    st.subheader("🎭 Quirks & Habits")
    quirks_text = st.text_area(
        "What makes your agent unique?",
        value='\n'.join(config.get('quirks', [])),
        placeholder="e.g., Always checks the time\nUses medical analogies\nAsk lots of questions",
        help="Small behavioral traits that make your agent memorable"
    )
    
    if quirks_text:
        config['quirks'] = [q.strip() for q in quirks_text.split('\n') if q.strip()]

def show_psychology_step():
    """Step 5: Psychology and emotions"""
    config = st.session_state.agent_config
    
    # Emotional triggers
    st.subheader("⚡ Emotional Triggers")
    triggers_text = st.text_area(
        "What makes your agent emotional?",
        value='\n'.join(config.get('emotional_triggers', [])),
        placeholder="e.g., Injustice\nWasted resources\nPeople in danger",
        help="Things that provoke strong emotional reactions"
    )
    
    if triggers_text:
        config['emotional_triggers'] = [t.strip() for t in triggers_text.split('\n') if t.strip()]
    
    # Fears
    st.subheader("😨 Fears & Concerns")
    fears_text = st.text_area(
        "What does your agent fear?",
        value='\n'.join(config.get('fears', [])),
        placeholder="e.g., Making mistakes\nBeing misunderstood\nLosing control",
        help="Deep fears that drive your agent's behavior"
    )
    
    if fears_text:
        config['fears'] = [f.strip() for f in fears_text.split('\n') if f.strip()]
    
    # Topics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💚 Preferred Topics")
        preferred_text = st.text_area(
            "Topics your agent loves",
            value='\n'.join(config.get('preferred_topics', [])),
            placeholder="e.g., Technology\nMusic\nPhilosophy"
        )
        if preferred_text:
            config['preferred_topics'] = [t.strip() for t in preferred_text.split('\n') if t.strip()]
    
    with col2:
        st.subheader("❌ Avoided Topics")
        avoided_text = st.text_area(
            "Topics your agent dislikes",
            value='\n'.join(config.get('avoided_topics', [])),
            placeholder="e.g., Politics\nViolence\nGossip"
        )
        if avoided_text:
            config['avoided_topics'] = [t.strip() for t in avoided_text.split('\n') if t.strip()]

def show_advanced_step():
    """Step 6: Advanced settings"""
    config = st.session_state.agent_config
    
    # Custom instructions
    st.subheader("📝 Custom Instructions")
    config['custom_instructions'] = st.text_area(
        "Special instructions for your agent",
        value=config.get('custom_instructions', ''),
        placeholder="Any specific instructions for how your agent should behave in conversations...",
        height=150,
        help="Advanced behavioral instructions"
    )
    
    # Conflict style
    config['conflict_style'] = st.selectbox(
        "Conflict Style",
        options=["aggressive", "diplomatic", "avoidant", "collaborative", "competitive"],
        index=["aggressive", "diplomatic", "avoidant", "collaborative", "competitive"].index(
            config.get('conflict_style', 'diplomatic')
        ),
        help="How does your agent handle disagreements?"
    )
    
    # Preview
    st.subheader("👁️ Preview")
    if st.button("Preview Agent Configuration"):
        show_agent_preview(config)

def show_agent_preview(config: Dict[str, Any]):
    """Show a preview of the agent configuration"""
    st.info("🔍 Agent Preview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Name:** {config.get('name', 'Unnamed Agent')}")
        st.write(f"**Role:** {config.get('role', 'custom')}")
        st.write(f"**Response Style:** {config.get('response_style', 'brief')}")
        st.write(f"**Conflict Style:** {config.get('conflict_style', 'diplomatic')}")
    
    with col2:
        traits = config.get('personality_traits', {})
        st.write("**Personality:**")
        for trait, value in traits.items():
            st.write(f"  • {trait.title()}: {value:.1f}")
    
    if config.get('description'):
        st.write(f"**Description:** {config['description']}")
    
    if config.get('goals'):
        st.write("**Goals:**")
        for goal in config['goals']:
            st.write(f"  • {goal}")

def create_final_agent(creator: AgentCreator):
    """Create the final agent from configuration"""
    config_dict = st.session_state.agent_config
    
    try:
        # Create CustomAgentConfig object
        agent_config = CustomAgentConfig(
            name=config_dict.get('name', 'Unnamed Agent'),
            role=config_dict.get('role', 'custom'),
            description=config_dict.get('description', ''),
            background_story=config_dict.get('background_story', ''),
            goals=config_dict.get('goals', []),
            personality_traits=config_dict.get('personality_traits', {}),
            speech_patterns=config_dict.get('speech_patterns', []),
            emotional_triggers=config_dict.get('emotional_triggers', []),
            fears=config_dict.get('fears', []),
            motivations=config_dict.get('motivations', []),
            quirks=config_dict.get('quirks', []),
            preferred_topics=config_dict.get('preferred_topics', []),
            avoided_topics=config_dict.get('avoided_topics', []),
            relationships={},
            custom_instructions=config_dict.get('custom_instructions', ''),
            response_style=config_dict.get('response_style', 'brief'),
            conflict_style=config_dict.get('conflict_style', 'diplomatic'),
            created_date=datetime.now().isoformat(),
            created_by="wizard"
        )
        
        # Validate configuration
        issues = creator.validate_agent_config(agent_config)
        
        if issues:
            st.error("Please fix the following issues:")
            for issue in issues:
                st.write(f"• {issue}")
            return
        
        # Save agent
        filepath = creator.save_agent_config(agent_config)
        
        st.success(f"🎉 Agent '{agent_config.name}' created successfully!")
        st.write(f"Saved to: {filepath}")
        
        # Reset wizard
        st.session_state.wizard_step = 1
        st.session_state.agent_config = {}
        
        if st.button("Create Another Agent"):
            st.rerun()
        
    except Exception as e:
        st.error(f"Error creating agent: {e}")

def show_template_browser(creator: AgentCreator):
    """Browse and use agent templates"""
    st.header("📚 Agent Templates")
    st.write("Start with a pre-built template and customize as needed")
    
    templates = creator._get_agent_templates()
    
    for template_name, template_data in templates.items():
        with st.expander(f"🎭 {template_name.replace('_', ' ').title()}"):
            st.write(f"**Role:** {template_data['role']}")
            st.write(f"**Description:** {template_data['description']}")
            st.write(f"**Style:** {template_data['response_style']} responses, {template_data['conflict_style']} in conflicts")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Goals:**")
                for goal in template_data['goals'][:3]:
                    st.write(f"• {goal}")
            
            with col2:
                st.write("**Speech Patterns:**")
                for pattern in template_data['speech_patterns'][:3]:
                    st.write(f"• {pattern}")
            
            if st.button(f"Use {template_name.replace('_', ' ').title()}", key=f"use_{template_name}"):
                use_template(creator, template_name)

def use_template(creator: AgentCreator, template_name: str):
    """Use a template to create an agent"""
    st.subheader(f"✨ Creating agent from {template_name.replace('_', ' ').title()} template")
    
    agent_name = st.text_input("Agent Name", placeholder="Enter a name for your agent")
    custom_instructions = st.text_area(
        "Additional Instructions (Optional)",
        placeholder="Any special instructions or modifications...",
        height=100
    )
    
    # Personality adjustments
    st.write("**Personality Adjustments (Optional):**")
    col1, col2 = st.columns(2)
    
    personality_adjustments = {}
    
    with col1:
        if st.checkbox("Adjust Openness"):
            personality_adjustments['openness'] = st.slider("Openness", 0.0, 1.0, 0.5)
        if st.checkbox("Adjust Conscientiousness"):
            personality_adjustments['conscientiousness'] = st.slider("Conscientiousness", 0.0, 1.0, 0.5)
    
    with col2:
        if st.checkbox("Adjust Extraversion"):
            personality_adjustments['extraversion'] = st.slider("Extraversion", 0.0, 1.0, 0.5)
        if st.checkbox("Adjust Agreeableness"):
            personality_adjustments['agreeableness'] = st.slider("Agreeableness", 0.0, 1.0, 0.5)
    
    if st.button("Create Agent from Template") and agent_name:
        try:
            config = creator.create_agent_from_template(
                template_name, agent_name, custom_instructions, personality_adjustments
            )
            
            filepath = creator.save_agent_config(config)
            st.success(f"✅ Agent '{agent_name}' created from template!")
            st.write(f"Saved to: {filepath}")
            
        except Exception as e:
            st.error(f"Error creating agent: {e}")

def show_saved_agents_manager(creator: AgentCreator):
    """Manage saved agents"""
    st.header("💾 Saved Agents")
    
    saved_agents = creator.list_saved_agents()
    
    if not saved_agents:
        st.info("No saved agents found. Create your first agent!")
        return
    
    for agent in saved_agents:
        with st.expander(f"🤖 {agent['name']} ({agent['role']})"):
            st.write(f"**Description:** {agent['description']}")
            st.write(f"**Created:** {agent['created_date'][:10]}")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📝 Edit", key=f"edit_{agent['filename']}"):
                    edit_agent(creator, agent['filename'])
            
            with col2:
                if st.button("👁️ View Details", key=f"view_{agent['filename']}"):
                    view_agent_details(creator, agent['filename'])
            
            with col3:
                if st.button("🗑️ Delete", key=f"delete_{agent['filename']}"):
                    if creator.delete_agent_config(agent['filename']):
                        st.success(f"Deleted {agent['name']}")
                        st.rerun()
                    else:
                        st.error("Failed to delete agent")

def view_agent_details(creator: AgentCreator, filename: str):
    """View detailed agent information"""
    try:
        config = creator.load_agent_config(filename)
        
        st.subheader(f"🔍 {config.name} - Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Role:** {config.role}")
            st.write(f"**Description:** {config.description}")
            st.write(f"**Response Style:** {config.response_style}")
            st.write(f"**Conflict Style:** {config.conflict_style}")
            
            if config.goals:
                st.write("**Goals:**")
                for goal in config.goals:
                    st.write(f"• {goal}")
        
        with col2:
            st.write("**Personality Traits:**")
            for trait, value in config.personality_traits.items():
                st.write(f"• {trait.title()}: {value:.1f}")
            
            if config.emotional_triggers:
                st.write("**Emotional Triggers:**")
                for trigger in config.emotional_triggers[:5]:
                    st.write(f"• {trigger}")
        
        if config.background_story:
            st.write("**Background:**")
            st.write(config.background_story)
        
        if config.custom_instructions:
            st.write("**Custom Instructions:**")
            st.write(config.custom_instructions)
            
    except Exception as e:
        st.error(f"Error loading agent: {e}")

def edit_agent(creator: AgentCreator, filename: str):
    """Edit an existing agent (simplified version)"""
    try:
        config = creator.load_agent_config(filename)
        
        st.subheader(f"✏️ Edit {config.name}")
        
        # Basic edits only for now
        new_name = st.text_input("Name", value=config.name)
        new_description = st.text_area("Description", value=config.description)
        new_instructions = st.text_area("Custom Instructions", value=config.custom_instructions)
        
        if st.button("Save Changes"):
            config.name = new_name
            config.description = new_description
            config.custom_instructions = new_instructions
            
            creator.save_agent_config(config)
            st.success("Agent updated successfully!")
            
    except Exception as e:
        st.error(f"Error editing agent: {e}")

def show_import_export_ui(creator: AgentCreator):
    """Import/Export agent functionality"""
    st.header("📦 Import/Export Agents")
    
    tab1, tab2 = st.tabs(["Export Agents", "Import Agents"])
    
    with tab1:
        st.subheader("📤 Export Agent Pack")
        
        saved_agents = creator.list_saved_agents()
        if saved_agents:
            agent_names = [agent['name'] for agent in saved_agents]
            selected_agents = st.multiselect("Select agents to export", agent_names)
            
            pack_name = st.text_input("Pack Name", placeholder="My Custom Agents")
            
            if st.button("Create Agent Pack") and selected_agents and pack_name:
                try:
                    pack_path = creator.export_agent_pack(selected_agents, pack_name)
                    st.success(f"Agent pack created: {pack_path}")
                    
                    # Offer download
                    with open(pack_path, 'r') as f:
                        st.download_button(
                            "📥 Download Pack",
                            data=f.read(),
                            file_name=f"{pack_name.lower().replace(' ', '_')}_pack.json",
                            mime="application/json"
                        )
                except Exception as e:
                    st.error(f"Error creating pack: {e}")
        else:
            st.info("No agents available for export")
    
    with tab2:
        st.subheader("📥 Import Agent Pack")
        
        uploaded_file = st.file_uploader("Choose agent pack file", type=['json'])
        
        if uploaded_file is not None:
            try:
                # Save uploaded file temporarily
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                
                if st.button("Import Agents"):
                    imported = creator.import_agent_pack(temp_path)
                    
                    if imported:
                        st.success(f"Successfully imported {len(imported)} agents:")
                        for name in imported:
                            st.write(f"• {name}")
                    else:
                        st.warning("No agents were imported")
                
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
            except Exception as e:
                st.error(f"Error importing pack: {e}")

if __name__ == "__main__":
    show_agent_creator_ui()
