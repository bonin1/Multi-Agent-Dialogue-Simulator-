"""
Custom Agent Creation System
Allows users to create their own agents with custom personalities, instructions, and behaviors
"""

from typing import Dict, List, Optional, Any
import json
import os
from datetime import datetime
from dataclasses import dataclass, asdict
from models.agent_models import AgentRole, PersonalityTrait, EmotionalState, EmotionType
from agents.agent import Agent
from models.model_manager import ModelManager

@dataclass
class CustomAgentConfig:
    """Configuration for a custom agent"""
    name: str
    role: str  # Custom role or predefined role
    description: str
    background_story: str
    goals: List[str]
    personality_traits: Dict[str, float]  # openness, conscientiousness, etc.
    speech_patterns: List[str]
    emotional_triggers: List[str]
    fears: List[str]
    motivations: List[str]
    quirks: List[str]
    preferred_topics: List[str]
    avoided_topics: List[str]
    relationships: Dict[str, float]  # relationships with other agents
    custom_instructions: str
    response_style: str  # "brief", "detailed", "emotional", "analytical", etc.
    conflict_style: str  # "aggressive", "diplomatic", "avoidant", "collaborative"
    created_date: str
    created_by: str

class AgentCreator:
    """System for creating and managing custom agents"""
    
    def __init__(self, custom_agents_dir: str = "custom_agents"):
        self.custom_agents_dir = custom_agents_dir
        os.makedirs(custom_agents_dir, exist_ok=True)
        self.predefined_roles = self._get_predefined_roles()
        self.personality_templates = self._get_personality_templates()
        self.agent_templates = self._get_agent_templates()
    
    def _get_predefined_roles(self) -> List[str]:
        """Get list of predefined agent roles"""
        return [
            "doctor", "engineer", "spy", "rebel", "diplomat", 
            "scientist", "journalist", "teacher", "custom"
        ]
    
    def _get_personality_templates(self) -> Dict[str, Dict[str, float]]:
        """Get personality templates for quick agent creation"""
        return {
            "analytical": {
                "openness": 0.8,
                "conscientiousness": 0.9,
                "extraversion": 0.4,
                "agreeableness": 0.6,
                "neuroticism": 0.3
            },
            "creative": {
                "openness": 0.9,
                "conscientiousness": 0.5,
                "extraversion": 0.7,
                "agreeableness": 0.7,
                "neuroticism": 0.4
            },
            "leader": {
                "openness": 0.7,
                "conscientiousness": 0.8,
                "extraversion": 0.9,
                "agreeableness": 0.6,
                "neuroticism": 0.2
            },
            "skeptic": {
                "openness": 0.6,
                "conscientiousness": 0.7,
                "extraversion": 0.5,
                "agreeableness": 0.3,
                "neuroticism": 0.6
            },
            "optimist": {
                "openness": 0.8,
                "conscientiousness": 0.6,
                "extraversion": 0.8,
                "agreeableness": 0.9,
                "neuroticism": 0.2
            },
            "pessimist": {
                "openness": 0.4,
                "conscientiousness": 0.7,
                "extraversion": 0.3,
                "agreeableness": 0.4,
                "neuroticism": 0.8
            }
        }
    
    def _get_agent_templates(self) -> Dict[str, Dict[str, Any]]:
        """Get pre-built agent templates for quick creation"""
        return {
            "tech_entrepreneur": {
                "role": "entrepreneur",
                "description": "A tech startup founder who's passionate about innovation",
                "goals": ["Disrupt traditional industries", "Build successful products", "Scale the business"],
                "speech_patterns": ["Let me think about the market opportunity...", "From a business perspective...", "We need to move fast on this"],
                "emotional_triggers": ["Market opportunities", "Competition", "Innovation potential"],
                "fears": ["Missing opportunities", "Falling behind competitors", "Running out of funding"],
                "motivations": ["Building something meaningful", "Creating value", "Changing the world"],
                "quirks": ["Always thinking about scalability", "References other successful companies", "Gets excited about new technologies"],
                "response_style": "brief",
                "conflict_style": "aggressive"
            },
            
            "environmental_activist": {
                "role": "activist",
                "description": "A passionate environmentalist fighting for climate action",
                "goals": ["Protect the environment", "Raise awareness about climate change", "Hold corporations accountable"],
                "speech_patterns": ["This is destroying our planet!", "Future generations will judge us", "We need action, not words"],
                "emotional_triggers": ["Environmental destruction", "Corporate greed", "Government inaction"],
                "fears": ["Irreversible climate damage", "Corporate power", "Public apathy"],
                "motivations": ["Saving the planet", "Justice for future generations", "Holding power accountable"],
                "quirks": ["References environmental statistics", "Gets emotional about nature", "Challenges authority"],
                "response_style": "emotional",
                "conflict_style": "aggressive"
            },
            
            "philosophy_professor": {
                "role": "academic",
                "description": "A thoughtful philosophy professor who questions everything",
                "goals": ["Seek truth", "Challenge assumptions", "Teach others to think critically"],
                "speech_patterns": ["But what do we really mean by...", "That raises an interesting question", "Let's examine this more carefully"],
                "emotional_triggers": ["Logical fallacies", "Unexamined assumptions", "Intellectual dishonesty"],
                "fears": ["Accepting false beliefs", "Stopping the search for truth", "Intellectual complacency"],
                "motivations": ["Understanding reality", "Teaching others", "Pursuing wisdom"],
                "quirks": ["Asks probing questions", "References philosophical concepts", "Thinks out loud"],
                "response_style": "detailed",
                "conflict_style": "diplomatic"
            },
            
            "social_media_influencer": {
                "role": "influencer",
                "description": "A social media personality focused on trends and engagement",
                "goals": ["Build audience", "Create engaging content", "Stay relevant"],
                "speech_patterns": ["OMG, this is so important!", "My followers would love this", "This is giving me major vibes"],
                "emotional_triggers": ["Trending topics", "Audience engagement", "Social media drama"],
                "fears": ["Losing followers", "Being irrelevant", "Missing trends"],
                "motivations": ["Building community", "Creating content", "Staying connected"],
                "quirks": ["Uses current slang", "References social media metrics", "Thinks about 'content'"],
                "response_style": "brief",
                "conflict_style": "avoidant"
            }
        }
    
    def create_custom_agent(self, config: CustomAgentConfig, model_manager: ModelManager) -> Agent:
        """Create a custom agent from configuration"""
        # Convert custom role to AgentRole enum or use CUSTOM
        try:
            role = AgentRole[config.role.upper()]
        except KeyError:
            role = AgentRole.CUSTOM if hasattr(AgentRole, 'CUSTOM') else AgentRole.TEACHER
        
        # Create personality trait object
        personality = PersonalityTrait(
            openness=config.personality_traits.get('openness', 0.5),
            conscientiousness=config.personality_traits.get('conscientiousness', 0.5),
            extraversion=config.personality_traits.get('extraversion', 0.5),
            agreeableness=config.personality_traits.get('agreeableness', 0.5),
            neuroticism=config.personality_traits.get('neuroticism', 0.5)
        )
        
        # Create the agent
        agent = Agent(
            name=config.name,
            role=role,
            personality=personality,
            model_manager=model_manager,
            background_story=config.background_story
        )
        
        # Set custom properties
        agent.state.goals = config.goals
        agent.state.relationships = config.relationships
        
        # Store custom instructions and behavior patterns
        agent.custom_config = config
        
        return agent
    
    def save_agent_config(self, config: CustomAgentConfig) -> str:
        """Save agent configuration to file"""
        filename = f"{config.name.lower().replace(' ', '_')}.json"
        filepath = os.path.join(self.custom_agents_dir, filename)
        
        # Convert config to dictionary
        config_dict = asdict(config)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def load_agent_config(self, filename: str) -> CustomAgentConfig:
        """Load agent configuration from file"""
        filepath = os.path.join(self.custom_agents_dir, filename)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        return CustomAgentConfig(**config_dict)
    
    def list_saved_agents(self) -> List[Dict[str, Any]]:
        """List all saved custom agents"""
        agents = []
        
        for filename in os.listdir(self.custom_agents_dir):
            if filename.endswith('.json'):
                try:
                    config = self.load_agent_config(filename)
                    agents.append({
                        'filename': filename,
                        'name': config.name,
                        'role': config.role,
                        'description': config.description,
                        'created_date': config.created_date
                    })
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
        
        return sorted(agents, key=lambda x: x['created_date'], reverse=True)
    
    def delete_agent_config(self, filename: str) -> bool:
        """Delete an agent configuration file"""
        filepath = os.path.join(self.custom_agents_dir, filename)
        
        try:
            os.remove(filepath)
            return True
        except Exception:
            return False
    
    def create_agent_from_template(self, template_name: str, agent_name: str, 
                                 custom_instructions: str = "", 
                                 personality_adjustments: Dict[str, float] = None) -> CustomAgentConfig:
        """Create an agent config from a template"""
        if template_name not in self.agent_templates:
            raise ValueError(f"Template '{template_name}' not found")
        
        template = self.agent_templates[template_name]
        
        # Start with template personality or default
        personality = self.personality_templates.get(template_name, {
            "openness": 0.5,
            "conscientiousness": 0.5,
            "extraversion": 0.5,
            "agreeableness": 0.5,
            "neuroticism": 0.5
        })
        
        # Apply personality adjustments
        if personality_adjustments:
            for trait, value in personality_adjustments.items():
                if trait in personality:
                    personality[trait] = max(0.0, min(1.0, value))
        
        config = CustomAgentConfig(
            name=agent_name,
            role=template["role"],
            description=template["description"],
            background_story=f"As a {template['role']}, {template['description']}",
            goals=template["goals"],
            personality_traits=personality,
            speech_patterns=template["speech_patterns"],
            emotional_triggers=template["emotional_triggers"],
            fears=template["fears"],
            motivations=template["motivations"],
            quirks=template["quirks"],
            preferred_topics=[],
            avoided_topics=[],
            relationships={},
            custom_instructions=custom_instructions,
            response_style=template["response_style"],
            conflict_style=template["conflict_style"],
            created_date=datetime.now().isoformat(),
            created_by="template"
        )
        
        return config
    
    def validate_agent_config(self, config: CustomAgentConfig) -> List[str]:
        """Validate agent configuration and return list of issues"""
        issues = []
        
        if not config.name or len(config.name.strip()) < 2:
            issues.append("Agent name must be at least 2 characters long")
        
        if not config.description or len(config.description.strip()) < 10:
            issues.append("Description must be at least 10 characters long")
        
        if not config.goals:
            issues.append("Agent must have at least one goal")
        
        # Validate personality traits
        for trait, value in config.personality_traits.items():
            if not 0.0 <= value <= 1.0:
                issues.append(f"Personality trait '{trait}' must be between 0.0 and 1.0")
        
        required_traits = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
        for trait in required_traits:
            if trait not in config.personality_traits:
                issues.append(f"Missing required personality trait: {trait}")
        
        return issues
    
    def export_agent_pack(self, agent_names: List[str], pack_name: str) -> str:
        """Export multiple agents as a pack"""
        pack_data = {
            'pack_name': pack_name,
            'created_date': datetime.now().isoformat(),
            'agents': []
        }
        
        for name in agent_names:
            filename = f"{name.lower().replace(' ', '_')}.json"
            try:
                config = self.load_agent_config(filename)
                pack_data['agents'].append(asdict(config))
            except Exception as e:
                print(f"Error loading agent {name}: {e}")
        
        pack_filename = f"{pack_name.lower().replace(' ', '_')}_pack.json"
        pack_filepath = os.path.join(self.custom_agents_dir, pack_filename)
        
        with open(pack_filepath, 'w', encoding='utf-8') as f:
            json.dump(pack_data, f, indent=2, ensure_ascii=False)
        
        return pack_filepath
    
    def import_agent_pack(self, pack_filepath: str) -> List[str]:
        """Import agents from a pack file"""
        with open(pack_filepath, 'r', encoding='utf-8') as f:
            pack_data = json.load(f)
        
        imported_agents = []
        
        for agent_data in pack_data.get('agents', []):
            try:
                config = CustomAgentConfig(**agent_data)
                self.save_agent_config(config)
                imported_agents.append(config.name)
            except Exception as e:
                print(f"Error importing agent: {e}")
        
        return imported_agents
    
    def get_agent_creation_wizard_steps(self) -> List[Dict[str, Any]]:
        """Get steps for the agent creation wizard"""
        return [
            {
                "step": 1,
                "title": "Basic Information",
                "fields": ["name", "description", "role"],
                "description": "Set up the basic identity of your agent"
            },
            {
                "step": 2,
                "title": "Personality",
                "fields": ["personality_traits", "personality_template"],
                "description": "Define how your agent thinks and feels"
            },
            {
                "step": 3,
                "title": "Background & Goals",
                "fields": ["background_story", "goals", "motivations"],
                "description": "Give your agent a history and purpose"
            },
            {
                "step": 4,
                "title": "Behavior Patterns",
                "fields": ["speech_patterns", "quirks", "response_style"],
                "description": "How your agent communicates and behaves"
            },
            {
                "step": 5,
                "title": "Psychology",
                "fields": ["emotional_triggers", "fears", "preferred_topics"],
                "description": "What makes your agent tick emotionally"
            },
            {
                "step": 6,
                "title": "Advanced Settings",
                "fields": ["custom_instructions", "conflict_style", "relationships"],
                "description": "Fine-tune your agent's behavior"
            }
        ]
