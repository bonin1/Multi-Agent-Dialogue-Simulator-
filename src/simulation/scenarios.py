"""
Scenario management for multi-agent conversations
"""
from typing import Dict, List, Any, Optional
import random
from datetime import datetime


class ScenarioManager:
    """Manages conversation scenarios and context"""
    
    def __init__(self):
        self.scenarios = self._load_default_scenarios()
    
    def _load_default_scenarios(self) -> Dict[str, Dict[str, Any]]:
        """Load default conversation scenarios"""
        return {
            "political_negotiation": {
                "name": "Political Negotiation",
                "description": "A tense political negotiation between opposing parties",
                "context": "The country faces a critical decision about environmental policy. Stakeholders with different interests must find common ground.",
                "goals": ["Find compromise", "Protect interests", "Maintain relationships"],
                "tension_level": 0.8,
                "suggested_agents": ["diplomat", "rebel", "engineer"],
                "duration_estimate": "20-30 minutes",
                "opening_prompts": [
                    "We need to address the environmental crisis, but we can't ignore economic concerns.",
                    "The proposed policy will devastate local industries. We need alternatives.",
                    "Time is running out. We must act decisively on climate change."
                ]
            },
            
            "team_building": {
                "name": "Team Building Exercise",
                "description": "A corporate team building exercise",
                "context": "New team members from different departments need to work together on an important project.",
                "goals": ["Build trust", "Establish roles", "Create synergy"],
                "tension_level": 0.3,
                "suggested_agents": ["engineer", "doctor", "diplomat"],
                "duration_estimate": "15-25 minutes",
                "opening_prompts": [
                    "Let's get to know each other and figure out how to work together effectively.",
                    "We each bring different skills to this project. How can we combine them?",
                    "What are everyone's strengths and how can we leverage them?"
                ]
            },
            
            "crisis_management": {
                "name": "Crisis Management",
                "description": "Emergency response to a developing crisis",
                "context": "A natural disaster has struck and the response team must coordinate rescue and relief efforts.",
                "goals": ["Save lives", "Coordinate resources", "Manage information"],
                "tension_level": 0.9,
                "suggested_agents": ["doctor", "engineer", "diplomat"],
                "duration_estimate": "10-20 minutes",
                "opening_prompts": [
                    "We have multiple casualties and infrastructure damage. What's our priority?",
                    "Resources are limited. We need to make tough decisions quickly.",
                    "Communication is breaking down. How do we coordinate our response?"
                ]
            },
            
            "scientific_collaboration": {
                "name": "Scientific Collaboration",
                "description": "Researchers collaborating on a breakthrough discovery",
                "context": "Scientists from different fields must combine their expertise to solve a complex problem.",
                "goals": ["Share knowledge", "Validate theories", "Plan experiments"],
                "tension_level": 0.4,
                "suggested_agents": ["engineer", "doctor", "spy"],
                "duration_estimate": "20-35 minutes",
                "opening_prompts": [
                    "Our preliminary results are promising, but we need to validate the methodology.",
                    "The implications of this discovery could be significant. How do we proceed?",
                    "We're seeing unexpected patterns in the data. What could explain this?"
                ]
            },
            
            "social_gathering": {
                "name": "Social Gathering",
                "description": "Casual social interaction at a community event",
                "context": "People from different backgrounds meet at a neighborhood gathering.",
                "goals": ["Build connections", "Share stories", "Have fun"],
                "tension_level": 0.2,
                "suggested_agents": ["diplomat", "rebel", "doctor"],
                "duration_estimate": "15-30 minutes",
                "opening_prompts": [
                    "It's great to see so many neighbors coming together like this.",
                    "This community has so much diversity. What brings everyone here?",
                    "I love events like this. You never know who you'll meet."
                ]
            },
            
            "business_meeting": {
                "name": "Business Strategy Meeting",
                "description": "Corporate executives discussing business strategy",
                "context": "Company leaders must decide on a major strategic direction amid market changes.",
                "goals": ["Analyze market", "Make decisions", "Align strategy"],
                "tension_level": 0.6,
                "suggested_agents": ["engineer", "spy", "diplomat"],
                "duration_estimate": "25-40 minutes",
                "opening_prompts": [
                    "Market conditions have changed dramatically. We need to adapt our strategy.",
                    "Our competitors are moving fast. How do we maintain our advantage?",
                    "The board is expecting results. What's our plan moving forward?"
                ]
            },
            
            "ethical_dilemma": {
                "name": "Ethical Dilemma Discussion",
                "description": "Debate over a complex ethical issue",
                "context": "A committee must decide on an ethically complex issue with no clear right answer.",
                "goals": ["Explore perspectives", "Consider consequences", "Find ethical path"],
                "tension_level": 0.7,
                "suggested_agents": ["doctor", "rebel", "diplomat"],
                "duration_estimate": "20-35 minutes",
                "opening_prompts": [
                    "This decision will affect many lives. We need to consider all perspectives.",
                    "Sometimes the right thing to do isn't clear. How do we proceed?",
                    "We have competing values at stake here. How do we prioritize?"
                ]
            },
            
            "creative_brainstorm": {
                "name": "Creative Brainstorming Session",
                "description": "Collaborative creative problem-solving session",
                "context": "A diverse team works together to generate innovative solutions to a creative challenge.",
                "goals": ["Generate ideas", "Build on concepts", "Think creatively"],
                "tension_level": 0.3,
                "suggested_agents": ["engineer", "rebel", "spy"],
                "duration_estimate": "20-30 minutes",
                "opening_prompts": [
                    "Let's think outside the box. What wild ideas can we come up with?",
                    "No idea is too crazy right now. Let's see what we can imagine.",
                    "How can we approach this problem from a completely different angle?"
                ]
            }
        }
    
    def get_scenario(self, scenario_name: str) -> Optional[Dict[str, Any]]:
        """Get a specific scenario by name"""
        return self.scenarios.get(scenario_name)
    
    def get_default_scenario(self) -> Dict[str, Any]:
        """Get the default scenario"""
        return self.scenarios["team_building"]
    
    def list_scenarios(self) -> List[str]:
        """Get list of available scenario names"""
        return list(self.scenarios.keys())
    
    def get_scenario_details(self, scenario_name: str) -> Dict[str, Any]:
        """Get detailed information about a scenario"""
        scenario = self.get_scenario(scenario_name)
        if not scenario:
            return {}
        
        return {
            "name": scenario["name"],
            "description": scenario["description"],
            "context": scenario["context"],
            "goals": scenario["goals"],
            "tension_level": scenario["tension_level"],
            "suggested_agents": scenario["suggested_agents"],
            "duration_estimate": scenario["duration_estimate"],
            "complexity": self._calculate_complexity(scenario)
        }
    
    def _calculate_complexity(self, scenario: Dict[str, Any]) -> str:
        """Calculate scenario complexity based on various factors"""
        complexity_score = 0
        
        # Tension level contributes to complexity
        complexity_score += scenario["tension_level"] * 30
        
        # Number of goals
        complexity_score += len(scenario["goals"]) * 10
        
        # Number of suggested agents
        complexity_score += len(scenario["suggested_agents"]) * 5
        
        if complexity_score < 30:
            return "Simple"
        elif complexity_score < 60:
            return "Moderate"
        else:
            return "Complex"
    
    def generate_opening_message(
        self,
        scenario: Dict[str, Any],
        agents: List[Any]
    ) -> str:
        """Generate an opening message for the conversation"""
        
        # Select an opening prompt from the scenario
        opening_prompts = scenario.get("opening_prompts", [])
        if opening_prompts:
            base_message = random.choice(opening_prompts)
        else:
            base_message = f"Welcome to this {scenario['name'].lower()}."
        
        # Add context
        context_message = f"Context: {scenario['context']}"
        
        # Add participant information
        agent_names = [agent.name for agent in agents]
        participant_message = f"Participants: {', '.join(agent_names)}"
        
        # Add goals
        goals = scenario.get("goals", [])
        if goals:
            goals_message = f"Goals: {', '.join(goals)}"
        else:
            goals_message = "Goal: Engage in meaningful dialogue"
        
        # Combine messages
        full_message = f"{base_message}\n\n{context_message}\n\n{participant_message}\n\n{goals_message}"
        
        return full_message
    
    def suggest_agents_for_scenario(self, scenario_name: str) -> List[str]:
        """Suggest agent roles for a scenario"""
        scenario = self.get_scenario(scenario_name)
        if scenario and "suggested_agents" in scenario:
            return scenario["suggested_agents"]
        
        # Default suggestions
        return ["diplomat", "engineer", "doctor"]
    
    def create_custom_scenario(
        self,
        name: str,
        description: str,
        context: str,
        goals: List[str],
        tension_level: float = 0.5,
        suggested_agents: Optional[List[str]] = None,
        opening_prompts: Optional[List[str]] = None
    ) -> str:
        """Create a custom scenario"""
        
        scenario_id = name.lower().replace(" ", "_")
        
        custom_scenario = {
            "name": name,
            "description": description,
            "context": context,
            "goals": goals,
            "tension_level": max(0.0, min(1.0, tension_level)),
            "suggested_agents": suggested_agents or ["diplomat", "engineer", "doctor"],
            "duration_estimate": "15-30 minutes",
            "opening_prompts": opening_prompts or [description],
            "custom": True,
            "created_at": datetime.now().isoformat()
        }
        
        self.scenarios[scenario_id] = custom_scenario
        return scenario_id
    
    def get_scenarios_by_complexity(self, complexity: str) -> List[str]:
        """Get scenarios filtered by complexity level"""
        matching_scenarios = []
        
        for scenario_name, scenario_data in self.scenarios.items():
            if self._calculate_complexity(scenario_data) == complexity:
                matching_scenarios.append(scenario_name)
        
        return matching_scenarios
    
    def get_scenarios_by_tension_level(
        self,
        min_tension: float = 0.0,
        max_tension: float = 1.0
    ) -> List[str]:
        """Get scenarios within a tension level range"""
        matching_scenarios = []
        
        for scenario_name, scenario_data in self.scenarios.items():
            tension = scenario_data.get("tension_level", 0.5)
            if min_tension <= tension <= max_tension:
                matching_scenarios.append(scenario_name)
        
        return matching_scenarios
    
    def get_random_scenario(
        self,
        exclude_high_tension: bool = False,
        complexity_preference: Optional[str] = None
    ) -> str:
        """Get a random scenario with optional filters"""
        available_scenarios = list(self.scenarios.keys())
        
        # Filter by tension level
        if exclude_high_tension:
            available_scenarios = [
                name for name in available_scenarios
                if self.scenarios[name].get("tension_level", 0.5) < 0.7
            ]
        
        # Filter by complexity
        if complexity_preference:
            available_scenarios = [
                name for name in available_scenarios
                if self._calculate_complexity(self.scenarios[name]) == complexity_preference
            ]
        
        if not available_scenarios:
            return "team_building"  # Fallback
        
        return random.choice(available_scenarios)
    
    def get_scenario_recommendations(self, agent_roles: List[str]) -> List[str]:
        """Recommend scenarios based on agent roles"""
        recommendations = []
        
        for scenario_name, scenario_data in self.scenarios.items():
            suggested_agents = scenario_data.get("suggested_agents", [])
            
            # Calculate match score
            common_roles = set(agent_roles) & set(suggested_agents)
            match_score = len(common_roles) / len(suggested_agents) if suggested_agents else 0
            
            if match_score > 0.5:  # At least 50% match
                recommendations.append(scenario_name)
        
        return recommendations
    
    def export_scenarios(self) -> Dict[str, Any]:
        """Export all scenarios for backup or sharing"""
        return {
            "scenarios": self.scenarios,
            "exported_at": datetime.now().isoformat(),
            "total_scenarios": len(self.scenarios)
        }
    
    def import_scenarios(self, scenario_data: Dict[str, Any]):
        """Import scenarios from exported data"""
        if "scenarios" in scenario_data:
            imported_scenarios = scenario_data["scenarios"]
            
            for scenario_id, scenario in imported_scenarios.items():
                if scenario_id not in self.scenarios:
                    self.scenarios[scenario_id] = scenario
    
    def validate_scenario(self, scenario: Dict[str, Any]) -> List[str]:
        """Validate a scenario structure and return any issues"""
        issues = []
        
        required_fields = ["name", "description", "context", "goals"]
        for field in required_fields:
            if field not in scenario:
                issues.append(f"Missing required field: {field}")
        
        # Check data types
        if "tension_level" in scenario:
            if not isinstance(scenario["tension_level"], (int, float)):
                issues.append("tension_level must be a number")
            elif not 0.0 <= scenario["tension_level"] <= 1.0:
                issues.append("tension_level must be between 0.0 and 1.0")
        
        if "goals" in scenario and not isinstance(scenario["goals"], list):
            issues.append("goals must be a list")
        
        if "suggested_agents" in scenario and not isinstance(scenario["suggested_agents"], list):
            issues.append("suggested_agents must be a list")
        
        return issues
