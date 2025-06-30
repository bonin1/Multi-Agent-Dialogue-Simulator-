from typing import List, Dict, Any, Optional
import json
from datetime import datetime
import random

# Predefined agent configurations
AGENT_CONFIGS = {
    "Dr. Sarah Chen": {
        "role": "doctor",
        "personality": {
            "openness": 0.8,
            "conscientiousness": 0.9,
            "extraversion": 0.6,
            "agreeableness": 0.8,
            "neuroticism": 0.3
        },
        "background": "Dr. Sarah Chen is a compassionate emergency physician with 15 years of experience. She believes in evidence-based medicine and always puts patient care first. She's worked in conflict zones and has seen the worst of humanity, making her both empathetic and realistic.",
        "goals": ["Ensure everyone's well-being", "Provide medical expertise", "Advocate for ethical decisions"]
    },
    "Marcus Steel": {
        "role": "engineer",
        "personality": {
            "openness": 0.7,
            "conscientiousness": 0.8,
            "extraversion": 0.4,
            "agreeableness": 0.6,
            "neuroticism": 0.4
        },
        "background": "Marcus Steel is a systems engineer who thinks in terms of efficiency and practical solutions. He has built infrastructure in developing countries and believes technology can solve most problems. He's logical, methodical, and sometimes struggles with emotional nuances.",
        "goals": ["Find practical solutions", "Optimize systems and processes", "Ensure technical feasibility"]
    },
    "Agent X": {
        "role": "spy",
        "personality": {
            "openness": 0.6,
            "conscientiousness": 0.7,
            "extraversion": 0.3,
            "agreeableness": 0.4,
            "neuroticism": 0.5
        },
        "background": "Agent X is a intelligence operative with a mysterious past. They excel at reading people and situations, always thinking three steps ahead. Their loyalty is to their mission, and they trust very few people completely.",
        "goals": ["Gather intelligence", "Protect classified information", "Complete the mission"]
    },
    "Maya Rodriguez": {
        "role": "rebel",
        "personality": {
            "openness": 0.9,
            "conscientiousness": 0.5,
            "extraversion": 0.8,
            "agreeableness": 0.4,
            "neuroticism": 0.6
        },
        "background": "Maya Rodriguez is a passionate activist who fights for social justice and environmental causes. She's not afraid to challenge authority and speak truth to power. Her idealism sometimes clashes with practical constraints, but her heart is always in the right place.",
        "goals": ["Fight for justice", "Challenge the status quo", "Give voice to the voiceless"]
    },
    "Ambassador Williams": {
        "role": "diplomat",
        "personality": {
            "openness": 0.7,
            "conscientiousness": 0.8,
            "extraversion": 0.7,
            "agreeableness": 0.9,
            "neuroticism": 0.2
        },
        "background": "Ambassador Williams has served in diplomatic roles for over 20 years, specializing in conflict resolution and international negotiations. They believe in the power of dialogue and have successfully mediated several international disputes.",
        "goals": ["Build consensus", "Maintain peace", "Foster understanding between parties"]
    }
}

# Scenario definitions
SCENARIOS = {
    "Political Negotiation": {
        "description": "A tense political negotiation where different parties must reach a compromise on a controversial issue.",
        "context": "A bill regarding environmental regulations vs. economic growth is being debated. Each party has different interests and constituencies to represent.",
        "phases": ["Opening statements", "Issue identification", "Negotiation", "Compromise seeking", "Final agreement"],
        "suggested_agents": ["Ambassador Williams", "Maya Rodriguez", "Dr. Sarah Chen"],
        "initial_prompt": "We need to discuss the proposed environmental regulations. The stakes are high for both the environment and the economy."
    },
    
    "Team Building": {
        "description": "A team of professionals working together to solve a complex problem.",
        "context": "A diverse team has been assembled to tackle a crisis situation that requires different expertise and perspectives.",
        "phases": ["Introductions", "Problem analysis", "Solution brainstorming", "Implementation planning", "Role assignment"],
        "suggested_agents": ["Dr. Sarah Chen", "Marcus Steel", "Agent X"],
        "initial_prompt": "We've been brought together to handle this crisis. Let's start by understanding what each of us brings to the table."
    },
    
    "Crisis Management": {
        "description": "A high-stakes crisis requiring immediate coordination and decision-making.",
        "context": "A natural disaster has struck, and the team must coordinate rescue efforts, resource allocation, and public communication.",
        "phases": ["Situation assessment", "Resource inventory", "Priority setting", "Action planning", "Execution coordination"],
        "suggested_agents": ["Dr. Sarah Chen", "Marcus Steel", "Ambassador Williams"],
        "initial_prompt": "The situation is critical. We need to act fast and coordinate our efforts. What's our immediate priority?"
    },
    
    "Corporate Espionage": {
        "description": "A covert operation where trust is scarce and everyone has hidden agendas.",
        "context": "Agents from different organizations are forced to work together while protecting their own interests and secrets.",
        "phases": ["Initial meeting", "Information sharing", "Trust building", "Revelation of motives", "Final confrontation"],
        "suggested_agents": ["Agent X", "Maya Rodriguez", "Marcus Steel"],
        "initial_prompt": "We all know why we're here, even if we can't say it openly. How do we proceed when trust is a luxury we can't afford?"
    },
    
    "Medical Ethics Committee": {
        "description": "A medical ethics committee debating a difficult case with moral implications.",
        "context": "A controversial medical case requires the committee to balance patient rights, medical ethics, resource allocation, and legal considerations.",
        "phases": ["Case presentation", "Ethical analysis", "Stakeholder perspectives", "Moral reasoning", "Decision making"],
        "suggested_agents": ["Dr. Sarah Chen", "Ambassador Williams", "Maya Rodriguez"],
        "initial_prompt": "We're here to discuss a case that challenges our understanding of medical ethics. Each perspective is important for reaching the right decision."
    },
    
    "Innovation Workshop": {
        "description": "A brainstorming session to develop innovative solutions to contemporary challenges.",
        "context": "A diverse group of experts is exploring cutting-edge solutions to global problems, balancing innovation with practical constraints.",
        "phases": ["Problem definition", "Creative exploration", "Feasibility analysis", "Prototype planning", "Implementation strategy"],
        "suggested_agents": ["Marcus Steel", "Dr. Sarah Chen", "Maya Rodriguez"],
        "initial_prompt": "Today we're thinking outside the box to solve problems that matter. What innovative approaches can we explore?"
    }
}

class ScenarioManager:
    """Manages scenario selection, context, and progression"""
    
    def __init__(self):
        self.current_scenario = None
        self.current_phase = 0
        self.scenario_history = []
    
    def get_available_scenarios(self) -> Dict[str, Dict]:
        """Get all available scenarios"""
        return SCENARIOS
    
    def get_available_agents(self) -> Dict[str, Dict]:
        """Get all available agent configurations"""
        return AGENT_CONFIGS
    
    def set_scenario(self, scenario_name: str):
        """Set the current scenario"""
        if scenario_name in SCENARIOS:
            self.current_scenario = SCENARIOS[scenario_name]
            self.current_phase = 0
            self.scenario_history.append({
                "scenario": scenario_name,
                "start_time": datetime.now(),
                "phases_completed": []
            })
        else:
            raise ValueError(f"Scenario '{scenario_name}' not found")
    
    def get_current_context(self) -> Dict[str, Any]:
        """Get current scenario context"""
        if not self.current_scenario:
            return {}
        
        return {
            "scenario_description": self.current_scenario["description"],
            "scenario_context": self.current_scenario["context"],
            "current_phase": self.current_scenario["phases"][self.current_phase] if self.current_phase < len(self.current_scenario["phases"]) else "Conclusion",
            "phase_number": self.current_phase + 1,
            "total_phases": len(self.current_scenario["phases"]),
            "initial_prompt": self.current_scenario.get("initial_prompt", "")
        }
    
    def advance_phase(self):
        """Move to the next phase of the scenario"""
        if self.current_scenario and self.current_phase < len(self.current_scenario["phases"]) - 1:
            self.current_phase += 1
            if self.scenario_history:
                self.scenario_history[-1]["phases_completed"].append(self.current_scenario["phases"][self.current_phase - 1])
    
    def get_suggested_agents(self, scenario_name: str) -> List[str]:
        """Get suggested agents for a scenario"""
        if scenario_name in SCENARIOS:
            return SCENARIOS[scenario_name].get("suggested_agents", [])
        return []
    
    def generate_random_scenario_context(self) -> str:
        """Generate additional random context for variety"""
        contexts = [
            "Tensions are high and time is running short.",
            "New information has just come to light that changes everything.",
            "An unexpected stakeholder has entered the conversation.",
            "Previous assumptions are being challenged.",
            "A deadline is approaching rapidly.",
            "External pressures are mounting.",
            "Public opinion is shifting.",
            "Technical constraints have been discovered.",
            "Budget limitations have been revealed.",
            "Legal implications are becoming clear."
        ]
        return random.choice(contexts)
    
    def get_phase_transition_prompt(self) -> str:
        """Get a prompt for transitioning between phases"""
        if not self.current_scenario:
            return ""
        
        current_phase_name = self.current_scenario["phases"][self.current_phase] if self.current_phase < len(self.current_scenario["phases"]) else "Conclusion"
        
        transitions = {
            "Opening statements": "Let's begin by having each party state their position clearly.",
            "Issue identification": "Now let's identify the key issues that need to be addressed.",
            "Problem analysis": "Let's analyze the core problems we're facing.",
            "Solution brainstorming": "It's time to brainstorm potential solutions.",
            "Negotiation": "Let's start the formal negotiation process.",
            "Compromise seeking": "We need to find middle ground that works for everyone.",
            "Implementation planning": "How do we put our ideas into action?",
            "Final agreement": "Let's work towards a final agreement.",
            "Conclusion": "Let's wrap up and summarize what we've accomplished."
        }
        
        return transitions.get(current_phase_name, f"We're now in the {current_phase_name} phase.")
    
    def export_scenario_history(self) -> str:
        """Export scenario history as JSON"""
        return json.dumps(self.scenario_history, default=str, indent=2)
