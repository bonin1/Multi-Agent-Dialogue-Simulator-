"""
Personality traits system for agents using the Big Five model
"""
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
import random
import numpy as np


class Personality(BaseModel):
    """Big Five personality traits model"""
    
    # Big Five traits (0.0 - 1.0 scale)
    openness: float = Field(default=0.5, ge=0.0, le=1.0, description="Openness to experience")
    conscientiousness: float = Field(default=0.5, ge=0.0, le=1.0, description="Conscientiousness")
    extraversion: float = Field(default=0.5, ge=0.0, le=1.0, description="Extraversion")
    agreeableness: float = Field(default=0.5, ge=0.0, le=1.0, description="Agreeableness")
    neuroticism: float = Field(default=0.5, ge=0.0, le=1.0, description="Neuroticism")
    
    # Additional traits
    traits: List[str] = Field(default_factory=list, description="Specific personality traits")
    background: str = Field(default="", description="Character background")
    
    def get_trait_description(self) -> str:
        """Generate a personality description based on traits"""
        descriptions = []
        
        # Openness
        if self.openness > 0.7:
            descriptions.append("highly creative and open to new experiences")
        elif self.openness < 0.3:
            descriptions.append("prefers routine and conventional approaches")
        else:
            descriptions.append("balanced in openness to new experiences")
            
        # Conscientiousness
        if self.conscientiousness > 0.7:
            descriptions.append("very organized and detail-oriented")
        elif self.conscientiousness < 0.3:
            descriptions.append("spontaneous and flexible")
        else:
            descriptions.append("moderately organized")
            
        # Extraversion
        if self.extraversion > 0.7:
            descriptions.append("outgoing and energetic")
        elif self.extraversion < 0.3:
            descriptions.append("reserved and thoughtful")
        else:
            descriptions.append("balanced in social energy")
            
        # Agreeableness
        if self.agreeableness > 0.7:
            descriptions.append("cooperative and trusting")
        elif self.agreeableness < 0.3:
            descriptions.append("competitive and skeptical")
        else:
            descriptions.append("balanced in cooperation")
            
        # Neuroticism
        if self.neuroticism > 0.7:
            descriptions.append("emotionally sensitive")
        elif self.neuroticism < 0.3:
            descriptions.append("emotionally stable")
        else:
            descriptions.append("moderately emotional")
            
        return ", ".join(descriptions)
    
    def get_response_style(self) -> Dict[str, float]:
        """Get response style modifiers based on personality"""
        return {
            "verbosity": self.extraversion * 0.8 + self.openness * 0.2,
            "formality": self.conscientiousness * 0.6 + (1 - self.extraversion) * 0.4,
            "empathy": self.agreeableness * 0.7 + (1 - self.neuroticism) * 0.3,
            "confidence": (1 - self.neuroticism) * 0.6 + self.extraversion * 0.4,
            "creativity": self.openness * 0.8 + (1 - self.conscientiousness) * 0.2,
            "directness": (1 - self.agreeableness) * 0.5 + self.conscientiousness * 0.3,
        }
    
    def generate_speaking_pattern(self) -> Dict[str, str]:
        """Generate speaking patterns based on personality"""
        patterns = {
            "sentence_starters": [],
            "filler_words": [],
            "emphasis_style": "",
            "question_tendency": "medium"
        }
        
        # High extraversion - more expressive
        if self.extraversion > 0.7:
            patterns["sentence_starters"].extend([
                "You know what,", "I think", "Let me tell you,", "Actually,"
            ])
            patterns["filler_words"].extend(["like", "you know", "I mean"])
            patterns["emphasis_style"] = "exclamatory"
            patterns["question_tendency"] = "high"
            
        # Low extraversion - more reserved
        elif self.extraversion < 0.3:
            patterns["sentence_starters"].extend([
                "Perhaps", "It seems", "I believe", "One might consider"
            ])
            patterns["filler_words"].extend(["um", "well", "actually"])
            patterns["emphasis_style"] = "understated"
            patterns["question_tendency"] = "low"
            
        # High agreeableness - more cooperative language
        if self.agreeableness > 0.7:
            patterns["sentence_starters"].extend([
                "I understand", "That's a good point", "I agree that"
            ])
            
        # Low agreeableness - more challenging language
        elif self.agreeableness < 0.3:
            patterns["sentence_starters"].extend([
                "I disagree", "That's not quite right", "Actually,", "But"
            ])
            
        # High neuroticism - more hedging
        if self.neuroticism > 0.7:
            patterns["sentence_starters"].extend([
                "I'm not sure but", "Maybe", "I worry that", "I hope"
            ])
            patterns["filler_words"].extend(["um", "uh", "well"])
            
        return patterns
    
    def calculate_compatibility(self, other: 'Personality') -> float:
        """Calculate compatibility with another personality (0.0 - 1.0)"""
        # Some traits are complementary, others work better when similar
        compatibility_score = 0.0
        
        # Agreeableness - similar levels work better
        agreeableness_diff = abs(self.agreeableness - other.agreeableness)
        compatibility_score += (1 - agreeableness_diff) * 0.3
        
        # Neuroticism - lower combined neuroticism is better
        combined_neuroticism = (self.neuroticism + other.neuroticism) / 2
        compatibility_score += (1 - combined_neuroticism) * 0.2
        
        # Conscientiousness - similar levels work better
        conscientiousness_diff = abs(self.conscientiousness - other.conscientiousness)
        compatibility_score += (1 - conscientiousness_diff) * 0.2
        
        # Extraversion - some difference can be good (complementary)
        extraversion_diff = abs(self.extraversion - other.extraversion)
        # Optimal difference is around 0.3
        optimal_diff = 0.3
        extraversion_score = 1 - abs(extraversion_diff - optimal_diff) / 0.7
        compatibility_score += extraversion_score * 0.2
        
        # Openness - similar levels work better
        openness_diff = abs(self.openness - other.openness)
        compatibility_score += (1 - openness_diff) * 0.1
        
        return max(0.0, min(1.0, compatibility_score))


class PersonalityGenerator:
    """Utility class for generating personalities"""
    
    @staticmethod
    def generate_random_personality() -> Personality:
        """Generate a random personality"""
        return Personality(
            openness=random.uniform(0.2, 0.8),
            conscientiousness=random.uniform(0.2, 0.8),
            extraversion=random.uniform(0.2, 0.8),
            agreeableness=random.uniform(0.2, 0.8),
            neuroticism=random.uniform(0.2, 0.8),
            traits=random.sample([
                "creative", "analytical", "empathetic", "confident", "curious",
                "patient", "energetic", "careful", "optimistic", "realistic",
                "diplomatic", "direct", "supportive", "independent", "collaborative"
            ], k=random.randint(2, 5))
        )
    
    @staticmethod
    def generate_complementary_personality(base: Personality) -> Personality:
        """Generate a personality that complements the base personality"""
        # Create some variation while maintaining compatibility
        complementary = Personality(
            openness=base.openness + random.uniform(-0.2, 0.2),
            conscientiousness=base.conscientiousness + random.uniform(-0.2, 0.2),
            extraversion=1 - base.extraversion + random.uniform(-0.3, 0.3),  # Complementary extraversion
            agreeableness=base.agreeableness + random.uniform(-0.1, 0.1),  # Similar agreeableness
            neuroticism=max(0.1, base.neuroticism - random.uniform(0.1, 0.3)),  # Lower neuroticism
        )
        
        # Clamp values to valid range
        for trait in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]:
            value = getattr(complementary, trait)
            setattr(complementary, trait, max(0.0, min(1.0, value)))
        
        return complementary
    
    @staticmethod
    def create_team_personalities(num_agents: int, diversity_level: float = 0.7) -> List[Personality]:
        """Create a diverse team of personalities
        
        Args:
            num_agents: Number of personalities to generate
            diversity_level: How diverse the team should be (0.0 - 1.0)
        """
        personalities = []
        
        if num_agents <= 0:
            return personalities
            
        # Generate the first personality randomly
        personalities.append(PersonalityGenerator.generate_random_personality())
        
        for _ in range(1, num_agents):
            if diversity_level > 0.5:
                # High diversity - generate random personalities
                personalities.append(PersonalityGenerator.generate_random_personality())
            else:
                # Lower diversity - generate complementary personalities
                base_personality = random.choice(personalities)
                personalities.append(PersonalityGenerator.generate_complementary_personality(base_personality))
        
        return personalities


def analyze_team_dynamics(personalities: List[Personality]) -> Dict[str, float]:
    """Analyze team dynamics based on personality composition"""
    if not personalities:
        return {}
    
    n = len(personalities)
    
    # Calculate average traits
    avg_traits = {
        "openness": sum(p.openness for p in personalities) / n,
        "conscientiousness": sum(p.conscientiousness for p in personalities) / n,
        "extraversion": sum(p.extraversion for p in personalities) / n,
        "agreeableness": sum(p.agreeableness for p in personalities) / n,
        "neuroticism": sum(p.neuroticism for p in personalities) / n,
    }
    
    # Calculate diversity (standard deviation)
    diversity = {}
    for trait in avg_traits:
        values = [getattr(p, trait) for p in personalities]
        diversity[trait] = float(np.std(values))
    
    # Calculate overall compatibility
    total_compatibility = 0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            total_compatibility += personalities[i].calculate_compatibility(personalities[j])
            pairs += 1
    
    avg_compatibility = total_compatibility / pairs if pairs > 0 else 0
    
    # Predict team outcomes
    team_creativity = avg_traits["openness"] + diversity["openness"] * 0.5
    team_reliability = avg_traits["conscientiousness"] + (1 - diversity["conscientiousness"]) * 0.3
    team_energy = avg_traits["extraversion"] + diversity["extraversion"] * 0.3
    team_harmony = avg_compatibility * avg_traits["agreeableness"]
    team_stability = (1 - avg_traits["neuroticism"]) * avg_compatibility
    
    return {
        "average_traits": avg_traits,
        "diversity": diversity,
        "compatibility": avg_compatibility,
        "predicted_creativity": min(1.0, team_creativity),
        "predicted_reliability": min(1.0, team_reliability),
        "predicted_energy": min(1.0, team_energy),
        "predicted_harmony": min(1.0, team_harmony),
        "predicted_stability": min(1.0, team_stability),
    }
