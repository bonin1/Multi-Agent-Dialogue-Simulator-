"""
Base agent class with memory, emotions, and conversation capabilities
"""
import uuid
import time
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum
import logging

from .personality import Personality
from ..memory.vector_memory import AgentMemory
from ..memory.reflection import ReflectionSystem


class EmotionalState(BaseModel):
    """Current emotional state of an agent"""
    joy: float = Field(default=0.5, ge=0.0, le=1.0)
    sadness: float = Field(default=0.0, ge=0.0, le=1.0)
    anger: float = Field(default=0.0, ge=0.0, le=1.0)
    fear: float = Field(default=0.0, ge=0.0, le=1.0)
    surprise: float = Field(default=0.0, ge=0.0, le=1.0)
    disgust: float = Field(default=0.0, ge=0.0, le=1.0)
    
    def update_emotion(self, emotion: str, intensity: float, decay_rate: float = 0.1):
        """Update a specific emotion with decay"""
        if hasattr(self, emotion):
            current_value = getattr(self, emotion)
            # Apply the new emotion
            new_value = min(1.0, current_value + intensity)
            setattr(self, emotion, new_value)
            
            # Apply decay to other emotions
            for other_emotion in ["joy", "sadness", "anger", "fear", "surprise", "disgust"]:
                if other_emotion != emotion:
                    current = getattr(self, other_emotion)
                    setattr(self, other_emotion, max(0.0, current - decay_rate))
    
    def get_dominant_emotion(self) -> Tuple[str, float]:
        """Get the most dominant current emotion"""
        emotions = {
            "joy": self.joy,
            "sadness": self.sadness,
            "anger": self.anger,
            "fear": self.fear,
            "surprise": self.surprise,
            "disgust": self.disgust
        }
        dominant = max(emotions.items(), key=lambda x: x[1])
        return dominant


class AgentIntent(BaseModel):
    """Current intent/goal of an agent"""
    primary_goal: str = Field(default="engage_in_conversation")
    secondary_goals: List[str] = Field(default_factory=list)
    current_strategy: str = Field(default="cooperative")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    urgency: float = Field(default=0.3, ge=0.0, le=1.0)


class RelationshipState(BaseModel):
    """Relationship state with another agent"""
    agent_id: str
    trust_level: float = Field(default=0.5, ge=0.0, le=1.0)
    respect_level: float = Field(default=0.5, ge=0.0, le=1.0)
    affinity: float = Field(default=0.5, ge=0.0, le=1.0)
    shared_experiences: int = Field(default=0)
    last_interaction: Optional[datetime] = None
    interaction_history: List[str] = Field(default_factory=list)
    
    def update_relationship(self, interaction_type: str, impact: float):
        """Update relationship based on interaction"""
        self.last_interaction = datetime.now()
        self.shared_experiences += 1
        self.interaction_history.append(interaction_type)
        
        # Keep only recent interactions
        if len(self.interaction_history) > 20:
            self.interaction_history = self.interaction_history[-20:]
        
        # Update relationship metrics based on interaction type
        if interaction_type in ["agreement", "support", "help"]:
            self.trust_level = min(1.0, self.trust_level + impact * 0.1)
            self.affinity = min(1.0, self.affinity + impact * 0.1)
        elif interaction_type in ["disagreement", "challenge"]:
            self.trust_level = max(0.0, self.trust_level - impact * 0.05)
            self.respect_level = min(1.0, self.respect_level + impact * 0.05)  # Respectful disagreement
        elif interaction_type in ["conflict", "betrayal"]:
            self.trust_level = max(0.0, self.trust_level - impact * 0.2)
            self.affinity = max(0.0, self.affinity - impact * 0.15)


class Agent(BaseModel):
    """Base agent class with personality, memory, and emotional intelligence"""
    
    # Core identity
    agent_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    role: str
    personality: Personality
    
    # State
    emotional_state: EmotionalState = Field(default_factory=EmotionalState)
    current_intent: AgentIntent = Field(default_factory=AgentIntent)
    energy_level: float = Field(default=1.0, ge=0.0, le=1.0)
    
    # Relationships
    relationships: Dict[str, RelationshipState] = Field(default_factory=dict)
    
    # Memory and reflection
    memory: Optional[AgentMemory] = None
    reflection_system: Optional[ReflectionSystem] = None
    
    # Conversation state
    conversation_context: List[Dict[str, Any]] = Field(default_factory=list)
    last_spoke: Optional[datetime] = None
    turn_count: int = 0
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, **data):
        super().__init__(**data)
        # Initialize memory and reflection systems
        self.memory = AgentMemory(agent_id=self.agent_id)
        self.reflection_system = ReflectionSystem(agent_id=self.agent_id)
        
        # Set initial intent based on role
        self._set_role_based_intent()
    
    def _set_role_based_intent(self):
        """Set initial intent based on agent role"""
        role_intents = {
            "doctor": {
                "primary_goal": "provide_medical_insight",
                "secondary_goals": ["help_others", "ensure_safety"],
                "current_strategy": "empathetic_analysis"
            },
            "engineer": {
                "primary_goal": "solve_technical_problems",
                "secondary_goals": ["optimize_solutions", "share_knowledge"],
                "current_strategy": "systematic_approach"
            },
            "spy": {
                "primary_goal": "gather_information",
                "secondary_goals": ["protect_secrets", "assess_threats"],
                "current_strategy": "cautious_observation"
            },
            "rebel": {
                "primary_goal": "challenge_status_quo",
                "secondary_goals": ["inspire_change", "expose_problems"],
                "current_strategy": "passionate_advocacy"
            },
            "diplomat": {
                "primary_goal": "build_consensus",
                "secondary_goals": ["maintain_peace", "find_compromise"],
                "current_strategy": "diplomatic_mediation"
            }
        }
        
        if self.role in role_intents:
            intent_data = role_intents[self.role]
            self.current_intent = AgentIntent(**intent_data)
    
    def get_agent_state(self) -> Dict[str, Any]:
        """Get current complete state of the agent"""
        dominant_emotion, emotion_intensity = self.emotional_state.get_dominant_emotion()
        
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "role": self.role,
            "energy_level": self.energy_level,
            "dominant_emotion": dominant_emotion,
            "emotion_intensity": emotion_intensity,
            "primary_goal": self.current_intent.primary_goal,
            "strategy": self.current_intent.current_strategy,
            "turn_count": self.turn_count,
            "relationship_count": len(self.relationships),
            "memory_size": len(self.conversation_context)
        }
    
    def update_relationship(self, other_agent_id: str, interaction_type: str, impact: float = 0.5):
        """Update relationship with another agent"""
        if other_agent_id not in self.relationships:
            self.relationships[other_agent_id] = RelationshipState(agent_id=other_agent_id)
        
        self.relationships[other_agent_id].update_relationship(interaction_type, impact)
    
    def get_relationship_context(self, other_agent_id: str) -> str:
        """Get relationship context for conversation"""
        if other_agent_id not in self.relationships:
            return "This is a new interaction."
        
        rel = self.relationships[other_agent_id]
        context_parts = []
        
        # Trust and affinity
        if rel.trust_level > 0.7:
            context_parts.append("I trust this person")
        elif rel.trust_level < 0.3:
            context_parts.append("I'm cautious around this person")
        
        if rel.affinity > 0.7:
            context_parts.append("I like working with them")
        elif rel.affinity < 0.3:
            context_parts.append("I find them difficult to work with")
        
        # Recent interactions
        if rel.interaction_history:
            recent = rel.interaction_history[-3:]
            if "conflict" in recent:
                context_parts.append("we had recent conflicts")
            elif "agreement" in recent:
                context_parts.append("we've been agreeing lately")
        
        return ". ".join(context_parts) if context_parts else "Our relationship is neutral"
    
    async def process_message(self, message: str, sender_id: str, context: Dict[str, Any]) -> str:
        """Process an incoming message and generate response"""
        # Update conversation context
        self.conversation_context.append({
            "timestamp": datetime.now(),
            "sender": sender_id,
            "message": message,
            "context": context
        })
        
        # Keep context window manageable
        if len(self.conversation_context) > 20:
            self.conversation_context = self.conversation_context[-20:]
        
        # Store in long-term memory
        await self.memory.store_conversation(
            content=f"{sender_id}: {message}",
            metadata={
                "sender": sender_id,
                "timestamp": datetime.now().isoformat(),
                "context": context
            }
        )
        
        # Analyze emotional impact
        self._analyze_emotional_impact(message, sender_id)
        
        # Update relationship
        interaction_type = self._classify_interaction(message)
        self.update_relationship(sender_id, interaction_type)
        
        # Generate response
        response = await self._generate_response(message, sender_id, context)
        
        # Store own response
        self.conversation_context.append({
            "timestamp": datetime.now(),
            "sender": self.agent_id,
            "message": response,
            "context": {"type": "response"}
        })
        
        # Update turn count and reflection
        self.turn_count += 1
        if self.turn_count % 5 == 0:  # Reflect every 5 turns
            await self._trigger_reflection()
        
        self.last_spoke = datetime.now()
        return response
    
    def _analyze_emotional_impact(self, message: str, sender_id: str):
        """Analyze emotional impact of received message"""
        # Simple emotion detection based on keywords and personality
        message_lower = message.lower()
        
        # Positive emotions
        if any(word in message_lower for word in ["great", "amazing", "wonderful", "excellent", "fantastic"]):
            self.emotional_state.update_emotion("joy", 0.2)
        
        # Negative emotions
        if any(word in message_lower for word in ["terrible", "awful", "horrible", "disaster", "crisis"]):
            self.emotional_state.update_emotion("sadness", 0.3)
            if self.personality.neuroticism > 0.6:
                self.emotional_state.update_emotion("fear", 0.2)
        
        # Anger triggers
        if any(word in message_lower for word in ["stupid", "wrong", "ridiculous", "absurd"]):
            if self.personality.agreeableness < 0.4:
                self.emotional_state.update_emotion("anger", 0.4)
            else:
                self.emotional_state.update_emotion("sadness", 0.2)
        
        # Surprise
        if any(word in message_lower for word in ["unexpected", "surprise", "shocking", "wow"]):
            self.emotional_state.update_emotion("surprise", 0.3)
    
    def _classify_interaction(self, message: str) -> str:
        """Classify the type of interaction"""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ["agree", "yes", "exactly", "absolutely", "right"]):
            return "agreement"
        elif any(word in message_lower for word in ["disagree", "no", "wrong", "incorrect"]):
            return "disagreement"
        elif any(word in message_lower for word in ["help", "support", "assist", "collaborate"]):
            return "support"
        elif any(word in message_lower for word in ["challenge", "question", "doubt"]):
            return "challenge"
        elif any(word in message_lower for word in ["stupid", "idiot", "ridiculous", "absurd"]):
            return "conflict"
        else:
            return "neutral"
    
    async def _generate_response(self, message: str, sender_id: str, context: Dict[str, Any]) -> str:
        """Generate response based on personality, emotions, and memory"""
        # Get relevant memories
        memories = await self.memory.retrieve_relevant_memories(message, limit=3)
        
        # Get relationship context
        relationship_context = self.get_relationship_context(sender_id)
        
        # Get personality-based response style
        response_style = self.personality.get_response_style()
        speaking_pattern = self.personality.generate_speaking_pattern()
        
        # Get emotional state
        dominant_emotion, emotion_intensity = self.emotional_state.get_dominant_emotion()
        
        # Build context for response generation
        response_context = {
            "agent_name": self.name,
            "agent_role": self.role,
            "personality_description": self.personality.get_trait_description(),
            "dominant_emotion": dominant_emotion,
            "emotion_intensity": emotion_intensity,
            "relationship_context": relationship_context,
            "current_goal": self.current_intent.primary_goal,
            "strategy": self.current_intent.current_strategy,
            "relevant_memories": [mem.get("content", "") for mem in memories],
            "response_style": response_style,
            "speaking_pattern": speaking_pattern,
            "received_message": message,
            "conversation_context": context
        }
        
        # This would connect to the LLM - for now return a structured response
        response = self._format_response_with_personality(response_context)
        
        return response
    
    def _format_response_with_personality(self, context: Dict[str, Any]) -> str:
        """Format response according to personality traits (placeholder for LLM integration)"""
        # This is a simplified version - in the full implementation this would use the LLM
        style = context["response_style"]
        pattern = context["speaking_pattern"]
        
        # Start with personality-appropriate opener
        starters = pattern.get("sentence_starters", ["Well,"])
        opener = starters[0] if starters else ""
        
        # Base response based on role and emotion
        role = context["agent_role"]
        emotion = context["dominant_emotion"]
        
        base_responses = {
            "doctor": "From a medical perspective, I think we need to consider the health implications here.",
            "engineer": "Let me analyze this systematically and find an optimal solution.",
            "spy": "I'm observing some interesting patterns in this situation.",
            "rebel": "We need to challenge the conventional thinking here.",
            "diplomat": "I believe we can find common ground if we listen to each other."
        }
        
        base_response = base_responses.get(role, "That's an interesting point.")
        
        # Modify based on emotion
        if emotion == "anger" and context["emotion_intensity"] > 0.5:
            if style["directness"] > 0.6:
                base_response = f"I strongly disagree. {base_response}"
            else:
                base_response = f"I have concerns about this approach. {base_response}"
        elif emotion == "joy" and context["emotion_intensity"] > 0.5:
            base_response = f"I'm excited about this! {base_response}"
        elif emotion == "fear" and context["emotion_intensity"] > 0.5:
            base_response = f"I'm worried that {base_response.lower()}"
        
        # Add personality flair
        if style["verbosity"] > 0.7:
            base_response += " Let me elaborate on this further."
        if style["empathy"] > 0.7:
            base_response = f"I understand your perspective. {base_response}"
        
        response = f"{opener} {base_response}".strip()
        
        # Add filler words occasionally
        if pattern.get("filler_words") and style["confidence"] < 0.5:
            filler = pattern["filler_words"][0]
            response = response.replace(".", f", {filler}.")
        
        return response
    
    async def _trigger_reflection(self):
        """Trigger long-term reflection process"""
        recent_conversations = self.conversation_context[-10:]  # Last 10 messages
        await self.reflection_system.generate_reflection(
            recent_experiences=recent_conversations,
            current_state=self.get_agent_state()
        )
    
    def should_interrupt(self, current_speaker_id: str, urgency_threshold: float = 0.7) -> bool:
        """Determine if agent should interrupt current speaker"""
        if current_speaker_id == self.agent_id:
            return False
        
        # Check personality traits
        interruption_tendency = (
            self.personality.extraversion * 0.4 +
            (1 - self.personality.agreeableness) * 0.3 +
            self.current_intent.urgency * 0.3
        )
        
        # Check emotional state
        dominant_emotion, intensity = self.emotional_state.get_dominant_emotion()
        if dominant_emotion in ["anger", "excitement"] and intensity > 0.6:
            interruption_tendency += 0.2
        
        return interruption_tendency > urgency_threshold
    
    def adjust_energy(self, delta: float):
        """Adjust agent energy level"""
        self.energy_level = max(0.0, min(1.0, self.energy_level + delta))
        
        # Low energy affects personality expression
        if self.energy_level < 0.3:
            # Reduce extraversion when tired
            self.emotional_state.update_emotion("sadness", 0.1)
    
    def get_conversation_summary(self) -> str:
        """Get summary of recent conversations"""
        if not self.conversation_context:
            return "No recent conversations"
        
        recent = self.conversation_context[-5:]
        summary_parts = []
        
        for msg in recent:
            sender = "I" if msg["sender"] == self.agent_id else "Someone"
            summary_parts.append(f"{sender}: {msg['message'][:50]}...")
        
        return "\n".join(summary_parts)


# Factory function for creating role-specific agents
def create_agent(name: str, role: str, personality: Optional[Personality] = None) -> Agent:
    """Create an agent with role-specific configuration"""
    try:
        from ..config.settings import AGENT_CONFIG
    except ImportError:
        from config.settings import AGENT_CONFIG
    
    if personality is None:
        # Use default personality for role
        default_personalities = AGENT_CONFIG["default_personalities"]
        if role in default_personalities:
            personality_data = default_personalities[role]
            personality = Personality(**personality_data)
        else:
            # Generate random personality
            from .personality import PersonalityGenerator
            personality = PersonalityGenerator.generate_random_personality()
    
    agent = Agent(
        name=name,
        role=role,
        personality=personality
    )
    
    return agent
