from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from enum import Enum
import uuid
from datetime import datetime

class EmotionType(str, Enum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    ANGRY = "angry"
    SAD = "sad"
    EXCITED = "excited"
    FRUSTRATED = "frustrated"
    CONFIDENT = "confident"
    ANXIOUS = "anxious"
    SUSPICIOUS = "suspicious"
    EMPATHETIC = "empathetic"

class AgentRole(str, Enum):
    DOCTOR = "doctor"
    ENGINEER = "engineer"
    SPY = "spy"
    REBEL = "rebel"
    DIPLOMAT = "diplomat"
    SCIENTIST = "scientist"
    JOURNALIST = "journalist"
    TEACHER = "teacher"
    CUSTOM = "custom"

class PersonalityTrait(BaseModel):
    openness: float = Field(ge=0, le=1, description="Open to new experiences")
    conscientiousness: float = Field(ge=0, le=1, description="Organized and disciplined")
    extraversion: float = Field(ge=0, le=1, description="Outgoing and energetic")
    agreeableness: float = Field(ge=0, le=1, description="Cooperative and trusting")
    neuroticism: float = Field(ge=0, le=1, description="Emotional instability")

class EmotionalState(BaseModel):
    primary_emotion: EmotionType = EmotionType.NEUTRAL
    intensity: float = Field(ge=0, le=1, default=0.5)
    secondary_emotions: Dict[EmotionType, float] = Field(default_factory=dict)
    emotional_history: List[tuple] = Field(default_factory=list)  # (emotion, intensity, timestamp)
    
    def update_emotion(self, emotion: EmotionType, intensity: float, trigger: str = ""):
        """Update emotional state with new emotion"""
        self.emotional_history.append((self.primary_emotion, self.intensity, datetime.now(), trigger))
        self.primary_emotion = emotion
        self.intensity = intensity
        
        # Keep only last 10 emotional states
        if len(self.emotional_history) > 10:
            self.emotional_history = self.emotional_history[-10:]

class Memory(BaseModel):
    content: str
    timestamp: datetime
    importance: float = Field(ge=0, le=1, default=0.5)
    related_agents: List[str] = Field(default_factory=list)
    emotional_context: EmotionType = EmotionType.NEUTRAL
    tags: List[str] = Field(default_factory=list)

class Belief(BaseModel):
    statement: str
    confidence: float = Field(ge=0, le=1, default=0.5)
    evidence: List[str] = Field(default_factory=list)
    last_updated: datetime = Field(default_factory=datetime.now)

class AgentState(BaseModel):
    agent_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    role: AgentRole
    personality: PersonalityTrait
    emotional_state: EmotionalState = Field(default_factory=EmotionalState)
    beliefs: Dict[str, Belief] = Field(default_factory=dict)
    goals: List[str] = Field(default_factory=list)
    relationships: Dict[str, float] = Field(default_factory=dict)  # agent_id -> trust_level
    conversation_context: List[str] = Field(default_factory=list)
    reflection_notes: List[str] = Field(default_factory=list)
    
    class Config:
        use_enum_values = True
        
    def add_belief(self, statement: str, confidence: float, evidence: List[str] = None):
        """Add or update a belief"""
        belief = Belief(
            statement=statement,
            confidence=confidence,
            evidence=evidence or [],
            last_updated=datetime.now()
        )
        self.beliefs[statement] = belief
        
    def update_relationship(self, agent_id: str, trust_change: float):
        """Update trust level with another agent"""
        current_trust = self.relationships.get(agent_id, 0.5)
        new_trust = max(0, min(1, current_trust + trust_change))
        self.relationships[agent_id] = new_trust
        
    def add_reflection(self, reflection: str):
        """Add a reflection note"""
        self.reflection_notes.append(f"[{datetime.now().strftime('%H:%M')}] {reflection}")
        # Keep only last 20 reflections
        if len(self.reflection_notes) > 20:
            self.reflection_notes = self.reflection_notes[-20:]
