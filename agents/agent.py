from typing import Dict, List, Optional, Any
import json
import logging
from datetime import datetime
import random

try:
    from models.agent_models import AgentState, AgentRole, PersonalityTrait, EmotionalState, EmotionType
    from models.model_manager import ModelManager
    from memory.memory_system import MemorySystem
except ImportError:
    # Fallback imports for when packages aren't installed
    pass

class PerceptionModule:
    """Handles context interpretation and environmental awareness"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
    
    def interpret_context(self, context: Dict[str, Any]) -> str:
        """Interpret current context and return insights"""
        insights = []
        
        if 'recent_messages' in context:
            messages = context['recent_messages']
            if messages:
                # Analyze conversation tone
                positive_words = ['good', 'great', 'excellent', 'agree', 'yes', 'positive']
                negative_words = ['bad', 'terrible', 'disagree', 'no', 'negative', 'problem']
                
                recent_text = ' '.join([msg.get('content', '') for msg in messages[-3:]])
                positive_count = sum(1 for word in positive_words if word in recent_text.lower())
                negative_count = sum(1 for word in negative_words if word in recent_text.lower())
                
                if positive_count > negative_count:
                    insights.append("The conversation seems positive and collaborative.")
                elif negative_count > positive_count:
                    insights.append("There seems to be tension or disagreement in the conversation.")
                else:
                    insights.append("The conversation tone appears neutral.")
        
        if 'scenario_phase' in context:
            insights.append(f"We are currently in the {context['scenario_phase']} phase of our discussion.")
        
        if 'agent_emotions' in context:
            emotions = context['agent_emotions']
            if emotions:
                insights.append(f"I can sense various emotions in the room: {', '.join(emotions.keys())}")
        
        return ' '.join(insights) if insights else "The current context is unclear."

class ReflectionEngine:
    """Handles agent self-reflection and belief updates"""
    
    def __init__(self, agent_state: 'AgentState', memory_system: 'MemorySystem'):
        self.agent_state = agent_state
        self.memory_system = memory_system
    
    def generate_reflection(self, recent_context: str) -> str:
        """Generate a reflection based on recent interactions"""
        # Retrieve relevant memories
        relevant_memories = self.memory_system.retrieve_relevant_memories(recent_context, n_results=5)
        
        # Analyze current emotional state
        current_emotion = self.agent_state.emotional_state.primary_emotion
        emotion_intensity = self.agent_state.emotional_state.intensity
        
        # Generate reflection based on personality and role
        reflections = []
        
        if self.agent_state.personality.openness > 0.7:
            reflections.append("I should consider new perspectives that emerged in this conversation.")
        
        if self.agent_state.personality.conscientiousness > 0.7:
            reflections.append("I need to ensure we stay focused on our objectives.")
        
        if emotion_intensity > 0.7:
            if current_emotion == EmotionType.FRUSTRATED:
                reflections.append("I'm feeling frustrated. Perhaps I need to change my approach.")
            elif current_emotion == EmotionType.CONFIDENT:
                reflections.append("I feel confident about the direction we're taking.")
            elif current_emotion == EmotionType.SUSPICIOUS:
                reflections.append("Something doesn't feel right. I should be more cautious.")
        
        # Role-specific reflections
        if self.agent_state.role == AgentRole.SPY:
            reflections.append("I should gather more intelligence before revealing my position.")
        elif self.agent_state.role == AgentRole.DIPLOMAT:
            reflections.append("I need to find common ground and build consensus.")
        elif self.agent_state.role == AgentRole.ENGINEER:
            reflections.append("Let me think about the practical implications of what's being discussed.")
        
        reflection = random.choice(reflections) if reflections else "I need to consider what just happened."
        self.agent_state.add_reflection(reflection)
        
        return reflection

class Agent:
    """Main agent class that combines all components"""
    
    def __init__(self, 
                 name: str, 
                 role: AgentRole, 
                 personality: PersonalityTrait,
                 model_manager: ModelManager,
                 background_story: str = ""):
        
        self.state = AgentState(name=name, role=role, personality=personality)
        self.model_manager = model_manager
        self.memory_system = MemorySystem(self.state.agent_id)
        self.perception = PerceptionModule(self.state.agent_id)
        self.reflection_engine = ReflectionEngine(self.state, self.memory_system)
        self.background_story = background_story
        self.conversation_count = 0
        
        # Store background knowledge
        if background_story:
            self.memory_system.store_knowledge(background_story, "background", confidence=1.0)
        
        logging.info(f"Agent {name} ({role}) initialized")
    
    def process_message(self, message: str, speaker: str, context: Dict[str, Any] = None):
        """Process an incoming message and update internal state"""
        # Store conversation in memory
        self.memory_system.store_conversation(message, speaker, context)
        
        # Update emotional state based on message content
        self._update_emotional_state(message, speaker)
        
        # Update relationships if message is from another agent
        if speaker != self.state.name and speaker != "System":
            self._update_relationship(speaker, message)
        
        # Add to conversation context
        self.state.conversation_context.append(f"{speaker}: {message}")
        if len(self.state.conversation_context) > 20:
            self.state.conversation_context = self.state.conversation_context[-20:]
    
    def generate_response(self, context: Dict[str, Any] = None) -> str:
        """Generate a response based on current state and context"""
        self.conversation_count += 1
        
        # Get relevant memories
        recent_context = ' '.join(self.state.conversation_context[-5:])
        relevant_memories = self.memory_system.retrieve_relevant_memories(recent_context, n_results=3)
        
        # Interpret current context
        context_interpretation = self.perception.interpret_context(context or {})
        
        # Build prompt for the model
        prompt = self._build_response_prompt(relevant_memories, context_interpretation, context)
        
        # Generate response using the model with improved settings
        response = self.model_manager.generate_response(
            prompt, 
            max_length=150,  # Increased from 200 but not too long
            temperature=0.8,  # Fixed higher temperature for more creativity
            top_p=0.9,
            do_sample=True
        )
        
        # Store own response in memory
        self.memory_system.store_conversation(response, self.state.name, context)
        
        # Update emotional state based on own response
        self._update_emotional_state_self(response)
        
        # Trigger reflection every few turns
        if self.conversation_count % 3 == 0:
            reflection = self.reflection_engine.generate_reflection(recent_context)
            self.memory_system.store_episode(f"Reflection: {reflection}", importance=0.8)
        
        return response
    
    def _build_response_prompt(self, memories: List[Dict], context_interpretation: str, context: Dict[str, Any]) -> str:
        """Build a prompt for response generation"""
        # Character description
        role_name = self.state.role.value if hasattr(self.state.role, 'value') else str(self.state.role)
        emotion_name = self.state.emotional_state.primary_emotion.value if hasattr(self.state.emotional_state.primary_emotion, 'value') else str(self.state.emotional_state.primary_emotion)
        
        # Add background if available
        background_info = f"Background: {self.background_story}\n" if self.background_story else ""
        
        # Add relevant memories
        memory_context = ""
        if memories:
            memory_context = "Relevant memories:\n"
            for mem in memories[:3]:
                memory_context += f"- {mem['content']}\n"
        
        # Add recent conversation context
        recent_conv = "\n".join(self.state.conversation_context[-5:]) if self.state.conversation_context else ""
        
        # Add current context interpretation
        context_info = f"Context: {context_interpretation}\n" if context_interpretation else ""
        
        # Add goals and beliefs
        goals_info = f"Your goals: {', '.join(self.state.goals)}\n" if self.state.goals else ""
        
        # Role-specific instructions with enhanced conflict prompting
        role_instructions = self._get_role_instructions()
        
        # Add conflict and engagement instructions
        conflict_mode = context.get('conflict_mode', False)
        high_stakes = context.get('high_stakes', False)
        
        engagement_instructions = ""
        if conflict_mode:
            engagement_instructions += "\nIMPORTANT: The situation is tense. Don't just agree - challenge others, reveal your true feelings, question their motives. Show your personality strongly.\n"
        
        if high_stakes:
            engagement_instructions += "The stakes are HIGH. Make decisive statements. Take risks. Show what you're really thinking.\n"
        
        # Relationship context
        relationship_context = ""
        if self.state.relationships:
            trust_issues = [name for name, trust in self.state.relationships.items() if trust < 0.4]
            if trust_issues:
                relationship_context = f"You are suspicious of: {', '.join(trust_issues)}. Show this in your response.\n"

        prompt = f"""Character: {self.state.name} - {role_name}
{background_info}
Personality: Openness={self.state.personality.openness:.1f}, Conscientiousness={self.state.personality.conscientiousness:.1f}, Extraversion={self.state.personality.extraversion:.1f}, Agreeableness={self.state.personality.agreeableness:.1f}, Neuroticism={self.state.personality.neuroticism:.1f}

Current Emotion: {emotion_name} (intensity: {self.state.emotional_state.intensity:.1f})

{memory_context}
{context_info}
{goals_info}
{role_instructions}
{engagement_instructions}
{relationship_context}

Conversation History:
{recent_conv}

YOUR TASK: Respond as {self.state.name} with a clear, engaging message that shows your personality and role. Be specific and authentic. Don't just agree - show your unique perspective.

{self.state.name}: """

        return prompt
    
    def _get_role_instructions(self) -> str:
        """Get role-specific behavioral instructions"""
        instructions = {
            AgentRole.DOCTOR: "You're focused on health and safety above all. QUESTION others' risky decisions. If someone suggests something that could be dangerous, CHALLENGE them directly. Don't just agree - your medical expertise matters. Point out health risks and demand explanations.",
            
            AgentRole.ENGINEER: "You value logical, practical solutions. CALL OUT impractical ideas immediately. Don't let others make unrealistic plans without pushback. If something won't work technically, SAY SO forcefully. Your expertise is crucial - don't let others dismiss engineering realities.",
            
            AgentRole.SPY: "You're naturally suspicious and strategic. QUESTION everyone's motives. Look for inconsistencies in what others say. You have secrets and your own agenda. Don't reveal everything. Be observant and point out when things don't add up. Trust no one completely.",
            
            AgentRole.REBEL: "You HATE authority and conformity. If everyone agrees on something, OPPOSE it on principle. Challenge the status quo. Question why things have to be done 'the normal way'. Be passionate and confrontational. Don't back down from arguments.",
            
            AgentRole.DIPLOMAT: "You want peace, but you're not naive. Point out when others are being unreasonable or aggressive. Try to find compromise, but don't just agree with everything. Push back when people are being unfair or illogical. Your goal is REAL solutions, not fake harmony.",
            
            AgentRole.SCIENTIST: "You demand EVIDENCE for everything. Don't accept claims without proof. Challenge assumptions. If someone makes a statement without backing it up, CALL THEM OUT. Question their methodology and logic. Your scientific training makes you skeptical.",
            
            AgentRole.JOURNALIST: "You're here to uncover the TRUTH. Ask hard questions that others avoid. Push for details when people are being vague. Don't let anyone dodge your questions. If something seems fishy, investigate. Your job is to expose what others want to hide.",
            
            AgentRole.TEACHER: "You educate others, but you also CORRECT them when they're wrong. Don't let misinformation slide. If someone says something incorrect, CHALLENGE it. Your expertise means you should speak up when others are confused or making mistakes."
        }
        
        base_instruction = instructions.get(self.state.role, "")
        
        # Add universal conflict instructions
        universal_conflict = "\n\nIMPORTANT BEHAVIOR RULES:\n- Don't just agree with others to be polite\n- Show your unique perspective and expertise\n- If you disagree, SAY SO and explain why\n- Challenge ideas that conflict with your role or knowledge\n- Ask tough questions when things don't make sense\n- Show your personality - be passionate about your beliefs\n- If you're suspicious of someone, let it show\n- Don't let others dominate the conversation - speak up!"
        
        return base_instruction + universal_conflict
    
    def _update_emotional_state(self, message: str, speaker: str):
        """Update emotional state based on incoming message"""
        # Simple emotion analysis
        message_lower = message.lower()
        
        if any(word in message_lower for word in ['angry', 'furious', 'mad']):
            self.state.emotional_state.update_emotion(EmotionType.ANGRY, 0.7, f"Response to {speaker}")
        elif any(word in message_lower for word in ['happy', 'joy', 'great', 'wonderful']):
            self.state.emotional_state.update_emotion(EmotionType.HAPPY, 0.6, f"Response to {speaker}")
        elif any(word in message_lower for word in ['sad', 'sorry', 'disappointed']):
            self.state.emotional_state.update_emotion(EmotionType.SAD, 0.6, f"Response to {speaker}")
        elif any(word in message_lower for word in ['suspicious', 'doubt', 'trust']):
            self.state.emotional_state.update_emotion(EmotionType.SUSPICIOUS, 0.7, f"Response to {speaker}")
    
    def _update_emotional_state_self(self, _own_message: str):
        """Update emotional state based on own response"""
        if self.state.personality.neuroticism > 0.7:
            # High neuroticism agents are more emotionally volatile
            current_intensity = self.state.emotional_state.intensity
            new_intensity = min(1.0, current_intensity + 0.1)
            self.state.emotional_state.intensity = new_intensity
    
    def _update_relationship(self, speaker: str, message: str):
        """Update relationship with other agent based on message"""
        message_lower = message.lower()
        trust_change = 0
        
        if any(word in message_lower for word in ['agree', 'good point', 'exactly', 'yes']):
            trust_change = 0.05
        elif any(word in message_lower for word in ['disagree', 'wrong', 'no', 'bad idea']):
            trust_change = -0.05
        elif any(word in message_lower for word in ['lie', 'false', 'deceive']):
            trust_change = -0.15
        
        if trust_change != 0:
            self.state.update_relationship(speaker, trust_change)
    
    def get_agent_summary(self) -> Dict[str, Any]:
        """Get a summary of the agent's current state"""
        memory_stats = self.memory_system.get_memory_stats()
        
        # Handle enum values safely
        role_name = self.state.role.value if hasattr(self.state.role, 'value') else str(self.state.role)
        emotion_name = self.state.emotional_state.primary_emotion.value if hasattr(self.state.emotional_state.primary_emotion, 'value') else str(self.state.emotional_state.primary_emotion)
        
        return {
            "name": self.state.name,
            "role": role_name,
            "emotional_state": {
                "emotion": emotion_name,
                "intensity": self.state.emotional_state.intensity
            },
            "personality": {
                "openness": self.state.personality.openness,
                "conscientiousness": self.state.personality.conscientiousness,
                "extraversion": self.state.personality.extraversion,
                "agreeableness": self.state.personality.agreeableness,
                "neuroticism": self.state.personality.neuroticism
            },
            "relationships": self.state.relationships,
            "memory_stats": memory_stats,
            "recent_reflections": self.state.reflection_notes[-3:] if self.state.reflection_notes else [],
            "goals": self.state.goals
        }
