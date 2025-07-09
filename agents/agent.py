from typing import Dict, List, Optional, Any
import json
import logging
from datetime import datetime
import random

try:
    from models.agent_models import AgentState, AgentRole, PersonalityTrait, EmotionalState, EmotionType
    from models.model_manager import ModelManager
    from memory.memory_system import MemorySystem
    from prompts.agent_prompts import AgentPromptManager
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
        """Generate a reflection based on recent interactions using advanced prompts"""
        # Simple reflection for now - could be enhanced to use the model for more sophisticated reflection
        
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
    
    def generate_advanced_reflection(self, recent_experiences: str) -> str:
        """Generate an advanced reflection using the prompt manager"""
        # Get emotion name safely
        emotion_name = self.agent_state.emotional_state.primary_emotion.value if hasattr(self.agent_state.emotional_state.primary_emotion, 'value') else str(self.agent_state.emotional_state.primary_emotion)
        
        # Prepare personality traits as a dict
        personality_dict = {
            'openness': self.agent_state.personality.openness,
            'conscientiousness': self.agent_state.personality.conscientiousness,
            'extraversion': self.agent_state.personality.extraversion,
            'agreeableness': self.agent_state.personality.agreeableness,
            'neuroticism': self.agent_state.personality.neuroticism
        }
        
        # Get reflection prompt
        reflection_prompt = self.agent_state.prompt_manager.get_reflection_prompt(
            agent_name=self.agent_state.name,
            recent_experiences=recent_experiences,
            personality=personality_dict,
            current_emotion=emotion_name
        )
        
        # Could generate using model for more sophisticated reflection
        # For now, use a simple approach
        return f"Reflecting on recent events... {emotion_name} about what happened."

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
        self.prompt_manager = AgentPromptManager()
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
        
        # Generate response using the model with human-like settings
        response = self.model_manager.generate_response(
            prompt, 
            max_length=150,  # Shorter responses for natural conversation
            temperature=0.9,  # Higher temperature for more natural variation
            top_p=0.95,  # Allow more variety in word choice
            do_sample=True,
            repetition_penalty=1.4  # Higher penalty to prevent repetitive responses
        )
        
        # Clean up response formatting
        response = self._clean_response(response)
        
        # Store own response in memory
        self.memory_system.store_conversation(response, self.state.name, context)
        
        # Update emotional state based on conversation content
        self._update_emotional_state_from_context(context)
        
        # Trigger reflection every few turns
        if self.conversation_count % 3 == 0:
            reflection = self.reflection_engine.generate_reflection(recent_context)
            self.memory_system.store_episode(f"Reflection: {reflection}", importance=0.8)
        
        return response
    
    def _build_response_prompt(self, memories: List[Dict], context_interpretation: str, context: Dict[str, Any]) -> str:
        """Build a prompt for response generation using the advanced prompt manager"""
        
        # Prepare personality traits as a dict
        personality_dict = {
            'openness': self.state.personality.openness,
            'conscientiousness': self.state.personality.conscientiousness,
            'extraversion': self.state.personality.extraversion,
            'agreeableness': self.state.personality.agreeableness,
            'neuroticism': self.state.personality.neuroticism
        }
        
        # Prepare recent conversation context - prioritize conversation flow
        recent_conversation = []
        if context and context.get('conversation_flow'):
            # Build conversation context from message history for better flow
            recent_messages = context.get('recent_messages', [])
            for msg in recent_messages[-4:]:  # Get last 4 messages
                if isinstance(msg, dict):
                    speaker = msg.get('speaker', 'Unknown')
                    message = msg.get('message', '')
                    if speaker and message:
                        recent_conversation.append(f"{speaker}: {message}")
        else:
            # Fallback to old method
            recent_conversation = self.state.conversation_context[-10:] if self.state.conversation_context else []
        
        # Get emotion name safely
        emotion_name = self.state.emotional_state.primary_emotion.value if hasattr(self.state.emotional_state.primary_emotion, 'value') else str(self.state.emotional_state.primary_emotion)
        
        # Prepare relationships
        relationships = self.state.relationships if hasattr(self.state, 'relationships') else {}
        
        # Prepare enhanced context with conversation flow priority
        enhanced_context = {
            'conflict_mode': context.get('conflict_mode', False),
            'high_stakes': context.get('high_stakes', False),
            'scenario_phase': context.get('scenario_phase', 'discussion'),
            'context_interpretation': context_interpretation,
            'memories': memories,
            'encourage_response': context.get('encourage_response', False),
            'conversation_flow': context.get('conversation_flow', False),
            'last_speaker': context.get('last_speaker', None),
            'last_message': context.get('last_message', None)
        }
        
        # Use the advanced prompt manager to build the prompt
        return self.prompt_manager.build_main_prompt(
            agent_name=self.state.name,
            role=self.state.role,
            personality=personality_dict,
            emotion=emotion_name,
            emotion_intensity=self.state.emotional_state.intensity,
            recent_conversation=recent_conversation,
            relationships=relationships,
            context=enhanced_context
        )
    
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
    
    def _clean_response(self, response: str) -> str:
        """Clean up the response to remove formatting artifacts and ensure pure dialogue"""
        import re
        
        # Remove bullet points and structural formatting
        response = response.strip()
        
        # Remove bullet points at the start
        if response.startswith('- '):
            response = response[2:].strip()
        
        # Remove numbered lists at the start
        if response and response[0].isdigit() and '. ' in response[:5]:
            first_dot = response.find('. ')
            if first_dot != -1:
                response = response[first_dot + 2:].strip()
        
        # Remove character count indicators
        if 'characters remaining' in response:
            lines = response.split('\n')
            response = '\n'.join([line for line in lines if 'characters remaining' not in line])
        
        # Remove structural markers
        response = response.replace('[to ', '').replace('] ', '')
        response = response.replace('**', '').replace('***', '')
        
        # CRITICAL: Remove stage directions and narrative text
        # Remove text in asterisks (like *crosses arms* or *sighs*)
        response = re.sub(r'\*[^*]*\*', '', response)
        
        # Remove text in parentheses that describes actions
        response = re.sub(r'\([^)]*\)', '', response)
        
        # Remove narrative patterns like "she says", "he responds", etc.
        narrative_patterns = [
            r'\b(he|she|they|[A-Z][a-z]+)\s+(says?|responds?|replies?|answers?|asks?|tells?|explains?|exclaims?|whispers?|shouts?|mutters?)\b',
            r'\b(says?|responds?|replies?|answers?|asks?|tells?|explains?|exclaims?|whispers?|shouts?|mutters?)\s+([A-Z][a-z]+)\b',
            r'\bas\s+(he|she|they|[A-Z][a-z]+)\s+(crosses?|folds?|waves?|points?|gestures?|looks?|stares?|glances?|turns?|walks?|steps?|moves?|leans?)',
            r'\bwhile\s+(crossing|folding|waving|pointing|gesturing|looking|staring|glancing|turning|walking|stepping|moving|leaning)',
            r'\bwith\s+a\s+(sigh|smile|frown|grimace|laugh|chuckle|smirk)',
            r'\bin\s+a\s+(frustrated|angry|sad|happy|excited|serious|stern|gentle|loud|quiet)\s+tone',
            r'\bwhile\s+(sighing|smiling|frowning|laughing|chuckling|smirking)',
            r'\b(angrily|sadly|happily|excitedly|seriously|sternly|gently|loudly|quietly)\s+(at|to|towards?)\s+',
            r'\bas\s+(he|she|they)\s+(lean|walk|step|move|turn)',
        ]
        
        for pattern in narrative_patterns:
            response = re.sub(pattern, '', response, flags=re.IGNORECASE)
        
        # Remove action descriptions at the beginning or end
        response = re.sub(r'^[^a-zA-Z0-9"\']*', '', response)  # Remove non-speech at start
        
        # Clean up multiple spaces and newlines
        response = re.sub(r'\s+', ' ', response).strip()
        
        # Ensure it doesn't start with quotes unless it's actual dialogue
        if response.startswith('"') and response.count('"') == 1:
            response = response[1:]
        
        # Remove incomplete sentences at the end
        if response and response[-1] not in '.!?':
            # Find the last complete sentence
            last_punct = max(
                response.rfind('.'),
                response.rfind('!'),
                response.rfind('?')
            )
            if last_punct > len(response) * 0.6:  # Only if it's not too short
                response = response[:last_punct + 1]
        
        # Final cleanup - remove any remaining empty text or just punctuation
        if not response or len(response.strip('.,!? ')) < 3:
            return "I see what you mean."  # Fallback response
        
        return response.strip()
    
    def _update_emotional_state_from_context(self, context: Dict[str, Any]):
        """Update emotional state based on conversation context"""
        import random
        
        # Get the scenario context
        scenario_phase = context.get('scenario_phase', '')
        recent_messages = context.get('recent_messages', [])
        
        # Analyze recent conversation for emotional triggers
        if recent_messages:
            last_messages = [msg.get('message', '') for msg in recent_messages[-2:]]
            combined_text = ' '.join(last_messages).lower()
            
            # Role-specific emotional triggers
            if self.state.role == AgentRole.REBEL:
                if any(word in combined_text for word in ['compromise', 'gradual', 'slow', 'economic concerns']):
                    self._set_emotion(EmotionType.FRUSTRATED, 0.7, "Frustrated by slow action")
                elif any(word in combined_text for word in ['bold', 'immediate', 'action', 'justice']):
                    self._set_emotion(EmotionType.EXCITED, 0.6, "Excited about bold action")
            
            elif self.state.role == AgentRole.SCIENTIST:
                if any(word in combined_text for word in ['data', 'evidence', 'study', 'research']):
                    self._set_emotion(EmotionType.CONFIDENT, 0.6, "Confident in data")
                elif any(word in combined_text for word in ['ignore', 'dismiss', 'politics']):
                    self._set_emotion(EmotionType.FRUSTRATED, 0.7, "Frustrated by politics over science")
            
            elif self.state.role == AgentRole.DIPLOMAT:
                if any(word in combined_text for word in ['agreement', 'compromise', 'together']):
                    self._set_emotion(EmotionType.HAPPY, 0.6, "Happy about collaboration")
                elif any(word in combined_text for word in ['conflict', 'disagreement', 'impossible']):
                    self._set_emotion(EmotionType.ANXIOUS, 0.6, "Anxious about breakdown")
            
            # General emotional responses
            if any(word in combined_text for word in ['deaths', 'dying', 'crisis', 'catastrophe']):
                if random.random() < 0.7:  # 70% chance to react emotionally
                    self._set_emotion(EmotionType.SAD, 0.6, "Sad about human cost")
            
            elif any(word in combined_text for word in ['jobs', 'economy', 'unemployment']):
                if self.state.personality.conscientiousness > 0.6:
                    self._set_emotion(EmotionType.ANXIOUS, 0.5, "Anxious about economic impact")
        
        # Gradual emotional decay toward neutral
        current_intensity = self.state.emotional_state.intensity
        if current_intensity > 0.1:
            new_intensity = max(0.1, current_intensity - 0.1)
            self.state.emotional_state.intensity = new_intensity
    
    def _set_emotion(self, emotion_type: EmotionType, intensity: float, reason: str):
        """Set a specific emotion with intensity"""
        self.state.emotional_state.update_emotion(emotion_type, intensity, reason)

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
