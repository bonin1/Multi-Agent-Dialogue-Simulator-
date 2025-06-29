"""
Dialogue manager for orchestrating multi-agent conversations
"""
import asyncio
import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
import random

from ..agents.base_agent import Agent, create_agent
from ..agents.personality import PersonalityGenerator
from ..utils.relationship_graph import RelationshipGraph
from ..utils.model_manager import get_model_manager
from .scenarios import ScenarioManager


class ConversationTurn:
    """Represents a single turn in the conversation"""
    
    def __init__(
        self,
        speaker_id: str,
        content: str,
        timestamp: datetime,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.speaker_id = speaker_id
        self.content = content
        self.timestamp = timestamp
        self.metadata = metadata or {}


class DialogueManager:
    """Manages multi-agent dialogue simulation"""
    
    def __init__(
        self,
        num_agents: int = 3,
        scenario: str = "team_building",
        config: Optional[Dict[str, Any]] = None
    ):
        self.config = config or {}
        self.logger = logging.getLogger("DialogueManager")
        
        # Core components
        self.agents: Dict[str, Agent] = {}
        self.relationship_graph = RelationshipGraph()
        self.scenario_manager = ScenarioManager()
        self.model_manager = None
        
        # Conversation state
        self.conversation_history: List[ConversationTurn] = []
        self.current_scenario = None
        self.is_running = False
        self.turn_order: List[str] = []
        self.current_turn_index = 0
        
        # Initialize simulation
        self._initialize_agents(num_agents)
        self._load_scenario(scenario)
        
    def _initialize_agents(self, num_agents: int):
        """Initialize agents with diverse personalities and roles"""
        available_roles = ["doctor", "engineer", "spy", "rebel", "diplomat"]
        
        # Select roles for agents
        if num_agents <= len(available_roles):
            selected_roles = available_roles[:num_agents]
        else:
            # Repeat roles if we need more agents
            selected_roles = available_roles * (num_agents // len(available_roles) + 1)
            selected_roles = selected_roles[:num_agents]
        
        # Generate personalities
        personalities = PersonalityGenerator.create_team_personalities(
            num_agents=num_agents,
            diversity_level=0.7
        )
        
        # Create agents
        for i, (role, personality) in enumerate(zip(selected_roles, personalities)):
            agent_name = f"{role.title()}{i+1}"
            agent = create_agent(
                name=agent_name,
                role=role,
                personality=personality
            )
            
            self.agents[agent.agent_id] = agent
            
            # Add to relationship graph
            self.relationship_graph.add_agent(
                agent.agent_id,
                {
                    "name": agent.name,
                    "role": agent.role,
                    "personality": personality.model_dump()
                }
            )
        
        # Set initial turn order
        self.turn_order = list(self.agents.keys())
        random.shuffle(self.turn_order)
        
        self.logger.info(f"Initialized {num_agents} agents with roles: {selected_roles}")
    
    def _load_scenario(self, scenario_name: str):
        """Load and set up scenario"""
        self.current_scenario = self.scenario_manager.get_scenario(scenario_name)
        if not self.current_scenario:
            self.logger.warning(f"Scenario '{scenario_name}' not found, using default")
            self.current_scenario = self.scenario_manager.get_default_scenario()
        
        self.logger.info(f"Loaded scenario: {self.current_scenario['description']}")
    
    async def initialize_model(self):
        """Initialize the language model"""
        self.model_manager = get_model_manager()
        if not self.model_manager.is_loaded:
            success = self.model_manager.load_model()
            if not success:
                self.logger.error("Failed to load language model")
                return False
        
        self.logger.info("Language model ready")
        return True
    
    async def start_conversation(self, opening_message: Optional[str] = None) -> str:
        """Start the conversation with an opening message"""
        if not self.current_scenario:
            raise RuntimeError("No scenario loaded")
        
        # Initialize model if needed
        if not self.model_manager:
            await self.initialize_model()
        
        # Generate opening message if not provided
        if not opening_message:
            opening_message = self.scenario_manager.generate_opening_message(
                self.current_scenario,
                list(self.agents.values())
            )
        
        # Add opening to conversation history
        opening_turn = ConversationTurn(
            speaker_id="system",
            content=opening_message,
            timestamp=datetime.now(),
            metadata={"type": "opening", "scenario": self.current_scenario["description"]}
        )
        self.conversation_history.append(opening_turn)
        
        self.is_running = True
        self.logger.info("Conversation started")
        
        return opening_message
    
    async def simulate_conversation(
        self,
        max_turns: int = 20,
        turn_timeout: float = 30.0
    ) -> List[Dict[str, Any]]:
        """Simulate a complete conversation"""
        
        # Start conversation
        opening = await self.start_conversation()
        results = [{"speaker": "System", "content": opening, "timestamp": datetime.now()}]
        
        for turn_num in range(max_turns):
            if not self.is_running:
                break
            
            try:
                # Get next speaker
                speaker_agent = await self._get_next_speaker()
                if not speaker_agent:
                    self.logger.info("No more speakers available")
                    break
                
                # Generate response
                response = await asyncio.wait_for(
                    self._generate_agent_response(speaker_agent),
                    timeout=turn_timeout
                )
                
                if not response:
                    self.logger.warning(f"No response from {speaker_agent.name}")
                    continue
                
                # Add to conversation
                turn = ConversationTurn(
                    speaker_id=speaker_agent.agent_id,
                    content=response,
                    timestamp=datetime.now(),
                    metadata={"turn": turn_num, "agent_name": speaker_agent.name}
                )
                self.conversation_history.append(turn)
                
                # Update relationship graph
                await self._update_relationships(speaker_agent, response)
                
                # Add to results
                results.append({
                    "speaker": speaker_agent.name,
                    "content": response,
                    "timestamp": turn.timestamp,
                    "agent_id": speaker_agent.agent_id,
                    "role": speaker_agent.role
                })
                
                # Check for conversation ending conditions
                if await self._should_end_conversation(turn_num):
                    self.logger.info("Conversation ended naturally")
                    break
                
                # Small delay between turns
                await asyncio.sleep(0.1)
                
            except asyncio.TimeoutError:
                self.logger.warning(f"Turn {turn_num} timed out")
                continue
            except Exception as e:
                self.logger.error(f"Error in turn {turn_num}: {e}")
                continue
        
        self.is_running = False
        self.logger.info(f"Conversation completed with {len(results)} turns")
        
        return results
    
    async def _get_next_speaker(self) -> Optional[Agent]:
        """Determine who should speak next"""
        if not self.agents:
            return None
        
        # Check for interruptions
        for agent in self.agents.values():
            if agent.should_interrupt(self._get_current_speaker_id()):
                self.logger.debug(f"{agent.name} interrupting")
                return agent
        
        # Normal turn order
        if self.turn_order:
            current_speaker_id = self.turn_order[self.current_turn_index]
            self.current_turn_index = (self.current_turn_index + 1) % len(self.turn_order)
            return self.agents.get(current_speaker_id)
        
        # Fallback: random selection
        return random.choice(list(self.agents.values()))
    
    def _get_current_speaker_id(self) -> Optional[str]:
        """Get the ID of the last speaker"""
        if self.conversation_history:
            return self.conversation_history[-1].speaker_id
        return None
    
    async def _generate_agent_response(self, agent: Agent) -> Optional[str]:
        """Generate response for a specific agent"""
        try:
            # Get conversation context
            recent_history = self.conversation_history[-5:]  # Last 5 turns
            
            # Build context for the agent
            context = {
                "scenario": self.current_scenario,
                "conversation_history": [
                    {
                        "speaker": self._get_speaker_name(turn.speaker_id),
                        "content": turn.content,
                        "timestamp": turn.timestamp
                    }
                    for turn in recent_history
                ],
                "other_agents": [
                    {
                        "name": other_agent.name,
                        "role": other_agent.role
                    }
                    for other_agent in self.agents.values()
                    if other_agent.agent_id != agent.agent_id
                ]
            }
            
            # Get the last message to respond to
            last_message = recent_history[-1].content if recent_history else ""
            last_speaker_id = recent_history[-1].speaker_id if recent_history else "system"
            
            # Generate response using agent's process_message method
            if self.model_manager and self.model_manager.is_loaded:
                response = await self._generate_llm_response(agent, last_message, context)
            else:
                # Fallback to agent's built-in response generation
                response = await agent.process_message(last_message, last_speaker_id, context)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating response for {agent.name}: {e}")
            return None
    
    async def _generate_llm_response(
        self,
        agent: Agent,
        message: str,
        context: Dict[str, Any]
    ) -> str:
        """Generate response using the language model"""
        try:
            # Build agent context for the model
            agent_context = {
                "agent_name": agent.name,
                "agent_role": agent.role,
                "personality_description": agent.personality.get_trait_description(),
                "dominant_emotion": agent.emotional_state.get_dominant_emotion()[0],
                "emotion_intensity": agent.emotional_state.get_dominant_emotion()[1],
                "current_goal": agent.current_intent.primary_goal,
                "strategy": agent.current_intent.current_strategy,
                "relevant_memories": [],  # Could add memory retrieval here
                "response_style": agent.personality.get_response_style(),
                "speaking_pattern": agent.personality.generate_speaking_pattern(),
            }
            
            # Get relationship context
            if context["conversation_history"]:
                last_speaker = context["conversation_history"][-1]["speaker"]
                # Find agent ID for last speaker
                last_speaker_id = None
                for aid, a in self.agents.items():
                    if a.name == last_speaker:
                        last_speaker_id = aid
                        break
                
                if last_speaker_id:
                    agent_context["relationship_context"] = agent.get_relationship_context(last_speaker_id)
            
            # Format prompt for the model
            prompt = self.model_manager.format_conversation_prompt(
                agent_context=agent_context,
                conversation_history=context["conversation_history"],
                current_message=message
            )
            
            # Generate response
            response = self.model_manager.generate_response(
                prompt=prompt,
                max_new_tokens=150,  # Use max_new_tokens instead of max_length
                temperature=0.7 + agent.personality.openness * 0.2,  # Vary by personality
                top_p=0.9,
                repetition_penalty=1.1
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"LLM generation failed for {agent.name}: {e}")
            # Fallback to agent's built-in response
            return await agent.process_message(message, "unknown", context)
    
    def _get_speaker_name(self, speaker_id: str) -> str:
        """Get speaker name from ID"""
        if speaker_id == "system":
            return "System"
        
        agent = self.agents.get(speaker_id)
        return agent.name if agent else "Unknown"
    
    async def _update_relationships(self, speaker: Agent, message: str):
        """Update relationship graph based on message content"""
        # Simple relationship update based on message sentiment
        # In a full implementation, this would be more sophisticated
        
        message_lower = message.lower()
        
        for other_agent in self.agents.values():
            if other_agent.agent_id == speaker.agent_id:
                continue
            
            # Determine relationship change
            relationship_change = 0.0
            
            if any(word in message_lower for word in ["agree", "good idea", "excellent", "yes"]):
                relationship_change = 0.1
            elif any(word in message_lower for word in ["disagree", "wrong", "no", "bad idea"]):
                relationship_change = -0.1
            elif any(word in message_lower for word in ["help", "support", "together"]):
                relationship_change = 0.15
            elif any(word in message_lower for word in ["stupid", "ridiculous", "absurd"]):
                relationship_change = -0.2
            
            if relationship_change != 0.0:
                # Update relationship in graph
                relationship_type = "cooperation" if relationship_change > 0 else "conflict"
                
                self.relationship_graph.add_relationship(
                    from_agent=speaker.agent_id,
                    to_agent=other_agent.agent_id,
                    relationship_type=relationship_type,
                    strength=relationship_change,
                    metadata={
                        "triggered_by": message[:100],
                        "timestamp": datetime.now().isoformat()
                    }
                )
                
                # Update agent's internal relationship tracking
                interaction_type = "agreement" if relationship_change > 0 else "disagreement"
                speaker.update_relationship(
                    other_agent.agent_id,
                    interaction_type,
                    abs(relationship_change)
                )
    
    async def _should_end_conversation(self, turn_num: int) -> bool:
        """Determine if conversation should end"""
        # Don't end too early - require at least 5 turns
        if turn_num < 5:
            return False
            
        # End if scenario goals are met (but be more strict about it)
        if self.current_scenario and turn_num > 8:  # Only check after substantial conversation
            goals = self.current_scenario.get("goals", [])
            if goals and self._are_goals_achieved(goals):
                return True
        
        # End if all agents agree on something (but require more consensus)
        if len(self.conversation_history) >= 6:  # Need at least 6 messages
            recent_messages = [turn.content.lower() for turn in self.conversation_history[-4:]]
            if all("agree" in msg or "consensus" in msg or "settled" in msg for msg in recent_messages):
                return True
        
        # End if conversation becomes repetitive (but be more lenient)
        if len(self.conversation_history) > 15:
            recent_content = [turn.content for turn in self.conversation_history[-8:]]
            if len(set(recent_content)) < 3:  # Too much repetition
                return True
        
        return False
    
    def _are_goals_achieved(self, goals: List[str]) -> bool:
        """Check if scenario goals are achieved (more conservative)"""
        # This is a placeholder - in a full implementation, this would be more sophisticated
        if not self.conversation_history or len(self.conversation_history) < 8:
            return False
        
        # Require more substantial conversation content
        recent_messages = " ".join([turn.content.lower() for turn in self.conversation_history[-8:]])
        
        # Count how many goals seem to be addressed
        goals_met = 0
        
        for goal in goals:
            goal_keywords = {
                "Build trust": ["trust", "reliable", "depend on", "confident", "trustworthy"],
                "Establish roles": ["role", "responsibility", "expertise", "specialization", "duties"],
                "Create synergy": ["together", "combine", "synergy", "collaborate", "teamwork", "cooperation"],
                "Find compromise": ["compromise", "middle ground", "agreement", "both sides", "balance"],
                "Save lives": ["rescue", "help", "save", "lives", "safety", "protect"],
                "Share knowledge": ["share", "explain", "teach", "knowledge", "learn", "expertise"]
            }
            
            keywords = goal_keywords.get(goal, goal.lower().split())
            # Require multiple keyword matches for each goal
            keyword_matches = sum(1 for keyword in keywords if keyword in recent_messages)
            if keyword_matches >= 2:  # Need at least 2 keyword matches per goal
                goals_met += 1
        
        # Only consider goals achieved if most goals are substantially addressed
        return goals_met >= len(goals) * 0.8  # 80% of goals need to be addressed
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of the conversation"""
        if not self.conversation_history:
            return {}
        
        # Basic statistics
        agent_participation = {}
        total_words = 0
        
        for turn in self.conversation_history:
            if turn.speaker_id != "system":
                speaker_name = self._get_speaker_name(turn.speaker_id)
                if speaker_name not in agent_participation:
                    agent_participation[speaker_name] = {"turns": 0, "words": 0}
                
                agent_participation[speaker_name]["turns"] += 1
                words = len(turn.content.split())
                agent_participation[speaker_name]["words"] += words
                total_words += words
        
        # Relationship summary
        relationships = self.relationship_graph.get_network_metrics()
        
        # Emotional analysis (simplified)
        emotional_turns = []
        for turn in self.conversation_history:
            content = turn.content.lower()
            if any(word in content for word in ["angry", "frustrated", "upset", "mad"]):
                emotional_turns.append({"type": "negative", "speaker": self._get_speaker_name(turn.speaker_id)})
            elif any(word in content for word in ["happy", "excited", "great", "wonderful"]):
                emotional_turns.append({"type": "positive", "speaker": self._get_speaker_name(turn.speaker_id)})
        
        return {
            "total_turns": len(self.conversation_history),
            "duration_minutes": (self.conversation_history[-1].timestamp - self.conversation_history[0].timestamp).total_seconds() / 60,
            "total_words": total_words,
            "agent_participation": agent_participation,
            "relationship_metrics": relationships,
            "emotional_moments": emotional_turns,
            "scenario": self.current_scenario["description"] if self.current_scenario else None,
            "final_relationships": {
                agent.name: len(agent.relationships) 
                for agent in self.agents.values()
            }
        }
    
    def export_conversation(self, format_type: str = "json") -> str:
        """Export conversation in various formats"""
        if format_type == "json":
            import json
            data = {
                "conversation": [
                    {
                        "timestamp": turn.timestamp.isoformat(),
                        "speaker": self._get_speaker_name(turn.speaker_id),
                        "content": turn.content,
                        "metadata": turn.metadata
                    }
                    for turn in self.conversation_history
                ],
                "summary": self.get_conversation_summary(),
                "agents": {
                    agent.name: agent.get_agent_state()
                    for agent in self.agents.values()
                },
                "relationships": self.relationship_graph.export_graph_data()
            }
            return json.dumps(data, indent=2, default=str)
        
        elif format_type == "text":
            lines = [f"Conversation: {self.current_scenario['description']}\n"]
            lines.append("=" * 50)
            
            for turn in self.conversation_history:
                speaker = self._get_speaker_name(turn.speaker_id)
                timestamp = turn.timestamp.strftime("%H:%M:%S")
                lines.append(f"\n[{timestamp}] {speaker}: {turn.content}")
            
            return "\n".join(lines)
        
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    async def add_agent(self, name: str, role: str, personality=None) -> str:
        """Add a new agent to the conversation"""
        agent = create_agent(name, role, personality)
        self.agents[agent.agent_id] = agent
        
        # Add to relationship graph
        self.relationship_graph.add_agent(
            agent.agent_id,
            {
                "name": agent.name,
                "role": agent.role,
                "personality": agent.personality.model_dump()
            }
        )
        
        # Update turn order
        self.turn_order.append(agent.agent_id)
        
        self.logger.info(f"Added new agent: {name} ({role})")
        return agent.agent_id
    
    def remove_agent(self, agent_id: str):
        """Remove an agent from the conversation"""
        if agent_id in self.agents:
            agent_name = self.agents[agent_id].name
            del self.agents[agent_id]
            
            # Remove from turn order
            if agent_id in self.turn_order:
                self.turn_order.remove(agent_id)
            
            self.logger.info(f"Removed agent: {agent_name}")
    
    def pause_conversation(self):
        """Pause the conversation"""
        self.is_running = False
        self.logger.info("Conversation paused")
    
    def resume_conversation(self):
        """Resume the conversation"""
        self.is_running = True
        self.logger.info("Conversation resumed")
