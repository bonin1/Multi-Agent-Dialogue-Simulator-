"""
Long-term reflection system for agents
"""
import asyncio
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging
import json


class Reflection:
    """Individual reflection entry"""
    
    def __init__(
        self,
        content: str,
        trigger_events: List[Dict[str, Any]],
        insights: List[str],
        emotional_state: Dict[str, float],
        timestamp: Optional[datetime] = None
    ):
        self.id = str(uuid.uuid4())
        self.content = content
        self.trigger_events = trigger_events
        self.insights = insights
        self.emotional_state = emotional_state
        self.timestamp = timestamp or datetime.now()
        self.impact_score = 0.5  # How much this reflection affects future behavior
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert reflection to dictionary"""
        return {
            "id": self.id,
            "content": self.content,
            "trigger_events": self.trigger_events,
            "insights": self.insights,
            "emotional_state": self.emotional_state,
            "timestamp": self.timestamp.isoformat(),
            "impact_score": self.impact_score
        }


class ReflectionSystem:
    """System for generating and managing agent reflections"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.reflections: List[Reflection] = []
        self.logger = logging.getLogger(f"ReflectionSystem-{agent_id}")
        self.last_reflection_time = datetime.now()
        self.reflection_triggers = {
            "conversation_count": 5,  # Reflect every 5 conversations
            "emotional_intensity": 0.7,  # Reflect on high emotions
            "relationship_change": 0.3,  # Reflect on significant relationship changes
            "goal_conflict": True,  # Reflect when goals conflict
        }
    
    async def generate_reflection(
        self,
        recent_experiences: List[Dict[str, Any]],
        current_state: Dict[str, Any]
    ) -> Optional[Reflection]:
        """Generate a reflection based on recent experiences"""
        
        if not recent_experiences:
            return None
        
        # Analyze recent experiences for reflection triggers
        should_reflect, trigger_reason = self._should_reflect(recent_experiences, current_state)
        
        if not should_reflect:
            return None
        
        # Generate reflection content
        reflection_content = await self._generate_reflection_content(
            recent_experiences, current_state, trigger_reason
        )
        
        # Extract insights
        insights = self._extract_insights(recent_experiences, current_state)
        
        # Create reflection
        reflection = Reflection(
            content=reflection_content,
            trigger_events=recent_experiences,
            insights=insights,
            emotional_state=current_state.get("emotional_state", {})
        )
        
        # Calculate impact score
        reflection.impact_score = self._calculate_impact_score(reflection)
        
        # Store reflection
        self.reflections.append(reflection)
        self.last_reflection_time = datetime.now()
        
        # Keep only recent reflections
        self._cleanup_old_reflections()
        
        self.logger.info(f"Generated reflection: {reflection_content[:100]}...")
        return reflection
    
    def _should_reflect(
        self,
        recent_experiences: List[Dict[str, Any]],
        current_state: Dict[str, Any]
    ) -> tuple[bool, str]:
        """Determine if agent should reflect"""
        
        # Time-based reflection
        time_since_last = datetime.now() - self.last_reflection_time
        if time_since_last > timedelta(hours=1):  # Reflect at least every hour
            return True, "time_based"
        
        # Emotional intensity trigger
        dominant_emotion_intensity = current_state.get("emotion_intensity", 0.0)
        if dominant_emotion_intensity > self.reflection_triggers["emotional_intensity"]:
            return True, "emotional_intensity"
        
        # Conversation count trigger
        if len(recent_experiences) >= self.reflection_triggers["conversation_count"]:
            return True, "conversation_count"
        
        # Conflict or disagreement trigger
        for experience in recent_experiences:
            message = experience.get("message", "").lower()
            if any(word in message for word in ["disagree", "conflict", "wrong", "argue"]):
                return True, "conflict_detected"
        
        # Relationship changes
        relationship_count = current_state.get("relationship_count", 0)
        if relationship_count > 0:  # New relationships formed
            return True, "relationship_change"
        
        return False, "no_trigger"
    
    async def _generate_reflection_content(
        self,
        recent_experiences: List[Dict[str, Any]],
        current_state: Dict[str, Any],
        trigger_reason: str
    ) -> str:
        """Generate reflection content based on experiences and state"""
        
        # Analyze the experiences
        analysis = self._analyze_experiences(recent_experiences)
        
        # Generate reflection based on trigger and analysis
        reflection_templates = {
            "emotional_intensity": [
                "I'm feeling quite {emotion} about recent events. I need to consider how this affects my interactions.",
                "The intensity of my {emotion} is notable. I should reflect on what's driving this.",
                "I'm experiencing strong {emotion}. This might be influencing my judgment."
            ],
            "conflict_detected": [
                "There seems to be some disagreement in our conversations. I should think about different perspectives.",
                "I notice conflict arising. Perhaps I need to adjust my approach to be more collaborative.",
                "Disagreements are occurring. I should consider whether I'm being too rigid in my thinking."
            ],
            "conversation_count": [
                "After several conversations, I'm noticing some patterns in how people respond to me.",
                "I've had multiple interactions now. I should consider what I'm learning about group dynamics.",
                "Looking back at recent conversations, I can see how my role affects the discussions."
            ],
            "relationship_change": [
                "My relationships with others are evolving. I should consider how this affects our collaboration.",
                "I'm building new connections. This changes how I should approach future interactions.",
                "The dynamics between us are shifting. I need to adapt my communication style."
            ],
            "time_based": [
                "Taking a moment to reflect on recent events and my responses.",
                "It's good to periodically consider how I'm progressing toward my goals.",
                "Regular reflection helps me understand my impact on others."
            ]
        }
        
        # Select appropriate template
        templates = reflection_templates.get(trigger_reason, reflection_templates["time_based"])
        base_reflection = templates[len(self.reflections) % len(templates)]
        
        # Fill in emotion if needed
        dominant_emotion = current_state.get("dominant_emotion", "neutral")
        reflection_content = base_reflection.format(emotion=dominant_emotion)
        
        # Add specific observations
        specific_observations = []
        
        if analysis["sentiment_trend"] == "negative":
            specific_observations.append("I notice conversations have been more challenging lately.")
        elif analysis["sentiment_trend"] == "positive":
            specific_observations.append("Recent interactions have been quite positive and productive.")
        
        if analysis["topic_diversity"] > 0.7:
            specific_observations.append("We've covered a wide range of topics, which shows good engagement.")
        elif analysis["topic_diversity"] < 0.3:
            specific_observations.append("Our conversations seem to be getting stuck on similar themes.")
        
        if analysis["interruption_frequency"] > 0.3:
            specific_observations.append("There's been quite a bit of interrupting. Maybe we need better turn-taking.")
        
        if specific_observations:
            reflection_content += " " + " ".join(specific_observations)
        
        return reflection_content
    
    def _analyze_experiences(self, experiences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze recent experiences for patterns"""
        if not experiences:
            return {}
        
        analysis = {
            "sentiment_trend": "neutral",
            "topic_diversity": 0.5,
            "interruption_frequency": 0.0,
            "message_length_trend": "stable",
            "emotional_volatility": 0.0
        }
        
        # Analyze sentiment
        positive_words = ["good", "great", "excellent", "agree", "yes", "perfect"]
        negative_words = ["bad", "terrible", "disagree", "no", "wrong", "awful"]
        
        positive_count = 0
        negative_count = 0
        
        for exp in experiences:
            message = exp.get("message", "").lower()
            positive_count += sum(1 for word in positive_words if word in message)
            negative_count += sum(1 for word in negative_words if word in message)
        
        if positive_count > negative_count * 1.5:
            analysis["sentiment_trend"] = "positive"
        elif negative_count > positive_count * 1.5:
            analysis["sentiment_trend"] = "negative"
        
        # Simple topic diversity (based on unique words)
        all_words = set()
        total_words = 0
        for exp in experiences:
            message = exp.get("message", "")
            words = message.lower().split()
            all_words.update(words)
            total_words += len(words)
        
        if total_words > 0:
            analysis["topic_diversity"] = len(all_words) / total_words
        
        # Message length trend
        lengths = [len(exp.get("message", "")) for exp in experiences]
        if len(lengths) > 1:
            if lengths[-1] > lengths[0] * 1.5:
                analysis["message_length_trend"] = "increasing"
            elif lengths[-1] < lengths[0] * 0.7:
                analysis["message_length_trend"] = "decreasing"
        
        return analysis
    
    def _extract_insights(
        self,
        recent_experiences: List[Dict[str, Any]],
        current_state: Dict[str, Any]
    ) -> List[str]:
        """Extract key insights from experiences"""
        insights = []
        
        # Role-based insights
        role = current_state.get("role", "unknown")
        if role == "doctor":
            insights.append("I should focus on the health and well-being aspects of our discussions.")
        elif role == "engineer":
            insights.append("I can contribute by breaking down complex problems systematically.")
        elif role == "spy":
            insights.append("I'm gathering valuable information about group dynamics.")
        elif role == "rebel":
            insights.append("I need to balance challenging ideas with maintaining relationships.")
        elif role == "diplomat":
            insights.append("My role is to help find common ground between different viewpoints.")
        
        # Emotional insights
        dominant_emotion = current_state.get("dominant_emotion", "neutral")
        if dominant_emotion == "anger":
            insights.append("My anger might be affecting my ability to collaborate effectively.")
        elif dominant_emotion == "fear":
            insights.append("I should address my concerns more directly rather than letting fear guide me.")
        elif dominant_emotion == "joy":
            insights.append("My positive mood is helping create a better atmosphere for everyone.")
        
        # Relationship insights
        relationship_count = current_state.get("relationship_count", 0)
        if relationship_count > 2:
            insights.append("I'm building a network of relationships that could be valuable for future collaboration.")
        elif relationship_count == 0:
            insights.append("I should focus more on building connections with others.")
        
        # Conversation pattern insights
        turn_count = current_state.get("turn_count", 0)
        if turn_count > 10:
            insights.append("I've been quite active in conversations. I should make sure others have space to contribute.")
        elif turn_count < 3:
            insights.append("I should participate more actively in discussions.")
        
        return insights
    
    def _calculate_impact_score(self, reflection: Reflection) -> float:
        """Calculate how much this reflection should impact future behavior"""
        impact = 0.5  # Base impact
        
        # Emotional intensity increases impact
        max_emotion = max(reflection.emotional_state.values()) if reflection.emotional_state else 0
        impact += max_emotion * 0.3
        
        # Number of insights increases impact
        impact += len(reflection.insights) * 0.05
        
        # Conflict-related reflections have higher impact
        if any("conflict" in insight.lower() or "disagree" in insight.lower() 
               for insight in reflection.insights):
            impact += 0.2
        
        return min(1.0, impact)
    
    def _cleanup_old_reflections(self, max_reflections: int = 50):
        """Remove old reflections to prevent memory bloat"""
        if len(self.reflections) > max_reflections:
            # Sort by timestamp and keep the most recent
            self.reflections.sort(key=lambda r: r.timestamp, reverse=True)
            self.reflections = self.reflections[:max_reflections]
    
    def get_recent_reflections(self, limit: int = 5) -> List[Reflection]:
        """Get recent reflections"""
        sorted_reflections = sorted(self.reflections, key=lambda r: r.timestamp, reverse=True)
        return sorted_reflections[:limit]
    
    def get_high_impact_reflections(self, limit: int = 5) -> List[Reflection]:
        """Get reflections with high impact scores"""
        sorted_reflections = sorted(self.reflections, key=lambda r: r.impact_score, reverse=True)
        return sorted_reflections[:limit]
    
    def get_reflection_summary(self) -> str:
        """Get a summary of recent reflections"""
        recent = self.get_recent_reflections(3)
        if not recent:
            return "No recent reflections available."
        
        summaries = []
        for reflection in recent:
            age_hours = (datetime.now() - reflection.timestamp).total_seconds() / 3600
            time_desc = f"{int(age_hours)} hours ago" if age_hours >= 1 else "recently"
            summaries.append(f"{time_desc}: {reflection.content[:100]}...")
        
        return "\n".join(summaries)
    
    def get_insights_by_category(self) -> Dict[str, List[str]]:
        """Get insights organized by category"""
        categories = {
            "emotional": [],
            "relationship": [],
            "role": [],
            "communication": [],
            "general": []
        }
        
        for reflection in self.reflections:
            for insight in reflection.insights:
                insight_lower = insight.lower()
                if any(word in insight_lower for word in ["feel", "emotion", "mood", "anger", "joy", "fear"]):
                    categories["emotional"].append(insight)
                elif any(word in insight_lower for word in ["relationship", "connection", "trust", "collaborate"]):
                    categories["relationship"].append(insight)
                elif any(word in insight_lower for word in ["role", "doctor", "engineer", "spy", "rebel", "diplomat"]):
                    categories["role"].append(insight)
                elif any(word in insight_lower for word in ["conversation", "discuss", "communicate", "listen"]):
                    categories["communication"].append(insight)
                else:
                    categories["general"].append(insight)
        
        return categories
    
    def get_reflection_stats(self) -> Dict[str, Any]:
        """Get statistics about reflections"""
        if not self.reflections:
            return {
                "total_reflections": 0,
                "avg_impact_score": 0.0,
                "total_insights": 0,
                "reflection_frequency": 0.0
            }
        
        total_impact = sum(r.impact_score for r in self.reflections)
        total_insights = sum(len(r.insights) for r in self.reflections)
        
        # Calculate reflection frequency (reflections per day)
        if len(self.reflections) > 1:
            time_span = (self.reflections[-1].timestamp - self.reflections[0].timestamp).total_seconds()
            frequency = len(self.reflections) / (time_span / 86400) if time_span > 0 else 0
        else:
            frequency = 0
        
        return {
            "total_reflections": len(self.reflections),
            "avg_impact_score": total_impact / len(self.reflections),
            "total_insights": total_insights,
            "reflection_frequency": frequency,
            "most_recent": self.reflections[-1].timestamp.isoformat() if self.reflections else None
        }
