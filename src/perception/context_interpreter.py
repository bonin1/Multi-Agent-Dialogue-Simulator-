"""
Context interpreter and perception module for agents
"""
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging


class ContextInterpreter:
    """Interprets conversational context and environmental cues"""
    
    def __init__(self):
        self.logger = logging.getLogger("ContextInterpreter")
        
        # Emotion keywords for sentiment analysis
        self.emotion_keywords = {
            "joy": ["happy", "excited", "thrilled", "delighted", "pleased", "cheerful", "joyful"],
            "sadness": ["sad", "depressed", "disappointed", "melancholy", "sorrowful", "unhappy"],
            "anger": ["angry", "furious", "irritated", "annoyed", "outraged", "mad", "frustrated"],
            "fear": ["afraid", "scared", "worried", "anxious", "nervous", "concerned", "terrified"],
            "surprise": ["surprised", "amazed", "astonished", "shocked", "stunned", "unexpected"],
            "disgust": ["disgusted", "revolted", "repulsed", "sickened", "appalled"]
        }
        
        # Intent keywords
        self.intent_keywords = {
            "agreement": ["agree", "yes", "absolutely", "exactly", "right", "correct", "definitely"],
            "disagreement": ["disagree", "no", "wrong", "incorrect", "never", "absolutely not"],
            "question": ["what", "how", "why", "when", "where", "who", "which", "?"],
            "suggestion": ["suggest", "recommend", "propose", "maybe", "perhaps", "could we"],
            "criticism": ["problem", "issue", "concern", "worry", "mistake", "error", "flaw"],
            "support": ["help", "assist", "support", "back", "endorse", "encourage"],
            "information": ["explain", "tell me", "describe", "clarify", "elaborate", "details"]
        }
        
        # Topic keywords
        self.topic_keywords = {
            "medical": ["health", "medical", "doctor", "patient", "treatment", "diagnosis", "symptoms"],
            "technical": ["system", "software", "technical", "engineering", "data", "algorithm", "code"],
            "political": ["policy", "government", "political", "election", "legislation", "public"],
            "social": ["community", "social", "people", "relationship", "family", "friends"],
            "business": ["business", "company", "market", "profit", "strategy", "customer", "sales"],
            "environmental": ["environment", "climate", "pollution", "sustainability", "green", "ecology"],
            "educational": ["education", "learning", "teaching", "school", "university", "knowledge"]
        }
    
    def analyze_message(self, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Comprehensive analysis of a message"""
        analysis = {
            "message": message,
            "timestamp": datetime.now(),
            "sentiment": self._analyze_sentiment(message),
            "emotions": self._detect_emotions(message),
            "intent": self._detect_intent(message),
            "topics": self._detect_topics(message),
            "urgency": self._assess_urgency(message),
            "formality": self._assess_formality(message),
            "complexity": self._assess_complexity(message),
            "questions": self._extract_questions(message),
            "key_phrases": self._extract_key_phrases(message),
            "meta_info": self._extract_meta_info(message)
        }
        
        # Add contextual analysis if context provided
        if context:
            analysis["contextual"] = self._analyze_context(message, context)
        
        return analysis
    
    def _analyze_sentiment(self, message: str) -> Dict[str, float]:
        """Analyze sentiment polarity and intensity"""
        message_lower = message.lower()
        
        positive_words = [
            "good", "great", "excellent", "wonderful", "fantastic", "amazing", "brilliant",
            "perfect", "outstanding", "superb", "love", "like", "enjoy", "appreciate"
        ]
        
        negative_words = [
            "bad", "terrible", "awful", "horrible", "hate", "dislike", "wrong",
            "stupid", "ridiculous", "absurd", "worst", "disgusting", "annoying"
        ]
        
        intensifiers = ["very", "extremely", "incredibly", "absolutely", "completely", "totally"]
        
        positive_score = 0
        negative_score = 0
        intensity_multiplier = 1.0
        
        # Check for intensifiers
        for intensifier in intensifiers:
            if intensifier in message_lower:
                intensity_multiplier = 1.5
                break
        
        # Count positive and negative words
        for word in positive_words:
            if word in message_lower:
                positive_score += intensity_multiplier
        
        for word in negative_words:
            if word in message_lower:
                negative_score += intensity_multiplier
        
        # Normalize scores
        total_words = len(message.split())
        if total_words > 0:
            positive_score = min(1.0, positive_score / total_words * 10)
            negative_score = min(1.0, negative_score / total_words * 10)
        
        # Calculate overall polarity
        if positive_score > negative_score:
            polarity = "positive"
            confidence = positive_score / (positive_score + negative_score) if (positive_score + negative_score) > 0 else 0.5
        elif negative_score > positive_score:
            polarity = "negative"
            confidence = negative_score / (positive_score + negative_score) if (positive_score + negative_score) > 0 else 0.5
        else:
            polarity = "neutral"
            confidence = 0.5
        
        return {
            "polarity": polarity,
            "confidence": confidence,
            "positive_score": positive_score,
            "negative_score": negative_score,
            "intensity": intensity_multiplier
        }
    
    def _detect_emotions(self, message: str) -> Dict[str, float]:
        """Detect emotions in the message"""
        message_lower = message.lower()
        emotion_scores = {}
        
        for emotion, keywords in self.emotion_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in message_lower:
                    # Check for context around the keyword
                    if f"not {keyword}" in message_lower or f"don't {keyword}" in message_lower:
                        score -= 0.5  # Negation reduces score
                    else:
                        score += 1
            
            # Normalize by message length
            total_words = len(message.split())
            if total_words > 0:
                emotion_scores[emotion] = min(1.0, score / total_words * 20)
            else:
                emotion_scores[emotion] = 0.0
        
        return emotion_scores
    
    def _detect_intent(self, message: str) -> Dict[str, float]:
        """Detect speaker intent"""
        message_lower = message.lower()
        intent_scores = {}
        
        for intent, keywords in self.intent_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in message_lower:
                    score += 1
            
            # Special handling for questions
            if intent == "question" and "?" in message:
                score += 2
            
            # Normalize
            total_words = len(message.split())
            if total_words > 0:
                intent_scores[intent] = min(1.0, score / total_words * 10)
            else:
                intent_scores[intent] = 0.0
        
        return intent_scores
    
    def _detect_topics(self, message: str) -> Dict[str, float]:
        """Detect topics being discussed"""
        message_lower = message.lower()
        topic_scores = {}
        
        for topic, keywords in self.topic_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in message_lower:
                    score += 1
            
            # Normalize
            total_words = len(message.split())
            if total_words > 0:
                topic_scores[topic] = min(1.0, score / total_words * 15)
            else:
                topic_scores[topic] = 0.0
        
        return topic_scores
    
    def _assess_urgency(self, message: str) -> float:
        """Assess urgency level of the message"""
        message_lower = message.lower()
        urgency_indicators = [
            "urgent", "emergency", "immediately", "asap", "critical", "crisis",
            "quickly", "hurry", "rush", "deadline", "time sensitive", "now"
        ]
        
        punctuation_urgency = message.count("!") * 0.2
        caps_urgency = sum(1 for char in message if char.isupper()) / len(message) if len(message) > 0 else 0
        
        keyword_urgency = 0
        for indicator in urgency_indicators:
            if indicator in message_lower:
                keyword_urgency += 0.3
        
        total_urgency = min(1.0, punctuation_urgency + caps_urgency + keyword_urgency)
        return total_urgency
    
    def _assess_formality(self, message: str) -> float:
        """Assess formality level of the message"""
        formal_indicators = [
            "please", "thank you", "sir", "madam", "respectfully", "sincerely",
            "accordingly", "furthermore", "therefore", "consequently"
        ]
        
        informal_indicators = [
            "yeah", "nah", "gonna", "wanna", "ain't", "lol", "omg", "btw",
            "cool", "awesome", "dude", "guys", "hey", "hi"
        ]
        
        message_lower = message.lower()
        formal_score = sum(1 for indicator in formal_indicators if indicator in message_lower)
        informal_score = sum(1 for indicator in informal_indicators if indicator in message_lower)
        
        # Check sentence structure
        complete_sentences = len([s for s in message.split('.') if len(s.strip()) > 5])
        total_parts = len(message.split('.'))
        
        structure_formality = complete_sentences / total_parts if total_parts > 0 else 0
        
        # Combine scores
        total_words = len(message.split())
        if total_words > 0:
            formal_ratio = formal_score / total_words * 10
            informal_ratio = informal_score / total_words * 10
            
            if formal_ratio > informal_ratio:
                formality = 0.5 + (formal_ratio * 0.3) + (structure_formality * 0.2)
            else:
                formality = 0.5 - (informal_ratio * 0.3) - ((1 - structure_formality) * 0.2)
            
            return max(0.0, min(1.0, formality))
        
        return 0.5
    
    def _assess_complexity(self, message: str) -> float:
        """Assess linguistic complexity of the message"""
        words = message.split()
        if not words:
            return 0.0
        
        # Average word length
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Sentence length
        sentences = [s.strip() for s in message.split('.') if s.strip()]
        avg_sentence_length = len(words) / len(sentences) if sentences else len(words)
        
        # Complex punctuation
        complex_punct = message.count(';') + message.count(':') + message.count('(') + message.count(')')
        
        # Calculate complexity score
        word_complexity = min(1.0, (avg_word_length - 3) / 7)  # Normalize around 3-10 letter words
        sentence_complexity = min(1.0, (avg_sentence_length - 5) / 20)  # Normalize around 5-25 word sentences
        punct_complexity = min(1.0, complex_punct / len(words) * 10) if words else 0
        
        total_complexity = (word_complexity + sentence_complexity + punct_complexity) / 3
        return max(0.0, min(1.0, total_complexity))
    
    def _extract_questions(self, message: str) -> List[str]:
        """Extract questions from the message"""
        # Split by sentence and find questions
        sentences = re.split(r'[.!]', message)
        questions = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence.endswith('?') or any(sentence.lower().startswith(qword) for qword in ['what', 'how', 'why', 'when', 'where', 'who', 'which']):
                questions.append(sentence)
        
        return questions
    
    def _extract_key_phrases(self, message: str) -> List[str]:
        """Extract key phrases and important terms"""
        # Simple key phrase extraction
        words = message.lower().split()
        
        # Filter out common stop words
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "is", "are", "was", "were", "be", "been", "have",
            "has", "had", "do", "does", "did", "will", "would", "could", "should",
            "may", "might", "can", "i", "you", "he", "she", "it", "we", "they"
        }
        
        key_words = [word for word in words if len(word) > 3 and word not in stop_words]
        
        # Find phrases (simple bigrams and trigrams)
        phrases = []
        for i in range(len(key_words) - 1):
            phrases.append(f"{key_words[i]} {key_words[i+1]}")
        
        for i in range(len(key_words) - 2):
            phrases.append(f"{key_words[i]} {key_words[i+1]} {key_words[i+2]}")
        
        # Return unique key words and phrases
        return list(set(key_words + phrases))
    
    def _extract_meta_info(self, message: str) -> Dict[str, Any]:
        """Extract meta information about the message"""
        return {
            "word_count": len(message.split()),
            "character_count": len(message),
            "sentence_count": len([s for s in message.split('.') if s.strip()]),
            "question_count": message.count('?'),
            "exclamation_count": message.count('!'),
            "capital_ratio": sum(1 for char in message if char.isupper()) / len(message) if message else 0,
            "punctuation_density": sum(1 for char in message if not char.isalnum() and not char.isspace()) / len(message) if message else 0
        }
    
    def _analyze_context(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze message in relation to provided context"""
        contextual_analysis = {
            "relevance_to_topic": 0.0,
            "response_appropriateness": 0.0,
            "conversation_flow": 0.0,
            "role_consistency": 0.0
        }
        
        # Analyze relevance to current topic
        if "current_topic" in context:
            current_topic = context["current_topic"].lower()
            message_lower = message.lower()
            
            # Simple keyword overlap
            topic_words = current_topic.split()
            message_words = message_lower.split()
            overlap = len(set(topic_words) & set(message_words))
            
            if topic_words:
                contextual_analysis["relevance_to_topic"] = min(1.0, overlap / len(topic_words))
        
        # Analyze response appropriateness
        if "previous_message" in context:
            prev_message = context["previous_message"].lower()
            message_lower = message.lower()
            
            # Check if response addresses previous message
            if any(word in message_lower for word in ["yes", "no", "agree", "disagree"]):
                contextual_analysis["response_appropriateness"] += 0.3
            
            # Check for question-answer pattern
            if "?" in prev_message and not "?" in message:
                contextual_analysis["response_appropriateness"] += 0.4
        
        # Analyze conversation flow
        if "conversation_history" in context:
            history = context["conversation_history"]
            if len(history) > 1:
                # Check for topic consistency
                recent_topics = [self._get_dominant_topic(turn.get("content", "")) for turn in history[-3:]]
                current_topic = self._get_dominant_topic(message)
                
                if current_topic in recent_topics:
                    contextual_analysis["conversation_flow"] = 0.8
                else:
                    contextual_analysis["conversation_flow"] = 0.3
        
        # Analyze role consistency
        if "agent_role" in context:
            role = context["agent_role"]
            role_consistency = self._check_role_consistency(message, role)
            contextual_analysis["role_consistency"] = role_consistency
        
        return contextual_analysis
    
    def _get_dominant_topic(self, message: str) -> str:
        """Get the dominant topic of a message"""
        topics = self._detect_topics(message)
        if topics:
            return max(topics.items(), key=lambda x: x[1])[0]
        return "general"
    
    def _check_role_consistency(self, message: str, role: str) -> float:
        """Check if message is consistent with agent role"""
        role_patterns = {
            "doctor": ["health", "medical", "patient", "treatment", "diagnosis", "care"],
            "engineer": ["system", "technical", "solution", "design", "build", "optimize"],
            "spy": ["observe", "analyze", "gather", "intelligence", "careful", "strategic"],
            "rebel": ["change", "challenge", "fight", "revolution", "system", "freedom"],
            "diplomat": ["negotiate", "compromise", "peace", "agreement", "understand", "mediate"]
        }
        
        if role not in role_patterns:
            return 0.5  # Neutral if role not recognized
        
        message_lower = message.lower()
        pattern_words = role_patterns[role]
        
        matches = sum(1 for word in pattern_words if word in message_lower)
        consistency = min(1.0, matches / len(pattern_words) * 3)
        
        return consistency
    
    def summarize_conversation_context(self, conversation_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize the overall context of a conversation"""
        if not conversation_history:
            return {}
        
        # Analyze all messages
        all_analyses = []
        for turn in conversation_history:
            if "content" in turn:
                analysis = self.analyze_message(turn["content"])
                analysis["speaker"] = turn.get("speaker", "Unknown")
                all_analyses.append(analysis)
        
        # Aggregate sentiment over time
        sentiment_over_time = []
        for analysis in all_analyses:
            sentiment = analysis["sentiment"]["polarity"]
            confidence = analysis["sentiment"]["confidence"]
            sentiment_over_time.append({"sentiment": sentiment, "confidence": confidence})
        
        # Dominant topics
        all_topics = {}
        for analysis in all_analyses:
            for topic, score in analysis["topics"].items():
                all_topics[topic] = all_topics.get(topic, 0) + score
        
        dominant_topics = sorted(all_topics.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Average metrics
        avg_urgency = sum(analysis["urgency"] for analysis in all_analyses) / len(all_analyses)
        avg_formality = sum(analysis["formality"] for analysis in all_analyses) / len(all_analyses)
        avg_complexity = sum(analysis["complexity"] for analysis in all_analyses) / len(all_analyses)
        
        # Conversation dynamics
        speaker_participation = {}
        for analysis in all_analyses:
            speaker = analysis["speaker"]
            if speaker not in speaker_participation:
                speaker_participation[speaker] = {"count": 0, "total_words": 0}
            
            speaker_participation[speaker]["count"] += 1
            speaker_participation[speaker]["total_words"] += analysis["meta_info"]["word_count"]
        
        return {
            "total_turns": len(all_analyses),
            "sentiment_trajectory": sentiment_over_time,
            "dominant_topics": dominant_topics,
            "average_urgency": avg_urgency,
            "average_formality": avg_formality,
            "average_complexity": avg_complexity,
            "speaker_participation": speaker_participation,
            "conversation_length": sum(analysis["meta_info"]["word_count"] for analysis in all_analyses),
            "question_density": sum(analysis["meta_info"]["question_count"] for analysis in all_analyses) / len(all_analyses),
            "emotional_intensity": max(max(analysis["emotions"].values()) for analysis in all_analyses) if all_analyses else 0
        }
