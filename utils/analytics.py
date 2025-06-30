from typing import Dict, List, Any
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import json

class RelationshipAnalyzer:
    """Analyzes and visualizes relationships between agents"""
    
    def __init__(self):
        self.relationship_graph = nx.Graph()
        self.interaction_history = []
    
    def add_interaction(self, agent1: str, agent2: str, interaction_type: str, valence: float):
        """Add an interaction between two agents"""
        interaction = {
            'agent1': agent1,
            'agent2': agent2,
            'type': interaction_type,
            'valence': valence,  # -1 to 1, negative to positive
            'timestamp': datetime.now()
        }
        self.interaction_history.append(interaction)
        
        # Update graph
        if not self.relationship_graph.has_edge(agent1, agent2):
            self.relationship_graph.add_edge(agent1, agent2, weight=0, interactions=0)
        
        edge_data = self.relationship_graph[agent1][agent2]
        edge_data['interactions'] += 1
        edge_data['weight'] = (edge_data['weight'] + valence) / edge_data['interactions']
    
    def get_relationship_strength(self, agent1: str, agent2: str) -> float:
        """Get relationship strength between two agents"""
        if self.relationship_graph.has_edge(agent1, agent2):
            return self.relationship_graph[agent1][agent2]['weight']
        return 0.0
    
    def analyze_group_dynamics(self, agents: List[str]) -> Dict[str, Any]:
        """Analyze overall group dynamics"""
        if len(agents) < 2:
            return {}
        
        # Calculate network metrics
        subgraph = self.relationship_graph.subgraph(agents)
        
        analysis = {
            'network_density': nx.density(subgraph),
            'average_relationship_strength': np.mean([data['weight'] for _, _, data in subgraph.edges(data=True)]) if subgraph.edges() else 0,
            'most_connected': max(agents, key=lambda x: subgraph.degree(x)) if subgraph.nodes() else None,
            'potential_conflicts': [],
            'strong_alliances': []
        }
        
        # Find conflicts and alliances
        for agent1 in agents:
            for agent2 in agents:
                if agent1 < agent2 and subgraph.has_edge(agent1, agent2):
                    weight = subgraph[agent1][agent2]['weight']
                    if weight < -0.5:
                        analysis['potential_conflicts'].append((agent1, agent2, weight))
                    elif weight > 0.5:
                        analysis['strong_alliances'].append((agent1, agent2, weight))
        
        return analysis

class ConversationAnalyzer:
    """Analyzes conversation patterns and dynamics"""
    
    def __init__(self):
        self.conversation_data = []
    
    def add_message(self, speaker: str, message: str, timestamp: datetime, emotional_state: str):
        """Add a message to analysis"""
        self.conversation_data.append({
            'speaker': speaker,
            'message': message,
            'timestamp': timestamp,
            'emotional_state': emotional_state,
            'word_count': len(message.split()),
            'message_length': len(message)
        })
    
    def analyze_participation(self) -> Dict[str, Any]:
        """Analyze participation patterns"""
        if not self.conversation_data:
            return {}
        
        speaker_stats = {}
        for msg in self.conversation_data:
            speaker = msg['speaker']
            if speaker not in speaker_stats:
                speaker_stats[speaker] = {
                    'message_count': 0,
                    'total_words': 0,
                    'total_characters': 0,
                    'emotions': []
                }
            
            speaker_stats[speaker]['message_count'] += 1
            speaker_stats[speaker]['total_words'] += msg['word_count']
            speaker_stats[speaker]['total_characters'] += msg['message_length']
            speaker_stats[speaker]['emotions'].append(msg['emotional_state'])
        
        # Calculate averages and dominance
        total_messages = len(self.conversation_data)
        for speaker, stats in speaker_stats.items():
            stats['participation_rate'] = stats['message_count'] / total_messages
            stats['avg_words_per_message'] = stats['total_words'] / stats['message_count']
            stats['dominant_emotion'] = max(set(stats['emotions']), key=stats['emotions'].count)
        
        return speaker_stats
    
    def analyze_conversation_flow(self) -> Dict[str, Any]:
        """Analyze conversation flow and turn-taking patterns"""
        if len(self.conversation_data) < 2:
            return {}
        
        turn_patterns = []
        speaker_sequences = [msg['speaker'] for msg in self.conversation_data]
        
        for i in range(len(speaker_sequences) - 1):
            current_speaker = speaker_sequences[i]
            next_speaker = speaker_sequences[i + 1]
            if current_speaker != next_speaker:
                turn_patterns.append((current_speaker, next_speaker))
        
        # Analyze interruption patterns
        interruption_count = {}
        for current, next_speaker in turn_patterns:
            if current not in interruption_count:
                interruption_count[current] = {'interrupted_by': {}, 'interrupts': {}}
            if next_speaker not in interruption_count:
                interruption_count[next_speaker] = {'interrupted_by': {}, 'interrupts': {}}
            
            interruption_count[current]['interrupted_by'][next_speaker] = interruption_count[current]['interrupted_by'].get(next_speaker, 0) + 1
            interruption_count[next_speaker]['interrupts'][current] = interruption_count[next_speaker]['interrupts'].get(current, 0) + 1
        
        return {
            'turn_patterns': turn_patterns,
            'interruption_patterns': interruption_count,
            'conversation_length': len(self.conversation_data),
            'unique_speakers': len(set(speaker_sequences))
        }
    
    def analyze_emotional_dynamics(self) -> Dict[str, Any]:
        """Analyze emotional patterns in conversation"""
        if not self.conversation_data:
            return {}
        
        emotions_over_time = [msg['emotional_state'] for msg in self.conversation_data]
        timestamps = [msg['timestamp'] for msg in self.conversation_data]
        
        # Emotional trajectory
        emotion_changes = []
        for i in range(1, len(emotions_over_time)):
            if emotions_over_time[i] != emotions_over_time[i-1]:
                emotion_changes.append({
                    'from': emotions_over_time[i-1],
                    'to': emotions_over_time[i],
                    'timestamp': timestamps[i],
                    'speaker': self.conversation_data[i]['speaker']
                })
        
        # Most common emotions
        emotion_frequency = {}
        for emotion in emotions_over_time:
            emotion_frequency[emotion] = emotion_frequency.get(emotion, 0) + 1
        
        return {
            'emotion_changes': emotion_changes,
            'emotion_frequency': emotion_frequency,
            'emotional_volatility': len(emotion_changes) / len(emotions_over_time) if emotions_over_time else 0,
            'dominant_emotion': max(emotion_frequency.items(), key=lambda x: x[1])[0] if emotion_frequency else None
        }

class PerformanceMetrics:
    """Track simulation performance and quality metrics"""
    
    def __init__(self):
        self.metrics = {
            'response_times': [],
            'conversation_quality_scores': [],
            'agent_coherence_scores': {},
            'memory_efficiency': {},
            'simulation_stability': []
        }
    
    def record_response_time(self, agent_name: str, response_time: float):
        """Record response generation time"""
        self.metrics['response_times'].append({
            'agent': agent_name,
            'time': response_time,
            'timestamp': datetime.now()
        })
    
    def rate_conversation_quality(self, score: float):
        """Rate overall conversation quality (0-1)"""
        self.metrics['conversation_quality_scores'].append({
            'score': score,
            'timestamp': datetime.now()
        })
    
    def assess_agent_coherence(self, agent_name: str, coherence_score: float):
        """Assess how coherent an agent's responses are"""
        if agent_name not in self.metrics['agent_coherence_scores']:
            self.metrics['agent_coherence_scores'][agent_name] = []
        
        self.metrics['agent_coherence_scores'][agent_name].append({
            'score': coherence_score,
            'timestamp': datetime.now()
        })
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        summary = {}
        
        # Average response time
        if self.metrics['response_times']:
            avg_response_time = np.mean([rt['time'] for rt in self.metrics['response_times']])
            summary['avg_response_time'] = avg_response_time
        
        # Conversation quality trend
        if self.metrics['conversation_quality_scores']:
            quality_scores = [cq['score'] for cq in self.metrics['conversation_quality_scores']]
            summary['avg_conversation_quality'] = np.mean(quality_scores)
            summary['quality_trend'] = 'improving' if len(quality_scores) > 1 and quality_scores[-1] > quality_scores[0] else 'stable'
        
        # Agent coherence
        coherence_summary = {}
        for agent, scores in self.metrics['agent_coherence_scores'].items():
            if scores:
                coherence_summary[agent] = {
                    'avg_coherence': np.mean([s['score'] for s in scores]),
                    'coherence_stability': np.std([s['score'] for s in scores])
                }
        summary['agent_coherence'] = coherence_summary
        
        return summary
    
    def export_metrics(self, filepath: str):
        """Export metrics to JSON file"""
        # Convert datetime objects to strings for JSON serialization
        serializable_metrics = {}
        for key, value in self.metrics.items():
            if isinstance(value, list):
                serializable_metrics[key] = [
                    {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in item.items()}
                    if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                serializable_metrics[key] = value
        
        with open(filepath, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
