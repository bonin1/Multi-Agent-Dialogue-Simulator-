from typing import Dict, List, Optional, Any, Tuple
import chromadb
from chromadb.config import Settings
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import os
from datetime import datetime
import logging

class MemorySystem:
    """Advanced memory system using ChromaDB for vector storage and retrieval"""
    
    def __init__(self, agent_id: str, memory_dir: str = "memory_db"):
        self.agent_id = agent_id
        self.memory_dir = memory_dir
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ChromaDB
        os.makedirs(memory_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=memory_dir)
        
        # Create collections for different types of memories
        self.conversation_memory = self._get_or_create_collection("conversations")
        self.episodic_memory = self._get_or_create_collection("episodes")
        self.semantic_memory = self._get_or_create_collection("knowledge")
        self.emotional_memory = self._get_or_create_collection("emotions")
        
        logging.info(f"Memory system initialized for agent {agent_id}")
    
    def _get_or_create_collection(self, collection_type: str):
        """Get or create a ChromaDB collection"""
        collection_name = f"{self.agent_id}_{collection_type}"
        try:
            return self.client.get_collection(collection_name)
        except:
            return self.client.create_collection(collection_name)
    
    def store_conversation(self, message: str, speaker: str, context: Dict[str, Any] = None):
        """Store a conversation message with context"""
        timestamp = datetime.now().isoformat()
        
        # Serialize context for ChromaDB (only simple types allowed)
        serialized_context = {}
        if context:
            for key, value in context.items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    serialized_context[key] = value
                elif isinstance(value, list):
                    # Convert list to string representation
                    serialized_context[key] = str(len(value)) + "_items"
                else:
                    # Convert complex objects to string
                    serialized_context[key] = str(value)
        
        metadata = {
            "speaker": speaker,
            "timestamp": timestamp,
            "type": "conversation",
            **serialized_context
        }
        
        # Generate embedding
        embedding = self.embedding_model.encode(message).tolist()
        
        # Store in ChromaDB
        self.conversation_memory.add(
            embeddings=[embedding],
            documents=[message],
            metadatas=[metadata],
            ids=[f"conv_{timestamp}_{hash(message)}"]
        )
    
    def store_episode(self, episode: str, importance: float, emotions: List[str] = None):
        """Store an episodic memory with importance weighting"""
        timestamp = datetime.now().isoformat()
        
        # Serialize emotions list to string for ChromaDB
        emotions_str = ",".join(emotions) if emotions else ""
        
        metadata = {
            "timestamp": timestamp,
            "importance": importance,
            "emotions": emotions_str,
            "type": "episode"
        }
        
        embedding = self.embedding_model.encode(episode).tolist()
        
        self.episodic_memory.add(
            embeddings=[embedding],
            documents=[episode],
            metadatas=[metadata],
            ids=[f"episode_{timestamp}_{hash(episode)}"]
        )
    
    def store_knowledge(self, knowledge: str, category: str, confidence: float = 0.5):
        """Store semantic knowledge"""
        timestamp = datetime.now().isoformat()
        metadata = {
            "category": category,
            "confidence": confidence,
            "timestamp": timestamp,
            "type": "knowledge"
        }
        
        embedding = self.embedding_model.encode(knowledge).tolist()
        
        self.semantic_memory.add(
            embeddings=[embedding],
            documents=[knowledge],
            metadatas=[metadata],
            ids=[f"knowledge_{timestamp}_{hash(knowledge)}"]
        )
    
    def store_emotional_memory(self, event: str, emotion: str, intensity: float, trigger: str = ""):
        """Store emotional memories"""
        timestamp = datetime.now().isoformat()
        metadata = {
            "emotion": emotion,
            "intensity": intensity,
            "trigger": trigger,
            "timestamp": timestamp,
            "type": "emotional"
        }
        
        embedding = self.embedding_model.encode(f"{event} {emotion} {trigger}").tolist()
        
        self.emotional_memory.add(
            embeddings=[embedding],
            documents=[event],
            metadatas=[metadata],
            ids=[f"emotion_{timestamp}_{hash(event)}"]
        )
    
    def retrieve_relevant_memories(self, query: str, memory_type: str = "all", n_results: int = 5) -> List[Dict]:
        """Retrieve relevant memories based on semantic similarity"""
        query_embedding = self.embedding_model.encode(query).tolist()
        
        all_results = []
        
        collections = {
            "conversation": self.conversation_memory,
            "episode": self.episodic_memory,
            "knowledge": self.semantic_memory,
            "emotional": self.emotional_memory
        }
        
        target_collections = collections if memory_type == "all" else {memory_type: collections[memory_type]}
        
        for mem_type, collection in target_collections.items():
            try:
                # Check if collection exists and has documents
                count = collection.count()
                if count == 0:
                    continue
                    
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=min(n_results, count)
                )
                
                if results['documents'] and results['documents'][0]:
                    for i, doc in enumerate(results['documents'][0]):
                        memory_item = {
                            "content": doc,
                            "metadata": results['metadatas'][0][i],
                            "distance": results['distances'][0][i],
                            "type": mem_type
                        }
                        all_results.append(memory_item)
            except Exception as e:
                logging.warning(f"Error retrieving from {mem_type} collection: {e}")
                # Return empty results if ChromaDB has issues
                continue
        
        # Sort by relevance (lower distance = higher relevance)
        all_results.sort(key=lambda x: x['distance'])
        return all_results[:n_results]
    
    def get_recent_conversations(self, n_recent: int = 10) -> List[Dict]:
        """Get recent conversation history"""
        try:
            all_conversations = self.conversation_memory.get()
            if not all_conversations['documents']:
                return []
            
            # Sort by timestamp
            conversations_with_time = []
            for i, doc in enumerate(all_conversations['documents']):
                metadata = all_conversations['metadatas'][i]
                conversations_with_time.append({
                    "content": doc,
                    "metadata": metadata,
                    "timestamp": metadata.get('timestamp', '')
                })
            
            conversations_with_time.sort(key=lambda x: x['timestamp'], reverse=True)
            return conversations_with_time[:n_recent]
        except Exception as e:
            logging.warning(f"Error getting recent conversations: {e}")
            return []
    
    def reflect_on_memories(self, reflection_prompt: str) -> str:
        """Generate reflection based on stored memories"""
        relevant_memories = self.retrieve_relevant_memories(reflection_prompt, n_results=10)
        
        if not relevant_memories:
            return "No relevant memories to reflect upon."
        
        # Format memories for reflection
        memory_context = "\n".join([
            f"- {mem['content']} (Type: {mem['type']}, Relevance: {1-mem['distance']:.2f})"
            for mem in relevant_memories
        ])
        
        return memory_context
    
    def get_memory_stats(self) -> Dict[str, int]:
        """Get statistics about stored memories"""
        stats = {}
        collections = {
            "conversations": self.conversation_memory,
            "episodes": self.episodic_memory,
            "knowledge": self.semantic_memory,
            "emotions": self.emotional_memory
        }
        
        for name, collection in collections.items():
            try:
                stats[name] = collection.count()
            except Exception:
                stats[name] = 0
        
        return stats
