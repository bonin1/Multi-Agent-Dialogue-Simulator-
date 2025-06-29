"""
Vector-based memory system using ChromaDB for agent memories
"""
import asyncio
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging
import json

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class MemoryItem:
    """Individual memory item"""
    
    def __init__(
        self,
        content: str,
        metadata: Dict[str, Any],
        timestamp: Optional[datetime] = None,
        importance: float = 0.5
    ):
        self.id = str(uuid.uuid4())
        self.content = content
        self.metadata = metadata
        self.timestamp = timestamp or datetime.now()
        self.importance = importance
        self.access_count = 0
        self.last_accessed = self.timestamp
    
    def access(self):
        """Record access to this memory"""
        self.access_count += 1
        self.last_accessed = datetime.now()
    
    def get_age_hours(self) -> float:
        """Get age of memory in hours"""
        return (datetime.now() - self.timestamp).total_seconds() / 3600
    
    def calculate_relevance_score(self, query_embedding: List[float], current_time: Optional[datetime] = None) -> float:
        """Calculate relevance score based on content similarity, recency, and importance"""
        if current_time is None:
            current_time = datetime.now()
        
        # Base relevance (would be calculated using embeddings in full implementation)
        base_relevance = 0.5  # Placeholder
        
        # Recency factor (exponential decay)
        age_hours = (current_time - self.timestamp).total_seconds() / 3600
        recency_factor = max(0.1, 0.99 ** age_hours)  # Decay over time
        
        # Importance factor
        importance_factor = self.importance
        
        # Access frequency factor
        access_factor = min(1.0, self.access_count * 0.1)
        
        return (
            base_relevance * 0.6 +
            recency_factor * 0.2 +
            importance_factor * 0.15 +
            access_factor * 0.05
        )


class AgentMemory:
    """Vector-based memory system for agents"""
    
    def __init__(self, agent_id: str, memory_config: Optional[Dict[str, Any]] = None):
        self.agent_id = agent_id
        self.config = memory_config or self._get_default_config()
        self.logger = logging.getLogger(f"AgentMemory-{agent_id}")
        
        # In-memory storage as fallback
        self.memories: List[MemoryItem] = []
        self.embedding_cache: Dict[str, List[float]] = {}
        
        # Initialize vector database if available
        self.chroma_client = None
        self.collection = None
        self.embedding_model = None
        
        self._initialize_storage()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default memory configuration"""
        return {
            "max_memories": 1000,
            "similarity_threshold": 0.7,
            "importance_threshold": 0.3,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "chromadb_path": "./data/agent_memories",
            "collection_name": f"agent_{self.agent_id}_memories"
        }
    
    def _initialize_storage(self):
        """Initialize storage backend"""
        if CHROMADB_AVAILABLE:
            try:
                self.chroma_client = chromadb.PersistentClient(
                    path=self.config["chromadb_path"]
                )
                collection_name = self.config["collection_name"].replace("-", "_")
                self.collection = self.chroma_client.get_or_create_collection(
                    name=collection_name
                )
                self.logger.info("ChromaDB initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize ChromaDB: {e}")
                self.chroma_client = None
        
        if SENTENCE_TRANSFORMERS_AVAILABLE and not self.embedding_model:
            try:
                self.embedding_model = SentenceTransformer(
                    self.config["embedding_model"]
                )
                self.logger.info("Sentence transformer model loaded")
            except Exception as e:
                self.logger.warning(f"Failed to load embedding model: {e}")
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text"""
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        if self.embedding_model:
            try:
                embedding = self.embedding_model.encode(text).tolist()
                self.embedding_cache[text] = embedding
                return embedding
            except Exception as e:
                self.logger.warning(f"Failed to generate embedding: {e}")
        
        # Fallback: simple hash-based pseudo-embedding
        import hashlib
        hash_obj = hashlib.md5(text.encode())
        pseudo_embedding = [
            float(int(hash_obj.hexdigest()[i:i+2], 16)) / 255.0
            for i in range(0, min(32, len(hash_obj.hexdigest())), 2)
        ]
        # Pad to minimum length
        while len(pseudo_embedding) < 16:
            pseudo_embedding.append(0.0)
        
        self.embedding_cache[text] = pseudo_embedding
        return pseudo_embedding
    
    async def store_memory(
        self,
        content: str,
        metadata: Dict[str, Any],
        importance: float = 0.5
    ) -> str:
        """Store a new memory"""
        memory_item = MemoryItem(
            content=content,
            metadata=metadata,
            importance=importance
        )
        
        # Add to in-memory storage
        self.memories.append(memory_item)
        
        # Store in vector database if available
        if self.collection:
            try:
                embedding = self._get_embedding(content)
                
                # Prepare metadata for ChromaDB (only simple types allowed)
                chroma_metadata = {
                    "memory_id": memory_item.id,
                    "timestamp": memory_item.timestamp.isoformat(),
                    "importance": importance,
                    "agent_id": self.agent_id
                }
                
                # Add simple metadata values only
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        chroma_metadata[key] = value
                    elif value is None:
                        chroma_metadata[key] = None
                    else:
                        # Convert complex objects to strings
                        chroma_metadata[f"{key}_str"] = str(value)
                
                self.collection.add(
                    embeddings=[embedding],
                    documents=[content],
                    metadatas=[chroma_metadata],
                    ids=[memory_item.id]
                )
            except Exception as e:
                self.logger.warning(f"Failed to store in ChromaDB: {e}")
        
        # Cleanup old memories if necessary
        await self._cleanup_memories()
        
        return memory_item.id
    
    async def store_conversation(
        self,
        content: str,
        metadata: Dict[str, Any]
    ) -> str:
        """Store conversation memory with automatic importance scoring"""
        # Calculate importance based on content and context
        importance = self._calculate_conversation_importance(content, metadata)
        
        return await self.store_memory(
            content=content,
            metadata={
                **metadata,
                "type": "conversation",
                "agent_id": self.agent_id
            },
            importance=importance
        )
    
    def _calculate_conversation_importance(
        self,
        content: str,
        metadata: Dict[str, Any]
    ) -> float:
        """Calculate importance score for conversation content"""
        importance = 0.3  # Base importance
        
        content_lower = content.lower()
        
        # Emotional content is more important
        emotional_words = [
            "love", "hate", "excited", "angry", "sad", "happy",
            "frustrated", "pleased", "worried", "confident"
        ]
        if any(word in content_lower for word in emotional_words):
            importance += 0.2
        
        # Questions and answers are important
        if "?" in content or content_lower.startswith(("what", "how", "why", "when", "where")):
            importance += 0.15
        
        # Personal information is important
        personal_indicators = ["i am", "i feel", "i think", "my", "me"]
        if any(indicator in content_lower for indicator in personal_indicators):
            importance += 0.1
        
        # Disagreements and conflicts are important
        conflict_words = ["disagree", "wrong", "no", "never", "against"]
        if any(word in content_lower for word in conflict_words):
            importance += 0.15
        
        # Length factor (longer messages tend to be more important)
        if len(content) > 100:
            importance += 0.1
        
        return min(1.0, importance)
    
    async def retrieve_relevant_memories(
        self,
        query: str,
        limit: int = 5,
        minimum_relevance: float = 0.3
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant memories based on query"""
        query_embedding = self._get_embedding(query)
        
        # Try ChromaDB first
        if self.collection:
            try:
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=limit * 2  # Get more to filter
                )
                
                memories = []
                for i, doc in enumerate(results["documents"][0]):
                    metadata = results["metadatas"][0][i]
                    distance = results["distances"][0][i] if "distances" in results else 0.5
                    relevance = 1.0 - distance  # Convert distance to similarity
                    
                    if relevance >= minimum_relevance:
                        memories.append({
                            "content": doc,
                            "metadata": metadata,
                            "relevance": relevance,
                            "id": metadata.get("memory_id", "unknown")
                        })
                
                # Sort by relevance and return top results
                memories.sort(key=lambda x: x["relevance"], reverse=True)
                return memories[:limit]
                
            except Exception as e:
                self.logger.warning(f"ChromaDB query failed: {e}")
        
        # Fallback to in-memory search
        scored_memories = []
        for memory in self.memories:
            relevance_score = memory.calculate_relevance_score(query_embedding)
            if relevance_score >= minimum_relevance:
                scored_memories.append({
                    "content": memory.content,
                    "metadata": memory.metadata,
                    "relevance": relevance_score,
                    "id": memory.id
                })
                memory.access()  # Record access
        
        # Sort by relevance and return top results
        scored_memories.sort(key=lambda x: x["relevance"], reverse=True)
        return scored_memories[:limit]
    
    async def get_recent_memories(
        self,
        hours_back: int = 24,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get recent memories within specified time window"""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        recent_memories = []
        for memory in self.memories:
            if memory.timestamp >= cutoff_time:
                recent_memories.append({
                    "content": memory.content,
                    "metadata": memory.metadata,
                    "timestamp": memory.timestamp,
                    "importance": memory.importance,
                    "id": memory.id
                })
        
        # Sort by timestamp (most recent first)
        recent_memories.sort(key=lambda x: x["timestamp"], reverse=True)
        return recent_memories[:limit]
    
    async def get_important_memories(
        self,
        importance_threshold: float = 0.7,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get memories above importance threshold"""
        important_memories = []
        for memory in self.memories:
            if memory.importance >= importance_threshold:
                important_memories.append({
                    "content": memory.content,
                    "metadata": memory.metadata,
                    "timestamp": memory.timestamp,
                    "importance": memory.importance,
                    "id": memory.id
                })
        
        # Sort by importance
        important_memories.sort(key=lambda x: x["importance"], reverse=True)
        return important_memories[:limit]
    
    async def _cleanup_memories(self):
        """Clean up old or less important memories"""
        if len(self.memories) <= self.config["max_memories"]:
            return
        
        # Sort memories by relevance score (considering age, importance, access)
        current_time = datetime.now()
        dummy_query = [0.0] * 16  # Dummy embedding for scoring
        
        memory_scores = []
        for memory in self.memories:
            score = memory.calculate_relevance_score(dummy_query, current_time)
            memory_scores.append((score, memory))
        
        # Sort by score and keep the top memories
        memory_scores.sort(key=lambda x: x[0], reverse=True)
        self.memories = [memory for _, memory in memory_scores[:self.config["max_memories"]]]
        
        self.logger.info(f"Cleaned up memories, kept {len(self.memories)} items")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about current memory state"""
        if not self.memories:
            return {
                "total_memories": 0,
                "avg_importance": 0.0,
                "oldest_memory": None,
                "newest_memory": None,
                "memory_types": {}
            }
        
        memory_types = {}
        total_importance = 0.0
        
        for memory in self.memories:
            memory_type = memory.metadata.get("type", "unknown")
            memory_types[memory_type] = memory_types.get(memory_type, 0) + 1
            total_importance += memory.importance
        
        oldest = min(self.memories, key=lambda m: m.timestamp)
        newest = max(self.memories, key=lambda m: m.timestamp)
        
        return {
            "total_memories": len(self.memories),
            "avg_importance": total_importance / len(self.memories),
            "oldest_memory": oldest.timestamp.isoformat(),
            "newest_memory": newest.timestamp.isoformat(),
            "memory_types": memory_types,
            "chromadb_available": self.collection is not None,
            "embedding_model_available": self.embedding_model is not None
        }
