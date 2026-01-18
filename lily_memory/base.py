"""
Base interfaces for Clara's memory system.

These abstract classes define the contract that any memory implementation
must follow, enabling easy swapping of backends.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


class MemoryTier(Enum):
    """Memory tiers with different persistence and recall characteristics."""
    SESSION = "session"      # Working memory, current conversation (weight: 1.0)
    DAILY = "daily"          # Today's consolidated memories (weight: 0.7)
    LONGTERM = "longterm"    # Permanent, high-importance facts (weight: 0.5)


# Tier weights for recall scoring
TIER_WEIGHTS = {
    MemoryTier.SESSION: 1.0,
    MemoryTier.DAILY: 0.7,
    MemoryTier.LONGTERM: 0.5,
}


@dataclass
class Memory:
    """
    A single memory unit.

    Attributes:
        id: Unique identifier
        content: The actual memory content (text) - may be encrypted
        tier: Which memory tier (session/daily/longterm)
        importance: 0.0-1.0 score for consolidation decisions
        timestamp: When the memory was created
        metadata: Additional context (entities, emotions, topics)
        embedding: Optional vector representation (HDC or traditional)
        encrypted: Whether content is encrypted
        user_id: Owner of this memory (for multi-user support)
    """
    id: str
    content: str
    tier: MemoryTier = MemoryTier.SESSION
    importance: float = 0.5
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    encrypted: bool = False
    user_id: str = "default"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage."""
        return {
            "id": self.id,
            "content": self.content,
            "tier": self.tier.value,
            "importance": self.importance,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "encrypted": self.encrypted,
            "user_id": self.user_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Memory":
        """Deserialize from storage."""
        return cls(
            id=data["id"],
            content=data["content"],
            tier=MemoryTier(data["tier"]),
            importance=data.get("importance", 0.5),
            timestamp=datetime.fromisoformat(data["timestamp"]) if isinstance(data["timestamp"], str) else data["timestamp"],
            metadata=data.get("metadata", {}),
            encrypted=data.get("encrypted", False),
            user_id=data.get("user_id", "default"),
        )


class MemoryStore(ABC):
    """
    Abstract base class for memory storage backends.

    Implement this interface to create swappable memory systems:
    - HDCMemory (hyperdimensional computing, in-memory)
    - SupabaseStore (PostgreSQL, long-term)
    - FalkorStore (graph database, relationships)
    """

    @abstractmethod
    async def store(self, memory: Memory) -> str:
        """
        Store a memory and return its ID.

        Args:
            memory: The Memory object to store

        Returns:
            The memory's unique ID
        """
        pass

    @abstractmethod
    async def recall(
        self,
        query: str,
        top_k: int = 5,
        tier_filter: Optional[List[MemoryTier]] = None,
        user_id: str = "default"
    ) -> List[Tuple[Memory, float]]:
        """
        Recall memories similar to the query.

        Args:
            query: The search query (text)
            top_k: Maximum number of memories to return
            tier_filter: Optional filter by memory tier
            user_id: Filter by user

        Returns:
            List of (Memory, similarity_score) tuples, sorted by relevance
        """
        pass

    @abstractmethod
    async def forget(self, memory_id: str) -> bool:
        """
        Remove a memory by ID.

        Args:
            memory_id: The ID of the memory to remove

        Returns:
            True if removed, False if not found
        """
        pass

    async def get_context(
        self,
        query: str,
        max_tokens: int = 1000,
        user_id: str = "default"
    ) -> str:
        """
        Build context string for prompt injection.

        Args:
            query: The current query to find relevant memories for
            max_tokens: Approximate token budget
            user_id: User to retrieve memories for

        Returns:
            Formatted string of relevant memories for prompt injection
        """
        memories = await self.recall(query, top_k=10, user_id=user_id)
        context_parts = []
        estimated_tokens = 0

        for memory, score in memories:
            # Rough token estimate: ~4 chars per token
            mem_tokens = len(memory.content) // 4
            if estimated_tokens + mem_tokens > max_tokens:
                break

            tier_label = memory.tier.value.upper()
            context_parts.append(f"[{tier_label}|{score:.2f}] {memory.content}")
            estimated_tokens += mem_tokens

        if not context_parts:
            return ""

        return "Relevant memories:\n" + "\n".join(context_parts)


class ConsolidationEngine(ABC):
    """
    Abstract base class for memory consolidation.

    Consolidation is the "sleep" process that:
    - Promotes important session memories to daily/longterm
    - Compresses and summarizes old memories
    - Creates associative bindings between related concepts

    Implementations:
    - ClassicalConsolidation: Traditional importance-based promotion
    - QuantumConsolidation: TFQ/Cirq entanglement for fuzzy binding
    """

    @abstractmethod
    async def process(
        self,
        memories: List[Memory]
    ) -> Tuple[List[Memory], Dict[str, Any]]:
        """
        Process memories through consolidation.

        Args:
            memories: List of memories to consolidate

        Returns:
            Tuple of (processed_memories, stats_dict)
        """
        pass

    @abstractmethod
    async def create_bindings(
        self,
        memories: List[Memory]
    ) -> Dict[str, List[str]]:
        """
        Create associative bindings between memories.

        Args:
            memories: Memories to find associations between

        Returns:
            Dict mapping memory_id -> [related_memory_ids]
        """
        pass
