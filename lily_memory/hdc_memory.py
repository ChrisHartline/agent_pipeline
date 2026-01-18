"""
Hyperdimensional Computing (HDC) Memory for Clara.

Fast, in-memory associative memory using high-dimensional vectors.
Used for SESSION tier - immediate conversation context.

Key concepts:
- Hypervectors: 64k-dimensional bipolar vectors (+1/-1)
- Binding (⊗): Element-wise multiplication for associations
- Bundling (+): Element-wise addition for superposition
- Similarity: Cosine similarity for retrieval
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import uuid
import hashlib

from .base import Memory, MemoryTier, MemoryStore, TIER_WEIGHTS


class HDCMemory(MemoryStore):
    """
    Hyperdimensional Computing memory store.

    Fast, in-memory storage for session-level memories.
    Uses high-dimensional vectors for similarity-based retrieval.

    Attributes:
        dimensions: Vector dimensionality (default: 64000)
        threshold: Minimum similarity for recall (default: 0.15)
    """

    def __init__(
        self,
        dimensions: int = 64000,
        threshold: float = 0.15,
        max_memories: int = 1000,
        seed: Optional[int] = None
    ):
        """
        Initialize HDC memory.

        Args:
            dimensions: Hypervector dimensions (higher = more capacity)
            threshold: Minimum cosine similarity for recall
            max_memories: Maximum memories before oldest are forgotten
            seed: Random seed for reproducibility
        """
        self.dimensions = dimensions
        self.threshold = threshold
        self.max_memories = max_memories

        if seed is not None:
            np.random.seed(seed)

        # Memory storage
        self._memories: Dict[str, Memory] = {}
        self._vectors: Dict[str, np.ndarray] = {}

        # Character-level encoding vectors (learned vocabulary)
        self._char_vectors: Dict[str, np.ndarray] = {}

        # Position encoding for word order
        self._position_vectors: List[np.ndarray] = [
            self._random_vector() for _ in range(100)  # Support up to 100 words
        ]

        # Tier context vectors
        self._tier_vectors = {
            MemoryTier.SESSION: self._random_vector(),
            MemoryTier.DAILY: self._random_vector(),
            MemoryTier.LONGTERM: self._random_vector(),
        }

    def _random_vector(self) -> np.ndarray:
        """Generate a random bipolar hypervector."""
        return np.random.choice([-1, 1], size=self.dimensions).astype(np.float32)

    def _get_char_vector(self, char: str) -> np.ndarray:
        """Get or create character vector."""
        if char not in self._char_vectors:
            self._char_vectors[char] = self._random_vector()
        return self._char_vectors[char]

    def _encode_word(self, word: str) -> np.ndarray:
        """Encode a word as a hypervector using character n-grams."""
        if not word:
            return np.zeros(self.dimensions, dtype=np.float32)

        word = word.lower()
        ngrams = []

        # Character trigrams
        padded = f"#{word}#"
        for i in range(len(padded) - 2):
            trigram = padded[i:i+3]
            # Bind characters with position
            v = self._get_char_vector(trigram[0])
            for j, c in enumerate(trigram[1:], 1):
                v = v * np.roll(self._get_char_vector(c), j)
            ngrams.append(v)

        # Bundle all n-grams
        if ngrams:
            word_vec = np.sum(ngrams, axis=0)
            word_vec = np.sign(word_vec)  # Normalize to bipolar
            word_vec[word_vec == 0] = 1
            return word_vec

        return self._random_vector()

    def _encode_text(self, text: str) -> np.ndarray:
        """Encode text as a hypervector."""
        words = text.lower().split()
        if not words:
            return np.zeros(self.dimensions, dtype=np.float32)

        # Encode each word and bind with position
        word_vectors = []
        for i, word in enumerate(words[:100]):  # Limit to 100 words
            word_vec = self._encode_word(word)
            pos_vec = self._position_vectors[i]
            bound = word_vec * pos_vec  # Bind word with position
            word_vectors.append(bound)

        # Bundle all word vectors
        text_vec = np.sum(word_vectors, axis=0)

        # Normalize
        norm = np.linalg.norm(text_vec)
        if norm > 0:
            text_vec = text_vec / norm

        return text_vec

    def _compute_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute cosine similarity between vectors."""
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

    async def store(self, memory: Memory) -> str:
        """Store a memory in HDC."""
        if not memory.id:
            memory.id = str(uuid.uuid4())

        # Encode content
        content_vec = self._encode_text(memory.content)

        # Bind with tier context
        tier_vec = self._tier_vectors[memory.tier]
        memory_vec = content_vec * tier_vec

        # Store
        self._memories[memory.id] = memory
        self._vectors[memory.id] = memory_vec
        memory.embedding = memory_vec

        # Enforce max memories (forget oldest)
        if len(self._memories) > self.max_memories:
            oldest_id = min(
                self._memories.keys(),
                key=lambda k: self._memories[k].timestamp
            )
            await self.forget(oldest_id)

        return memory.id

    async def recall(
        self,
        query: str,
        top_k: int = 5,
        tier_filter: Optional[List[MemoryTier]] = None,
        user_id: str = "default"
    ) -> List[Tuple[Memory, float]]:
        """Recall memories similar to query."""
        if not self._memories:
            return []

        query_vec = self._encode_text(query)
        results = []

        for memory_id, memory in self._memories.items():
            # Filter by tier
            if tier_filter and memory.tier not in tier_filter:
                continue

            # Filter by user
            if memory.user_id != user_id:
                continue

            # Compute similarity
            memory_vec = self._vectors[memory_id]
            similarity = self._compute_similarity(query_vec, memory_vec)

            # Apply tier weight
            tier_weight = TIER_WEIGHTS.get(memory.tier, 0.5)
            weighted_score = similarity * tier_weight

            # Apply recency boost (memories from last hour get boost)
            age_hours = (datetime.now() - memory.timestamp).total_seconds() / 3600
            recency_factor = 1.0 + max(0, 1.0 - age_hours) * 0.2  # Up to 20% boost
            weighted_score *= recency_factor

            if weighted_score >= self.threshold:
                results.append((memory, weighted_score))

        # Sort by score and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    async def forget(self, memory_id: str) -> bool:
        """Remove a memory."""
        if memory_id in self._memories:
            del self._memories[memory_id]
            del self._vectors[memory_id]
            return True
        return False

    async def consolidate(self) -> Dict[str, int]:
        """
        Run consolidation (promote important memories).

        Returns stats about what was consolidated.
        """
        stats = {"promoted": 0, "forgotten": 0}

        now = datetime.now()
        to_promote = []
        to_forget = []

        for memory_id, memory in self._memories.items():
            age_hours = (now - memory.timestamp).total_seconds() / 3600

            # Session memories older than 1 hour with high importance -> promote
            if memory.tier == MemoryTier.SESSION and age_hours > 1:
                if memory.importance > 0.7:
                    to_promote.append(memory_id)
                elif age_hours > 4:  # Forget old low-importance session memories
                    to_forget.append(memory_id)

        # Promote
        for memory_id in to_promote:
            self._memories[memory_id].tier = MemoryTier.DAILY
            stats["promoted"] += 1

        # Forget
        for memory_id in to_forget:
            await self.forget(memory_id)
            stats["forgotten"] += 1

        return stats

    def get_all_memories(self, user_id: str = "default") -> List[Memory]:
        """Get all memories for a user (for external consolidation)."""
        return [m for m in self._memories.values() if m.user_id == user_id]

    def clear_session(self, user_id: str = "default"):
        """Clear session memories for a user."""
        to_remove = [
            mid for mid, m in self._memories.items()
            if m.user_id == user_id and m.tier == MemoryTier.SESSION
        ]
        for mid in to_remove:
            del self._memories[mid]
            del self._vectors[mid]
