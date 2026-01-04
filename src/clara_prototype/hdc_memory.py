"""Hyperdimensional Computing (HDC) Memory System for Clara.

This module implements Clara's episodic memory using Hyperdimensional Computing,
providing efficient storage and retrieval of conversational memories.

Features:
- Semantic encoding via sentence transformers
- HDC operations (bind, bundle, similarity)
- Personality-weighted importance scoring
- Entity extraction and indexing
- Persistent storage (JSON)

Example:
    >>> from clara_prototype.hdc_memory import ClaraHDCMemory
    >>> memory = ClaraHDCMemory(dimensions=10000)
    >>> memory.store("User offered coffee", memory_type="user_input")
    >>> results = memory.recall("What about the coffee?")
"""
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .config import HDCConfig, PersonalityConfig

logger = logging.getLogger(__name__)


# Personality trait definitions for semantic encoding
TRAIT_DEFINITIONS = {
    "warmth": {
        "high": """friendly caring supportive empathetic kind gentle understanding
                   compassionate nurturing affectionate welcoming comforting helpful
                   I'd be happy to help you with that
                   That's a great question, let me explain
                   I understand how you feel
                   It's wonderful to hear from you
                   I really appreciate you sharing that with me""",
        "low": """cold distant dismissive indifferent aloof detached impersonal
                  formal reserved standoffish unfriendly curt
                  Here is the answer
                  That is incorrect
                  Do it yourself
                  Not my concern
                  Figure it out""",
    },
    "patience": {
        "high": """patient calm understanding tolerant unhurried steady persistent
                   thorough methodical take your time no rush let me explain again
                   happy to go over this as many times as you need
                   that's okay, learning takes time
                   don't worry, we'll work through this together
                   there's no hurry, let's make sure you understand""",
        "low": """impatient rushed hurried abrupt curt short dismissive quick
                  frustrated annoyed already explained this
                  I told you before just do it
                  hurry up we don't have time
                  I already answered that""",
    },
    "curiosity": {
        "high": """curious interested fascinated intrigued eager to learn exploring
                   tell me more that's interesting I wonder why how does that work
                   what made you think of that I'd love to understand better
                   that's fascinating can you elaborate
                   I'm curious about your perspective on this""",
        "low": """bored disinterested uninterested indifferent apathetic dull
                  whatever doesn't matter who cares not my problem
                  I don't care about that
                  that's irrelevant""",
    },
    "encouragement": {
        "high": """encouraging supportive motivating uplifting positive you can do it
                   great job well done I believe in you keep going you're doing great
                   that's excellent progress don't give up
                   you're making wonderful progress
                   I'm proud of how far you've come
                   that's a brilliant insight""",
        "low": """discouraging negative critical harsh judgmental disappointing
                  that's wrong you failed not good enough why bother
                  you'll never get it
                  that's a terrible idea
                  don't waste my time""",
    },
}

# Entity patterns for extraction
ENTITY_PATTERNS = [
    r"\bcoffee\b", r"\btea\b", r"\bwater\b", r"\bdrink\b",
    r"\bfood\b", r"\blunch\b", r"\bdinner\b", r"\bbreakfast\b",
    r"\bproject\b", r"\bwork\b", r"\bmeeting\b", r"\btask\b",
    r"\bDocker\b", r"\bPython\b", r"\bcode\b", r"\bnotebook\b",
    r"\bhelp\b", r"\bthanks\b", r"\bplease\b",
]


@dataclass
class Memory:
    """Single memory unit with metadata."""

    text: str
    timestamp: float
    memory_type: str  # 'user_input', 'clara_response', 'preference', 'fact'
    importance: float = 0.5
    domain: str = "general"
    entities: List[str] = field(default_factory=list)
    turn_id: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "timestamp": self.timestamp,
            "memory_type": self.memory_type,
            "importance": self.importance,
            "domain": self.domain,
            "entities": self.entities,
            "turn_id": self.turn_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Memory":
        return cls(
            text=data["text"],
            timestamp=data["timestamp"],
            memory_type=data.get("memory_type", "interaction"),
            importance=data["importance"],
            domain=data.get("domain", "general"),
            entities=data.get("entities", []),
            turn_id=data.get("turn_id", 0),
        )


class ClaraHDCMemory:
    """Hyperdimensional Computing Memory System for Clara.

    This system uses HDC for efficient episodic memory with:
    - Semantic encoding via sentence transformers
    - Personality-weighted importance scoring
    - Entity extraction for fast lookup
    - Similarity-based recall

    HDC Operations:
    - Bind (⊗): Element-wise multiplication - creates associations
    - Bundle (+): Element-wise addition + sign - superposition
    - Similarity: Cosine similarity for retrieval

    Args:
        config: HDC configuration (dimensions, thresholds, etc.)
        embedder: Sentence transformer for text encoding (optional)
        personality_weights: Dict of personality trait weights
    """

    VERSION = "2.1"

    def __init__(
        self,
        config: Optional[HDCConfig] = None,
        embedder: Optional[Any] = None,
        personality_weights: Optional[Dict[str, float]] = None,
    ):
        self.config = config or HDCConfig()
        self.dim = self.config.dimensions
        self.debug = self.config.debug
        self.rng = np.random.RandomState(self.config.seed)

        # Initialize embedder
        self._embedder = embedder
        self._embedder_dim = 384  # Default for all-MiniLM-L6-v2

        # Random projection matrix: embedder_dim → HDC dim
        logger.info("Initializing projection matrix...")
        self.projection = self.rng.randn(self._embedder_dim, self.dim).astype(np.float32)
        self.projection /= np.linalg.norm(self.projection, axis=1, keepdims=True)

        # Memory stores
        self.memories: List[Tuple[np.ndarray, Memory]] = []
        self.memory_bundle = np.zeros(self.dim, dtype=np.float32)
        self.current_turn = 0

        # Symbol library for structured binding
        logger.info("Building symbol library...")
        self.symbols: Dict[str, np.ndarray] = {}
        self._init_symbols()

        # Entity index for fast lookup
        self.entity_index: Dict[str, List[int]] = {}

        # Personality vectors (semantic encoding)
        self.personality_weights = personality_weights or {
            "warmth": 0.85,
            "patience": 0.90,
            "curiosity": 0.75,
            "encouragement": 0.85,
        }
        logger.info("Encoding semantic personality vectors...")
        self.personality = self._init_personality()

        logger.info(f"HDC Memory v{self.VERSION} initialized (dim={self.dim})")

    @property
    def embedder(self) -> Any:
        """Lazy-load sentence transformer embedder."""
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
                logger.info("Loaded all-MiniLM-L6-v2 embedder")
            except ImportError:
                raise RuntimeError(
                    "sentence-transformers is required for HDC memory. "
                    "Install with: pip install sentence-transformers"
                )
        return self._embedder

    def _init_symbols(self) -> None:
        """Create base symbol vocabulary."""
        base_symbols = [
            # Roles
            "ROLE_USER", "ROLE_CLARA", "ROLE_TOPIC", "ROLE_OUTCOME", "ROLE_ENTITY",
            # Actions
            "ASKED", "ANSWERED", "OFFERED", "RECEIVED", "DISCUSSED",
            # Domains
            "MEDICAL", "CODING", "TEACHING", "QUANTUM", "PERSONALITY",
            # Common conversational entities
            "COFFEE", "TEA", "FOOD", "DRINK", "HELP", "THANKS",
            # Outcomes
            "HELPFUL", "CONFUSED", "SATISFIED", "FRUSTRATED",
            # Time markers
            "RECENT", "TODAY", "THIS_SESSION", "EARLIER"
        ]
        for s in base_symbols:
            self.symbols[s] = self._random_hv()

    def _init_personality(self) -> Dict[str, np.ndarray]:
        """Encode Clara's personality traits as semantic hypervectors."""
        personality = {}

        for trait, weight in self.personality_weights.items():
            if trait not in TRAIT_DEFINITIONS:
                personality[trait] = self._random_hv() * weight
                if self.debug:
                    logger.debug(f"[PERSONALITY] {trait}: using random HV (no definition)")
                continue

            descriptions = TRAIT_DEFINITIONS[trait]

            # Encode high and low trait descriptions
            high_hv = self._text_to_hv(descriptions["high"])
            low_hv = self._text_to_hv(descriptions["low"])

            # Clara's trait vector: weighted toward high end
            trait_hv = self.bundle([
                high_hv * weight,
                low_hv * (1 - weight)
            ])

            personality[trait] = trait_hv

            # Verify semantic encoding
            test_sim = self.similarity(trait_hv, high_hv)
            logger.debug(f"[PERSONALITY] {trait}: weight={weight:.2f}, alignment={test_sim:.3f}")

        # Composite personality = bundle of all traits
        all_traits = [personality[t] for t in self.personality_weights.keys() if t in personality]
        personality["composite"] = self.bundle(all_traits)

        return personality

    # === HDC Core Operations ===

    def _random_hv(self) -> np.ndarray:
        """Generate random bipolar hypervector {-1, +1}."""
        return self.rng.choice([-1, 1], size=self.dim).astype(np.float32)

    def _text_to_hv(self, text: str) -> np.ndarray:
        """Convert text to hypervector using semantic embedding."""
        embedding = self.embedder.encode(text)
        hv = embedding @ self.projection
        return np.sign(hv).astype(np.float32)

    def _get_symbol(self, name: str) -> np.ndarray:
        """Get or create symbol hypervector."""
        name_upper = name.upper().replace(" ", "_")
        if name_upper not in self.symbols:
            self.symbols[name_upper] = self._random_hv()
        return self.symbols[name_upper]

    def bind(self, hv1: np.ndarray, hv2: np.ndarray) -> np.ndarray:
        """Bind two hypervectors (⊗) - creates association."""
        return hv1 * hv2

    def bundle(self, hvs: List[np.ndarray]) -> np.ndarray:
        """Bundle hypervectors (+) - superposition."""
        if not hvs:
            return np.zeros(self.dim, dtype=np.float32)
        return np.sign(np.sum(hvs, axis=0)).astype(np.float32)

    def similarity(self, hv1: np.ndarray, hv2: np.ndarray) -> float:
        """Cosine similarity between hypervectors."""
        n1, n2 = np.linalg.norm(hv1), np.linalg.norm(hv2)
        if n1 == 0 or n2 == 0:
            return 0.0
        return float(np.dot(hv1, hv2) / (n1 * n2))

    # === Entity Extraction ===

    def _extract_entities(self, text: str) -> List[str]:
        """Extract key entities from text for indexing."""
        entities = []
        text_lower = text.lower()

        for pattern in ENTITY_PATTERNS:
            match = re.search(pattern, text_lower)
            if match:
                entities.append(match.group().upper())

        return list(set(entities))

    # === Personality Operations ===

    def personality_alignment(self, text: str) -> Dict[str, float]:
        """Check how well text aligns with each personality trait.

        Returns dict of trait -> alignment score (-1 to 1).
        Positive = aligns with Clara's personality.
        """
        text_hv = self._text_to_hv(text)

        alignments = {}
        for trait in self.personality_weights.keys():
            if trait in self.personality:
                alignments[trait] = self.similarity(text_hv, self.personality[trait])

        return alignments

    def emotional_importance(self, text: str) -> float:
        """Calculate importance boost based on emotional/personality resonance.

        Warm, encouraging, patient interactions are more memorable for Clara.
        """
        alignments = self.personality_alignment(text)

        if not alignments:
            return 0.0

        total = 0.0
        weight_sum = 0.0
        for trait, alignment in alignments.items():
            trait_weight = self.personality_weights.get(trait, 0.5)
            total += alignment * trait_weight
            weight_sum += trait_weight

        avg_alignment = total / weight_sum if weight_sum > 0 else 0.0

        # Convert to importance boost (0 to 0.2)
        boost = max(0, avg_alignment * 0.2)
        return boost

    # === Memory Operations ===

    def store(
        self,
        text: str,
        memory_type: str = "user_input",
        importance: float = 0.5,
        domain: str = "general",
        increment_turn: bool = False,
        **bindings: Any,
    ) -> None:
        """Store a memory with entity extraction and optional bindings.

        Args:
            text: The actual conversational content
            memory_type: 'user_input', 'clara_response', 'preference', 'fact'
            importance: 0.0-1.0 importance score
            domain: 'medical', 'coding', 'teaching', 'quantum', 'personality'
            increment_turn: Whether this starts a new conversation turn
            **bindings: Structured role-filler pairs
        """
        if increment_turn:
            self.current_turn += 1

        # Extract entities
        entities = self._extract_entities(text)

        # Calculate personality-based importance boost
        personality_boost = self.emotional_importance(text)
        adjusted_importance = min(1.0, importance + personality_boost)

        if self.debug and personality_boost > 0.01:
            logger.debug(f"[PERSONALITY] Importance boosted: {importance:.2f} → {adjusted_importance:.2f}")

        # Create semantic hypervector from text
        text_hv = self._text_to_hv(text)

        # Add domain binding
        domain_hv = self._get_symbol(domain.upper())
        text_hv = self.bind(text_hv, domain_hv)

        # Add entity bindings
        if entities:
            entity_hvs = []
            for entity in entities:
                entity_hv = self._get_symbol(entity)
                role_hv = self._get_symbol("ROLE_ENTITY")
                entity_hvs.append(self.bind(role_hv, entity_hv))
            if entity_hvs:
                entity_bundle = self.bundle(entity_hvs)
                text_hv = self.bundle([text_hv, entity_bundle])

        # Add structural bindings if provided
        if bindings:
            bound_parts = []
            for role, filler in bindings.items():
                role_hv = self._get_symbol(f"ROLE_{role.upper()}")
                filler_hv = self._get_symbol(str(filler).upper())
                bound_parts.append(self.bind(role_hv, filler_hv))
            if bound_parts:
                structure_hv = self.bundle(bound_parts)
                text_hv = self.bundle([text_hv, structure_hv])

        memory = Memory(
            text=text,
            timestamp=time.time(),
            memory_type=memory_type,
            importance=adjusted_importance,
            domain=domain,
            entities=entities,
            turn_id=self.current_turn,
        )

        memory_idx = len(self.memories)
        self.memories.append((text_hv, memory))

        # Update entity index
        for entity in entities:
            if entity not in self.entity_index:
                self.entity_index[entity] = []
            self.entity_index[entity].append(memory_idx)

        # Update bundled representation
        self.memory_bundle = self.bundle([self.memory_bundle, text_hv])

        if self.debug:
            logger.debug(f"[MEM] Stored: '{text[:50]}...' Entities: {entities}")

    def recall(
        self,
        query: str,
        top_k: int = 5,
        domain_filter: Optional[str] = None,
        include_entities: bool = True,
    ) -> List[Tuple[Memory, float]]:
        """Retrieve memories similar to query.

        Args:
            query: Query text
            top_k: Maximum number of results
            domain_filter: Only return memories from this domain
            include_entities: Boost memories with matching entities

        Returns:
            List of (Memory, score) tuples sorted by relevance
        """
        if not self.memories:
            return []

        query_hv = self._text_to_hv(query)
        query_entities = self._extract_entities(query) if include_entities else []

        if self.debug:
            logger.debug(f"[RECALL] Query: '{query[:40]}...' Entities: {query_entities}")

        results = []
        for idx, (hv, memory) in enumerate(self.memories):
            # Apply domain filter
            if domain_filter and memory.domain != domain_filter:
                continue

            # Base similarity
            sim = self.similarity(query_hv, hv)

            # Entity overlap boost
            entity_boost = 0.0
            if query_entities and memory.entities:
                overlap = set(query_entities) & set(memory.entities)
                if overlap:
                    entity_boost = 0.15 * len(overlap)

            # Recency boost (stronger for recent turns)
            turn_diff = self.current_turn - memory.turn_id
            recency = 1.0 / (1.0 + turn_diff * 0.1)

            # Time-based recency (for cross-session)
            age_hours = (time.time() - memory.timestamp) / 3600
            time_recency = 1.0 / (1.0 + age_hours / 24)

            # Combined score
            weighted = (
                sim + entity_boost
            ) * (0.6 + 0.2 * memory.importance) * (0.7 + 0.15 * recency + 0.15 * time_recency)

            results.append((memory, weighted))

            if self.debug and sim > 0.1:
                logger.debug(f"[{idx}] sim={sim:.3f} final={weighted:.3f} '{memory.text[:30]}...'")

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def recall_by_entity(self, entity: str) -> List[Memory]:
        """Fast lookup by entity name."""
        entity_upper = entity.upper()
        if entity_upper not in self.entity_index:
            return []

        indices = self.entity_index[entity_upper]
        return [self.memories[i][1] for i in indices]

    def get_context_string(self, query: str, max_memories: int = 3) -> str:
        """Get memory context as string for prompt injection."""
        relevant = self.recall(query, top_k=max_memories)
        if not relevant:
            return ""

        good_memories = [(m, s) for m, s in relevant if s > self.config.recall_threshold]

        if self.debug:
            logger.debug(f"[CONTEXT] {len(relevant)} retrieved, {len(good_memories)} above threshold")

        if not good_memories:
            return ""

        context_parts = ["[Conversation context:"]
        for mem, score in good_memories:
            if mem.memory_type == "user_input":
                context_parts.append(f"- User said: {mem.text[:100]}")
            elif mem.memory_type == "clara_response":
                context_parts.append(f"- Clara said: {mem.text[:100]}")
            else:
                context_parts.append(f"- {mem.text[:100]}")
        context_parts.append("]")

        return "\n".join(context_parts)

    def get_recent_context(self, n_turns: int = 2) -> str:
        """Get context from most recent N conversation turns."""
        if not self.memories:
            return ""

        min_turn = max(0, self.current_turn - n_turns)
        recent = [m for _, m in self.memories if m.turn_id >= min_turn]

        if not recent:
            return ""

        context_parts = ["[Recent conversation:"]
        for mem in recent[-6:]:
            if mem.memory_type == "user_input":
                context_parts.append(f"- User: {mem.text[:80]}")
            elif mem.memory_type == "clara_response":
                context_parts.append(f"- Clara: {mem.text[:80]}")
        context_parts.append("]")

        return "\n".join(context_parts)

    def get_routing_context(self, query: str) -> np.ndarray:
        """Get memory-augmented context for routing decisions."""
        query_hv = self._text_to_hv(query)

        relevant = self.recall(query, top_k=3)
        if relevant:
            memory_hvs = [self._text_to_hv(m.text) for m, _ in relevant]
            memory_context = self.bundle(memory_hvs)
        else:
            memory_context = np.zeros(self.dim, dtype=np.float32)

        routing_hv = self.bundle([
            query_hv,
            memory_context * 0.3,
            self.personality["composite"] * 0.2
        ])

        return routing_hv

    # === Statistics ===

    def stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        domain_counts: Dict[str, int] = {}
        type_counts: Dict[str, int] = {}
        for _, m in self.memories:
            domain_counts[m.domain] = domain_counts.get(m.domain, 0) + 1
            type_counts[m.memory_type] = type_counts.get(m.memory_type, 0) + 1

        return {
            "total_memories": len(self.memories),
            "by_domain": domain_counts,
            "by_type": type_counts,
            "symbols": len(self.symbols),
            "entities_tracked": len(self.entity_index),
            "current_turn": self.current_turn,
            "size_kb": self.size_bytes() / 1024,
            "personality_traits": list(self.personality_weights.keys()),
            "version": self.VERSION,
        }

    def size_bytes(self) -> int:
        """Memory footprint (excluding embedder)."""
        bundle_size = self.memory_bundle.nbytes
        memories_size = sum(hv.nbytes for hv, _ in self.memories)
        symbols_size = sum(hv.nbytes for hv in self.symbols.values())
        personality_size = sum(hv.nbytes for hv in self.personality.values())
        projection_size = self.projection.nbytes
        return bundle_size + memories_size + symbols_size + personality_size + projection_size

    # === Persistence ===

    def save(self, path: str) -> None:
        """Save memory state to file."""
        data = {
            "version": self.VERSION,
            "dim": self.dim,
            "current_turn": int(self.current_turn),
            "recall_threshold": float(self.config.recall_threshold),
            "memories": [
                {
                    "hv": hv.tolist(),
                    **m.to_dict()
                }
                for hv, m in self.memories
            ],
            "symbols": {k: v.tolist() for k, v in self.symbols.items()},
            "entity_index": self.entity_index,
            "memory_bundle": self.memory_bundle.tolist(),
            "projection": self.projection.tolist(),
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)

        logger.info(f"Saved {len(self.memories)} memories to {path}")

    def load(self, path: str) -> bool:
        """Load memory state from file."""
        if not Path(path).exists():
            logger.warning(f"No memory file found at {path}")
            return False

        with open(path) as f:
            data = json.load(f)

        version = data.get("version", "1.0")
        logger.info(f"Loading memory file version {version}")

        if data.get("dim") != self.dim:
            logger.warning(f"Dimension mismatch: file has {data.get('dim')}, expected {self.dim}")
            return False

        # Load memories
        self.memories = []
        for m in data["memories"]:
            memory = Memory(
                text=m["text"],
                timestamp=m["timestamp"],
                memory_type=m.get("memory_type", "interaction"),
                importance=m["importance"],
                domain=m.get("domain", "general"),
                entities=m.get("entities", []),
                turn_id=m.get("turn_id", 0),
            )
            hv = np.array(m["hv"], dtype=np.float32)
            self.memories.append((hv, memory))

        # Load other state
        self.symbols = {k: np.array(v, dtype=np.float32) for k, v in data["symbols"].items()}
        self.entity_index = data.get("entity_index", {})
        self.memory_bundle = np.array(data["memory_bundle"], dtype=np.float32)
        self.current_turn = data.get("current_turn", 0)

        if "projection" in data:
            self.projection = np.array(data["projection"], dtype=np.float32)

        logger.info(f"Loaded {len(self.memories)} memories from {path}")
        return True

    def clear(self) -> None:
        """Clear all memories."""
        self.memories = []
        self.memory_bundle = np.zeros(self.dim, dtype=np.float32)
        self.entity_index = {}
        self.current_turn = 0
        logger.info("Memory cleared")
