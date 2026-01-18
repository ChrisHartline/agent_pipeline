"""
FalkorDB (Graph) store for Clara's entity relationships.

Stores semantic connections between entities (people, places, concepts).
Enables queries like "What do I know about X?" or "How are X and Y related?"

Uses client-side encryption for sensitive entity data.

Required environment variables:
    FALKORDB_HOST: FalkorDB host (or use FALKORDB_URL for cloud)
    FALKORDB_PORT: FalkorDB port (default: 6379)
    FALKORDB_HOSTED_KEY: API key for FalkorDB Cloud
    CLARA_MEMORY_KEY: Base64-encoded encryption key for client-side encryption
"""

import os
import uuid
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Set

from .base import Memory, MemoryTier, MemoryStore, TIER_WEIGHTS
from .encryption import get_encryption, MemoryEncryption

try:
    from falkordb import FalkorDB
    FALKOR_AVAILABLE = True
except ImportError:
    FALKOR_AVAILABLE = False


class FalkorStore(MemoryStore):
    """
    FalkorDB-backed graph store for entity relationships.

    Stores:
    - Entities (people, places, things, concepts)
    - Relationships between entities
    - Memory nodes linked to entities they mention

    Graph schema:
        (:Entity {name, type, encrypted_data})
        (:Memory {id, content, tier, importance})
        (:Entity)-[:RELATED_TO {strength}]->(:Entity)
        (:Memory)-[:MENTIONS {salience}]->(:Entity)
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        password: Optional[str] = None,
        graph_name: str = "clara_memory",
        encryption: Optional[MemoryEncryption] = None
    ):
        """
        Initialize FalkorDB connection.

        Args:
            host: FalkorDB host (or FALKOR_HOST env var)
            port: FalkorDB port (or FALKOR_PORT env var)
            password: FalkorDB password (or FALKOR_PASSWORD env var)
            graph_name: Name of the graph to use
            encryption: Encryption instance for sensitive data
        """
        if not FALKOR_AVAILABLE:
            raise ImportError(
                "falkordb package required. Install with: pip install falkordb"
            )

        self.host = host or os.environ.get("FALKORDB_HOST", "localhost")
        self.port = port or int(os.environ.get("FALKORDB_PORT", "6379"))
        self.password = password or os.environ.get("FALKORDB_HOSTED_KEY")

        # Connect to FalkorDB
        self.db = FalkorDB(
            host=self.host,
            port=self.port,
            password=self.password
        )
        self.graph = self.db.select_graph(graph_name)
        self.encryption = encryption or get_encryption()

        # Initialize schema
        self._init_schema()

    def _init_schema(self):
        """Create indexes for efficient queries."""
        try:
            # Index on entity name for fast lookups
            self.graph.query("CREATE INDEX FOR (e:Entity) ON (e.name)")
        except Exception:
            pass  # Index may already exist

        try:
            # Index on memory ID
            self.graph.query("CREATE INDEX FOR (m:Memory) ON (m.id)")
        except Exception:
            pass

        try:
            # Index on user_id
            self.graph.query("CREATE INDEX FOR (m:Memory) ON (m.user_id)")
        except Exception:
            pass

    def _extract_entities(self, text: str) -> List[Dict[str, str]]:
        """
        Extract entities from text.

        Simple rule-based extraction. For production, consider using
        spaCy NER or a similar library.

        Returns list of {name, type} dicts.
        """
        entities = []

        # Capitalized words (potential proper nouns)
        words = text.split()
        for i, word in enumerate(words):
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word and clean_word[0].isupper() and len(clean_word) > 1:
                # Skip common sentence starters
                if i == 0 or words[i-1].endswith(('.', '!', '?')):
                    continue
                entities.append({"name": clean_word, "type": "unknown"})

        # Common patterns
        # "my [relation] [Name]" -> Person
        relation_pattern = r"my\s+(friend|brother|sister|mom|dad|wife|husband|partner|colleague|boss)\s+(\w+)"
        for match in re.finditer(relation_pattern, text, re.IGNORECASE):
            entities.append({"name": match.group(2), "type": "person"})

        # Places after prepositions
        place_pattern = r"(?:in|at|to|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)"
        for match in re.finditer(place_pattern, text):
            entities.append({"name": match.group(1), "type": "place"})

        # Deduplicate
        seen = set()
        unique = []
        for e in entities:
            if e["name"].lower() not in seen:
                seen.add(e["name"].lower())
                unique.append(e)

        return unique

    async def store(self, memory: Memory) -> str:
        """Store a memory and link it to mentioned entities."""
        if not memory.id:
            memory.id = str(uuid.uuid4())

        # Encrypt content for storage
        encrypted_content, was_encrypted = self.encryption.encrypt_if_personal(
            memory.content,
            is_personal=True
        )

        # Create memory node
        self.graph.query(
            """
            MERGE (m:Memory {id: $id})
            SET m.content = $content,
                m.tier = $tier,
                m.importance = $importance,
                m.created_at = $created_at,
                m.user_id = $user_id,
                m.encrypted = $encrypted
            """,
            {
                "id": memory.id,
                "content": encrypted_content,
                "tier": memory.tier.value,
                "importance": memory.importance,
                "created_at": memory.timestamp.isoformat(),
                "user_id": memory.user_id,
                "encrypted": was_encrypted,
            }
        )

        # Extract and link entities
        entities = self._extract_entities(memory.content)
        for entity in entities:
            # Create or update entity node
            self.graph.query(
                """
                MERGE (e:Entity {name: $name})
                SET e.type = $type,
                    e.user_id = $user_id
                """,
                {
                    "name": entity["name"],
                    "type": entity["type"],
                    "user_id": memory.user_id,
                }
            )

            # Link memory to entity
            self.graph.query(
                """
                MATCH (m:Memory {id: $memory_id})
                MATCH (e:Entity {name: $entity_name})
                MERGE (m)-[r:MENTIONS]->(e)
                SET r.salience = $salience
                """,
                {
                    "memory_id": memory.id,
                    "entity_name": entity["name"],
                    "salience": 1.0,  # Could compute based on position/frequency
                }
            )

        memory.encrypted = was_encrypted
        return memory.id

    async def recall(
        self,
        query: str,
        top_k: int = 5,
        tier_filter: Optional[List[MemoryTier]] = None,
        user_id: str = "default"
    ) -> List[Tuple[Memory, float]]:
        """
        Recall memories related to query.

        Uses entity matching and graph traversal.
        """
        # Extract entities from query
        query_entities = self._extract_entities(query)
        entity_names = [e["name"] for e in query_entities]

        # Also check for keyword matches
        query_words = set(query.lower().split())

        memories_scores: Dict[str, Tuple[Memory, float]] = {}

        # Search by entity mentions
        if entity_names:
            result = self.graph.query(
                """
                MATCH (m:Memory)-[:MENTIONS]->(e:Entity)
                WHERE e.name IN $names AND m.user_id = $user_id
                RETURN m, count(e) as entity_matches
                ORDER BY entity_matches DESC, m.importance DESC
                LIMIT $limit
                """,
                {
                    "names": entity_names,
                    "user_id": user_id,
                    "limit": top_k * 2,
                }
            )

            for record in result.result_set:
                node = record[0]
                entity_matches = record[1]

                memory = self._node_to_memory(node)

                # Filter by tier
                if tier_filter and memory.tier not in tier_filter:
                    continue

                # Score based on entity matches
                score = entity_matches * 0.5

                # Apply tier weight
                score *= TIER_WEIGHTS.get(memory.tier, 0.5)

                memories_scores[memory.id] = (memory, score)

        # Also do keyword search on all memories
        result = self.graph.query(
            """
            MATCH (m:Memory)
            WHERE m.user_id = $user_id
            RETURN m
            ORDER BY m.importance DESC, m.created_at DESC
            LIMIT $limit
            """,
            {
                "user_id": user_id,
                "limit": top_k * 3,
            }
        )

        for record in result.result_set:
            node = record[0]
            memory = self._node_to_memory(node)

            if memory.id in memories_scores:
                continue  # Already scored

            # Filter by tier
            if tier_filter and memory.tier not in tier_filter:
                continue

            # Decrypt for matching
            plaintext = self.encryption.decrypt_if_needed(
                memory.content,
                memory.encrypted
            )

            # Keyword matching
            content_words = set(plaintext.lower().split())
            overlap = len(query_words & content_words)

            if overlap > 0:
                score = overlap / max(len(query_words), 1) * 0.3
                score *= TIER_WEIGHTS.get(memory.tier, 0.5)
                memory.content = plaintext  # Use decrypted content
                memories_scores[memory.id] = (memory, score)

        # Sort by score
        results = list(memories_scores.values())
        results.sort(key=lambda x: x[1], reverse=True)

        # Decrypt content for top results
        final_results = []
        for memory, score in results[:top_k]:
            if memory.encrypted and not memory.content.startswith("Relevant"):
                memory.content = self.encryption.decrypt_if_needed(
                    memory.content,
                    memory.encrypted
                )
            final_results.append((memory, score))

        return final_results

    async def forget(self, memory_id: str) -> bool:
        """Delete a memory and its entity links."""
        result = self.graph.query(
            """
            MATCH (m:Memory {id: $id})
            DETACH DELETE m
            RETURN count(m) as deleted
            """,
            {"id": memory_id}
        )

        if result.result_set:
            return result.result_set[0][0] > 0
        return False

    async def consolidate(self) -> Dict[str, int]:
        """Clean up orphaned entities and old memories."""
        stats = {"forgotten": 0, "entities_cleaned": 0}

        # Remove orphaned entities (no memories mention them)
        result = self.graph.query(
            """
            MATCH (e:Entity)
            WHERE NOT (e)<-[:MENTIONS]-()
            DELETE e
            RETURN count(e) as deleted
            """
        )
        if result.result_set:
            stats["entities_cleaned"] = result.result_set[0][0]

        return stats

    async def recall_by_entity(
        self,
        entity_name: str,
        user_id: str = "default"
    ) -> List[Memory]:
        """Get all memories mentioning a specific entity."""
        result = self.graph.query(
            """
            MATCH (m:Memory)-[:MENTIONS]->(e:Entity {name: $name})
            WHERE m.user_id = $user_id
            RETURN m
            ORDER BY m.importance DESC, m.created_at DESC
            """,
            {"name": entity_name, "user_id": user_id}
        )

        memories = []
        for record in result.result_set:
            memory = self._node_to_memory(record[0])
            memory.content = self.encryption.decrypt_if_needed(
                memory.content,
                memory.encrypted
            )
            memories.append(memory)

        return memories

    async def get_related_entities(
        self,
        entity_name: str,
        user_id: str = "default"
    ) -> List[Dict[str, Any]]:
        """Find entities related to a given entity (co-mentioned in memories)."""
        result = self.graph.query(
            """
            MATCH (e1:Entity {name: $name})<-[:MENTIONS]-(m:Memory)-[:MENTIONS]->(e2:Entity)
            WHERE m.user_id = $user_id AND e1 <> e2
            RETURN e2.name as name, e2.type as type, count(m) as co_mentions
            ORDER BY co_mentions DESC
            """,
            {"name": entity_name, "user_id": user_id}
        )

        return [
            {"name": r[0], "type": r[1], "co_mentions": r[2]}
            for r in result.result_set
        ]

    def _node_to_memory(self, node) -> Memory:
        """Convert FalkorDB node to Memory object."""
        props = node.properties
        return Memory(
            id=props.get("id", ""),
            content=props.get("content", ""),
            tier=MemoryTier(props.get("tier", "longterm")),
            importance=float(props.get("importance", 0.5)),
            timestamp=datetime.fromisoformat(props["created_at"]) if "created_at" in props else datetime.now(),
            encrypted=props.get("encrypted", True),
            user_id=props.get("user_id", "default"),
        )
