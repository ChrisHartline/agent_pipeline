"""
Supabase (PostgreSQL) store for Clara's long-term memory.

Stores encrypted personal facts, preferences, and conversation summaries.
Uses client-side encryption - Supabase never sees plaintext personal data.

Required environment variables:
    SUPABASE_URL: Your Supabase project URL
    SUPABASE_KEY: Your Supabase anon/service key
    CLARA_MEMORY_KEY: Base64-encoded 32-byte encryption key
"""

import os
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from .base import Memory, MemoryTier, MemoryStore, TIER_WEIGHTS
from .encryption import get_encryption, MemoryEncryption

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False


# SQL schema for memories table
MEMORIES_SCHEMA = """
-- Run this in Supabase SQL editor to create the table

CREATE TABLE IF NOT EXISTS clara_memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL DEFAULT 'default',
    content TEXT NOT NULL,  -- Encrypted content
    tier TEXT NOT NULL DEFAULT 'longterm',
    importance FLOAT NOT NULL DEFAULT 0.5,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB DEFAULT '{}',
    encrypted BOOLEAN NOT NULL DEFAULT true,

    -- Indexes for efficient queries
    CONSTRAINT valid_tier CHECK (tier IN ('session', 'daily', 'longterm'))
);

-- Index for user queries
CREATE INDEX IF NOT EXISTS idx_memories_user_id ON clara_memories(user_id);

-- Index for tier filtering
CREATE INDEX IF NOT EXISTS idx_memories_tier ON clara_memories(tier);

-- Index for importance-based queries
CREATE INDEX IF NOT EXISTS idx_memories_importance ON clara_memories(importance DESC);

-- Row Level Security (optional but recommended)
ALTER TABLE clara_memories ENABLE ROW LEVEL SECURITY;

-- Policy: Users can only access their own memories
CREATE POLICY "Users can manage own memories" ON clara_memories
    FOR ALL
    USING (auth.uid()::text = user_id OR user_id = 'default');
"""


class SupabaseStore(MemoryStore):
    """
    Supabase-backed memory store with client-side encryption.

    All personal content is encrypted before being sent to Supabase.
    Only you (with the encryption key) can read your memories.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        key: Optional[str] = None,
        encryption: Optional[MemoryEncryption] = None,
        table_name: str = "clara_memories"
    ):
        """
        Initialize Supabase store.

        Args:
            url: Supabase project URL (or set SUPABASE_URL env var)
            key: Supabase anon key (or set SUPABASE_KEY env var)
            encryption: Encryption instance (or loaded from CLARA_MEMORY_KEY)
            table_name: Name of the memories table
        """
        if not SUPABASE_AVAILABLE:
            raise ImportError(
                "supabase package required. Install with: pip install supabase"
            )

        self.url = url or os.environ.get("SUPABASE_URL")
        self.key = key or os.environ.get("SUPABASE_KEY")

        if not self.url or not self.key:
            raise ValueError(
                "Supabase credentials required. Set SUPABASE_URL and SUPABASE_KEY "
                "environment variables or pass url/key parameters."
            )

        self.client: Client = create_client(self.url, self.key)
        self.table_name = table_name
        self.encryption = encryption or get_encryption()

    async def store(self, memory: Memory) -> str:
        """Store an encrypted memory in Supabase."""
        if not memory.id:
            memory.id = str(uuid.uuid4())

        # Encrypt content
        encrypted_content, was_encrypted = self.encryption.encrypt_if_personal(
            memory.content,
            is_personal=True  # Always encrypt for Supabase (long-term storage)
        )

        data = {
            "id": memory.id,
            "user_id": memory.user_id,
            "content": encrypted_content,
            "tier": memory.tier.value,
            "importance": memory.importance,
            "created_at": memory.timestamp.isoformat(),
            "updated_at": datetime.now().isoformat(),
            "metadata": memory.metadata,
            "encrypted": was_encrypted,
        }

        self.client.table(self.table_name).upsert(data).execute()
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
        Recall memories from Supabase.

        Note: Since content is encrypted, we can't do server-side similarity search.
        We fetch recent memories and do client-side matching after decryption.
        """
        # Build query
        q = self.client.table(self.table_name).select("*").eq("user_id", user_id)

        if tier_filter:
            tier_values = [t.value for t in tier_filter]
            q = q.in_("tier", tier_values)

        # Order by importance and recency
        q = q.order("importance", desc=True).order("created_at", desc=True)

        # Fetch more than top_k since we'll filter client-side
        q = q.limit(top_k * 3)

        result = q.execute()

        if not result.data:
            return []

        # Decrypt and score
        memories_with_scores = []
        query_lower = query.lower()
        query_words = set(query_lower.split())

        for row in result.data:
            memory = self._row_to_memory(row)

            # Decrypt content for matching
            plaintext = self.encryption.decrypt_if_needed(
                memory.content,
                memory.encrypted
            )

            # Simple keyword matching score (since we can't use embeddings on encrypted data)
            content_lower = plaintext.lower()
            content_words = set(content_lower.split())

            # Word overlap score
            overlap = len(query_words & content_words)
            if overlap == 0:
                # Check substring match
                if query_lower in content_lower:
                    overlap = 1
                else:
                    continue  # No match

            # Normalize by query length
            score = overlap / max(len(query_words), 1)

            # Apply tier weight
            tier_weight = TIER_WEIGHTS.get(memory.tier, 0.5)
            score *= tier_weight

            # Apply importance weight
            score *= (0.5 + memory.importance * 0.5)

            # Update memory content with decrypted version for return
            memory.content = plaintext

            memories_with_scores.append((memory, score))

        # Sort by score
        memories_with_scores.sort(key=lambda x: x[1], reverse=True)

        return memories_with_scores[:top_k]

    async def forget(self, memory_id: str) -> bool:
        """Delete a memory from Supabase."""
        result = self.client.table(self.table_name).delete().eq("id", memory_id).execute()
        return len(result.data) > 0 if result.data else False

    async def consolidate(self) -> Dict[str, int]:
        """
        Consolidation for Supabase is mainly cleanup.

        Long-term memories don't need tier promotion, but we can:
        - Remove very old low-importance memories
        - Update importance based on access patterns
        """
        stats = {"forgotten": 0}

        # Delete old low-importance memories (older than 30 days, importance < 0.3)
        from datetime import timedelta
        cutoff = (datetime.now() - timedelta(days=30)).isoformat()

        result = (
            self.client.table(self.table_name)
            .delete()
            .lt("created_at", cutoff)
            .lt("importance", 0.3)
            .execute()
        )

        if result.data:
            stats["forgotten"] = len(result.data)

        return stats

    def _row_to_memory(self, row: Dict[str, Any]) -> Memory:
        """Convert Supabase row to Memory object."""
        return Memory(
            id=row["id"],
            content=row["content"],
            tier=MemoryTier(row["tier"]),
            importance=row["importance"],
            timestamp=datetime.fromisoformat(row["created_at"].replace("Z", "+00:00")),
            metadata=row.get("metadata", {}),
            encrypted=row.get("encrypted", True),
            user_id=row["user_id"],
        )

    async def get_all_for_user(self, user_id: str = "default") -> List[Memory]:
        """Get all memories for a user."""
        result = (
            self.client.table(self.table_name)
            .select("*")
            .eq("user_id", user_id)
            .execute()
        )

        memories = []
        for row in result.data or []:
            memory = self._row_to_memory(row)
            # Decrypt
            memory.content = self.encryption.decrypt_if_needed(
                memory.content,
                memory.encrypted
            )
            memories.append(memory)

        return memories

    @staticmethod
    def get_schema() -> str:
        """Return SQL schema for setting up the table."""
        return MEMORIES_SCHEMA
