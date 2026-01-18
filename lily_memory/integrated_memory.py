"""
Integrated Memory System for Clara.

Combines three memory stores:
- HDC (Session): Fast, in-memory, current conversation
- FalkorDB (Graph): Entity relationships, semantic connections
- Supabase (Long-term): Persistent facts, encrypted personal data

Provides a unified interface for the ADK agent.
"""

import os
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import uuid

from .base import Memory, MemoryTier, MemoryStore, TIER_WEIGHTS
from .encryption import get_encryption, MemoryEncryption
from .hdc_memory import HDCMemory

# Lazy imports for optional dependencies
SupabaseStore = None
FalkorStore = None


def _load_supabase():
    global SupabaseStore
    if SupabaseStore is None:
        from .supabase_store import SupabaseStore as _SupabaseStore
        SupabaseStore = _SupabaseStore
    return SupabaseStore


def _load_falkor():
    global FalkorStore
    if FalkorStore is None:
        from .falkor_store import FalkorStore as _FalkorStore
        FalkorStore = _FalkorStore
    return FalkorStore


class IntegratedMemory(MemoryStore):
    """
    Unified memory system combining HDC, FalkorDB, and Supabase.

    Memory flow:
    1. New memories go to HDC (session tier)
    2. Important session memories get promoted to daily (HDC + Supabase)
    3. High-importance daily memories become long-term (Supabase)
    4. Entity relationships are tracked in FalkorDB

    Recall searches all stores and merges results.
    """

    def __init__(
        self,
        encryption: Optional[MemoryEncryption] = None,
        hdc_dimensions: int = 64000,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        falkor_host: Optional[str] = None,
        falkor_port: Optional[int] = None,
        enable_supabase: bool = True,
        enable_falkor: bool = True,
        user_id: str = "default"
    ):
        """
        Initialize integrated memory system.

        Args:
            encryption: Encryption instance (or loaded from env)
            hdc_dimensions: Dimensions for HDC vectors
            supabase_url: Supabase URL (or SUPABASE_URL env)
            supabase_key: Supabase key (or SUPABASE_KEY env)
            falkor_host: FalkorDB host (or FALKOR_HOST env)
            falkor_port: FalkorDB port (or FALKOR_PORT env)
            enable_supabase: Whether to use Supabase
            enable_falkor: Whether to use FalkorDB
            user_id: Default user ID for memories
        """
        self.encryption = encryption or get_encryption()
        self.user_id = user_id

        # Always use HDC for session memory (in-memory, fast)
        self.hdc = HDCMemory(dimensions=hdc_dimensions)

        # Optional: Supabase for long-term storage
        self.supabase: Optional[MemoryStore] = None
        if enable_supabase:
            try:
                Store = _load_supabase()
                self.supabase = Store(
                    url=supabase_url,
                    key=supabase_key,
                    encryption=self.encryption
                )
            except Exception as e:
                print(f"Warning: Supabase not available: {e}")

        # Optional: FalkorDB for graph relationships
        self.falkor: Optional[MemoryStore] = None
        if enable_falkor:
            try:
                Store = _load_falkor()
                self.falkor = Store(
                    host=falkor_host,
                    port=falkor_port,
                    encryption=self.encryption
                )
            except Exception as e:
                print(f"Warning: FalkorDB not available: {e}")

    async def store(
        self,
        memory: Memory = None,
        content: str = None,
        importance: float = 0.5,
        tier: MemoryTier = MemoryTier.SESSION,
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        Store a memory across appropriate backends.

        Can pass either a Memory object or individual fields.

        Args:
            memory: Memory object to store
            content: Memory content (if not passing Memory object)
            importance: Importance score 0-1
            tier: Memory tier
            metadata: Additional metadata

        Returns:
            Memory ID
        """
        if memory is None:
            if content is None:
                raise ValueError("Either memory or content required")
            memory = Memory(
                id=str(uuid.uuid4()),
                content=content,
                tier=tier,
                importance=importance,
                timestamp=datetime.now(),
                metadata=metadata or {},
                user_id=self.user_id,
            )

        # Always store in HDC for fast session access
        await self.hdc.store(memory)

        # Store in FalkorDB for entity relationships
        if self.falkor and tier in [MemoryTier.DAILY, MemoryTier.LONGTERM]:
            try:
                await self.falkor.store(memory)
            except Exception as e:
                print(f"Warning: FalkorDB store failed: {e}")

        # Store in Supabase for long-term persistence
        if self.supabase and tier == MemoryTier.LONGTERM:
            try:
                await self.supabase.store(memory)
            except Exception as e:
                print(f"Warning: Supabase store failed: {e}")

        return memory.id

    async def recall(
        self,
        query: str,
        top_k: int = 5,
        tier_filter: Optional[List[MemoryTier]] = None,
        user_id: str = None
    ) -> List[Tuple[Memory, float]]:
        """
        Recall memories from all stores, merged by relevance.

        Searches HDC, FalkorDB, and Supabase, then merges results.
        """
        user_id = user_id or self.user_id
        all_results: Dict[str, Tuple[Memory, float]] = {}

        # Search HDC (always available)
        hdc_results = await self.hdc.recall(query, top_k * 2, tier_filter, user_id)
        for memory, score in hdc_results:
            all_results[memory.id] = (memory, score)

        # Search FalkorDB (entity-based)
        if self.falkor:
            try:
                falkor_results = await self.falkor.recall(query, top_k, tier_filter, user_id)
                for memory, score in falkor_results:
                    if memory.id in all_results:
                        # Boost score if found in multiple stores
                        existing = all_results[memory.id]
                        all_results[memory.id] = (memory, max(existing[1], score) * 1.2)
                    else:
                        all_results[memory.id] = (memory, score)
            except Exception as e:
                print(f"Warning: FalkorDB recall failed: {e}")

        # Search Supabase (long-term)
        if self.supabase and (tier_filter is None or MemoryTier.LONGTERM in tier_filter):
            try:
                supabase_results = await self.supabase.recall(query, top_k, tier_filter, user_id)
                for memory, score in supabase_results:
                    if memory.id in all_results:
                        existing = all_results[memory.id]
                        all_results[memory.id] = (memory, max(existing[1], score) * 1.2)
                    else:
                        all_results[memory.id] = (memory, score)
            except Exception as e:
                print(f"Warning: Supabase recall failed: {e}")

        # Sort by score and return top_k
        results = list(all_results.values())
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    async def forget(self, memory_id: str) -> bool:
        """Remove a memory from all stores."""
        forgotten = False

        # Remove from HDC
        if await self.hdc.forget(memory_id):
            forgotten = True

        # Remove from FalkorDB
        if self.falkor:
            try:
                if await self.falkor.forget(memory_id):
                    forgotten = True
            except Exception:
                pass

        # Remove from Supabase
        if self.supabase:
            try:
                if await self.supabase.forget(memory_id):
                    forgotten = True
            except Exception:
                pass

        return forgotten

    async def consolidate(self) -> Dict[str, Any]:
        """
        Run consolidation across all stores.

        Promotes important memories between tiers and cleans up.
        """
        stats = {
            "hdc": {},
            "falkor": {},
            "supabase": {},
            "promoted_to_longterm": 0,
        }

        # HDC consolidation (session -> daily)
        stats["hdc"] = await self.hdc.consolidate()

        # Find memories ready for long-term storage
        daily_memories = [
            m for m in self.hdc.get_all_memories(self.user_id)
            if m.tier == MemoryTier.DAILY and m.importance > 0.7
        ]

        # Promote to long-term (Supabase)
        for memory in daily_memories:
            memory.tier = MemoryTier.LONGTERM

            if self.supabase:
                try:
                    await self.supabase.store(memory)
                    stats["promoted_to_longterm"] += 1
                except Exception as e:
                    print(f"Warning: Failed to promote to long-term: {e}")

            if self.falkor:
                try:
                    await self.falkor.store(memory)
                except Exception:
                    pass

        # FalkorDB consolidation
        if self.falkor:
            try:
                stats["falkor"] = await self.falkor.consolidate()
            except Exception:
                pass

        # Supabase consolidation
        if self.supabase:
            try:
                stats["supabase"] = await self.supabase.consolidate()
            except Exception:
                pass

        return stats

    async def remember_fact(
        self,
        fact: str,
        importance: float = 0.8
    ) -> str:
        """
        Store a long-term fact about the user.

        Convenience method for ADK agent tools.
        """
        return await self.store(
            content=fact,
            importance=importance,
            tier=MemoryTier.LONGTERM
        )

    async def remember_preference(
        self,
        preference: str,
        importance: float = 0.7
    ) -> str:
        """Store a user preference."""
        return await self.store(
            content=f"User preference: {preference}",
            importance=importance,
            tier=MemoryTier.LONGTERM,
            metadata={"type": "preference"}
        )

    async def remember_conversation(
        self,
        content: str,
        importance: float = 0.5
    ) -> str:
        """Store a conversation snippet (session memory)."""
        return await self.store(
            content=content,
            importance=importance,
            tier=MemoryTier.SESSION
        )

    async def recall_about(
        self,
        entity: str,
        user_id: str = None
    ) -> List[Memory]:
        """Recall everything known about an entity."""
        user_id = user_id or self.user_id

        if self.falkor:
            try:
                return await self.falkor.recall_by_entity(entity, user_id)
            except Exception:
                pass

        # Fallback to keyword search
        results = await self.recall(entity, top_k=10, user_id=user_id)
        return [m for m, _ in results]

    async def get_context_for_prompt(
        self,
        query: str,
        max_memories: int = 5,
        max_tokens: int = 800
    ) -> str:
        """
        Build context string for LLM prompt injection.

        Returns formatted memories relevant to the query.
        """
        memories = await self.recall(query, top_k=max_memories)

        if not memories:
            return ""

        context_parts = []
        estimated_tokens = 0

        for memory, score in memories:
            mem_tokens = len(memory.content) // 4
            if estimated_tokens + mem_tokens > max_tokens:
                break

            tier_label = memory.tier.value[0].upper()  # S/D/L
            context_parts.append(f"[{tier_label}] {memory.content}")
            estimated_tokens += mem_tokens

        return "Relevant memories:\n" + "\n".join(context_parts)


def create_memory_system(
    mode: str = "full",
    encryption_key: Optional[bytes] = None,
    encryption_password: Optional[str] = None,
    **kwargs
) -> IntegratedMemory:
    """
    Factory function to create a memory system.

    Args:
        mode: "full" (all stores), "minimal" (HDC only), "test" (no encryption)
        encryption_key: Raw encryption key
        encryption_password: Password to derive key from
        **kwargs: Additional arguments for IntegratedMemory

    Returns:
        Configured IntegratedMemory instance
    """
    # Set up encryption
    if mode == "test":
        encryption = get_encryption(disabled=True)
    elif encryption_key:
        encryption = get_encryption(key=encryption_key)
    elif encryption_password:
        encryption = get_encryption(password=encryption_password)
    else:
        encryption = get_encryption()  # From env or disabled

    # Configure stores based on mode
    if mode == "minimal":
        return IntegratedMemory(
            encryption=encryption,
            enable_supabase=False,
            enable_falkor=False,
            **kwargs
        )
    elif mode == "test":
        return IntegratedMemory(
            encryption=encryption,
            enable_supabase=False,
            enable_falkor=False,
            hdc_dimensions=1000,  # Smaller for testing
            **kwargs
        )
    else:  # full
        return IntegratedMemory(
            encryption=encryption,
            enable_supabase=True,
            enable_falkor=True,
            **kwargs
        )
