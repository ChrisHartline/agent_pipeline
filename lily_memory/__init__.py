"""
Clara Memory System for ADK Agent

A three-tier memory architecture with client-side encryption:
- Session (HDC): Fast, in-memory, conversation context
- Graph (FalkorDB): Entity relationships and semantic connections
- Long-term (Supabase): Persistent facts, preferences, encrypted personal data

All personal data is encrypted client-side before storage.
"""

from .base import Memory, MemoryTier, MemoryStore
from .encryption import MemoryEncryption
from .hdc_memory import HDCMemory
from .supabase_store import SupabaseStore
from .falkor_store import FalkorStore
from .integrated_memory import IntegratedMemory, create_memory_system

__all__ = [
    "Memory",
    "MemoryTier",
    "MemoryStore",
    "MemoryEncryption",
    "HDCMemory",
    "SupabaseStore",
    "FalkorStore",
    "IntegratedMemory",
    "create_memory_system",
]
