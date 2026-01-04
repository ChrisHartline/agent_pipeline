# Clara HDC Architecture Roadmap

## Executive Summary

This document outlines proposed architectural upgrades for Clara, an AI agent with a dual-brain architecture (Mistral personality + Phi-3 knowledge) using Hyperdimensional Computing (HDC) for memory. These recommendations emerged from development sessions focused on improving memory recall, personality consistency, and scalability.

---

## Current Architecture

```
                              USER QUERY
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │    SEMANTIC ROUTER      │
                    │  (all-MiniLM-L6-v2)     │
                    │                         │
                    │  Embeds query, compares │
                    │  to domain descriptions │
                    └────────────┬────────────┘
                                 │
            ┌────────────────────┴────────────────────┐
            │                                         │
            ▼                                         ▼
   ┌─────────────────┐                     ┌─────────────────┐
   │ PERSONALITY     │                     │ KNOWLEDGE       │
   │ BRAIN           │                     │ BRAIN           │
   │                 │                     │                 │
   │ Mistral 7B      │                     │ Phi-3 (merged)  │
   │ + LoRA adapters │                     │                 │
   │   • warmth      │                     │ Domains:        │
   │   • playful     │                     │   • medical     │
   │   • encouragement│                    │   • coding      │
   │                 │                     │   • teaching    │
   │                 │                     │   • quantum     │
   └─────────────────┘                     └─────────────────┘
```

**HDC Memory Layer (v2.1):**
- 10,000-dimension bipolar hypervectors
- Semantic personality vectors (encoded from trait descriptions)
- Entity extraction and indexing
- Personality-based importance boosting
- Memory context injection into prompts

---

## Proposed Upgrades

### 1. 64k-Dimension Vectors + Bundle Merging

**Current:** 10,000-dimension vectors  
**Proposed:** 64,000-dimension vectors (configurable)

| Dimension | Memory/vector | Noise Tolerance | Use Case |
|-----------|---------------|-----------------|----------|
| 10k-dim   | 40 KB         | Good            | Edge (Jetson), <1000 memories |
| 64k-dim   | 256 KB        | Excellent       | Server, long-term, 10k+ memories |

**Benefits:**
- HDC capacity scales as O(d / log d) — 64k provides ~6x more binding capacity
- Finer clustering for distinguishing similar but distinct memories
- Better regime detection (recognizing conversation states)
- Improved noise resistance

**Bundle Merging (HDC's Killer Feature):**
```python
# Vector DB: Must rebuild index or use approximate methods
# HDC: 1-shot update, O(d) operation
memory_bundle = bundle([memory_bundle, new_memory_hv])  # Done!
```

**Implementation:**
```python
class ClaraHDCMemory:
    def __init__(self, embedder, dim: int = 10000, ...):
        # Easy to swap: dim=64000 for production
```

**Effort:** Low | **Impact:** Medium | **Priority:** ✅ Easy win

---

### 2. Alternative Router: Nemotron-Nano-2B

**Current:** Semantic router using all-MiniLM-L6-v2 embeddings  
**Alternative:** NVIDIA Nemotron-Nano-2B (open-source, Hugging Face)

| Router Type | Latency | Accuracy | Interpretability |
|-------------|---------|----------|------------------|
| Semantic (current) | ~5ms | 93%+ | High |
| Nemotron-Nano-2B | ~50-100ms | Higher? | Medium |
| HDC Router | ~2ms | TBD | Very high |

**Where Nemotron Shines:**
- Complex/ambiguous queries
- Multi-intent detection
- Context-dependent routing

**Recommended Hybrid Approach:**
```
Query → Fast semantic router → Confidence check
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
              Conf > 0.4                      Conf < 0.4
                    │                               │
                    ▼                               ▼
            Direct routing              Nemotron deliberation
              (5ms)                          (50ms)
```

**Alternative Architecture (Simpler):**
Use Nemotron as unified base with domain LoRA adapters:
```
Nemotron-Nano-2B (base)
├── personality_lora
├── coding_lora
├── medical_lora
├── teaching_lora
└── quantum_lora
```

**Effort:** High | **Impact:** Medium | **Priority:** ⏳ Evaluate vs. current

---

### 3. Voice Fine-Tuning from Chat History

**Goal:** Create distinctive "Clara voice" from conversation history  
**Data:** ~100k tokens of chat history  
**Method:** LoRA fine-tuning on base model

| Token Count | Quality |
|-------------|---------|
| 10k         | Basic patterns, inconsistent |
| 50k         | Recognizable voice, some gaps |
| 100k        | Strong voice adapter ✓ |
| 500k+       | Very consistent, risk of overfitting |

**Key Considerations:**
- Data quality > quantity
- Diversity of topics, emotions, response lengths
- Lower LoRA rank for style vs. knowledge (r=16 or r=32)

**Recommended Configuration:**
```python
voice_lora_config = LoraConfig(
    r=16,                    # Lower rank for style
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # Attention only
    lora_dropout=0.05,
)

# Training format:
# Human: [message]
# Clara: [response in Clara's voice]
```

**100k tokens ≈ 2,000-3,000 conversation turns** — sufficient if representative.

**Effort:** Medium | **Impact:** High | **Priority:** ✅ Differentiator

---

### 4. Recursive Reflection

**Concept:** Generate response, then self-edit for tone/consistency  
**Implementation:** Conditional "System 2" thinking for complex queries

**Pipeline:**
```
┌─────────────────────────────────────────────────────┐
│                  GENERATION PIPELINE                 │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Query → [Simple queries: fast path]               │
│              │                                      │
│              ▼                                      │
│         Generate response ──────────────────────▶ Output
│              │                                      │
│         [Complex/important: slow path]             │
│              │                                      │
│              ▼                                      │
│         Reflect & refine                            │
│         "Review for warmth, accuracy, tone"        │
│              │                                      │
│              ▼                                      │
│         Revised response ───────────────────────▶ Output
│                                                     │
└─────────────────────────────────────────────────────┘
```

**Reflection Triggers:**
- Response > 200 tokens
- Mixed emotion + technical query
- Low routing confidence
- User explicitly requests detail

**Implementation Sketch:**
```python
def clara_with_reflection(query: str, reflect: bool = "auto") -> str:
    response = clara(query, store_interaction=False)
    
    if reflect == "auto":
        reflect = should_reflect(query, response)
    
    if reflect:
        reflection_prompt = f"""Review this response for Clara's voice:
        
Original: {response}

Check for:
1. Warmth and encouragement
2. Patience
3. Clarity

Provide improved response:"""
        
        response = generate(reflection_prompt)
    
    store_interaction(query, response)
    return response
```

**Tradeoff:** ~2x latency for reflected responses

**Effort:** Medium | **Impact:** Medium | **Priority:** ⏳ After basics work

---

### 5. Chain-of-Thought Prompting

**Concept:** Structured thinking before response generation  
**Implementation:** Prompt engineering rather than architectural change

**CoT Prompt Template:**
```python
COT_PROMPT = """Think through this step by step:
1. What is the user really asking?
2. What do I know from memory that's relevant?
3. What domain expertise applies?
4. How would Clara (warm, patient, encouraging) phrase this?

User query: {query}

My thinking:
"""
```

**Complexity-Gated Generation:**
```python
def smart_generate(query, complexity_score):
    if complexity_score < 0.3:
        return direct_generate(query)        # Fast path
    elif complexity_score < 0.7:
        return cot_generate(query)           # Think first
    else:
        return cot_reflect_generate(query)   # Full pipeline
```

**Effort:** Low | **Impact:** Low | **Priority:** ⏳ Prompt engineering

---

### 6. Memory Tiers with HDC

**Concept:** Session / Daily / Long-term memory with elegant HDC-based retrieval  
**Key Insight:** HDC naturally blends tiers via similarity weighting — no rigid boundaries

**Tier Structure:**
```
┌─────────────────────────────────────────────────────────────┐
│                    HDC MEMORY TIERS                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  SESSION (Working Memory)                                   │
│  ├─ Last N turns (N=10)                                    │
│  ├─ Highest recall weight (1.0)                            │
│  ├─ No persistence                                         │
│  └─ Bound with: SESSION ⊗ TURN_N ⊗ content                │
│                                                             │
│  DAILY (Episodic Buffer)                                   │
│  ├─ Today's consolidated memories                          │
│  ├─ Medium recall weight (0.7)                             │
│  ├─ Persists until "sleep" cycle                          │
│  └─ Bound with: TODAY ⊗ TOPIC ⊗ content                   │
│                                                             │
│  LONG-TERM (Semantic Memory)                               │
│  ├─ High-importance consolidated facts                     │
│  ├─ Lower recall weight (0.5)                              │
│  ├─ Survives consolidation cycles                         │
│  └─ Bound with: PERMANENT ⊗ DOMAIN ⊗ content              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**HDC Retrieval (No Rigid Boundaries):**
```python
def recall(self, query, tier_weights=None):
    tier_weights = tier_weights or {
        'session': 1.0,
        'daily': 0.7,
        'longterm': 0.5,
    }
    
    query_hv = self._text_to_hv(query)
    
    results = []
    for hv, memory in self.memories:
        base_sim = self.similarity(query_hv, hv)
        tier_weight = tier_weights.get(memory.tier, 0.5)
        
        # Blend: relevance × tier × importance × recency
        score = base_sim * tier_weight * memory.importance * recency(memory)
        results.append((memory, score))
    
    return sorted(results, reverse=True)[:top_k]
```

**Consolidation ("Sleep" Cycle):**
```python
def consolidate(self):
    """Nightly consolidation cycle"""
    
    # 1. Session → Daily
    for mem in self.session_memories:
        mem.tier = 'daily'
    
    # 2. Daily → Long-term (high importance only)
    for mem in self.daily_memories:
        if mem.importance > 0.7:
            mem.tier = 'longterm'
            mem.text = extract_key_facts(mem.text)  # Compress
        elif mem.age_days > 7:
            self.forget(mem)  # Decay
    
    # 3. Pattern extraction
    patterns = extract_patterns(self.daily_memories)
    for pattern in patterns:
        self.store(pattern, tier='longterm', memory_type='pattern')
```

**Effort:** Medium | **Impact:** High | **Priority:** ✅ Core feature

---

## Implementation Priority

| Upgrade | Effort | Impact | Priority | Notes |
|---------|--------|--------|----------|-------|
| 64k-dim (configurable) | Low | Medium | ✅ P1 | Easy win, backward compatible |
| Memory tiers + consolidation | Medium | High | ✅ P1 | Core feature for long-term use |
| Voice LoRA (100k tokens) | Medium | High | ✅ P1 | Key differentiator |
| Recursive reflection | Medium | Medium | ⏳ P2 | After basics work |
| Nemotron router evaluation | High | Medium | ⏳ P2 | Compare against current |
| CoT loops | Low | Low | ⏳ P3 | Prompt engineering task |

---

## Current Implementation Status

### Completed (v2.1)
- ✅ Semantic router with 93%+ accuracy
- ✅ Dual-brain architecture (Mistral personality + Phi-3 knowledge)
- ✅ HDC memory with 10k-dim vectors
- ✅ Entity extraction and indexing
- ✅ Semantic personality vectors (encoded from trait descriptions)
- ✅ Personality-based importance boosting
- ✅ Memory context injection
- ✅ Memory persistence (save/load)
- ✅ Fixed recall threshold (0.15) for conversational follow-ups

### In Progress
- 🔄 Memory tier implementation
- 🔄 Consolidation cycle design

### Planned
- ⏳ 64k-dim vector support
- ⏳ Voice LoRA training pipeline
- ⏳ Recursive reflection (conditional)
- ⏳ Nemotron router evaluation

---

## Appendix: Clara Personality Specification

```python
CLARA_PERSONALITY_WEIGHTS = {
    'warmth': 0.85,       # Very warm, friendly
    'patience': 0.90,     # Very patient
    'curiosity': 0.75,    # Intellectually curious
    'encouragement': 0.85, # Very supportive
}
```

Each trait is encoded from rich semantic descriptions, enabling:
- Personality-aligned memory storage (warm interactions remembered more)
- Response alignment checking
- Consistent behavioral patterns

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Dec 2024 | Initial architecture roadmap |

---

*This document reflects architectural decisions for the Clara AI agent project, part of the D.Eng research on HDC-based memory systems for edge AI deployment.*
