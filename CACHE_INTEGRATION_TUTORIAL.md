# 🎓 CACHE INTEGRATION TUTORIAL

**Learn by Doing: Add Caching to Your RAG API**

This tutorial will teach you HOW and WHY to integrate caching into your chatbot API.

---

## 📚 LEARNING OBJECTIVES

By the end of this tutorial, you'll understand:
1. ✅ What caching is and why it matters
2. ✅ How to check cache before expensive operations
3. ✅ How to store results in cache for next time
4. ✅ How to initialize caches at startup
5. ✅ How to monitor cache performance

---

## 🎯 PART 1: Understanding the Flow

### **Without Caching (Current State)**
```
User Query: "How does suspension work?"
    ↓
Step 1: Embed query (100ms) 🐌
    ↓
Step 2: Search vector DB (300ms) 🐌
    ↓
Step 3: Rerank with cross-encoder (800ms) 🐌🐌🐌
    ↓
Total: 1200ms ❌ Slow!
```

### **With Caching (What We're Building)**
```
User Query: "How does suspension work?" (first time)
    ↓
Step 1: Check cache → MISS
    ↓
Step 2: Embed query (100ms)
    ↓
Step 3: Store in cache ✅
    ↓
Step 4: Search vector DB (300ms)
    ↓
Step 5: Store in cache ✅
    ↓
Step 6: Rerank (800ms)
    ↓
Step 7: Store in cache ✅
    ↓
Total: 1200ms (same as before)

---

User Query: "How does suspension work?" (second time)
    ↓
Step 1: Check cache → HIT! Return cached results
    ↓
Total: 1ms ⚡ 1200x faster!
```

---

## 🔧 PART 2: Integration Steps

### **STEP 2.1: Add Import to retrieval_segmented.py**

**What to do:**
Open `Pipeline_chatbot_v1/rag/retrieval_segmented.py`

**Find this line (around line 8):**
```python
from rag.router import QueryRouter
```

**Add this line RIGHT AFTER it:**
```python
from rag.cache import get_cache_manager  # NEW: Import cache access function
```

**Why?**
- `get_cache_manager()` gives us access to all caches
- We import it at the top so it's available everywhere in the file

---

### **STEP 2.2: Modify the `retrieve()` Method**

**Location:** Find the `retrieve()` method around line 103

**Current code looks like this:**
```python
def retrieve(
    self,
    query: str,
    top_n: int = 6,
    k_per_collection: int = 15,
    use_routing: bool = True
) -> Tuple[List[Document], List[float]]:
    """Main retrieval method with routing and reranking."""

    # Determine which collections to search
    if use_routing:
        collection_names = self.router.route_with_fallback(...)
    else:
        collection_names = list(self.collections.keys())

    # Retrieve from collections
    docs = self.retrieve_from_collections(query, collection_names, k_per_collection)

    if not docs:
        return [], []

    # Deduplicate
    docs = self.deduplicate_documents(docs)

    # Rerank with cross-encoder
    top_docs, scores = return_top_doc(query, docs, top_n=top_n)

    return top_docs, scores
```

---

### **STEP 2.2.1: Add Cache Manager Access**

**RIGHT AFTER the docstring, add:**
```python
# Get access to the global cache manager
cache_mgr = get_cache_manager()
```

**What this does:**
- Gets the singleton cache instance we'll initialize at startup
- If caching isn't initialized, returns `None` (safe)

---

### **STEP 2.2.2: Cache Retrieval Results**

**FIND THIS SECTION:**
```python
# Retrieve from collections
docs = self.retrieve_from_collections(query, collection_names, k_per_collection)
```

**REPLACE IT WITH THIS:**
```python
# CACHING LAYER 1: Try to get retrieval results from cache
docs = None
if cache_mgr and cache_mgr.retrieval_cache:
    # Create cache key from query + collections
    collections_tuple = tuple(sorted(collection_names))
    docs = cache_mgr.retrieval_cache.get_retrieval(query, collections_tuple, k_per_collection)

    if docs is not None:
        print(f"[Cache HIT] Retrieved {len(docs)} docs from cache (skipped DB search)")

# If cache miss, do the expensive retrieval
if docs is None:
    docs = self.retrieve_from_collections(query, collection_names, k_per_collection)

    if not docs:
        print("[Retrieval] No documents found")
        return [], []

    # Deduplicate
    docs = self.deduplicate_documents(docs)
    print(f"[Retrieval] {len(docs)} unique documents retrieved from DB")

    # Store in cache for next time
    if cache_mgr and cache_mgr.retrieval_cache:
        collections_tuple = tuple(sorted(collection_names))
        cache_mgr.retrieval_cache.set_retrieval(query, collections_tuple, k_per_collection, docs)
        print(f"[Cache] Stored retrieval results")
```

**Understanding the Code:**

1. **Check cache first:**
   ```python
   docs = cache_mgr.retrieval_cache.get_retrieval(query, collections_tuple, k_per_collection)
   ```
   - If found → `docs` contains cached results (skip DB!)
   - If not found → `docs` is `None` (need to search DB)

2. **Why `tuple(sorted(collection_names))`?**
   - Cache keys must be consistent
   - `["vehicle", "api"]` and `["api", "vehicle"]` should be the same
   - Sorting ensures same order every time

3. **Store results after retrieval:**
   ```python
   cache_mgr.retrieval_cache.set_retrieval(query, collections_tuple, k_per_collection, docs)
   ```
   - Next time same query comes → instant cache hit!

---

### **STEP 2.2.3: Cache Reranking Results**

**FIND THIS SECTION:**
```python
# Rerank with cross-encoder
top_docs, scores = return_top_doc(query, docs, top_n=top_n)

return top_docs, scores
```

**REPLACE IT WITH THIS:**
```python
# CACHING LAYER 2: Try to get reranked results from cache
cached_rerank = None
if cache_mgr and cache_mgr.rerank_cache:
    cached_rerank = cache_mgr.rerank_cache.get_reranked(query, docs, top_n)

    if cached_rerank is not None:
        top_docs, scores = cached_rerank
        print(f"[Cache HIT] Retrieved reranked results from cache (skipped cross-encoder)")
        return top_docs, scores

# If cache miss, do expensive reranking
print(f"[Reranking] Cross-encoding {len(docs)} documents...")
top_docs, scores = return_top_doc(query, docs, top_n=top_n)

# Store reranked results in cache
if cache_mgr and cache_mgr.rerank_cache:
    cache_mgr.rerank_cache.set_reranked(query, docs, top_n, top_docs, scores)
    print(f"[Cache] Stored reranked results")

return top_docs, scores
```

**Understanding the Code:**

1. **Check rerank cache:**
   ```python
   cached_rerank = cache_mgr.rerank_cache.get_reranked(query, docs, top_n)
   ```
   - Reranking is VERY expensive (500-2000ms)
   - If we already reranked these docs for this query → use cached results!

2. **Cache key includes docs:**
   - Same query + same docs = same ranking
   - Cache creates "fingerprint" of docs (first 200 chars of each)
   - If fingerprint matches → safe to use cached ranking

3. **Store after reranking:**
   ```python
   cache_mgr.rerank_cache.set_reranked(query, docs, top_n, top_docs, scores)
   ```
   - Saves both reranked docs AND scores

---

### **COMPLETE MODIFIED `retrieve()` METHOD**

Here's what your complete method should look like after changes:

```python
def retrieve(
    self,
    query: str,
    top_n: int = 6,
    k_per_collection: int = 15,
    use_routing: bool = True
) -> Tuple[List[Document], List[float]]:
    """
    Main retrieval method with routing and reranking.
    Now with two-layer caching for performance!
    """
    # Get cache manager
    cache_mgr = get_cache_manager()

    # Determine which collections to search
    if use_routing:
        collection_names = self.router.route_with_fallback(
            query,
            list(self.collections.keys())
        )
        print(f"[Routing] Searching in: {collection_names}")
    else:
        collection_names = list(self.collections.keys())
        print(f"[No Routing] Searching all {len(collection_names)} collections")

    # LAYER 1: Try retrieval cache
    docs = None
    if cache_mgr and cache_mgr.retrieval_cache:
        collections_tuple = tuple(sorted(collection_names))
        docs = cache_mgr.retrieval_cache.get_retrieval(query, collections_tuple, k_per_collection)
        if docs is not None:
            print(f"[Cache HIT] Retrieved {len(docs)} docs from cache")

    # If cache miss, retrieve from DB
    if docs is None:
        docs = self.retrieve_from_collections(query, collection_names, k_per_collection)

        if not docs:
            print("[Retrieval] No documents found")
            return [], []

        docs = self.deduplicate_documents(docs)
        print(f"[Retrieval] {len(docs)} unique documents from DB")

        # Store in cache
        if cache_mgr and cache_mgr.retrieval_cache:
            collections_tuple = tuple(sorted(collection_names))
            cache_mgr.retrieval_cache.set_retrieval(query, collections_tuple, k_per_collection, docs)
            print(f"[Cache] Stored retrieval results")

    # LAYER 2: Try rerank cache
    cached_rerank = None
    if cache_mgr and cache_mgr.rerank_cache:
        cached_rerank = cache_mgr.rerank_cache.get_reranked(query, docs, top_n)
        if cached_rerank is not None:
            top_docs, scores = cached_rerank
            print(f"[Cache HIT] Reranked results from cache")
            return top_docs, scores

    # If cache miss, rerank
    print(f"[Reranking] Cross-encoding {len(docs)} documents...")
    top_docs, scores = return_top_doc(query, docs, top_n=top_n)

    # Store in cache
    if cache_mgr and cache_mgr.rerank_cache:
        cache_mgr.rerank_cache.set_reranked(query, docs, top_n, top_docs, scores)
        print(f"[Cache] Stored reranked results")

    return top_docs, scores
```

---

## 🚀 PART 3: Initialize Cache at Startup

### **STEP 3.1: Add Import to api_segmented.py**

**Open:** `Pipeline_chatbot_v1/api_segmented.py`

**Find the imports section (around line 36):**
```python
from rag.retrieval_segmented import SegmentedRetriever
from rag.retrieval import (...)
from rag.prompt import Chatbot_prompt
from rag.memory import HybridMemory
```

**Add this import:**
```python
from rag.cache import init_cache_manager, get_cache_manager  # NEW: Cache initialization
```

---

### **STEP 3.2: Initialize Cache in startup() Function**

**Find the `startup()` function (around line 362):**

**Current code:**
```python
@app.on_event("startup")
def startup():
    """Initialize based on ENABLE_SEGMENTATION flag."""
    global vectordb, retriever, segmented_retriever, collections, memory, clarify_state

    if ENABLE_SEGMENTATION:
        print("\n" + "="*60)
        print("STARTING IN SEGMENTED MODE")
        # ... rest of initialization
```

**ADD THIS RIGHT AFTER the global declaration:**
```python
@app.on_event("startup")
def startup():
    """Initialize based on ENABLE_SEGMENTATION flag."""
    global vectordb, retriever, segmented_retriever, collections, memory, clarify_state

    # NEW: Initialize caching system
    print("\n" + "="*60)
    print("INITIALIZING CACHE SYSTEM")
    print("="*60)
    cache_mgr = init_cache_manager(
        embedding_cache_size=500,      # Store 500 query embeddings
        retrieval_cache_size=300,      # Store 300 retrieval results
        rerank_cache_size=200,         # Store 200 reranking results
        enable_embedding_cache=True,   # Enable all caches
        enable_retrieval_cache=True,
        enable_rerank_cache=True
    )
    print(f"[Cache] Initialization complete!")

    # Rest of existing code...
    if ENABLE_SEGMENTATION:
        print("\n" + "="*60)
        print("STARTING IN SEGMENTED MODE")
        # ...
```

**What this does:**
- Creates global cache instance before anything else
- All caches are ready when retrieval code needs them
- Prints confirmation so you know caching is active

---

### **STEP 3.3: Add Cache Statistics Endpoint**

**Add this NEW endpoint to api_segmented.py (anywhere after @app.on_event("startup")):**

```python
@app.get("/cache/stats")
def cache_stats():
    """
    Get cache performance statistics

    Returns hit rates, sizes, and utilization for all caches
    """
    cache_mgr = get_cache_manager()

    if not cache_mgr:
        return {"error": "Cache not initialized"}

    return {
        "stats": cache_mgr.get_all_stats(),
        "performance_note": "High hit rates = good caching effectiveness"
    }


@app.post("/cache/clear")
def cache_clear():
    """
    Clear all caches

    Use this after re-indexing documents or for testing
    """
    cache_mgr = get_cache_manager()

    if not cache_mgr:
        return {"error": "Cache not initialized"}

    cache_mgr.clear_all()

    return {
        "status": "success",
        "message": "All caches cleared"
    }
```

**What these endpoints do:**
- `/cache/stats` - See how well caching is working
- `/cache/clear` - Reset caches when needed

---

## 📊 PART 4: Testing Your Implementation

### **TEST 1: Verify Cache Initialization**

**Start your API:**
```bash
cd d:\GJ\Chatbot_Api\Pipeline_chatbot_v1
python api_segmented.py
```

**Look for these logs:**
```
============================================================
INITIALIZING CACHE SYSTEM
============================================================
[Cache] Initializing caching system...
[Cache] Enabled: Embedding(500), Retrieval(300), Rerank(200)
[Cache] Initialization complete!
```

✅ **If you see this → Cache system is working!**

---

### **TEST 2: Test Cache Miss (First Query)**

**Send a query:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "How does suspension work?"}'
```

**Look for these logs:**
```
[Routing] Searching in: ['vehicle', 'studio']
[Retrieval] 35 unique documents from DB
[Cache] Stored retrieval results
[Reranking] Cross-encoding 35 documents...
[Cache] Stored reranked results
```

✅ **No cache hits (first time) - results stored for next time**

---

### **TEST 3: Test Cache Hit (Same Query Again)**

**Send SAME query again:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "How does suspension work?"}'
```

**Look for these logs:**
```
[Routing] Searching in: ['vehicle', 'studio']
[Cache HIT] Retrieved 35 docs from cache
[Cache HIT] Reranked results from cache
```

✅ **Both caches hit! Query processed in <10ms instead of >1000ms**

---

### **TEST 4: Check Cache Statistics**

**Get cache stats:**
```bash
curl http://localhost:8000/cache/stats
```

**Expected response:**
```json
{
  "stats": {
    "embedding": {
      "size": 2,
      "max_size": 500,
      "hits": 1,
      "misses": 1,
      "hit_rate": "50.0%",
      "utilization": "0.4%"
    },
    "retrieval": {
      "size": 1,
      "max_size": 300,
      "hits": 1,
      "misses": 1,
      "hit_rate": "50.0%"
    },
    "rerank": {
      "size": 1,
      "max_size": 200,
      "hits": 1,
      "misses": 1,
      "hit_rate": "50.0%"
    }
  }
}
```

✅ **Hit rate of 50% = caching is working!**

---

## 🎓 PART 5: Understanding Cache Behavior

### **When Does Cache Hit?**

✅ **EXACT same query:**
```
Query 1: "How does suspension work?"
Query 2: "How does suspension work?"
→ Cache HIT! (100% match)
```

✅ **Same query, different case:**
```
Query 1: "How does suspension work?"
Query 2: "how does suspension work?"
→ Cache HIT! (normalized to lowercase)
```

❌ **Similar but different queries:**
```
Query 1: "How does suspension work?"
Query 2: "How does the suspension work?"
→ Cache MISS (different text)
```

❌ **Same query, different collections:**
```
Query 1: "suspension" → searches ["vehicle"]
Query 2: "suspension" → searches ["vehicle", "api"]
→ Cache MISS (different collections)
```

---

### **Cache Key Components**

**Retrieval Cache Key:**
```
query: "suspension"
collections: ("vehicle", "studio")  # sorted tuple
k: 15

Key = MD5("suspension|('studio', 'vehicle')|15")
    = "a4f9c2e8..."
```

**Rerank Cache Key:**
```
query: "suspension"
doc_fingerprint: "abc123..."  # hash of first 200 chars of each doc
top_n: 6

Key = MD5("suspension|abc123...|6")
    = "7d3e1f4a..."
```

---

## 🚀 PART 6: Performance Monitoring

### **Add This Helper to api_segmented.py:**

```python
@app.middleware("http")
async def log_cache_performance(request: Request, call_next):
    """Log cache hit rates for each request"""
    cache_mgr = get_cache_manager()

    # Get stats before request
    if cache_mgr:
        stats_before = cache_mgr.get_all_stats()

    # Process request
    response = await call_next(request)

    # Get stats after request
    if cache_mgr and request.url.path == "/chat":
        stats_after = cache_mgr.get_all_stats()

        # Print cache performance for this request
        print("\n" + "="*50)
        print("CACHE PERFORMANCE (This Request)")
        print("="*50)
        for cache_name in stats_after:
            before = stats_before.get(cache_name, {})
            after = stats_after[cache_name]

            hits_delta = after["hits"] - before.get("hits", 0)
            if hits_delta > 0:
                print(f"{cache_name}: {hits_delta} cache hits ")
        print("="*50 + "\n")

    return response
```

**What this does:**
- Tracks cache hits per request
- Prints summary after each /chat call
- Shows you exactly what was cached

---

## 🎯 PART 7: Advanced: When to Clear Cache

### **Clear Cache When:**

1. **After re-indexing documents:**
   ```bash
   curl -X POST http://localhost:8000/cache/clear
   ```
   Why? Old cached retrieval results point to old documents

2. **During development/testing:**
   - Clear between test runs
   - Ensure you're testing real performance, not cached

3. **If memory usage is high:**
   - Caches automatically evict old items (LRU)
   - But manual clear gives instant memory back

### **DON'T Clear Cache When:**

❌ **During normal operation** - defeats the purpose!
❌ **After every query** - you want cache hits!
❌ **Just because** - cache helps performance

---

## 📈 EXPECTED RESULTS

### **After 100 Queries:**

**Without Cache:**
- Average latency: 1200ms per query
- Total time: 120 seconds

**With Cache (50% hit rate):**
- 50 queries from cache: 50 × 2ms = 100ms
- 50 queries from DB: 50 × 1200ms = 60,000ms
- Total time: 60.1 seconds
- **Speedup: 2x faster!**

**With Cache (80% hit rate):**
- 80 queries from cache: 80 × 2ms = 160ms
- 20 queries from DB: 20 × 1200ms = 24,000ms
- Total time: 24.16 seconds
- **Speedup: 5x faster!**

---

## ✅ CHECKLIST: Did You Do Everything?

- [ ] Created `cache.py` with all cache classes
- [ ] Added `from rag.cache import get_cache_manager` to `retrieval_segmented.py`
- [ ] Modified `retrieve()` method with retrieval caching
- [ ] Modified `retrieve()` method with reranking caching
- [ ] Added `from rag.cache import init_cache_manager` to `api_segmented.py`
- [ ] Called `init_cache_manager()` in `startup()` function
- [ ] Added `/cache/stats` endpoint
- [ ] Added `/cache/clear` endpoint
- [ ] Tested with first query (cache miss)
- [ ] Tested with repeat query (cache hit)
- [ ] Checked cache statistics
- [ ] Celebrated your performance gains! 🎉

---

## 🤔 COMMON ISSUES & SOLUTIONS

### **Issue: "Cache not initialized" error**

**Problem:** Trying to use cache before startup
**Solution:** Make sure `init_cache_manager()` runs before anything else

---

### **Issue: Cache never hits**

**Problem 1:** Queries are slightly different
**Solution:** Check logs - queries must be EXACTLY the same

**Problem 2:** Collections are different
**Solution:** Same query with different routing → different cache key

---

### **Issue: Hit rate is 0%**

**Problem:** Each query is unique
**Solution:** This is normal if users ask many different questions
           Cache works best for repeated/similar questions

---

## 🎓 KEY TAKEAWAYS

1. **Caching = Store expensive results for reuse**
2. **Check cache BEFORE expensive operation**
3. **Store results AFTER expensive operation**
4. **Cache keys must include ALL relevant parameters**
5. **LRU = automatic memory management**
6. **Monitor hit rates to measure effectiveness**

---

## 🚀 NEXT STEPS

After you've successfully integrated caching:

1. **Monitor performance in production**
   - Check `/cache/stats` regularly
   - Aim for >30% hit rate

2. **Tune cache sizes based on usage**
   - More cache = better hit rate
   - But uses more memory

3. **Consider semantic caching** (advanced)
   - Cache similar queries, not just exact matches
   - Use embedding similarity to find "close enough" queries

4. **Add cache warming** (advanced)
   - Pre-populate cache with common queries
   - Faster response for first users

---

**Good luck! You're building a production-grade caching system! 🎉**
