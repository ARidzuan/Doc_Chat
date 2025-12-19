"""
cache.py - Multi-level caching for RAG pipeline

LEARNING CONCEPTS:
==================
1. LRU Cache: Least Recently Used eviction policy
   - Items move to front when accessed
   - Oldest unused items are removed when full

2. Cache Keys: How we identify cached items
   - Hashed for speed (MD5 digest)
   - Include all relevant parameters

3. Performance: Cache hit vs miss
   - Hit: Found in cache (fast!)
   - Miss: Not in cache (slow, need to compute)
   - Hit rate: % of requests served from cache

WHY CACHE?
==========
- Embedding: 100-500ms → 0.1ms (1000x faster!)
- Retrieval: 200-1000ms → 0.5ms (500x faster!)
- Reranking: 500-2000ms → 0.5ms (1000x faster!)
"""

from functools import lru_cache
import hashlib
import pickle
from typing import Tuple, List, Optional, Any
from collections import OrderedDict
from langchain.schema import Document


class LRUCache:
    """
    Least Recently Used (LRU) Cache

    How it works (visual example):

    Initial: []
    Add 'a': [a]
    Add 'b': [b, a]
    Add 'c': [c, b, a]
    Access 'a': [a, c, b]  <- 'a' moved to front!
    Add 'd' (max=3): [d, a, c]  <- 'b' removed (was at back)

    This ensures frequently used items stay in cache.
    """

    def __init__(self, max_size: int = 1000):
        """
        Initialize LRU cache

        Args:
            max_size: Maximum items to store
        """
        self.max_size = max_size
        # OrderedDict maintains insertion order + fast reordering
        self.cache = OrderedDict()

        # Metrics (for monitoring performance)
        self.hits = 0    # How many times we found item in cache
        self.misses = 0  # How many times item wasn't in cache

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache

        Returns:
            Value if found, None if not found
        """
        if key not in self.cache:
            self.misses += 1
            return None

        # Move to end = mark as recently used
        self.cache.move_to_end(key)
        self.hits += 1
        return self.cache[key]

    def set(self, key: str, value: Any):
        """
        Store value in cache

        If cache is full, removes oldest item first
        """
        # Update existing key
        if key in self.cache:
            self.cache.move_to_end(key)

        self.cache[key] = value

        # Evict oldest if over limit
        if len(self.cache) > self.max_size:
            # Remove first item (oldest)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

    def clear(self):
        """Clear all cached items"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    def get_stats(self) -> dict:
        """Get cache performance statistics"""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0

        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.1f}%",
            "utilization": f"{len(self.cache) / self.max_size * 100:.1f}%"
        }


class EmbeddingCache(LRUCache):
    """
    Cache for query embeddings

    WHAT IT CACHES:
    - Input: "How does suspension work?"
    - Output: [0.23, 0.45, 0.12, ...] (384-dim vector)

    WHY IT'S EXPENSIVE:
    - Neural network inference (100-500ms)
    - Done for EVERY query
    - Same query = same embedding (deterministic)

    USAGE:
        cache = EmbeddingCache()

        # First time (slow)
        emb = embedder.encode(query)
        cache.set_embedding(query, "model-name", emb)

        # Next time (fast!)
        cached = cache.get_embedding(query, "model-name")
        if cached is not None:
            return cached  # Skip expensive encoding!
    """

    def __init__(self, max_size: int = 500):
        super().__init__(max_size)

    def _make_key(self, query: str, model_name: str) -> str:
        """
        Create cache key from query + model

        Why hash?
        - Long queries are bad dict keys
        - MD5 gives fixed 32-char string
        - Fast lookup (O(1))

        Example:
            query = "How to configure Callas vehicle suspension?"
            model = "BAAI/bge-small-en-v1.5"
            key = "7f3a8c9d..." (MD5 hash)
        """
        # Normalize: lowercase + strip whitespace
        normalized = query.lower().strip()
        key_str = f"{model_name}:{normalized}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def get_embedding(self, query: str, model_name: str) -> Optional[Any]:
        """Get cached embedding"""
        key = self._make_key(query, model_name)
        return self.get(key)

    def set_embedding(self, query: str, model_name: str, embedding: Any):
        """Store embedding in cache"""
        key = self._make_key(query, model_name)
        self.set(key, embedding)


class RetrievalCache(LRUCache):
    """
    Cache for retrieved documents

    WHAT IT CACHES:
    - Input: query="suspension", collections=("vehicle", "studio"), k=15
    - Output: [Document1, Document2, ...] (15 docs)

    WHY IT'S EXPENSIVE:
    - Vector database similarity search (200-1000ms)
    - Done multiple times per query
    - Results stable unless docs re-indexed

    KEY INSIGHT:
    Same query + same collections = same results!

    USAGE:
        cache = RetrievalCache()

        # First time (slow - vector DB search)
        docs = vectordb.similarity_search(query, k=15)
        cache.set_retrieval(query, ("vehicle",), 15, docs)

        # Next time (fast - from cache!)
        cached = cache.get_retrieval(query, ("vehicle",), 15)
        if cached is not None:
            return cached  # Skip DB search!
    """

    def __init__(self, max_size: int = 300):
        super().__init__(max_size)

    def _make_key(self, query: str, collections: Tuple[str, ...], k: int) -> str:
        """
        Create cache key

        Key includes:
        - query: Different queries = different results
        - collections: Different collections = different results
        - k: Different k = different number of docs

        Example:
            query = "suspension"
            collections = ("vehicle", "studio")
            k = 15
            key = "a4f9c2..." (MD5 hash)
        """
        normalized = query.lower().strip()
        # Sort collections for consistent key
        sorted_colls = tuple(sorted(collections))
        key_str = f"{normalized}|{sorted_colls}|{k}"
        key_hash = hashlib.md5(key_str.encode()).hexdigest()
        print(f"[DEBUG RetrievalCache] key_str='{key_str[:80]}...' → hash={key_hash[:8]}...")
        return key_hash

    def get_retrieval(self, query: str, collections: Tuple[str, ...], k: int) -> Optional[List[Document]]:
        """Get cached retrieval results"""
        key = self._make_key(query, collections, k)
        cached = self.get(key)

        if cached is not None:
            # Return deep copy to prevent mutations
            return [self._copy_doc(d) for d in cached]
        return None

    def set_retrieval(self, query: str, collections: Tuple[str, ...], k: int, docs: List[Document]):
        """Store retrieval results"""
        key = self._make_key(query, collections, k)
        # Store deep copy
        self.set(key, [self._copy_doc(d) for d in docs])

    @staticmethod
    def _copy_doc(doc: Document) -> Document:
        """Deep copy Document to prevent external modifications"""
        return Document(
            page_content=doc.page_content,
            metadata=doc.metadata.copy()
        )


class RerankCache(LRUCache):
    """
    Cache for reranked results

    WHAT IT CACHES:
    - Input: query="suspension", docs=[d1,d2,...,d50], top_n=6
    - Output: ([best6docs], [scores])

    WHY IT'S VERY EXPENSIVE:
    - Cross-encoder scores EVERY (query, doc) pair
    - 50 docs = 50 neural network calls (500-2000ms)
    - Most expensive operation in pipeline!

    THE PROBLEM:
    Same query + same docs = same ranking every time
    But we compute it from scratch each time (wasteful!)

    THE SOLUTION:
    Create "fingerprint" of docs (first 200 chars)
    If same fingerprint = cache the reranking!

    USAGE:
        cache = RerankCache()

        # First time (very slow - 50 cross-encoder calls)
        ranked, scores = cross_encoder.predict(pairs)
        cache.set_reranked(query, docs, 6, ranked, scores)

        # Next time (instant!)
        cached = cache.get_reranked(query, docs, 6)
        if cached is not None:
            return cached  # Skip expensive reranking!
    """

    def __init__(self, max_size: int = 200):
        super().__init__(max_size)

    def _make_doc_fingerprint(self, docs: List[Document]) -> str:
        """
        Create fingerprint of document list

        Why not just use doc IDs?
        - Docs don't have stable IDs
        - Need content-based identifier

        Strategy:
        - Take first 200 chars of each doc
        - Include source file
        - Hash the combination

        If fingerprints match → same docs → can use cached ranking!
        """
        fingerprints = []
        for doc in docs:
            snippet = doc.page_content[:200]
            source = doc.metadata.get("source", "")
            fingerprints.append(f"{source}:{snippet}")

        combined = "|".join(fingerprints)
        return hashlib.md5(combined.encode()).hexdigest()

    def _make_key(self, query: str, docs: List[Document], top_n: int) -> str:
        """Create cache key from query, docs fingerprint, and top_n"""
        normalized = query.lower().strip()
        fingerprint = self._make_doc_fingerprint(docs)
        key_str = f"{normalized}|{fingerprint}|{top_n}"
        key_hash = hashlib.md5(key_str.encode()).hexdigest()
        print(f"[DEBUG RerankCache] query='{normalized[:40]}...', fingerprint={fingerprint[:8]}..., top_n={top_n} → hash={key_hash[:8]}...")
        return key_hash

    def get_reranked(self, query: str, docs: List[Document], top_n: int) -> Optional[Tuple[List[Document], List[float]]]:
        """Get cached reranking results"""
        key = self._make_key(query, docs, top_n)
        return self.get(key)

    def set_reranked(self, query: str, docs: List[Document], top_n: int,
                     reranked_docs: List[Document], scores: List[float]):
        """Store reranking results"""
        key = self._make_key(query, docs, top_n)
        self.set(key, (reranked_docs, scores))


class CacheManager:
    """
    Central manager for all caches

    WHY A MANAGER?
    - Single point to initialize all caches
    - Unified statistics
    - Easy to clear all caches at once
    - Toggle caches on/off for testing

    USAGE IN API:
        # In api_segmented.py startup():
        from rag.cache import init_cache_manager

        cache_mgr = init_cache_manager(
            embedding_cache_size=500,
            retrieval_cache_size=300,
            rerank_cache_size=200
        )

        # Later, in retrieval code:
        from rag.cache import get_cache_manager

        cache = get_cache_manager()
        cached_emb = cache.embedding_cache.get_embedding(query, model)
    """

    def __init__(
        self,
        embedding_cache_size: int = 500,
        retrieval_cache_size: int = 300,
        rerank_cache_size: int = 200,
        enable_embedding_cache: bool = True,
        enable_retrieval_cache: bool = True,
        enable_rerank_cache: bool = True
    ):
        """Initialize all caches"""
        print("\n[Cache] Initializing caching system...")

        self.embedding_cache = EmbeddingCache(embedding_cache_size) if enable_embedding_cache else None
        self.retrieval_cache = RetrievalCache(retrieval_cache_size) if enable_retrieval_cache else None
        self.rerank_cache = RerankCache(rerank_cache_size) if enable_rerank_cache else None

        enabled = []
        if self.embedding_cache:
            enabled.append(f"Embedding({embedding_cache_size})")
        if self.retrieval_cache:
            enabled.append(f"Retrieval({retrieval_cache_size})")
        if self.rerank_cache:
            enabled.append(f"Rerank({rerank_cache_size})")

        print(f"[Cache] Enabled: {', '.join(enabled)}")

    def clear_all(self):
        """Clear all caches (useful after re-indexing)"""
        if self.embedding_cache:
            self.embedding_cache.clear()
        if self.retrieval_cache:
            self.retrieval_cache.clear()
        if self.rerank_cache:
            self.rerank_cache.clear()
        print("[Cache] All caches cleared")

    def get_all_stats(self) -> dict:
        """Get statistics from all caches"""
        stats = {}
        if self.embedding_cache:
            stats["embedding"] = self.embedding_cache.get_stats()
        if self.retrieval_cache:
            stats["retrieval"] = self.retrieval_cache.get_stats()
        if self.rerank_cache:
            stats["rerank"] = self.rerank_cache.get_stats()
        return stats

    def print_stats(self):
        """Print formatted cache statistics"""
        stats = self.get_all_stats()
        print("\n" + "="*50)
        print("CACHE STATISTICS")
        print("="*50)
        for name, stat in stats.items():
            print(f"\n{name.upper()} CACHE:")
            for key, value in stat.items():
                print(f"  {key}: {value}")
        print("="*50 + "\n")


# Global cache instance (singleton pattern)
_global_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> Optional[CacheManager]:
    """
    Get the global cache manager instance

    Returns None if not initialized yet

    Usage:
        cache = get_cache_manager()
        if cache and cache.embedding_cache:
            emb = cache.embedding_cache.get_embedding(query, model)
    """
    return _global_cache_manager


def init_cache_manager(**kwargs) -> CacheManager:
    """
    Initialize global cache manager (call once at startup)

    Args:
        **kwargs: Passed to CacheManager constructor

    Returns:
        CacheManager instance

    Usage:
        # In api_segmented.py startup():
        cache_mgr = init_cache_manager(
            embedding_cache_size=500,
            retrieval_cache_size=300,
            rerank_cache_size=200
        )
    """
    global _global_cache_manager
    _global_cache_manager = CacheManager(**kwargs)
    return _global_cache_manager