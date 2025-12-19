from typing import List, Dict, Tuple
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma
from langchain.schema import Document

from rag.router import QueryRouter
from rag.retrieval import return_top_doc, get_related_image_docs, build_image_notes
from rag.cache import get_cache_manager  
from config import MMR_K, MMR_FETCH_K, MMR_LAMBDA


class SegmentedRetriever:
    """
    Retriever that queries multiple segmented collections based on query routing.
    Uses the QueryRouter to determine which collections are most relevant.
    """

    def __init__(self, collections: Dict[str, Chroma]):
        """
        Initialize the segmented retriever.

        Args:
            collections: Dictionary mapping collection names to Chroma instances
        """
        self.collections = collections
        self.router = QueryRouter()

    def retrieve_from_collections(
        self,
        query: str,
        collection_names: List[str],
        k_per_collection: int = 15
    ) -> List[Document]:
        """
        Retrieve documents from specified collections.

        Args:
            query: Search query
            collection_names: List of collection names to search
            k_per_collection: Number of documents to retrieve from each collection

        Returns:
            Combined list of documents from all collections
        """
        all_docs = []

        for coll_name in collection_names:
            if coll_name not in self.collections:
                print(f"[Retrieval] Collection '{coll_name}' not found, skipping")
                continue

            vectordb = self.collections[coll_name]

            try:
                # Use MMR for diversity within each collection
                docs = vectordb.similarity_search(
                    query,
                    k=k_per_collection,
                    # Can add MMR here if needed:
                    # search_type="mmr",
                    # search_kwargs={"k": k_per_collection, "fetch_k": k_per_collection * 2, "lambda_mult": 0.9}
                )

                # Add collection metadata if not present
                for doc in docs:
                    if "collection" not in doc.metadata:
                        doc.metadata["collection"] = coll_name

                all_docs.extend(docs)

            except Exception as e:
                print(f"[Retrieval Error] {coll_name}: {e}")
                continue

        return all_docs

    def deduplicate_documents(self, docs: List[Document]) -> List[Document]:
        """
        Remove duplicate documents based on source and content prefix.

        Args:
            docs: List of documents to deduplicate

        Returns:
            Deduplicated list of documents
        """
        seen = set()
        unique = []

        for doc in docs:
            source = doc.metadata.get("source", "")
            short_text = doc.page_content[:200]
            key = (source, short_text)

            if key not in seen:
                seen.add(key)
                unique.append(doc)

        return unique

    def retrieve(
        self,
        query: str,
        top_n: int = 6,
        k_per_collection: int = 15,
        use_routing: bool = True
    ) -> Tuple[List[Document], List[float]]:
        """
        Main retrieval method with routing and reranking.

        CACHING STRATEGY:
        1. Check retrieval cache for (query, collections, k)
        2. If cache hit → skip vector DB search
        3. If cache miss → search DB and cache results
        4. Check rerank cache for (query, docs, top_n)
        5. If cache hit → skip expensive reranking
        6. If cache miss → rerank and cache results

        Args:
            query: User query
            top_n: Number of top documents to return after reranking
            k_per_collection: Documents to retrieve from each collection
            use_routing: Whether to use router (True) or search all collections (False)

        Returns:
            Tuple of (top_docs, scores)
        """
        print(f"\n{'='*60}")
        print(f"[RETRIEVE CALLED] query='{query[:60]}...', top_n={top_n}, k={k_per_collection}, routing={use_routing}")
        print(f"{'='*60}")

        # Get cache manager
        cache_mgr = get_cache_manager()
        print(f"[DEBUG] Cache manager exists: {cache_mgr is not None}")
        if cache_mgr:
            print(f"[DEBUG] Retrieval cache enabled: {cache_mgr.retrieval_cache is not None}")
            print(f"[DEBUG] Rerank cache enabled: {cache_mgr.rerank_cache is not None}")

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
        #Layer 1
        # STEP 1: Try to get retrieval results from cache
        docs = None
        if cache_mgr and cache_mgr.retrieval_cache:
            collections_tuple = tuple(sorted(collection_names))  # Sort for consistent cache key
            print(f"[DEBUG] Checking retrieval cache: query='{query[:50]}...', collections={collections_tuple}, k={k_per_collection}")
            docs = cache_mgr.retrieval_cache.get_retrieval(query, collections_tuple, k_per_collection)
            if docs is not None:
                print(f"[Cache HIT] Retrieved {len(docs)} docs from cache (skipped DB search)")
            else:
                print(f"[Cache MISS] Retrieval cache - will query DB")

        # STEP 2: If cache miss, retrieve from collections
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
                print(f"[Cache] Stored retrieval results (query + {len(collection_names)} collections)")
        #Layer 2
        # STEP 3: Try to get reranked results from cache
        cached_rerank = None
        if cache_mgr and cache_mgr.rerank_cache:
            print(f"[DEBUG] Checking rerank cache: query='{query[:50]}...', num_docs={len(docs)}, top_n={top_n}")
            cached_rerank = cache_mgr.rerank_cache.get_reranked(query, docs, top_n)
            if cached_rerank is not None:
                top_docs, scores = cached_rerank
                print(f"[Cache HIT] Retrieved reranked results from cache (skipped cross-encoder)")
                return top_docs, scores
            else:
                print(f"[Cache MISS] Rerank cache - will perform cross-encoding")

        # STEP 4: If cache miss, rerank with cross-encoder
        print(f"[Reranking] Cross-encoding {len(docs)} documents...")

        top_docs, scores = return_top_doc(query, docs, top_n=top_n)

        # Store reranked results in cache
        if cache_mgr and cache_mgr.rerank_cache:
            cache_mgr.rerank_cache.set_reranked(query, docs, top_n, top_docs, scores)
            print(f"[Cache] Stored reranked results")

        return top_docs, scores

    def retrieve_with_entity(
        self,
        query: str,
        entity: str,
        top_n: int = 6,
        k_base: int = 15,
        k_entity: int = 10
    ) -> Tuple[List[Document], List[float]]:
        """
        Retrieve documents with additional entity-based search.

        Args:
            query: Main user query
            entity: Specific entity/term to search for
            top_n: Number of top documents to return
            k_base: Documents per collection for main query
            k_entity: Documents per collection for entity search

        Returns:
            Tuple of (top_docs, scores)
        """
        print(f"\n{'='*60}")
        print(f"[RETRIEVE_WITH_ENTITY CALLED] query='{query[:40]}...', entity='{entity}', top_n={top_n}, k_base={k_base}, k_entity={k_entity}")
        print(f"{'='*60}")

        cache_mgr = get_cache_manager()

        # Get base results from routing
        collection_names = self.router.route_with_fallback(
            query,
            list(self.collections.keys())
        )
        print(f"[Routing] Searching in: {collection_names}")

        # LAYER 1: Try retrieval cache for entity-based retrieval
        # Cache key includes query, entity, collections, k_base, and k_entity
        docs = None
        if cache_mgr and cache_mgr.retrieval_cache:
            collections_tuple = tuple(sorted(collection_names))
            # Create unique cache key that includes entity
            cache_key_query = f"{query}|entity:{entity or 'none'}"
            print(f"[DEBUG] Checking retrieval cache (with entity): query='{query[:40]}...', entity='{entity}', collections={collections_tuple}")
            # Use a composite k value for cache key
            cache_k = f"{k_base}+{k_entity}"

            # Try to get from cache using the entity-aware query
            from rag.cache import hashlib
            normalized = cache_key_query.lower().strip()
            sorted_colls = tuple(sorted(collection_names))
            key_str = f"{normalized}|{sorted_colls}|{cache_k}"
            cache_key = hashlib.md5(key_str.encode()).hexdigest()

            docs = cache_mgr.retrieval_cache.cache.get(cache_key)
            if docs is not None:
                # Deep copy like the original get_retrieval does
                docs = [cache_mgr.retrieval_cache._copy_doc(d) for d in docs]
                print(f"[Cache HIT] Retrieved {len(docs)} docs from cache (with entity, skipped DB search)")
                cache_mgr.retrieval_cache.hits += 1
            else:
                print(f"[Cache MISS] Retrieval cache (with entity) - will query DB")
                cache_mgr.retrieval_cache.misses += 1

        # If cache miss, retrieve from collections
        if docs is None:
            # Retrieve with main query
            docs = self.retrieve_from_collections(query, collection_names, k_base)

            # Add entity-specific results if provided
            if entity:
                entity_docs = self.retrieve_from_collections(entity, collection_names, k_entity)
                docs.extend(entity_docs)

            # Deduplicate
            docs = self.deduplicate_documents(docs)
            print(f"[Retrieval] {len(docs)} unique documents retrieved from DB (with entity)")

            # Store in cache
            if cache_mgr and cache_mgr.retrieval_cache:
                cache_mgr.retrieval_cache.cache[cache_key] = [cache_mgr.retrieval_cache._copy_doc(d) for d in docs]
                print(f"[Cache] Stored retrieval results (with entity: '{entity}')")

        # LAYER 2: Try rerank cache
        cached_rerank = None
        if cache_mgr and cache_mgr.rerank_cache:
            print(f"[DEBUG] Checking rerank cache (with entity): query='{query[:50]}...', num_docs={len(docs)}, top_n={top_n}")
            cached_rerank = cache_mgr.rerank_cache.get_reranked(query, docs, top_n)
            if cached_rerank is not None:
                top_docs, scores = cached_rerank
                print(f"[Cache HIT] Retrieved reranked results from cache (skipped cross-encoder)")
                return top_docs, scores
            else:
                print(f"[Cache MISS] Rerank cache (with entity) - will perform cross-encoding")

        # If cache miss, do expensive reranking
        print(f"[Reranking] Cross-encoding {len(docs)} documents...")
        top_docs, scores = return_top_doc(query, docs, top_n=top_n)

        # Store reranked results in cache
        if cache_mgr and cache_mgr.rerank_cache:
            cache_mgr.rerank_cache.set_reranked(query, docs, top_n, top_docs, scores)
            print(f"[Cache] Stored reranked results (with entity)")

        return top_docs, scores

    def get_related_images(self, query: str, k: int = 3) -> List[Document]:
        """
        Retrieve image-caption documents across all collections.

        Args:
            query: Search query
            k: Number of images to retrieve

        Returns:
            List of image-caption documents
        """
        all_images = []

        for coll_name, vectordb in self.collections.items():
            try:
                img_docs = get_related_image_docs(vectordb, query, k=k)
                all_images.extend(img_docs)
            except Exception as e:
                print(f"[Image Retrieval] {coll_name}: {e}")

        # Deduplicate by image path
        seen = set()
        unique_images = []
        for doc in all_images:
            img_path = doc.metadata.get("image_path") or doc.metadata.get("source")
            if img_path and img_path not in seen:
                seen.add(img_path)
                unique_images.append(doc)
                if len(unique_images) >= k:
                    break

        return unique_images[:k]
