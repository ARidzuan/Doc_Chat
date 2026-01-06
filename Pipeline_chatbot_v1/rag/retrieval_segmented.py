from typing import List, Dict, Tuple
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma
from langchain.schema import Document

from rag.router import QueryRouter
from rag.retrieval import return_top_doc, get_related_image_docs, build_image_notes
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

        # Retrieve from collections
        docs = self.retrieve_from_collections(query, collection_names, k_per_collection)

        if not docs:
            print("[Retrieval] No documents found")
            return [], []

        # Deduplicate
        docs = self.deduplicate_documents(docs)
        print(f"[Retrieval] {len(docs)} unique documents retrieved from DB")

        # Rerank with cross-encoder
        print(f"[Reranking] Cross-encoding {len(docs)} documents...")
        top_docs, scores = return_top_doc(query, docs, top_n=top_n)

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

        # Get base results from routing
        collection_names = self.router.route_with_fallback(
            query,
            list(self.collections.keys())
        )
        print(f"[Routing] Searching in: {collection_names}")

        # Retrieve with main query
        docs = self.retrieve_from_collections(query, collection_names, k_base)

        # Add entity-specific results if provided
        if entity:
            entity_docs = self.retrieve_from_collections(entity, collection_names, k_entity)
            docs.extend(entity_docs)

        # Deduplicate
        docs = self.deduplicate_documents(docs)
        print(f"[Retrieval] {len(docs)} unique documents retrieved from DB (with entity)")

        # Rerank with cross-encoder
        print(f"[Reranking] Cross-encoding {len(docs)} documents...")
        top_docs, scores = return_top_doc(query, docs, top_n=top_n)

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
