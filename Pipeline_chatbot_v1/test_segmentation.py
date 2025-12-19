"""
Test script for document segmentation and multi-collection retrieval.

Usage:
    python test_segmentation.py build    # Build segmented collections
    python test_segmentation.py test     # Test retrieval with sample queries
"""

import sys
from rag.build_segmented import build_segmented_collections
from rag.retrieval_segmented import SegmentedRetriever


def build_collections():
    """Build or load segmented collections"""
    print("=" * 60)
    print("Building/Loading Segmented Collections")
    print("=" * 60)

    collections = build_segmented_collections()

    if collections:
        print("\n" + "=" * 60)
        print("Collections Summary:")
        print("=" * 60)
        for name, db in collections.items():
            # Get collection count
            try:
                count = db._collection.count()
                print(f"  [{name}]: {count} documents")
            except:
                print(f"  [{name}]: loaded successfully")
        print("=" * 60)
    else:
        print("\nNo collections created. Check your configuration.")

    return collections


def test_retrieval(collections):
    """Test retrieval with sample queries"""
    if not collections:
        print("No collections available for testing.")
        return

    print("\n" + "=" * 60)
    print("Testing Multi-Collection Retrieval")
    print("=" * 60)

    retriever = SegmentedRetriever(collections)

    # Sample test queries
    test_queries = [
        "How do I configure vehicle dynamics?",
        "What API functions are available for simulation control?",
        "How to create terrain in SCANeR Studio?",
        "Unreal Engine integration setup",
        "Export simulation results for analysis"
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")

        # Retrieve documents
        top_docs, scores = retriever.retrieve(
            query,
            top_n=3,
            k_per_collection=10,
            use_routing=True
        )

        if top_docs:
            print(f"\nTop {len(top_docs)} Results:")
            for i, (doc, score) in enumerate(zip(top_docs, scores), 1):
                source = doc.metadata.get("source", "unknown")
                collection = doc.metadata.get("collection", "unknown")
                content_preview = doc.page_content[:150].replace("\n", " ")

                print(f"\n  {i}. [Score: {score:.3f}] [{collection}]")
                print(f"     Source: {source}")
                print(f"     Preview: {content_preview}...")
        else:
            print("\nNo results found.")


def interactive_search(collections):
    """Interactive search mode"""
    if not collections:
        print("No collections available.")
        return

    print("\n" + "=" * 60)
    print("Interactive Search Mode")
    print("=" * 60)
    print("Enter your queries (type 'quit' to exit)")

    retriever = SegmentedRetriever(collections)

    while True:
        print("\n" + "-" * 60)
        query = input("\nYour query: ").strip()

        if query.lower() in ['quit', 'exit', 'q']:
            print("Exiting interactive mode.")
            break

        if not query:
            continue

        print(f"\nSearching for: {query}")
        print("-" * 60)

        top_docs, scores = retriever.retrieve(
            query,
            top_n=5,
            k_per_collection=15,
            use_routing=True
        )

        if top_docs:
            print(f"\nFound {len(top_docs)} results:")
            for i, (doc, score) in enumerate(zip(top_docs, scores), 1):
                source = doc.metadata.get("source", "unknown")
                collection = doc.metadata.get("collection", "unknown")
                content_preview = doc.page_content[:200].replace("\n", " ")

                print(f"\n{i}. [{collection}] Score: {score:.3f}")
                print(f"   {source}")
                print(f"   {content_preview}...")
        else:
            print("\nNo results found.")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "build":
        collections = build_collections()
    elif command == "test":
        collections = build_collections()
        test_retrieval(collections)
    elif command == "interactive":
        collections = build_collections()
        interactive_search(collections)
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
