import os
from pathlib import Path
from typing import Dict, List
from langchain.schema import Document

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

from rag.build_chroma import (
    embedding_fn, splitter, 
    PyPDFLoader, UnstructuredLoader, UnstructuredHTMLLoader
)
from utils.sanitize import sanitize_documents
from utils.html_images import extract_image_captions
from config import (
    DOC_FOLDER, CHROMA_BASE_PATH, DOC_CATEGORIES, DEFAULT_COLLECTION,
    Text_format, Html_format, Pdf_format, ENABLE_SEGMENTATION
)


def determine_collection(file_path: str) -> str:
    """Determine which collection a document belongs to based on its path"""
    rel_path = os.path.relpath(file_path, DOC_FOLDER)
    
    # Check if file is in a categorized subfolder
    for category in DOC_CATEGORIES:
        if rel_path.startswith(f"help{os.sep}html{os.sep}{category}") or \
           rel_path.startswith(f"help\\html\\{category}"):
            return category.lower()
    
    # Root-level files go to default collection
    return DEFAULT_COLLECTION


def load_document(file_path: str) -> List[Document]:
    """Load and chunk a single document"""
    ext = Path(file_path).suffix.lower()
    docs = []
    
    try:
        if ext == Pdf_format:
            docs = PyPDFLoader(file_path).load()
            
        elif ext in Text_format:
            docs = UnstructuredLoader(file_path).load()
            
        elif ext in Html_format:
            docs = UnstructuredHTMLLoader(file_path).load()
            # Also extract image captions
            img_docs = extract_image_captions(file_path)
            docs.extend(img_docs)
    
    except Exception as e:
        print(f"[Load error] {file_path}: {e}")
        return []
    
    # Split into chunks
    chunks = splitter.split_documents(docs)
    
    # Add collection metadata
    collection = determine_collection(file_path)
    for chunk in chunks:
        chunk.metadata["collection"] = collection
    
    return chunks


def build_segmented_collections() -> Dict[str, Chroma]:
    """
    Build separate Chroma collections for each document category.
    Returns a dictionary mapping collection names to Chroma instances.
    """
    if not ENABLE_SEGMENTATION:
        print("Segmentation disabled. Use build_chroma.py instead.")
        return {}
    
    # Check if collections already exist
    if os.path.exists(CHROMA_BASE_PATH):
        existing = os.listdir(CHROMA_BASE_PATH)
        if existing:
            print(f"Loading {len(existing)} existing collections...")
            collections = {}
            for coll_name in existing:
                coll_path = os.path.join(CHROMA_BASE_PATH, coll_name)
                if os.path.isdir(coll_path):
                    collections[coll_name] = Chroma(
                        persist_directory=coll_path,
                        embedding_function=embedding_fn
                    )
            print(f"Loaded collections: {list(collections.keys())}")
            return collections
    
    print("Building new segmented collections...")
    
    # Group documents by collection
    docs_by_collection = {cat.lower(): [] for cat in DOC_CATEGORIES}
    docs_by_collection[DEFAULT_COLLECTION] = []
    
    # Walk through document folder
    file_count = 0
    for root, _, files in os.walk(DOC_FOLDER):
        for filename in files:
            file_path = os.path.join(root, filename)
            ext = Path(filename).suffix.lower()
            
            # Only process supported formats
            if ext not in (Text_format | Html_format | {Pdf_format}):
                continue
            
            chunks = load_document(file_path)
            if chunks:
                collection = chunks[0].metadata["collection"]
                docs_by_collection[collection].extend(chunks)
                file_count += 1
            
            if file_count % 100 == 0:
                print(f"Processed {file_count} files...")
    
    # Create Chroma collections
    collections = {}
    os.makedirs(CHROMA_BASE_PATH, exist_ok=True)
    
    for coll_name, docs in docs_by_collection.items():
        if not docs:
            print(f"[{coll_name}] No documents, skipping.")
            continue
        
        # Sanitize documents
        docs = sanitize_documents(docs)
        
        # Create collection
        coll_path = os.path.join(CHROMA_BASE_PATH, coll_name)
        vectordb = Chroma.from_documents(
            docs,
            embedding_fn,
            persist_directory=coll_path
        )
        vectordb.persist()
        collections[coll_name] = vectordb
        
        print(f"[{coll_name}] Indexed {len(docs)} chunks")
    
    print(f"\nTotal: {len(collections)} collections created")
    return collections