import os
from pathlib import Path
from langchain_community.document_loaders import (
                                                    PyPDFLoader, 
                                                    UnstructuredHTMLLoader, 
                                                    UnstructuredFileLoader
                                                )
try:
    from langchain_unstructured import UnstructuredLoader
except Exception:
    UnstructuredLoader = UnstructuredFileLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from utils.sanitize import sanitize_documents
from utils.html_images import extract_image_captions
from config import (
                    DOC_FOLDER, CHROMA_PATH,
                    EMBEDDING_MODEL,
                    CHUNK_SIZE, CHUNK_OVERLAP,
                    Text_format, Html_format, Pdf_format
                    )

embedding_fn = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    encode_kwargs={"normalize_embeddings": True}
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""]
)

def build_or_load_chroma():
    """
    Builds or loads a Chroma vector store for the document collection.
    This function performs the following high-level steps:
    - Checks whether a persisted Chroma database exists at CHROMA_PATH and loads it if present.
        # os.path.exists(CHROMA_PATH): returns True if the path exists on disk.
        # os.listdir(CHROMA_PATH): returns a list of entries in the directory; used to ensure it's not empty.
        # Chroma(..., embedding_function=embedding_fn): constructs a Chroma client that uses the provided embedding function
        #     to embed queries and (optionally) documents; this returns a ready-to-use vector store instance.
    - If no persisted DB is found, it walks DOC_FOLDER and loads files of supported types.
        # os.walk(DOC_FOLDER): traverses the directory tree rooted at DOC_FOLDER, yielding (root, dirs, files).
        # Path(filename).suffix.lower(): obtains the lowercase file extension for type dispatch.
    - For each supported file:
        - PDFs: uses PyPDFLoader(file_path).load() to extract pages/text.
            # PyPDFLoader: loader that extracts text from PDF files and returns a list of Document objects.
        - Plain text and other text-based files: uses UnstructuredLoader(file_path).load().
            # UnstructuredLoader: generic loader for many text-like file types via the unstructured package.
        - HTML: uses UnstructuredHTMLLoader(file_path).load() to extract text content from HTML.
            # UnstructuredHTMLLoader: HTML-specific loader that extracts body text and metadata.
            # Additionally, extract_image_captions(file_path) is called to produce documents from image captions in HTML.
                # extract_image_captions: custom utility (user-defined) that should return Documents representing image captions.
        - splitter.split_documents(docs) is used to break loaded Documents into smaller chunks suitable for embedding.
            # splitter.split_documents: a text splitter (e.g., RecursiveCharacterTextSplitter) that returns chunked Document objects.
    - Errors encountered while loading individual files are caught and logged (printed) but do not stop the overall build process.
        # try/except Exception as e: prevents a single bad file from aborting the indexing.
    - After walking all files:
        - If no documents were found, the function prints a message and returns None.
        - Otherwise, it sanitizes documents via sanitize_documents(all_docs) before indexing.
            # sanitize_documents: user-defined function that should clean, normalize, and/or deduplicate Documents prior to embedding.
        - It constructs a Chroma vector store from the sanitized documents:
            # Chroma.from_documents(all_docs, embedding_fn, persist_directory=CHROMA_PATH)
            #     - all_docs: iterable of Document objects to be embedded and persisted.
            #     - embedding_fn: callable used to convert text into embedding vectors (must be compatible with Chroma).
            #     - persist_directory: on-disk location used by Chroma to store its database files.
        - Calls vectordb.persist() to ensure the on-disk database is written, prints the number of indexed chunks,
            and returns the Chroma instance.
    Returns:
            Chroma | None
            - A Chroma vector store instance (either loaded from disk or newly created and persisted) if documents
                were found or an existing DB was present.
            - None if no documents were available to index.
    Side effects:
    - Reads files from DOC_FOLDER.
    - May write files to CHROMA_PATH when persisting a newly built index.
    - Uses several global or outer-scope names: CHROMA_PATH, DOC_FOLDER, embedding_fn, splitter,
        Pdf_format, Text_format, Html_format, PyPDFLoader, UnstructuredLoader, UnstructuredHTMLLoader,
        extract_image_captions, sanitize_documents, and Chroma. These must be defined in the module scope.
    Notes and recommendations:
    - Ensure embedding_fn is compatible with the Chroma client being used (signature and return type).
    - The text splitter should be tuned (chunk size/overlap) to balance retrieval accuracy and index size.
    - If processing many files or large documents, consider adding progress logging and more granular error handling.
    - For reproducibility, ensure CHROMA_PATH and DOC_FOLDER are absolute paths or otherwise consistently resolved.
    """

    if os.path.exists(CHROMA_PATH) and os.listdir(CHROMA_PATH):
        print("Loaded existing Chroma DB.")
        return Chroma(persist_directory=CHROMA_PATH, 
                      embedding_function=embedding_fn)

    print("Building a new one...")
    all_docs = []
    for root, _, files in os.walk(DOC_FOLDER):
        for filename in files:
            file_path = os.path.join(root, filename)
            ext = Path(filename).suffix.lower()

            try:
                if ext == Pdf_format:
                    docs = PyPDFLoader(file_path).load()
                    all_docs.extend(splitter.split_documents(docs))

                elif ext in Text_format:
                    docs = UnstructuredLoader(file_path).load()
                    all_docs.extend(splitter.split_documents(docs))

                elif ext in Html_format:
                    docs = UnstructuredHTMLLoader(file_path).load()
                    all_docs.extend(splitter.split_documents(docs))
                    img_docs = extract_image_captions(file_path)
                    all_docs.extend(img_docs)

            except Exception as e:
                print(f"[Load error] {file_path}: {e}")

    if not all_docs:
        print(" No documents found to index.")
        return None

    all_docs = sanitize_documents(all_docs)
    vectordb = Chroma.from_documents(all_docs, 
                                     embedding_fn, 
                                     persist_directory=CHROMA_PATH)
    vectordb.persist()
    print(f" Indexed {len(all_docs)} chunks into Chroma.")
    return vectordb
