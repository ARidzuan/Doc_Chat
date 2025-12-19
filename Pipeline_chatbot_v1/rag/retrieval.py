from typing import List, Tuple, Union, Optional
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_ollama import OllamaLLM
from sentence_transformers import CrossEncoder
from langchain.schema import Document
from config import (
    MMR_K, MMR_FETCH_K, MMR_LAMBDA,
    RERANKER_MODEL, USE_CUDA
)

# Select device for CrossEncoder (CUDA if available)
_device = "cuda" if USE_CUDA else "cpu"

# Cross-encoder used to re-rank retrieved docs given a (query, doc) pair
cross_encoder = CrossEncoder(RERANKER_MODEL, device=_device)


def make_retriever(vectordb: Chroma):
    """
    Create a retriever that uses MMR search on the vector DB and wraps it
    with a MultiQueryRetriever powered by an LLM for multi-query expansion.
    """
    # Base retriever using MMR (maximal marginal relevance) for diversity
    base = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": MMR_K,
            "fetch_k": MMR_FETCH_K,
            "lambda_mult": MMR_LAMBDA
        }
    )

    # Wrap base retriever with MultiQueryRetriever using Ollama LLM
    # OllamaLLM can be swapped out or configured as needed
    multiquery = MultiQueryRetriever.from_llm(
        retriever=base,
        llm=OllamaLLM(model="mistral", temperature=0.3)
    )
    return multiquery


def return_top_doc(query: str, docs: List[Document], top_n=6) -> Tuple[List[Document], List[float]]:
    """
    Re-rank a list of documents by relevance to `query` using the cross-encoder.
    Returns the top_n documents and their scores.
    """
    if not docs:
        return [], []

    #using rank method of cross encoder to get top n documents
    results = cross_encoder.rank(query, [d.page_content for d in docs], top_k=top_n)
    top_docs = [docs[hit['corpus_id']] for hit in results]
    top_scores = [hit['score'] for hit in results]
    return top_docs, top_scores


def get_related_image_docs(vectordb: Chroma, question: str, k=3) -> List[Document]:
    """
    Retrieve image-caption documents related to the question.
    Uses a metadata filter to only fetch docs where modality == "image_caption".
    Deduplicates results by image path (metadata["image_path"] or metadata["source"]).
    """
    try:
        # Use vector DB similarity search with filter for image captions
        image_docs = vectordb.similarity_search(
            question, k=max(k, 1),
            filter={"modality": "image_caption"}
        )

        # Deduplicate by image path or source to avoid returning multiple captions for same image
        seen, unique = set(), []
        for d in image_docs:
            p = d.metadata.get("image_path") or d.metadata.get("source")
            if p and p not in seen:
                seen.add(p)
                unique.append(d)
            if len(unique) >= k:
                break
        return unique
    except Exception as e:
        # Log and return empty list on errors (e.g., connectivity or query issues)
        print(f"[Images Retrieval] {e}")
        return []


def build_image_notes(image_docs: List[Document]) -> Union[str, None]:
    """
    Build a short notes string from image-caption documents.
    Returns "None" if no docs were passed, None if no captions found, or the joined captions.
    """
    if not image_docs:
        return "None"

    lines = []
    for d in image_docs:
        # Use source metadata as a label (fallback to "image")
        src = d.metadata.get("source", "image")
        cap = (d.page_content or "").strip()
        if cap:
            lines.append(f"[{src}] {cap}")

    # If we collected caption lines, join them; otherwise return None to indicate empty content
    if lines:
        return "\n".join(lines)
    else:
        return None


def entity_pinned_candidates(
    vectordb: Chroma,
    retriever,
    question: str,
    entity: Optional[str],
    k_base: int = 40,
    k_entity: int = 20
) -> List[Document]:
    """
    Gather candidate documents combining:
      - documents returned by the main retriever for the question (e.g., MMR + multi-query)
      - additional similarity search results for the `entity` (if provided)
    Merge results and deduplicate by (source, truncated page_content) to avoid near-duplicates.
    Limits results to k_base documents.
    """
    # Get base candidates from the retriever
    base_docs = retriever.invoke(question)

    docs = list(base_docs)

    # If an entity is provided, include additional results for that entity
    if entity:
        try:
            extra = vectordb.similarity_search(entity, k=k_entity)
            docs.extend(extra)
        except Exception:
            # Ignore failures on the auxiliary entity search
            pass

    # Deduplicate by (source, short_text) key and cap to k_base
    seen, uniq = set(), []
    for d in docs:
        source = d.metadata.get("source", "")
        short_text = d.page_content[:180]  # use prefix to avoid heavy comparisons
        key = (source, short_text)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(d)
        if len(uniq) >= k_base:
            break
    return uniq
