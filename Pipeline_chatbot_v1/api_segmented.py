"""
api_segmented.py

Modified version of api.py that uses segmented collections with smart routing.
Can be toggled between single collection (legacy) and segmented mode via config.

Usage:
    Set ENABLE_SEGMENTATION=True in config.py to use segmented collections
    Set ENABLE_SEGMENTATION=False to use the original single collection approach
"""
import os
import traceback
from typing import List, Optional, Union, Tuple, Dict
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# Import both single and segmented builders
from rag.build_chroma import build_or_load_chroma
from rag.build_segmented import build_segmented_collections
from rag.retrieval_segmented import SegmentedRetriever
from rag.retrieval import (
    make_retriever,
    return_top_doc,
    get_related_image_docs,
    build_image_notes,
    entity_pinned_candidates,
)
from rag.prompt import Chatbot_prompt
from rag.memory import HybridMemory
from rag.clarify import Clarifier, ClarifyState
from rag.focus import rewrite_followup_to_standalone
from rag.cache import init_cache_manager, get_cache_manager
from tracking.local_tracker import get_tracker
from tracking.callbacks import UsageTrackingCallback

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain.schema import Document
from sentence_transformers import SentenceTransformer

from config import (
    DOC_FOLDER,
    TOPIC_EMBEDDING_MODEL,
    USE_CUDA,
    URL,
    LLM_Model,
    LLM_TEMPERATURE,
    CLARIFY_MAX_TURNS,
    Image_format,
    ENABLE_SEGMENTATION,  # New config flag
)

# LLM setup
usage_callback = UsageTrackingCallback(model_name=LLM_Model)
text_llm = OllamaLLM(
    model=LLM_Model,
    temperature=LLM_TEMPERATURE,
    callbacks=[usage_callback]
)
clarifier = Clarifier()

# FastAPI app
app = FastAPI(title="Chatbot with Segmented Collections")
#Maybe some text here

# Global state
vectordb = None

retriever = None
segmented_retriever = None  # New: for segmented mode
collections = None  # New: dict of collections
memory = None
clarify_state = None
MAX_LEN = 200

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
app.mount("/static", StaticFiles(directory=DOC_FOLDER), name="static")
WEB_DIR = os.path.join(os.path.dirname(__file__), "web")
app.mount("/app", StaticFiles(directory=WEB_DIR), name="app")


@app.get("/")
def root():
    index_path = os.path.join(WEB_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return RedirectResponse(url="/docs")


@app.get("/favicon.ico")
def favicon():
    favicon_path = os.path.join(WEB_DIR, "scaner.png")
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path)
    return RedirectResponse(url="/docs")


def answer_or_clarify_segmented(
    user_q: str,
    segmented_retriever: SegmentedRetriever,
    memory: HybridMemory,
    cstate: ClarifyState,
) -> Tuple[Dict[str, str], List[Document], List[Document]]:
    """
    Modified version of answer_or_clarify that uses SegmentedRetriever.
    """
    # Compact history and get focus
    compact_history = memory.reset_if_new_topic(user_q)
    focus_terms = memory.get_focus()
    entity = focus_terms[0] if focus_terms else None

    print(f"[DEBUG Memory] focus_terms={focus_terms}, entity='{entity}'")

    # Rewrite query
    rewritten_q = rewrite_followup_to_standalone(user_q, focus_terms, compact_history)

    print(f"[DEBUG Query] Original: '{user_q[:60]}...', Rewritten: '{rewritten_q[:60]}...'")

    # Check if entity is already in the rewritten query
    # If the entity is part of the query itself, don't use entity-based retrieval
    # This prevents redundant searches and allows cache hits
    use_entity = entity and entity.lower() not in rewritten_q.lower()

    print(f"[DEBUG Retrieval Strategy] entity='{entity}', use_entity={use_entity}")

    # Retrieve with segmented retriever
    if use_entity:
        docs, scores = segmented_retriever.retrieve_with_entity(
            rewritten_q,
            entity,
            top_n=6,
            k_base=15,
            k_entity=10
        )
    else:
        docs, scores = segmented_retriever.retrieve(
            rewritten_q,
            top_n=6,
            k_per_collection=15,
            use_routing=True
        )

    context = "\n\n".join([d.page_content for d in docs]) if docs else ""

    # Get images across all collections
    image_docs = segmented_retriever.get_related_images(rewritten_q, k=2)
    image_notes = build_image_notes(image_docs)

    # Clarification logic
    can_clarify = not cstate.active or cstate.turns_done < CLARIFY_MAX_TURNS
    needs_clarification = clarifier.needs_clarification(docs, scores)
    if can_clarify and needs_clarification:
        cstate.active = True
        if cstate.turns_done == 0:
            cstate.original_question = rewritten_q
        c_q = clarifier.make_clarifying_question(rewritten_q, docs)
        cstate.last_bot_question = c_q
        cstate.turns_done += 1
        return ({"clarify": True, "clarify_question": c_q}, docs, image_docs)

    # Memory recall
    recalled = memory.recall(rewritten_q)
    memory_context = "\n".join([compact_history, "[Relevant Past]", *recalled]) if recalled else compact_history

    # Generate answer
    focus_terms_p = ", ".join(focus_terms) if focus_terms else "None"
    prompt = Chatbot_prompt.format(
        context=context,
        image_notes=image_notes,
        memory_context=memory_context,
        question=rewritten_q,
        focus_terms=focus_terms_p,
    )

    answer = text_llm.invoke(prompt).strip()
    memory.append(user_q, answer)

    # Reset clarify state
    cstate.active = False
    cstate.turns_done = 0
    cstate.original_question = ""
    cstate.last_bot_question = ""

    return ({"clarify": False, "answer": answer}, docs, image_docs)


def answer_or_clarify(
    user_q: str,
    retriever,
    vectordb: Chroma,
    memory: HybridMemory,
    cstate: ClarifyState,
    k_base: int = 60,
    k_entity: int = 24,
) -> Tuple[Dict[str, str], List[Document], List[Document]]:
    """
    Original single-collection answer_or_clarify (kept for backward compatibility).
    """
    compact_history = memory.reset_if_new_topic(user_q)
    focus_terms = memory.get_focus()
    entity = focus_terms[0] if focus_terms else None
    rewritten_q = rewrite_followup_to_standalone(user_q, focus_terms, compact_history)

    candidates = entity_pinned_candidates(
        vectordb, retriever, rewritten_q, entity, k_base=k_base, k_entity=k_entity
    )

    docs, scores = return_top_doc(rewritten_q, candidates, top_n=6)
    context = "\n\n".join([d.page_content for d in docs]) if docs else ""

    image_docs = get_related_image_docs(vectordb, rewritten_q, k=2)
    image_notes = build_image_notes(image_docs)

    can_clarify = not cstate.active or cstate.turns_done < CLARIFY_MAX_TURNS
    needs_clarification = clarifier.needs_clarification(docs, scores)
    if can_clarify and needs_clarification:
        cstate.active = True
        if cstate.turns_done == 0:
            cstate.original_question = rewritten_q
        c_q = clarifier.make_clarifying_question(rewritten_q, docs)
        cstate.last_bot_question = c_q
        cstate.turns_done += 1
        return ({"clarify": True, "clarify_question": c_q}, docs, image_docs)

    recalled = memory.recall(rewritten_q)
    memory_context = "\n".join([compact_history, "[Relevant Past]", *recalled]) if recalled else compact_history

    focus_terms_p = ", ".join(focus_terms) if focus_terms else "None"
    prompt = Chatbot_prompt.format(
        context=context,
        image_notes=image_notes,
        memory_context=memory_context,
        question=rewritten_q,
        focus_terms=focus_terms_p,
    )

    answer = text_llm.invoke(prompt).strip()
    memory.append(user_q, answer)

    cstate.active = False
    cstate.turns_done = 0
    cstate.original_question = ""
    cstate.last_bot_question = ""

    return ({"clarify": False, "answer": answer}, docs, image_docs)


# Utility functions (unchanged)
def like_image(path: str) -> bool:
    _, ext = os.path.splitext(path or "")
    return ext.lower() in Image_format


def _rel_path_under_doc(abs_path: str) -> Union[str, None]:
    if not abs_path:
        return None
    root = os.path.abspath(DOC_FOLDER)
    ap = os.path.abspath(abs_path)
    try:
        rel = os.path.relpath(ap, root)
    except ValueError:
        return None
    if rel.startswith(".."):
        return None
    return rel.replace("\\", "/")


def _absolute_static_url(rel: str, request: Request) -> str:
    if URL:
        return f"{URL.rstrip('/')}/static/{rel.lstrip('/')}"
    return str(request.url_for("static", path=rel))


def path_to_static_url(abs_path: str, request: Request) -> Union[str, None]:
    rel = _rel_path_under_doc(abs_path)
    if rel is None:
        return None
    return _absolute_static_url(rel, request)


def image_absolute_path(d) -> Union[str, None]:
    p = d.metadata.get("image_path", "")
    if not p:
        candidate = d.metadata.get("source", "")
        if like_image(candidate):
            p = candidate

    if not p:
        return None

    if os.path.isabs(p):
        return p

    anchor = d.metadata.get("source", "")
    if anchor:
        base_dir = os.path.dirname(anchor)
    else:
        base_dir = DOC_FOLDER
    return os.path.abspath(os.path.join(base_dir, p))


# Request/response models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    clarify_reply: Optional[bool] = False
    forced_version: Optional[str] = None


class Source(BaseModel):
    source: str
    collection: Optional[str] = None  # New: show which collection it came from


class ImageSource(BaseModel):
    path: str
    url: str
    caption: str


class ChatResponse(BaseModel):
    clarify: bool
    bot: str
    sources: List[Source] = []
    images: List[ImageSource] = []
    mode: Optional[str] = None  # New: indicates "segmented" or "single"


def pack_sources(docs):
    seen, out = set(), []
    for d in docs or []:
        src = d.metadata.get("source", "unknown")
        coll = d.metadata.get("collection", None)  # Get collection if available
        if src not in seen:
            out.append(Source(source=src, collection=coll))
            seen.add(src)
    return out


def pack_images(image_docs, request: Request):
    out = []
    for d in image_docs or []:
        abs_path = image_absolute_path(d)
        if not abs_path:
            continue
        url = path_to_static_url(abs_path, request)
        if not url:
            continue
        out.append(
            ImageSource(
                path=abs_path,
                url=url,
                caption=(d.page_content or "")[:MAX_LEN],
            )
        )
    return out

#Cache middleware and startup helper
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
                print(f"{cache_name}: {hits_delta} cache hits")
        print("="*50 + "\n")

    return response

@app.on_event("startup")
def startup():
    """
    Initialize based on ENABLE_SEGMENTATION flag.
    """
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
    if ENABLE_SEGMENTATION:
        print("\n" + "="*60)
        print("STARTING IN SEGMENTED MODE")
        print("="*60)
        collections = build_segmented_collections()
        if collections:
            segmented_retriever = SegmentedRetriever(collections)
            print(f"Loaded {len(collections)} collections: {list(collections.keys())}")
        else:
            print("WARNING: No collections loaded. Check your document folder.")
    else:
        print("\n" + "="*60)
        print("STARTING IN SINGLE COLLECTION MODE (Legacy)")
        print("="*60)
        vectordb = build_or_load_chroma()
        retriever = make_retriever(vectordb)

    # Initialize memory and clarify state (same for both modes)
    topic_embedder = SentenceTransformer(
        TOPIC_EMBEDDING_MODEL, device="cuda" if USE_CUDA else "cpu"
    )
    memory = HybridMemory(embedder=topic_embedder, llm=text_llm)
    clarify_state = ClarifyState()
    print("="*60 + "\n")

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


@app.post("/memory/clear")
def memory_clear():
    """
    Clear conversation memory

    Use this to start fresh testing without historical context
    """
    global memory

    if not memory:
        return {"error": "Memory not initialized"}

    # Clear all threads
    memory.threads = []
    memory.curr_idx = -1
    memory.save()

    return {
        "status": "success",
        "message": "Conversation memory cleared",
        "threads_cleared": len(memory.threads)
    }


@app.post("/clear-all")
def clear_all():
    """
    Clear both cache and memory for completely fresh testing
    """
    global memory

    cache_mgr = get_cache_manager()

    results = {}

    # Clear cache
    if cache_mgr:
        cache_mgr.clear_all()
        results["cache"] = "cleared"
    else:
        results["cache"] = "not initialized"

    # Clear memory
    if memory:
        thread_count = len(memory.threads)
        memory.threads = []
        memory.curr_idx = -1
        memory.save()
        results["memory"] = f"cleared {thread_count} threads"
    else:
        results["memory"] = "not initialized"

    return {
        "status": "success",
        "message": "All cache and memory cleared",
        "details": results
    }

@app.get("/metrics")
def metrics(
    hours: int = 24,
    model: Optional[str] = None,
    include_cumulative: bool = True,
    include_context_events: bool = True
):
    """
    Get LLM usage metrics and statistics.
    Query params:
    - hours: time window for stats (default: 24)
    - model: filter by specific model name (optional)
    - include_cumulative: include cumulative token counts across all time (default: True)
    - include_context_events: include context window clearing/reset events (default: True)
    """
    tracker = get_tracker()

    result = {
        "summary": tracker.get_stats(model_name=model, hours=hours),
        "model_comparison": tracker.get_model_comparison(hours=hours),
        "recent_calls": tracker.get_recent_calls(limit=10, model_name=model),
        "hourly_usage": tracker.get_hourly_usage(hours=hours)
    }

    # Add cumulative token consumption across all time
    if include_cumulative:
        result["cumulative"] = tracker.get_cumulative_tokens(model_name=model)

    # Add context window event tracking
    if include_context_events:
        result["context_events"] = {
            "stats": tracker.get_context_stats(hours=hours),
            "recent_events": tracker.get_context_events(hours=hours, limit=20)
        }

    return result


@app.get("/status")
def status():
    """Get system status and configuration"""
    return {
        "mode": "segmented" if ENABLE_SEGMENTATION else "single",
        "collections": list(collections.keys()) if collections else None,
        "single_db_loaded": vectordb is not None,
        "memory_initialized": memory is not None,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(chatrequest: ChatRequest, request: Request):
    """
    Chat endpoint that works with both single and segmented modes.
    """
    try:
        mode = "segmented" if ENABLE_SEGMENTATION else "single"

        # Check if system is ready
        if ENABLE_SEGMENTATION:
            if not segmented_retriever:
                return ChatResponse(
                    clarify=False,
                    bot="No segmented collections available. Please build collections first.",
                    sources=[],
                    images=[],
                    mode=mode
                )
        else:
            if vectordb is None:
                return ChatResponse(
                    clarify=False,
                    bot="No DB available. Please index documents.",
                    sources=[],
                    images=[],
                    mode=mode
                )

        # Handle clarify reply
        if chatrequest.clarify_reply and clarify_state and clarify_state.active:
            clarified_question = f"{clarify_state.original_question}\nAdditional details: {chatrequest.message}"
            if ENABLE_SEGMENTATION:
                result, docs, image_docs = answer_or_clarify_segmented(
                    clarified_question, segmented_retriever, memory, clarify_state
                )
            else:
                result, docs, image_docs = answer_or_clarify(
                    clarified_question, retriever, vectordb, memory, clarify_state
                )
        else:
            # Normal query
            if ENABLE_SEGMENTATION:
                result, docs, image_docs = answer_or_clarify_segmented(
                    chatrequest.message, segmented_retriever, memory, clarify_state
                )
            else:
                result, docs, image_docs = answer_or_clarify(
                    chatrequest.message, retriever, vectordb, memory, clarify_state
                )

        # Return clarification if needed
        if result.get("clarify"):
            return ChatResponse(
                clarify=True,
                bot=result["clarify_question"],
                sources=[],
                images=[],
                mode=mode
            )

        # Return answer
        return ChatResponse(
            clarify=False,
            bot=result.get("answer", "Sorry, I couldn't generate an answer."),
            sources=pack_sources(docs),
            images=pack_images(image_docs, request),
            mode=mode
        )

    except Exception as e:
        traceback.print_exc()
        return ChatResponse(
            clarify=False,
            bot=f"Server error: {e}",
            sources=[],
            images=[],
            mode="error"
        )


if __name__ == "__main__":
    uvicorn.run("api_segmented:app", host="0.0.0.0", port=8000, reload=True)
