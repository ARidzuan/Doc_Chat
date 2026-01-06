"""
Chainlit App for RAG-based Chatbot
Simplified version using api_segmented logic directly
No thread persistence or conversation history - each session is independent
"""
import os
# Fix OpenMP duplicate library issue (must be set FIRST before any library imports)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import traceback
from typing import List

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import chainlit as cl
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

# Import RAG components directly
from rag.build_segmented import build_segmented_collections
from rag.retrieval_segmented import SegmentedRetriever
from rag.prompt import Chatbot_prompt
from rag.memory import HybridMemory
from rag.clarify import Clarifier, ClarifyState
from rag.focus import rewrite_followup_to_standalone
from rag.retrieval import build_image_notes
from tracking.callbacks import UsageTrackingCallback

from config import (
    LLM_Model,
    LLM_TEMPERATURE,
    ENABLE_SEGMENTATION,
    TOPIC_EMBEDDING_MODEL,
    USE_CUDA,
    CLARIFY_MAX_TURNS,
    DOC_FOLDER,
)

# Global RAG components (initialized once at startup)
segmented_retriever = None
collections = None
text_llm = None
clarifier = None
topic_embedder = None


@cl.on_chat_start
async def start():
    """Initialize chat session (no persistence - fresh session each time)"""

    # Check if system is ready
    if not segmented_retriever:
        await cl.Message(
            content="⚠️ System not ready. Segmented collections not loaded."
        ).send()
        return

    # Create fresh memory and clarify state for this session
    memory = HybridMemory(embedder=topic_embedder, llm=text_llm)
    clarify_state = ClarifyState()

    # Store in session (no database persistence)
    cl.user_session.set("memory", memory)
    cl.user_session.set("clarify_state", clarify_state)

    # Welcome message
    mode = "segmented" if ENABLE_SEGMENTATION else "single collection"
    welcome_msg = f"""Welcome to the RAG Chatbot!

**Mode:** {mode}
**Model:** {LLM_Model}
**Collections:** {list(collections.keys()) if collections else "Single collection"}

Ask me anything about your documents!
"""
    await cl.Message(content=welcome_msg).send()


@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages"""
    user_query = message.content

    # Get session state (no database - only in-memory)
    memory = cl.user_session.get("memory")
    clarify_state = cl.user_session.get("clarify_state")

    # Check if this is a clarify reply
    is_clarify_reply = cl.user_session.get("awaiting_clarification", False)

    try:
        # Show processing indicator
        msg = cl.Message(content="")
        await msg.send()

        # Process query using api_segmented logic
        result, docs, image_docs = process_query(
            user_query,
            memory,
            clarify_state,
            is_clarify_reply
        )

        # Handle clarification needed
        if result.get("clarify"):
            msg.content = result["clarify_question"]
            await msg.update()
            cl.user_session.set("awaiting_clarification", True)
            return

        # Reset clarification flag
        cl.user_session.set("awaiting_clarification", False)

        # Build response
        answer = result.get("answer", "Sorry, I couldn't generate an answer.")

        # Update message with answer
        msg.content = answer
        await msg.update()

        # Add source elements
        elements = build_elements(docs, image_docs)
        if elements:
            await cl.Message(content="", elements=elements).send()

    except Exception as e:
        traceback.print_exc()
        error_msg = f"An error occurred: {str(e)}"
        await cl.Message(content=error_msg).send()


def process_query(user_query, memory, clarify_state, is_clarify_reply):
    """
    Process user query using api_segmented logic.
    This is the same logic as answer_or_clarify_segmented from api_segmented.py
    """
    # Handle clarify reply
    if is_clarify_reply and clarify_state and clarify_state.active:
        clarified_question = f"{clarify_state.original_question}\nAdditional details: {user_query}"
        user_query = clarified_question

    # Compact history and get focus
    compact_history = memory.reset_if_new_topic(user_query)
    focus_terms = memory.get_focus()
    entity = focus_terms[0] if focus_terms else None

    print(f"[Memory] focus_terms={focus_terms}, entity='{entity}'")

    # Rewrite query
    rewritten_q = rewrite_followup_to_standalone(user_query, focus_terms, compact_history)
    print(f"[Query] Original: '{user_query[:60]}...', Rewritten: '{rewritten_q[:60]}...'")

    # Check if entity is already in the rewritten query
    use_entity = entity and entity.lower() not in rewritten_q.lower()
    print(f"[Retrieval] entity='{entity}', use_entity={use_entity}")

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

    # Get images
    image_docs = segmented_retriever.get_related_images(rewritten_q, k=2)
    image_notes = build_image_notes(image_docs)

    # Clarification logic
    can_clarify = not clarify_state.active or clarify_state.turns_done < CLARIFY_MAX_TURNS
    needs_clarification = clarifier.needs_clarification(docs, scores)
    if can_clarify and needs_clarification:
        clarify_state.active = True
        if clarify_state.turns_done == 0:
            clarify_state.original_question = rewritten_q
        c_q = clarifier.make_clarifying_question(rewritten_q, docs)
        clarify_state.last_bot_question = c_q
        clarify_state.turns_done += 1
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
    memory.append(user_query, answer)

    # Reset clarify state
    clarify_state.active = False
    clarify_state.turns_done = 0
    clarify_state.original_question = ""
    clarify_state.last_bot_question = ""

    return ({"clarify": False, "answer": answer}, docs, image_docs)


def build_elements(docs: List[Document], image_docs: List[Document]):
    """Build UI elements for sources and images"""
    elements = []

    # Add document sources
    if docs:
        sources_text = "\n\n**Sources:**\n"
        seen_sources = set()
        for doc in docs:
            source = doc.metadata.get("source", "unknown")
            collection = doc.metadata.get("collection", None)

            if source not in seen_sources:
                if collection:
                    sources_text += f"- {source} (from {collection})\n"
                else:
                    sources_text += f"- {source}\n"
                seen_sources.add(source)

        elements.append(
            cl.Text(name="Sources", content=sources_text, display="side")
        )

    # Add image elements
    if image_docs:
        for idx, img_doc in enumerate(image_docs):
            abs_path = image_absolute_path(img_doc)
            if abs_path and os.path.exists(abs_path):
                elements.append(
                    cl.Image(
                        name=f"Image {idx+1}",
                        path=abs_path,
                        display="inline",
                        size="medium"
                    )
                )

    return elements


def like_image(path: str) -> bool:
    """Check if file is an image"""
    from config import Image_format
    _, ext = os.path.splitext(path or "")
    return ext.lower() in Image_format


def image_absolute_path(d) -> str:
    """Get absolute path for image from document metadata"""
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


@cl.on_chat_end
async def end():
    """Handle chat session end (no persistence needed)"""
    print("Chat session ended")


# Initialize system at startup (before any requests)
def initialize_system():
    """Initialize RAG system components"""
    global segmented_retriever, collections, text_llm, clarifier, topic_embedder

    if ENABLE_SEGMENTATION:
        print("\n" + "="*60)
        print("INITIALIZING CHAINLIT APP IN SEGMENTED MODE")
        print("="*60)

        # Build collections
        collections = build_segmented_collections()
        if collections:
            segmented_retriever = SegmentedRetriever(collections)
            print(f"Loaded {len(collections)} collections: {list(collections.keys())}")
        else:
            print("WARNING: No collections loaded. Check your document folder.")

        # Initialize LLM
        usage_callback = UsageTrackingCallback(model_name=LLM_Model)
        text_llm = OllamaLLM(
            model=LLM_Model,
            temperature=LLM_TEMPERATURE,
            callbacks=[usage_callback]
        )

        # Initialize clarifier
        clarifier = Clarifier()

        # Initialize topic embedder
        topic_embedder = SentenceTransformer(
            TOPIC_EMBEDDING_MODEL,
            device="cuda" if USE_CUDA else "cpu"
        )

        print("="*60 + "\n")
        print("System initialized successfully!")
    else:
        print("\n" + "="*60)
        print("ERROR: This app requires ENABLE_SEGMENTATION=True in config.py")
        print("="*60 + "\n")


# Initialize when module is loaded
initialize_system()


if __name__ == "__main__":
    # This is used for chainlit run command
    pass
