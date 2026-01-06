from dataclasses import dataclass
from typing import List
from langchain_ollama import OllamaLLM
from langchain.schema import Document
from config import (
    ENABLE_CLARIFICATION,
    CLARIFY_MIN_DOCS,
    CLARIFY_MIN_SCORE,
    LLM_Model,
)

# Simple state holder for clarification dialog progress
@dataclass
class ClarifyState:
    active: bool = False              # whether clarification flow is currently active
    turns_done: int = 0               # how many clarification turns have been taken
    original_question: str = ""       # original user question before clarifying
    last_bot_question: str = ""       # last clarification question asked by the bot

class Clarifier:
    """
    Decides when to ask a follow-up question, and generates one.
    Uses an LLM (OllamaLLM) to produce a single short clarifying question.
    """
    def __init__(self, model: str = LLM_Model, temperature: float = 0.3):
        # Initialize the language model used to generate clarifying questions
        self.llm = OllamaLLM(model=model, temperature=temperature)

    def needs_clarification(self, ranked_docs: List[Document], scores: List[float]) -> bool:
        """
        Determine whether a clarification question should be asked.

        Conditions:
        - Global config can disable clarification entirely.
        - If there are no docs at all, ask to clarify.
        - If relevance scores are reasonably good, don't clarify (let the model try to answer).
        - Otherwise, check top score and average score against thresholds.
        """
        # If clarification globally disabled, never ask
        if not ENABLE_CLARIFICATION:
            return False

        # If absolutely no documents were retrieved, clarification is needed
        if len(ranked_docs) == 0:
            return True

        # If we have at least one document, be more lenient and let the model try
        if len(ranked_docs) >= CLARIFY_MIN_DOCS:
            # We have enough docs, don't clarify
            return False

        # If no relevance scores but we have documents, let the model try to answer
        if not scores:
            return False

        # Check whether the top result meets the threshold
        top_ok = scores[0] >= CLARIFY_MIN_SCORE

        # Compute average score and compare to a lower threshold (0.7x instead of 0.9x)
        avg_ok = (sum(scores) / max(1, len(scores))) >= (CLARIFY_MIN_SCORE * 0.7)

        # Ask for clarification only if both top and average are poor
        return not (top_ok or avg_ok)

    def make_clarifying_question(self, user_q: str, top_docs: List[Document]) -> str:
        """
        Generate a single short clarifying question using the LLM.

        - Include up to the first two document snippets to give context.
        - Prompt instructs the model to produce one specific sentence ending with a question mark.
        """
        # Take up to two snippets, truncating each to 400 chars to keep prompt small
        snippets = "\n\n".join([(d.page_content[:400]) for d in top_docs[:2]])

        # Construct the prompt for the LLM with clear instructions and context
        prompt = f"""
You are helping to clrify a user's question to find the right section in technical documents.
Ask ONE short,  specific question that will best clarify the user's intended meaning. Do not answer yet.

User question:
{user_q}

Possibly relevant snippets (may be incomplete):
{snippets if snippets else "[none]"}

Rules:
- ONE sentence
- Be specific (e.g., version, component, location, type, steps)
- No small talk
- End with a question mark
"""
        # Invoke the LLM and return the trimmed result
        return self.llm.invoke(prompt).strip()
