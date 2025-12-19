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
        - If there are fewer docs than the configured minimum, ask to clarify.
        - If no score information is available, ask to clarify.
        - Otherwise, check top score and average score against thresholds.
        """
        # If clarification globally disabled, never ask
        if not ENABLE_CLARIFICATION:
            return False

        # If not enough documents were retrieved, clarification is needed
        if len(ranked_docs) < CLARIFY_MIN_DOCS:
            return True

        # If no relevance scores, be conservative and ask for clarification
        if not scores:
            return True

        # Check whether the top result meets the threshold
        top_ok = scores[0] >= CLARIFY_MIN_SCORE

        # Compute average score and compare to a slightly lower threshold
        avg_ok = (sum(scores) / max(1, len(scores))) >= (CLARIFY_MIN_SCORE * 0.9)

        # Ask for clarification unless both top and average are acceptable
        return not (top_ok and avg_ok)

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
