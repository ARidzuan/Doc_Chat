from typing import List
import re

# Common generic terms that often need disambiguation
GENERAL_TERMS = ["model", "component", "system"]

# Matches vague follow-up pronouns or short referents like "this", "that", "it", or "the <word>"
PRONOUN_FOLLOWUP = re.compile(r"\b(this|that|it|the \s+\w+)\b", re.I)

# Matches explicit model-like specifications, e.g. "model: GPT-4", "type-abc123", "ref: xyz"
EXPLICIT_MODEL = re.compile(
    r"\b(model|type|version|ref(?:erence)?)\s*[:\-]?\s*([A-Za-z0-9._\-]+)\b",
    re.I,
)


def rewrite_followup_to_standalone(
    question: str, focus: List[str], history_snippet: str = ""
) -> str:
    """
    Rewrite a possibly ambiguous follow-up question into a standalone form by injecting
    the focused entity when needed.

    Parameters:
    - question: the raw user question (may be a follow-up).
    - focus: list of entity identifiers (most-recent first) to use for disambiguation.
    - history_snippet: optional context from conversation history (not used currently).

    Behavior (priority order):
    1) If the question contains an explicit entity (EXPLICIT_MODEL) return it unchanged
       but tag the found entity.
    2) If the question contains a vague pronoun and we have a focus entity, inject it.
    3) If the question mentions a general term (e.g., "model") and we have a focus,
       make the question explicitly about that focus.
    4) Otherwise return the original question unchanged.
    """
    q = question.strip()

    # 1) If explicit entity is in the question → keep it and annotate which entity was found
    m = EXPLICIT_MODEL.search(q)
    if m:
        ent = m.group(1)
        return f"{q} (entity: {ent})"

    # 2) If vague pronoun and we have focus → inject last focus entity
    #    (e.g. "Is it updated?" -> "Is it updated? (referring to <focus>)")
    if PRONOUN_FOLLOWUP.search(q) and focus:
        ent = focus[0]
        return f"{q} (referring to {ent})"

    # 3) If general term like "model", "component", "system" but no explicit entity → use focus
    #    (e.g. "How does the model perform?" -> "How does the model perform? — specifically about <focus>")
    if any(word in q.lower() for word in GENERAL_TERMS) and focus:
        ent = focus[0]
        return f"{q} — specifically about {ent}"

    # 4) Otherwise return the original question unchanged
    return q
