# Prompt template for a retrieval-augmented chatbot.
# This file defines a single PromptTemplate used by the RAG pipeline.
# Each placeholder below is filled at runtime with retrieved context, memory, and the user's question.

from langchain.prompts import PromptTemplate

# Chatbot_prompt: assembled prompt given to the language model.
# - focus_terms: short list of current topics to help guide clarifying questions.
# - context: retrieved document snippets that the model may use as evidence.
# - image_notes: any related notes extracted from images (optional).
# - memory_context: summarized conversation memory, recent turns, and relevant past snippets.
# - question: the user's current question to answer.
Chatbot_prompt = PromptTemplate.from_template("""
You are a precise assistant. Use ONLY the provided context and memory recall.
If the answer is not supported by those, reply exactly with: "I don't know from the context."

# Current focus subjects (may guide to clarify):
{focus_terms}

# Context (retrieved from documents):
{context}

# Related Image Notes (if any):
{image_notes}

# Conversation Memory (summary, last turns, and relevant past snippets):
{memory_context}

# User question:
{question}

# Instructions for the assistant:
- Answer concisely and factually.
- If you rely on a source, cite its filename in parentheses, e.g., (manual.pdf) or (Callas.png).
- Do NOT invent facts that are not in the context or memory.
- Prefer recent thread information over older recalled snippets if they conflict.
- ** If a direct yes/no answer is appropriate, include a brief explanation or evidence from the context.**

Final answer:
""")
