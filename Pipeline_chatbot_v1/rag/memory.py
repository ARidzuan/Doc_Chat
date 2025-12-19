import os
import json
import time
import re
from dataclasses import dataclass, asdict
from tracking.local_tracker import get_tracker
from typing import List, Optional, Union
from sentence_transformers import SentenceTransformer
import numpy as np

from langchain_ollama import OllamaLLM

from config import (
    MEMORY_PATH, MEMORY_WINDOW_TURNS,
    MEMORY_SIM_THRESHOLD, MEMORY_MIN_TOPIC_LEN,
    MEMORY_RECALL_TOP_K
)

# Simple container for a single turn in a conversation
@dataclass
class Turn:
    ts: float                   # timestamp for the turn
    user: str                   # user text
    bot: str                    # bot text
    emb: Optional[List[float]] = None  # optional embedding for the user text

# Represents a threaded conversation (a "topic") with metadata
@dataclass
class Thread:
    id: str
    title: str
    turns: List[Turn]
    summary: str = ""           # rolling compressed summary of the thread
    focus: List[str] = None     # extracted focus tokens (e.g., model names, keywords)

class HybridMemory:
    """
    Hybrid memory combining:
    - topic detection (cosine similarity on user turns),
    - a window buffer for recent turns,
    - rolling summary compression via LLM,
    - vector recall across all threads,
    - persistence to disk (JSON).
    """
    def __init__(self, embedder: SentenceTransformer,
                 llm: Optional[OllamaLLM] = None):
        self.embedder = embedder
        self.llm = llm
        self.threads: List[Thread] = []   # list of Thread objects
        self.curr_idx: int = -1           # index of the current thread
        os.makedirs(os.path.dirname(MEMORY_PATH), exist_ok=True)
        self.load()                       # load persisted memory if present

    ############### Persistence ###################
    def load(self):
        """Load memory from disk (MEMORY_PATH)."""
        if not os.path.exists(MEMORY_PATH):
            return
        try:
            with open(MEMORY_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self.threads = []
            # Reconstruct Thread and Turn objects from JSON data
            for t in raw.get("threads", []):
                turns_data = t.get("turns", []) or []
                for turn in turns_data:
                    turns = [Turn(**turn)]
                self.threads.append(
                    Thread(id=t["id"],
                           title=t["title"],
                           turns=turns,
                           summary=t.get("summary", "")))
            self.curr_idx = raw.get("curr_idx", -1)
        except Exception as e:
            # Fail gracefully on load errors
            print(f"[Memory load error] {e}")

    def save(self):
        """Persist memory (threads + current index) to disk as JSON."""
        try:
            data = {
                "threads": [asdict(t) for t in self.threads],
                "curr_idx": self.curr_idx
            }
            with open(MEMORY_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[Memory save error] {e}")

    # ################# Threading ##################
    def new_thread(self, first_user_text: str):
        """Create a new thread and set it as current."""
        tid = f"t-{int(time.time())}"
        title = self._propose_title(first_user_text)
        self.threads.append(
            Thread(id=tid,
                   title=title,
                   turns=[],
                   summary="",
                   focus=[]))
        self.curr_idx = len(self.threads) - 1
        tracker = get_tracker()
        tracker.log_context_event(
            event_type="new_thread",
            thread_id=tid,
            turns_cleared=0,
            tokens_cleared=0,
            reason="User started new conversation",
            metadata={"first_message_preview": first_user_text[:100]}
        )

    ###############################################
    # Regular expressions used to extract focus tokens (e.g., model names)
    MODEL_CODE = re.compile(r"\b([A-Z][A-Za-z0-9._\-]{1,}|[A-Za-z]+[0-9][A-Za-z0-9._\-]*)\b", re.I)
    PATTERN_MODEL = re.compile(r"\b(model|type|version|ref(?:erence)?)\s*[:\-]?\s*([A-Za-z0-9._\-]+)\b", re.I)

    def extract_focus(self, text: str) -> list:
        """
        Extract short focus tokens from text: explicit "model X" patterns and
        tokens that look like model codes/names. Return up to 8 unique tokens.
        """
        if not text:
            return []
        hits = []
        # explicit “model X” / “type X”
        for m in self.PATTERN_MODEL.finditer(text):
            hits.append(m.group(2))
        # generic model/code tokens
        for m in self.MODEL_CODE.finditer(text):
            token = m.group(1)
            # Filter out common question words and articles
            if token.lower() in {"model", "type", "version", "this", "that", "what", "where", "when", "how", "why", "who", "which", "the", "a", "an", "is", "are", "do", "does", "can", "could", "would", "should"}:
                continue
            hits.append(token)
        # deduplicate while preserving order (case-insensitive)
        seen, out = set(), []
        for h in hits:
            k = h.strip()
            if not k or k.lower() in seen:
                continue
            seen.add(k.lower())
            out.append(k)
        return out[:8]

    ####################################
    def update_focus(self, user_text: str, bot_text: str):
        """
        Add extracted focus tokens from the latest user/bot texts into the current
        thread's focus list (up to 8 items) and persist.
        """
        th = self.current_thread()
        new_focus = []
        if not th:
            return
        candidates = self.extract_focus(user_text) + self.extract_focus(bot_text)
        if not candidates:
            return
        if th.focus is None:
            th.focus = []
        # Prioritize newest candidates, then keep previous focus tokens not present
        for c in candidates:
            new_focus.append(c)
        focus = th.focus or []
        for f in focus:
            if f not in candidates:
                new_focus.append(f)
        th.focus = new_focus[:8]
        self.save()

    def get_focus(self) -> list:
        """Return focus tokens for current thread, or empty list."""
        th = self.current_thread()
        if th:
            return (th.focus or [])
        return []

    def _propose_title(self, text: str) -> str:
        """Create a short title for a thread from given text."""
        txt = text.strip().replace("\n", " ")
        if not txt:
            return "New thread"
        if len(txt) > 50:
            return txt[:50] + "…"
        return txt

    def startnew_thread(self):
        """Convenience to start an empty thread."""
        self.new_thread("")

    def current_thread(self) -> Union[Thread, None]:
        """Return the currently active thread or None."""
        if 0 <= self.curr_idx < len(self.threads):
            return self.threads[self.curr_idx]
        return None

    ################Topic detection######################
    def is_new_topic(self, user_text: str) -> bool:
        """
        Decide whether the incoming user_text starts a new topic:
        - Compare embedding of the new user text to the most recent previous user turn.
        - If cosine similarity < MEMORY_SIM_THRESHOLD -> treat as new topic.
        - Special-case: if last bot answer was a simple "yes"/"no", avoid starting a new topic.
        """
        th = self.current_thread()
        if not th or not th.turns:
            return False
        # Avoid new topic if last answer was just "Yes" or "No"
        last_answer = th.turns[-1].bot
        if last_answer and last_answer.strip().lower().rstrip('.') in {"yes", "no"}:
            return False
        # Find the last user turn (skip empty ones)
        last_user = None
        for turn in reversed(th.turns):
            if turn.user:
                last_user = turn
                break
        if not last_user:
            return False
        # Compute embeddings (use cached embedding if present)
        question_emb = self.embedder.encode(user_text,
                                            convert_to_numpy=True,
                                            normalize_embeddings=True)
        if last_user.emb is None:
            last_emb = self.embedder.encode(
                last_user.user,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
        else:
            last_emb = np.array(last_user.emb)
        sim = float(np.dot(question_emb, last_emb))  # cosine since embeddings are normalized
        return sim < MEMORY_SIM_THRESHOLD

    def reset_if_new_topic(self, user_text: str) -> str:
        """
        Ensure a thread exists for the incoming user_text and create a new one if
        a new topic is detected. Optionally compress the current thread with LLM
        when it grows, then return a recent-window context string (summary + window).
        """
        th = self.current_thread()

        if th is None:
            self.new_thread(user_text)

        if self.is_new_topic(user_text):
            # Start a new thread if the new text represents a different topic
            self.new_thread(user_text)

        if not th:
            return ""

        # Periodically compress the thread into a rolling summary using the LLM
        if self.llm and len(th.turns) >= MEMORY_MIN_TOPIC_LEN and (len(th.turns) % 6 == 0):
            th.summary = self.summarize_thread(th.summary, th.turns)
            self.save()

        # Build a context window: optional summary + last N turns
        window = th.turns[-MEMORY_WINDOW_TURNS:]
        lines = []
        if th.summary:
            lines.append(f"[Summary] {th.summary}")
        for t in window:
            if t.user:
                lines.append(f"User: {t.user}")
            if t.bot:
                lines.append(f"Bot: {t.bot}")
        return "\n".join(lines)

    def append(self, user_text: str, bot_text: str):
        """
        Append a new turn to the current thread:
        - Ensure a thread exists.
        - Compute and store the user embedding.
        - Update focus tokens and persist memory.
        """
        th = self.current_thread()
        if th is None:
            self.new_thread(user_text)
            th = self.current_thread()
        emb = self.embedder.encode(user_text,
                                   convert_to_numpy=True,
                                   normalize_embeddings=True).tolist()
        th.turns.append(
            Turn(ts=time.time(),
                 user=user_text,
                 bot=bot_text,
                 emb=emb)
        )
        self.update_focus(user_text, bot_text)
        self.save()

    def recall(self, question: str, top_k: int = MEMORY_RECALL_TOP_K) -> List[str]:
        """
        Retrieve up to top_k most relevant past snippets (across all threads)
        to the given question using cosine similarity on stored embeddings.
        """
        if not self.threads:
            return []
        q = self.embedder.encode(question,
                                 convert_to_numpy=True,
                                 normalize_embeddings=True)
        candidates = []
        for t in self.threads:
            for turn in t.turns:
                if not turn.emb:
                    continue
                # Skip short/affirmative bot replies that are not informative
                if turn.bot and turn.bot.strip().lower().rstrip('.') in {"yes", "no"}:
                    continue
                sim = float(np.dot(q, np.array(turn.emb)))
                candidates.append((sim, t.title, turn))
        if not candidates:
            return []
        # Sort by descending similarity
        candidates.sort(key=lambda x: x[0], reverse=True)
        out = []
        for sim, title, tr in candidates[:top_k]:
            out.append(f"[{title}] User: {tr.user} | Bot: {tr.bot}")
        return out

    def summarize_thread(self, prev_summary: str, turns: List[Turn]) -> str:
        """
        Summarize a short conversation thread, optionally incorporating a previous summary.

        This method builds a compression prompt from an optional previous summary and up to
        the last 12 turns of the provided conversation, then asks the instance LLM to
        produce a concise, factual summary that preserves key intents, decisions, and
        domain-specific terms while avoiding fluff.

        Args:
            prev_summary (str): An existing summary of earlier context. If provided, it is
                included at the top of the prompt to allow the LLM to produce an updated
                or consolidated summary.
            turns (List[Turn]): A sequence of Turn objects (expected to have `.user` and
                `.bot` attributes). Only the final 12 turns from this list are included
                in the prompt to bound prompt size.

        Returns:
            str: The new summary produced by the LLM (leading/trailing whitespace removed).
                If no LLM is configured on the instance or if the LLM invocation fails,
                the method returns the original prev_summary unchanged.

        Behavior/Notes:
            - The prompt format prefixes the previous summary (if any) as "Previous summary:"
              and formats each turn as "U: <user>" and "A: <bot>".
            - The method intentionally limits the number of included turns to the most
              recent 12 to reduce prompt length.
            - All exceptions raised during the LLM invocation are caught; on error the
              function falls back to returning prev_summary.
        """
        if not self.llm:
            return prev_summary
        content = []
        if prev_summary:
            content.append(f"Previous summary: {prev_summary}")
        # Include only the most recent up to 12 turns to limit prompt length
        for t in turns[-12:]:
            content.append(f"U: {t.user}")
            content.append(f"A: {t.bot}")
        prompt = (
            "Compress the following conversation into a short, factual summary that preserves key intent, "
            "decisions, and domain terms. Avoid fluff.\n\n" + "\n".join(content)
        )
        try:
            return self.llm.invoke(prompt).strip()
        except Exception:
            # On any LLM error, keep prior summary unchanged
            return prev_summary
