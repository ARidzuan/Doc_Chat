"""
Example integration of context event tracking with your HybridMemory system.
This shows how to log context window events when memory is cleared, reset, or modified.
"""
from Pipeline_chatbot_v1.tracking.local_tracker import get_tracker


# Example 1: Track when a new thread/conversation is started
def on_new_thread_created(thread_id: str, first_message: str):
    """Call this when creating a new conversation thread."""
    tracker = get_tracker()
    tracker.log_context_event(
        event_type="new_thread",
        thread_id=thread_id,
        turns_cleared=0,
        tokens_cleared=0,
        reason="User started new conversation",
        metadata={"first_message_preview": first_message[:100]}
    )


# Example 2: Track when context window is trimmed (old messages removed)
def on_context_window_trimmed(thread_id: str, turns_removed: int, estimated_tokens: int):
    """Call this when old turns are removed from the context window."""
    tracker = get_tracker()
    tracker.log_context_event(
        event_type="window_trim",
        thread_id=thread_id,
        turns_cleared=turns_removed,
        tokens_cleared=estimated_tokens,
        reason="Context window size limit reached",
        metadata={"window_policy": "rolling", "max_turns": 6}
    )


# Example 3: Track when memory/thread is completely cleared/reset
def on_thread_cleared(thread_id: str, total_turns: int, total_tokens: int):
    """Call this when an entire thread is cleared or reset."""
    tracker = get_tracker()
    tracker.log_context_event(
        event_type="thread_clear",
        thread_id=thread_id,
        turns_cleared=total_turns,
        tokens_cleared=total_tokens,
        reason="User requested conversation reset"
    )


# Example 4: Track when switching between threads
def on_thread_switch(from_thread_id: str, to_thread_id: str, context_size: int):
    """Call this when switching between conversation threads."""
    tracker = get_tracker()
    tracker.log_context_event(
        event_type="thread_switch",
        thread_id=to_thread_id,
        turns_cleared=0,
        tokens_cleared=0,
        reason=f"Switched from {from_thread_id}",
        metadata={
            "from_thread": from_thread_id,
            "to_thread": to_thread_id,
            "new_context_size": context_size
        }
    )


# Example 5: Track when summary compression happens
def on_summary_compression(thread_id: str, turns_compressed: int, tokens_before: int, tokens_after: int):
    """Call this when turns are compressed into a summary."""
    tracker = get_tracker()
    tracker.log_context_event(
        event_type="summary_compression",
        thread_id=thread_id,
        turns_cleared=turns_compressed,
        tokens_cleared=tokens_before - tokens_after,
        reason="Summarized old turns to reduce context size",
        metadata={
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
            "compression_ratio": round(tokens_after / tokens_before, 2) if tokens_before > 0 else 0
        }
    )


# Example 6: Integration point in HybridMemory class
"""
To integrate into your Pipeline_chatbot_v1/rag/memory.py HybridMemory class:

class HybridMemory:
    def __init__(self, embedder, llm=None):
        # ... existing code ...
        self.tracker = get_tracker()  # Add this line

    def new_thread(self, first_user_text: str):
        # ... existing code to create thread ...

        # Add tracking after thread creation:
        self.tracker.log_context_event(
            event_type="new_thread",
            thread_id=tid,
            turns_cleared=0,
            tokens_cleared=0,
            reason="New conversation started"
        )

    def _trim_old_turns(self, thread: Thread):
        # If you implement a method to trim old turns from memory
        turns_before = len(thread.turns)
        # ... code that removes old turns ...
        turns_after = len(thread.turns)

        if turns_before > turns_after:
            self.tracker.log_context_event(
                event_type="window_trim",
                thread_id=thread.id,
                turns_cleared=turns_before - turns_after,
                tokens_cleared=0,  # Calculate if you track token counts
                reason="Context window limit exceeded"
            )
"""
