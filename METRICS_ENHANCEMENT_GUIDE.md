# Metrics Enhancement Guide

## Overview

Your metrics system has been enhanced with two major features:

1. **Cumulative Token Consumption** - Track total token usage across all time
2. **Context Window Event Tracking** - Monitor when LLM context is cleared, reset, or modified

## New Features

### 1. Cumulative Token Tracking

The system now tracks your total token consumption across all time, giving you visibility into:
- Total prompt tokens ever used
- Total completion tokens ever generated
- Total tokens across all LLM calls
- First and last call timestamps
- Total number of API calls

#### Database Schema Addition
A new table `context_events` has been added to track context window operations:

```sql
CREATE TABLE context_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    event_type TEXT NOT NULL,           -- 'clear', 'reset', 'new_thread', etc.
    thread_id TEXT,                     -- Conversation/thread ID
    session_id TEXT,                    -- Session ID if applicable
    turns_cleared INTEGER DEFAULT 0,    -- Number of conversation turns removed
    tokens_cleared INTEGER DEFAULT 0,   -- Estimated tokens cleared from context
    reason TEXT,                        -- Human-readable reason
    metadata TEXT                       -- Additional JSON metadata
)
```

### 2. Context Window Event Types

The following event types are supported:

| Event Type | Description | When to Use |
|-----------|-------------|-------------|
| `new_thread` | New conversation started | When creating a new thread/conversation |
| `thread_clear` | Entire thread cleared | User resets or deletes conversation |
| `thread_switch` | Switched between threads | When user changes conversation topic |
| `window_trim` | Old turns removed from context | When context window size limit reached |
| `summary_compression` | Turns compressed to summary | When summarizing old messages |
| `session_end` | Session terminated | When user ends session |
| `memory_reset` | Full memory reset | When clearing all memory |

## API Endpoints

### GET /metrics

Enhanced with new query parameters:

```bash
GET /metrics?hours=24&model=llama3&include_cumulative=true&include_context_events=true
```

**Query Parameters:**
- `hours` (int, default: 24) - Time window for recent stats
- `model` (string, optional) - Filter by model name
- `include_cumulative` (bool, default: true) - Include cumulative totals
- `include_context_events` (bool, default: true) - Include context event data

**Response Structure:**

```json
{
  "summary": {
    "total_calls": 150,
    "total_prompt_tokens": 45000,
    "total_completion_tokens": 75000,
    "total_tokens": 120000,
    "avg_latency_ms": 342.5,
    "max_latency_ms": 1250.0,
    "min_latency_ms": 89.3,
    "error_count": 2
  },
  "model_comparison": [...],
  "recent_calls": [...],
  "hourly_usage": [...],

  "cumulative": {
    "total_prompt_tokens": 1250000,
    "total_completion_tokens": 2100000,
    "total_tokens": 3350000,
    "total_calls": 5420,
    "first_call": "2025-01-01T10:00:00.000000",
    "last_call": "2025-12-18T15:30:00.000000"
  },

  "context_events": {
    "stats": {
      "total_events": 45,
      "total_turns_cleared": 230,
      "total_tokens_cleared": 89000,
      "affected_threads": 12,
      "event_breakdown": [
        {"event_type": "window_trim", "count": 25},
        {"event_type": "new_thread", "count": 12},
        {"event_type": "thread_clear", "count": 5},
        {"event_type": "summary_compression", "count": 3}
      ]
    },
    "recent_events": [
      {
        "id": 45,
        "timestamp": "2025-12-18T15:25:00.000000",
        "event_type": "window_trim",
        "thread_id": "t-1734531900",
        "session_id": null,
        "turns_cleared": 4,
        "tokens_cleared": 3200,
        "reason": "Context window size limit reached",
        "metadata": "{\"window_policy\": \"rolling\", \"max_turns\": 6}"
      }
    ]
  }
}
```

## Python API Usage

### Import the Tracker

```python
from Pipeline_chatbot_v1.tracking.local_tracker import get_tracker

tracker = get_tracker()
```

### Get Cumulative Token Stats

```python
# Get all-time totals
cumulative = tracker.get_cumulative_tokens()
print(f"Total tokens used: {cumulative['total_tokens']:,}")
print(f"Total API calls: {cumulative['total_calls']:,}")

# Get totals for specific model
cumulative_llama = tracker.get_cumulative_tokens(model_name="llama3")

# Get totals since a specific date
from datetime import datetime
since = "2025-12-01T00:00:00.000000"
cumulative_recent = tracker.get_cumulative_tokens(since_timestamp=since)
```

### Log Context Window Events

```python
# Example 1: New conversation started
tracker.log_context_event(
    event_type="new_thread",
    thread_id="t-1734531900",
    reason="User started new conversation"
)

# Example 2: Context window trimmed (old messages removed)
tracker.log_context_event(
    event_type="window_trim",
    thread_id="t-1734531900",
    turns_cleared=4,
    tokens_cleared=3200,
    reason="Context window size limit reached",
    metadata={"max_turns": 6, "window_policy": "rolling"}
)

# Example 3: Thread cleared/reset
tracker.log_context_event(
    event_type="thread_clear",
    thread_id="t-1734531900",
    turns_cleared=15,
    tokens_cleared=12000,
    reason="User requested conversation reset"
)

# Example 4: Summary compression
tracker.log_context_event(
    event_type="summary_compression",
    thread_id="t-1734531900",
    turns_cleared=10,
    tokens_cleared=8500,
    reason="Compressed old turns to summary",
    metadata={
        "tokens_before": 15000,
        "tokens_after": 6500,
        "compression_ratio": 0.43
    }
)
```

### Query Context Events

```python
# Get recent context events (last 24 hours)
events = tracker.get_context_events(hours=24, limit=50)

# Get events by type
clear_events = tracker.get_context_events(
    hours=168,  # Last 7 days
    event_type="thread_clear",
    limit=20
)

# Get events for specific thread
thread_events = tracker.get_context_events(
    hours=None,  # All time
    thread_id="t-1734531900",
    limit=100
)

# Get context event statistics
stats = tracker.get_context_stats(hours=24)
print(f"Events: {stats['total_events']}")
print(f"Turns cleared: {stats['total_turns_cleared']}")
print(f"Tokens cleared: {stats['total_tokens_cleared']}")
print(f"Event breakdown: {stats['event_breakdown']}")
```

## Integration with HybridMemory

To integrate context event tracking into your existing `Pipeline_chatbot_v1/rag/memory.py` HybridMemory class:

### Step 1: Initialize tracker in HybridMemory

```python
from Pipeline_chatbot_v1.tracking.local_tracker import get_tracker

class HybridMemory:
    def __init__(self, embedder: SentenceTransformer, llm: Optional[OllamaLLM] = None):
        self.embedder = embedder
        self.llm = llm
        self.threads: List[Thread] = []
        self.curr_idx: int = -1

        # Add tracker initialization
        self.tracker = get_tracker()

        os.makedirs(os.path.dirname(MEMORY_PATH), exist_ok=True)
        self.load()
```

### Step 2: Track new thread creation

```python
def new_thread(self, first_user_text: str):
    """Create a new thread and set it as current."""
    tid = f"t-{int(time.time())}"
    title = self._propose_title(first_user_text)
    self.threads.append(
        Thread(id=tid, title=title, turns=[], summary="", focus=[])
    )
    self.curr_idx = len(self.threads) - 1

    # Track the new thread event
    self.tracker.log_context_event(
        event_type="new_thread",
        thread_id=tid,
        turns_cleared=0,
        tokens_cleared=0,
        reason="New conversation started",
        metadata={"title": title}
    )

    self.save()
    return tid
```

### Step 3: Track context window trimming

Add this method to track when old turns are removed:

```python
def _trim_context_window(self, thread: Thread, max_turns: int = 6):
    """Remove old turns beyond the window size."""
    if len(thread.turns) > max_turns:
        turns_before = len(thread.turns)
        removed_turns = thread.turns[:-max_turns]
        thread.turns = thread.turns[-max_turns:]

        # Estimate tokens cleared (rough approximation)
        tokens_cleared = sum(
            len(turn.user.split()) * 1.3 + len(turn.bot.split()) * 1.3
            for turn in removed_turns
        )

        # Track the trim event
        self.tracker.log_context_event(
            event_type="window_trim",
            thread_id=thread.id,
            turns_cleared=turns_before - max_turns,
            tokens_cleared=int(tokens_cleared),
            reason="Context window size limit reached",
            metadata={"max_turns": max_turns}
        )
```

### Step 4: Track thread switching

```python
def switch_thread(self, thread_id: str):
    """Switch to a different conversation thread."""
    old_thread_id = self.threads[self.curr_idx].id if self.curr_idx >= 0 else None

    # Find and switch to new thread
    for idx, thread in enumerate(self.threads):
        if thread.id == thread_id:
            self.curr_idx = idx

            # Track thread switch
            self.tracker.log_context_event(
                event_type="thread_switch",
                thread_id=thread_id,
                turns_cleared=0,
                tokens_cleared=0,
                reason=f"Switched from {old_thread_id}",
                metadata={
                    "from_thread": old_thread_id,
                    "to_thread": thread_id,
                    "context_turns": len(thread.turns)
                }
            )
            return True

    return False
```

## Important Notes

### Token Count Accuracy

**Current Implementation:** Token counts are **estimated** using word count × 1.3 (see [callbacks.py:65-66](d:\GJ\Chatbot_Api\Pipeline_chatbot_v1\tracking\callbacks.py#L65-L66)).

**Limitation:** This is a rough approximation and not actual token counts from the LLM API.

**To Get Accurate Token Counts:**

1. **For Ollama models:** Check if your Ollama API returns token counts in the response metadata
2. **For OpenAI/Claude APIs:** Extract `usage.prompt_tokens` and `usage.completion_tokens` from API responses
3. **Manual calculation:** Use a tokenizer library like `tiktoken` (OpenAI) or model-specific tokenizers

Example for more accurate token counting:

```python
# Install: pip install tiktoken
import tiktoken

def count_tokens_accurately(text: str, model: str = "gpt-3.5-turbo"):
    """Count tokens using tiktoken."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# Use in callbacks.py:
prompt_tokens = count_tokens_accurately(prompt, model=self.model_name)
completion_tokens = count_tokens_accurately(response_text, model=self.model_name)
```

### Context Window Deletion vs. Trimming

- **Deletion** = Entire conversation/thread is cleared (event: `thread_clear`)
- **Trimming** = Old messages removed to stay within window size (event: `window_trim`)
- **Summary Compression** = Old turns compressed into summary (event: `summary_compression`)

All three reduce the context sent to the LLM, but serve different purposes.

## Testing

See [CONTEXT_TRACKING_EXAMPLE.py](d:\GJ\Chatbot_Api\Pipeline_chatbot_v1\tracking\CONTEXT_TRACKING_EXAMPLE.py) for complete integration examples.

### Quick Test

```python
from Pipeline_chatbot_v1.tracking.local_tracker import get_tracker

tracker = get_tracker()

# Test cumulative tracking
cumulative = tracker.get_cumulative_tokens()
print("Cumulative tokens:", cumulative)

# Test context event logging
tracker.log_context_event(
    event_type="new_thread",
    thread_id="test-thread-123",
    reason="Testing context tracking"
)

# Verify event was logged
events = tracker.get_context_events(hours=1, limit=5)
print("Recent events:", events)
```

## Cost Estimation (Optional Enhancement)

To add cost tracking based on token consumption, you can extend the tracker:

```python
# Define pricing per model (example prices)
MODEL_PRICING = {
    "llama3": {"prompt": 0.0, "completion": 0.0},  # Free/local
    "gpt-4": {"prompt": 0.03 / 1000, "completion": 0.06 / 1000},
    "claude-3": {"prompt": 0.015 / 1000, "completion": 0.075 / 1000}
}

def calculate_cost(model_name: str, prompt_tokens: int, completion_tokens: int):
    """Calculate cost based on token usage."""
    pricing = MODEL_PRICING.get(model_name, {"prompt": 0, "completion": 0})
    prompt_cost = prompt_tokens * pricing["prompt"]
    completion_cost = completion_tokens * pricing["completion"]
    return prompt_cost + completion_cost

# Use in your metrics endpoint
cumulative = tracker.get_cumulative_tokens(model_name="gpt-4")
total_cost = calculate_cost(
    "gpt-4",
    cumulative["total_prompt_tokens"],
    cumulative["total_completion_tokens"]
)
print(f"Total cost: ${total_cost:.2f}")
```

## Visualization Ideas

The new metrics enable powerful visualizations:

1. **Token consumption over time** - Line chart of cumulative usage
2. **Context event timeline** - When and why context was cleared
3. **Cost projection** - Estimate monthly costs based on usage trends
4. **Memory efficiency** - How often context is trimmed vs. summarized
5. **Thread lifecycle** - Visualize conversation length before reset

## Summary

Your metrics system now provides complete visibility into:
- ✅ Total token consumption across all time
- ✅ Context window operations (clear, trim, reset)
- ✅ Per-thread memory tracking
- ✅ Event history and statistics
- ✅ Cost estimation capabilities

This gives you full observability into LLM usage and memory management!
