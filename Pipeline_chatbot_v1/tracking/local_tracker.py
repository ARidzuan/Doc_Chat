"""
Local tracking system for LLM usage metrics.
Stores all LLM calls in SQLite for unlimited, offline analysis.
"""
import sqlite3
import json
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path
from contextlib import contextmanager


class LocalTracker:
    """Track LLM usage locally in SQLite database."""

    def __init__(self, db_path: str = "data/llm_tracking.db"):
        """Initialize tracker with SQLite database."""
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Create tables if they don't exist."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS llm_calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    operation_type TEXT,
                    prompt TEXT,
                    response TEXT,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    total_tokens INTEGER,
                    latency_ms REAL,
                    temperature REAL,
                    metadata TEXT,
                    error TEXT
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp
                ON llm_calls(timestamp)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_model
                ON llm_calls(model_name)
            """)

            # Table for tracking context window events
            conn.execute("""
                CREATE TABLE IF NOT EXISTS context_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    thread_id TEXT,
                    session_id TEXT,
                    turns_cleared INTEGER DEFAULT 0,
                    tokens_cleared INTEGER DEFAULT 0,
                    reason TEXT,
                    metadata TEXT
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_context_timestamp
                ON context_events(timestamp)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_context_thread
                ON context_events(thread_id)
            """)

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def log_call(
        self,
        model_name: str,
        operation_type: str = "generate",
        prompt: Optional[str] = None,
        response: Optional[str] = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        latency_ms: float = 0.0,
        temperature: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ):
        """Log an LLM call to the database."""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO llm_calls (
                    timestamp, model_name, operation_type, prompt, response,
                    prompt_tokens, completion_tokens, total_tokens,
                    latency_ms, temperature, metadata, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                model_name,
                operation_type,
                prompt,
                response,
                prompt_tokens,
                completion_tokens,
                prompt_tokens + completion_tokens,
                latency_ms,
                temperature,
                json.dumps(metadata) if metadata else None,
                error
            ))

    def get_stats(
        self,
        model_name: Optional[str] = None,
        hours: int = 24
    ) -> Dict[str, Any]:
        """Get usage statistics for the specified time period."""
        with self._get_connection() as conn:
            where_clause = "WHERE timestamp >= datetime('now', ?)"
            params = [f'-{hours} hours']

            if model_name:
                where_clause += " AND model_name = ?"
                params.append(model_name)

            cursor = conn.execute(f"""
                SELECT
                    COUNT(*) as total_calls,
                    SUM(prompt_tokens) as total_prompt_tokens,
                    SUM(completion_tokens) as total_completion_tokens,
                    SUM(total_tokens) as total_tokens,
                    AVG(latency_ms) as avg_latency_ms,
                    MAX(latency_ms) as max_latency_ms,
                    MIN(latency_ms) as min_latency_ms,
                    COUNT(CASE WHEN error IS NOT NULL THEN 1 END) as error_count
                FROM llm_calls
                {where_clause}
            """, params)

            row = cursor.fetchone()
            return dict(row) if row else {}

    def get_model_comparison(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Compare usage across different models."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT
                    model_name,
                    COUNT(*) as calls,
                    SUM(total_tokens) as total_tokens,
                    AVG(latency_ms) as avg_latency_ms,
                    COUNT(CASE WHEN error IS NOT NULL THEN 1 END) as errors
                FROM llm_calls
                WHERE timestamp >= datetime('now', ?)
                GROUP BY model_name
                ORDER BY calls DESC
            """, (f'-{hours} hours',))

            return [dict(row) for row in cursor.fetchall()]

    def get_recent_calls(self,limit: int = 10,model_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get recent LLM calls with details."""
        with self._get_connection() as conn:
            where_clause = ""
            params = []

            if model_name:
                where_clause = "WHERE model_name = ?"
                params.append(model_name)

            cursor = conn.execute(f"""
                SELECT
                    id, timestamp, model_name, operation_type,
                    prompt_tokens, completion_tokens, total_tokens,
                    latency_ms, error
                FROM llm_calls
                {where_clause}
                ORDER BY timestamp DESC
                LIMIT ?
            """, params + [limit])

            return [dict(row) for row in cursor.fetchall()]

    def get_hourly_usage(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get token usage grouped by hour."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT
                    strftime('%Y-%m-%d %H:00:00', timestamp) as hour,
                    COUNT(*) as calls,
                    SUM(total_tokens) as tokens,
                    AVG(latency_ms) as avg_latency
                FROM llm_calls
                WHERE timestamp >= datetime('now', ?)
                GROUP BY hour
                ORDER BY hour DESC
            """, (f'-{hours} hours',))

            return [dict(row) for row in cursor.fetchall()]

    def log_context_event(
        self,
        event_type: str,
        thread_id: Optional[str] = None,
        session_id: Optional[str] = None,
        turns_cleared: int = 0,
        tokens_cleared: int = 0,
        reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log a context window event (deletion, reset, clear, new thread, etc.).

        Args:
            event_type: Type of event ('clear', 'reset', 'new_thread', 'window_trim', etc.)
            thread_id: Thread/conversation ID affected
            session_id: Session ID if applicable
            turns_cleared: Number of conversation turns cleared/removed
            tokens_cleared: Estimated tokens cleared from context
            reason: Human-readable reason for the event
            metadata: Additional metadata as dict
        """
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO context_events (
                    timestamp, event_type, thread_id, session_id,
                    turns_cleared, tokens_cleared, reason, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                event_type,
                thread_id,
                session_id,
                turns_cleared,
                tokens_cleared,
                reason,
                json.dumps(metadata) if metadata else None
            ))

    def get_cumulative_tokens(
        self,
        model_name: Optional[str] = None,
        since_timestamp: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get cumulative token consumption across all time (or since timestamp).

        Args:
            model_name: Optional filter by model name
            since_timestamp: Optional ISO timestamp to calculate from

        Returns:
            Dict with total_prompt_tokens, total_completion_tokens, total_tokens,
            total_calls, first_call, last_call
        """
        with self._get_connection() as conn:
            where_clauses = []
            params = []

            if since_timestamp:
                where_clauses.append("timestamp >= ?")
                params.append(since_timestamp)

            if model_name:
                where_clauses.append("model_name = ?")
                params.append(model_name)

            where_clause = " WHERE " + " AND ".join(where_clauses) if where_clauses else ""

            cursor = conn.execute(f"""
                SELECT
                    SUM(prompt_tokens) as total_prompt_tokens,
                    SUM(completion_tokens) as total_completion_tokens,
                    SUM(total_tokens) as total_tokens,
                    COUNT(*) as total_calls,
                    MIN(timestamp) as first_call,
                    MAX(timestamp) as last_call
                FROM llm_calls
                {where_clause}
            """, params)

            row = cursor.fetchone()
            result = dict(row) if row else {}

            # Convert None to 0 for token counts
            result['total_prompt_tokens'] = result.get('total_prompt_tokens') or 0
            result['total_completion_tokens'] = result.get('total_completion_tokens') or 0
            result['total_tokens'] = result.get('total_tokens') or 0
            result['total_calls'] = result.get('total_calls') or 0

            return result

    def get_context_events(
        self,
        hours: Optional[int] = 24,
        event_type: Optional[str] = None,
        thread_id: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get context window events (clears, resets, etc.).

        Args:
            hours: Look back window in hours (None for all time)
            event_type: Filter by event type
            thread_id: Filter by thread ID
            limit: Max number of events to return

        Returns:
            List of context event records
        """
        with self._get_connection() as conn:
            where_clauses = []
            params = []

            if hours is not None:
                where_clauses.append("timestamp >= datetime('now', ?)")
                params.append(f'-{hours} hours')

            if event_type:
                where_clauses.append("event_type = ?")
                params.append(event_type)

            if thread_id:
                where_clauses.append("thread_id = ?")
                params.append(thread_id)

            where_clause = " WHERE " + " AND ".join(where_clauses) if where_clauses else ""

            cursor = conn.execute(f"""
                SELECT
                    id, timestamp, event_type, thread_id, session_id,
                    turns_cleared, tokens_cleared, reason, metadata
                FROM context_events
                {where_clause}
                ORDER BY timestamp DESC
                LIMIT ?
            """, params + [limit])

            return [dict(row) for row in cursor.fetchall()]

    def get_context_stats(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get statistics about context window events.

        Returns:
            Summary stats about context clearing/resets
        """
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT
                    COUNT(*) as total_events,
                    SUM(turns_cleared) as total_turns_cleared,
                    SUM(tokens_cleared) as total_tokens_cleared,
                    COUNT(DISTINCT thread_id) as affected_threads
                FROM context_events
                WHERE timestamp >= datetime('now', ?)
            """, (f'-{hours} hours',))

            row = cursor.fetchone()
            result = dict(row) if row else {}

            # Event type breakdown
            cursor = conn.execute("""
                SELECT
                    event_type,
                    COUNT(*) as count
                FROM context_events
                WHERE timestamp >= datetime('now', ?)
                GROUP BY event_type
                ORDER BY count DESC
            """, (f'-{hours} hours',))

            result['event_breakdown'] = [dict(row) for row in cursor.fetchall()]

            return result


# Global tracker instance
_tracker: Optional[LocalTracker] = None


def get_tracker() -> LocalTracker:
    """Get or create the global tracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = LocalTracker()
    return _tracker
