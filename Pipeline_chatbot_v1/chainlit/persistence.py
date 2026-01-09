# persistence.py
import sqlite3
import os

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

conn = sqlite3.connect("data/chat.db", check_same_thread=False)

def init_db():
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            role TEXT,
            content TEXT,
            ts DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()

def save_message(user_id, role, content):
    conn.execute(
        "INSERT INTO messages (user_id, role, content) VALUES (?, ?, ?)",
        (user_id, role, content),
    )
    conn.commit()

def load_messages(user_id, limit=20):
    cur = conn.execute(
        """
        SELECT role, content
        FROM messages
        WHERE user_id = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (user_id, limit),
    )
    return list(reversed(cur.fetchall()))

# Initialize once at import time
init_db()
