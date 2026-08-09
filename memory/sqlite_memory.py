import sqlite3
import json
from datetime import datetime

DB = "alphavest_memory.db"

def init_db():
    with sqlite3.connect(DB) as con:
        con.execute("""CREATE TABLE IF NOT EXISTS conversations(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT, role TEXT, content TEXT, created_at TEXT)""")
        con.execute("""CREATE TABLE IF NOT EXISTS investor_profile(
            key TEXT PRIMARY KEY, value TEXT)""")

def save_message(session_id, role, content):
    init_db()
    with sqlite3.connect(DB) as con:
        con.execute("INSERT INTO conversations(session_id,role,content,created_at) VALUES(?,?,?,?)",
                    (session_id, role, content, datetime.now().isoformat()))

def recent_messages(session_id, limit=10):
    init_db()
    with sqlite3.connect(DB) as con:
        rows = con.execute(
            "SELECT role,content FROM conversations WHERE session_id=? ORDER BY id DESC LIMIT ?",
            (session_id, limit)
        ).fetchall()
    return list(reversed(rows))

def set_profile(key, value):
    init_db()
    with sqlite3.connect(DB) as con:
        con.execute("INSERT OR REPLACE INTO investor_profile(key,value) VALUES(?,?)",
                    (key, json.dumps(value)))

def get_profile():
    init_db()
    with sqlite3.connect(DB) as con:
        rows = con.execute("SELECT key,value FROM investor_profile").fetchall()
    result = {}
    for k, v in rows:
        try:
            result[k] = json.loads(v)
        except Exception:
            result[k] = v
    return result

def clear_memory():
    init_db()
    with sqlite3.connect(DB) as con:
        con.execute("DELETE FROM conversations")
        con.execute("DELETE FROM investor_profile")
