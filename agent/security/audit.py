import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path


class AuditLogger:
    def __init__(self, db_path: str = "./storage/audit.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS queries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    query TEXT NOT NULL,
                    intent TEXT,
                    tools_used TEXT,
                    tokens_used INTEGER,
                    response_summary TEXT,
                    agent_type TEXT
                )
            """)

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def log(self, query: str, intent: str, tools_used: list[str] | None = None,
            tokens_used: int | None = None, response_summary: str = "",
            agent_type: str = ""):
        with self._lock:
            with self._get_conn() as conn:
                conn.execute(
                    """INSERT INTO queries
                       (timestamp, query, intent, tools_used, tokens_used, response_summary, agent_type)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (datetime.now().isoformat(), query[:500], intent,
                     json.dumps(tools_used or []), tokens_used,
                     response_summary[:200], agent_type)
                )

    def recent(self, limit: int = 10) -> list[dict]:
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM queries ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]
