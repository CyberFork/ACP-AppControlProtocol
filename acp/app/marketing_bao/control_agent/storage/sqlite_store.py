"""营销宝 SQLite 持久化。"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable

from acp.app.marketing_bao.schemas import ChatMessage, ContactSession, ExecutionResult, now_iso


class SQLiteStore:
    def __init__(self, path: str | Path = "logs/marketing_bao/marketing_bao.sqlite3") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.init_schema()

    def init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS contacts (
                contact_id TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                contact_id TEXT NOT NULL,
                direction TEXT NOT NULL,
                text TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                data TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS execution_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                status TEXT NOT NULL,
                data TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        self.conn.commit()

    def upsert_contact(self, session: ContactSession) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO contacts(contact_id, data, updated_at) VALUES (?, ?, ?)",
            (session.contact_id, session.model_dump_json(), session.updated_at),
        )
        self.conn.commit()

    def get_contact(self, contact_id: str) -> ContactSession | None:
        row = self.conn.execute("SELECT data FROM contacts WHERE contact_id=?", (contact_id,)).fetchone()
        if not row:
            return None
        return ContactSession.model_validate_json(row["data"])

    def list_contacts(self, limit: int = 100) -> list[ContactSession]:
        rows = self.conn.execute("SELECT data FROM contacts ORDER BY updated_at DESC LIMIT ?", (limit,)).fetchall()
        return [ContactSession.model_validate_json(r["data"]) for r in rows]

    def add_messages(self, messages: Iterable[ChatMessage]) -> None:
        for msg in messages:
            self.conn.execute(
                "INSERT OR REPLACE INTO messages(id, contact_id, direction, text, timestamp, data) VALUES (?, ?, ?, ?, ?, ?)",
                (msg.id, msg.contact_id, msg.direction.value, msg.text, msg.timestamp, msg.model_dump_json()),
            )
        self.conn.commit()

    def list_messages(self, contact_id: str, limit: int = 50) -> list[ChatMessage]:
        rows = self.conn.execute(
            "SELECT data FROM messages WHERE contact_id=? ORDER BY timestamp DESC LIMIT ?",
            (contact_id, limit),
        ).fetchall()
        return [ChatMessage.model_validate_json(r["data"]) for r in reversed(rows)]

    def add_execution_log(self, result: ExecutionResult) -> None:
        self.conn.execute(
            "INSERT INTO execution_logs(task_id, status, data, created_at) VALUES (?, ?, ?, ?)",
            (result.task_id, result.status.value, result.model_dump_json(), now_iso()),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()
