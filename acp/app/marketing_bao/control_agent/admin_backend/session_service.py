"""后台会话管理查询服务。"""

from __future__ import annotations

from pathlib import Path

from acp.app.marketing_bao.control_agent.admin_backend.schemas import SessionListItem
from acp.app.marketing_bao.schemas import ChatMessage, ContactSession
from acp.app.marketing_bao.control_agent.storage.sqlite_store import SQLiteStore


class SessionService:
    def __init__(self, db_path: str | Path = "logs/marketing_bao/marketing_bao.sqlite3") -> None:
        self.store = SQLiteStore(db_path)

    def list_sessions(self, limit: int = 100) -> list[SessionListItem]:
        sessions = self.store.list_contacts(limit=limit)
        return [
            SessionListItem(
                contact_id=s.contact_id,
                app=s.app,
                display_name=s.display_name,
                current_stage=s.current_stage,
                sub_status=s.sub_status,
                status=s.status.value,
                intent_score=s.intent_score,
                memory_summary=s.memory_summary,
                updated_at=s.updated_at,
            )
            for s in sessions
        ]

    def get_session(self, contact_id: str) -> ContactSession | None:
        return self.store.get_contact(contact_id)

    def list_messages(self, contact_id: str, limit: int = 50) -> list[ChatMessage]:
        return self.store.list_messages(contact_id, limit=limit)
