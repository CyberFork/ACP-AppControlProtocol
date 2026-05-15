"""营销宝共享业务数据结构。

设计目标：
- ControlAgent 只处理业务语义：阶段、记忆、任务、策略。
- ExecutionAgent 只处理执行语义：会话、消息、动作结果。
- 两层通过这些稳定 schema 通信，避免业务逻辑耦合到具体 App/UI。
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class StageID(str, Enum):
    """营销宝默认三阶段。用户可通过 config/stages.yaml 扩展/改名。"""

    BROAD_REACH = "broad_reach"      # 撒网期
    INTEREST_HIT = "interest_hit"    # 意向命中期
    CONVERSION = "conversion"        # 递进成交期


class ContactStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    CONVERTED = "converted"
    REJECTED = "rejected"
    BLOCKED = "blocked"


class MessageDirection(str, Enum):
    INBOUND = "inbound"
    OUTBOUND = "outbound"
    SYSTEM = "system"


class TaskIntent(str, Enum):
    SYNC_DIALOGS = "sync_dialogs"
    READ_MESSAGES = "read_messages"
    SEND_MESSAGE = "send_message"
    SEARCH_USER = "search_user"
    ADD_FRIEND = "add_friend"


class ExecutionStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class ChatMessage(BaseModel):
    id: str = Field(default_factory=lambda: f"msg_{uuid4().hex[:12]}")
    contact_id: str
    direction: MessageDirection
    text: str
    timestamp: str = Field(default_factory=now_iso)
    raw: dict[str, Any] = Field(default_factory=dict)


class ContactSession(BaseModel):
    contact_id: str
    app: str = "telegram"
    app_user_id: Optional[str] = None
    username: Optional[str] = None
    display_name: str = ""
    current_stage: str = StageID.BROAD_REACH.value
    sub_status: str = "new"
    status: ContactStatus = ContactStatus.ACTIVE
    intent_score: float = 0.0
    memory_summary: str = ""
    last_interaction_at: Optional[str] = None
    created_at: str = Field(default_factory=now_iso)
    updated_at: str = Field(default_factory=now_iso)
    metadata: dict[str, Any] = Field(default_factory=dict)


class StageDefinition(BaseModel):
    id: str
    name: str
    description: str = ""
    goal: str = ""
    sub_statuses: list[str] = Field(default_factory=list)
    entry_keywords: list[str] = Field(default_factory=list)
    exit_keywords: list[str] = Field(default_factory=list)
    knowledge_tags: list[str] = Field(default_factory=list)
    allowed_actions: list[str] = Field(default_factory=list)


class PlaybookRule(BaseModel):
    stage_id: str
    tone: str = "专业、自然、低压力"
    objective: str = "推进一次有效互动"
    templates: list[str] = Field(default_factory=list)
    forbidden: list[str] = Field(default_factory=list)


class ActionTask(BaseModel):
    task_id: str = Field(default_factory=lambda: f"task_{uuid4().hex[:12]}")
    contact_id: Optional[str] = None
    app: str = "telegram"
    intent: TaskIntent
    payload: dict[str, Any] = Field(default_factory=dict)
    requires_human_review: bool = False
    created_at: str = Field(default_factory=now_iso)


class ExecutionResult(BaseModel):
    task_id: str
    status: ExecutionStatus
    observations: dict[str, Any] = Field(default_factory=dict)
    messages: list[ChatMessage] = Field(default_factory=list)
    error: Optional[str] = None
    created_at: str = Field(default_factory=now_iso)


class AgentRunMode(str, Enum):
    HUMAN_REVIEW = "human_review"
    AUTO_SEND = "auto_send"


class RuntimeConfig(BaseModel):
    app: str = "telegram"
    run_mode: AgentRunMode = AgentRunMode.AUTO_SEND
    max_contacts_per_run: int = 20
    max_messages_per_contact: int = 20
    dry_run: bool = True
    llm_enabled_control: bool = False
    llm_enabled_execution: bool = False


class DriverKind(str, Enum):
    TELEGRAM_API = "telegram_api"
    TELEGRAM_ANDROID = "telegram_android"
    FEISHU_API = "feishu_api"
