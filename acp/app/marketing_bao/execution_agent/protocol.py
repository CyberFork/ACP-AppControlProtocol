"""ExecutionAgent 与 App Driver 协议。"""

from __future__ import annotations

from abc import ABC, abstractmethod

from acp.app.marketing_bao.schemas import ActionTask, ChatMessage, ContactSession, ExecutionResult


class AppDriver(ABC):
    """每个目标 App 的特化出口都实现这个协议。"""

    app_name: str

    @abstractmethod
    async def list_dialogs(self, limit: int = 20) -> list[ContactSession]:
        """列出最近会话/联系人。"""

    @abstractmethod
    async def read_messages(self, contact_id: str, limit: int = 20) -> list[ChatMessage]:
        """读取指定联系人的最近消息。"""

    @abstractmethod
    async def send_message(self, contact_id: str, text: str) -> ExecutionResult:
        """发送消息。"""

    async def search_user(self, keyword: str) -> list[ContactSession]:
        return []

    async def add_friend(self, user_id: str, message: str = "") -> ExecutionResult:
        return ExecutionResult(task_id="add_friend", status="skipped", error="driver 未实现 add_friend")


class PlatformDriver(ABC):
    """Android/iOS/模拟器平台能力。"""

    @abstractmethod
    async def check_ready(self) -> dict:
        """检查执行环境是否就绪。"""
