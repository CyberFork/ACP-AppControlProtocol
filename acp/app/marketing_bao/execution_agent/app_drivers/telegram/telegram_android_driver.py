"""Telegram Android UI 自动化 driver 占位。

用于 M0/M1 验证 Android 模拟器 + UIAutomator2/ADB。第一业务闭环先走 TelegramAPIDriver。
"""

from __future__ import annotations

from acp.app.marketing_bao.execution_agent.protocol import AppDriver
from acp.app.marketing_bao.schemas import ChatMessage, ContactSession, ExecutionResult, ExecutionStatus


class TelegramAndroidDriver(AppDriver):
    app_name = "telegram"

    async def list_dialogs(self, limit: int = 20) -> list[ContactSession]:
        raise NotImplementedError("TelegramAndroidDriver 将在 Android 模拟器验证后实现")

    async def read_messages(self, contact_id: str, limit: int = 20) -> list[ChatMessage]:
        raise NotImplementedError("TelegramAndroidDriver 将在 Android 模拟器验证后实现")

    async def send_message(self, contact_id: str, text: str) -> ExecutionResult:
        return ExecutionResult(task_id="send_message", status=ExecutionStatus.FAILED, error="TelegramAndroidDriver 尚未实现")
