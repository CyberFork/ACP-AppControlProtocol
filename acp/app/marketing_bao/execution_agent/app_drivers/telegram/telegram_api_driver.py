"""Telegram 用户账号 API driver。

注意：不使用 Bot API。Bot API 不能主动发起未互动用户对话，不适合作为本项目主线。

MVP 实现策略：
- 如果安装了 telethon 且配置了 TELEGRAM_API_ID/API_HASH，可连接真实 Telegram 用户账号。
- 默认 dry_run=True，不真实发送，只返回成功模拟结果，方便先跑通 ControlAgent 闭环。
- 后续可替换为 TDLib driver；上层 AppDriver 协议不变。
"""

from __future__ import annotations

import os
from typing import Any

from acp.app.marketing_bao.execution_agent.app_drivers.telegram.schemas import TelegramConfig
from acp.app.marketing_bao.execution_agent.protocol import AppDriver
from acp.app.marketing_bao.schemas import (
    ChatMessage,
    ContactSession,
    ExecutionResult,
    ExecutionStatus,
    MessageDirection,
)


class TelegramAPIDriver(AppDriver):
    app_name = "telegram"

    def __init__(self, config: TelegramConfig | None = None) -> None:
        self.config = config or TelegramConfig()
        self._client: Any = None
        self._telethon_available: bool | None = None

    @property
    def dry_run(self) -> bool:
        return self.config.dry_run

    def _get_api_credentials(self) -> tuple[str, str]:
        return os.getenv(self.config.api_id_env, ""), os.getenv(self.config.api_hash_env, "")

    async def _ensure_client(self) -> Any:
        if self.dry_run:
            return None
        api_id, api_hash = self._get_api_credentials()
        if not api_id or not api_hash:
            raise RuntimeError(f"缺少 Telegram API 环境变量: {self.config.api_id_env}/{self.config.api_hash_env}")
        try:
            from telethon import TelegramClient  # type: ignore
        except ImportError as exc:
            raise RuntimeError("需要安装 telethon 才能使用 telegram_api driver: pip install telethon") from exc

        if self._client is None:
            self._client = TelegramClient(self.config.session_name, int(api_id), api_hash)
            await self._client.start(phone=os.getenv(self.config.phone_env) or None)
        return self._client

    async def list_dialogs(self, limit: int = 20) -> list[ContactSession]:
        if self.dry_run:
            return [
                ContactSession(
                    contact_id="tg_dryrun_001",
                    app="telegram",
                    app_user_id="tg_dryrun_001",
                    username="dryrun_user",
                    display_name="Telegram 测试用户",
                    metadata={"driver": "telegram_api", "dry_run": True},
                )
            ][:limit]

        client = await self._ensure_client()
        sessions: list[ContactSession] = []
        async for dialog in client.iter_dialogs(limit=limit):
            entity = dialog.entity
            contact_id = str(getattr(entity, "id", dialog.id))
            sessions.append(
                ContactSession(
                    contact_id=contact_id,
                    app="telegram",
                    app_user_id=contact_id,
                    username=getattr(entity, "username", None),
                    display_name=getattr(dialog, "name", "") or getattr(entity, "first_name", "") or contact_id,
                    metadata={"is_user": bool(getattr(entity, "bot", None) is not None)},
                )
            )
        return sessions

    async def read_messages(self, contact_id: str, limit: int = 20) -> list[ChatMessage]:
        if self.dry_run:
            return [
                ChatMessage(
                    contact_id=contact_id,
                    direction=MessageDirection.INBOUND,
                    text="你好，我想了解一下你们的方案和价格。",
                    raw={"driver": "telegram_api", "dry_run": True},
                )
            ]

        client = await self._ensure_client()
        messages: list[ChatMessage] = []
        async for msg in client.iter_messages(int(contact_id) if contact_id.isdigit() else contact_id, limit=limit):
            text = getattr(msg, "message", "") or ""
            if not text:
                continue
            direction = MessageDirection.OUTBOUND if getattr(msg, "out", False) else MessageDirection.INBOUND
            messages.append(
                ChatMessage(
                    id=f"tg_{getattr(msg, 'id', '')}",
                    contact_id=contact_id,
                    direction=direction,
                    text=text,
                    timestamp=getattr(msg, "date", None).isoformat() if getattr(msg, "date", None) else None,
                    raw={"telegram_id": getattr(msg, "id", None)},
                )
            )
        return list(reversed(messages))

    async def send_message(self, contact_id: str, text: str) -> ExecutionResult:
        if self.dry_run:
            return ExecutionResult(
                task_id="send_message",
                status=ExecutionStatus.SUCCESS,
                observations={"dry_run": True, "contact_id": contact_id, "text": text},
            )

        client = await self._ensure_client()
        sent = await client.send_message(int(contact_id) if contact_id.isdigit() else contact_id, text)
        return ExecutionResult(
            task_id="send_message",
            status=ExecutionStatus.SUCCESS,
            observations={"telegram_message_id": getattr(sent, "id", None)},
        )

    async def search_user(self, keyword: str) -> list[ContactSession]:
        # Telethon/TDLib 搜索策略后续特化，这里先提供协议占位。
        return []

    async def add_friend(self, user_id: str, message: str = "") -> ExecutionResult:
        return ExecutionResult(
            task_id="add_friend",
            status=ExecutionStatus.SKIPPED,
            error="telegram_api add_friend 尚未实现；后续按 username/phone/contact import 策略特化",
            observations={"user_id": user_id, "message": message},
        )
