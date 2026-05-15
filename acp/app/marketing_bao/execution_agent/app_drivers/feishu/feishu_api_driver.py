"""飞书开放平台 API driver。

定位：API 型 app_driver 样板。
- dry_run=True：无需飞书配置，用于验证营销宝双 Agent 闭环。
- dry_run=False：使用飞书企业自建应用 app_id/app_secret 获取 tenant_access_token，调用 IM v1 发送消息。

说明：飞书不适合作为“陌生客户主动添加好友”的主样板，适合作为国内 API 型 IM/协作应用验证样板。
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

from acp.app.marketing_bao.execution_agent.app_drivers.feishu.schemas import FeishuConfig
from acp.app.marketing_bao.execution_agent.protocol import AppDriver
from acp.app.marketing_bao.schemas import (
    ChatMessage,
    ContactSession,
    ExecutionResult,
    ExecutionStatus,
    MessageDirection,
)


class FeishuAPIDriver(AppDriver):
    app_name = "feishu"

    def __init__(self, config: FeishuConfig | None = None) -> None:
        self.config = config or FeishuConfig()
        self._tenant_access_token: str | None = None

    @property
    def dry_run(self) -> bool:
        return self.config.dry_run

    def _env(self, name: str) -> str:
        return os.getenv(name, "").strip()

    async def _get_tenant_access_token(self) -> str:
        if self._tenant_access_token:
            return self._tenant_access_token

        app_id = self._env(self.config.app_id_env)
        app_secret = self._env(self.config.app_secret_env)
        if not app_id or not app_secret:
            raise RuntimeError(f"缺少飞书应用环境变量: {self.config.app_id_env}/{self.config.app_secret_env}")

        url = f"{self.config.base_url}/auth/v3/tenant_access_token/internal"
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(url, json={"app_id": app_id, "app_secret": app_secret})
            resp.raise_for_status()
            data = resp.json()

        if data.get("code") != 0:
            raise RuntimeError(f"获取 tenant_access_token 失败: {data}")
        self._tenant_access_token = data["tenant_access_token"]
        return self._tenant_access_token

    async def list_dialogs(self, limit: int = 20) -> list[ContactSession]:
        if self.dry_run:
            return [
                ContactSession(
                    contact_id="feishu_dryrun_001",
                    app="feishu",
                    app_user_id="feishu_dryrun_001",
                    display_name="飞书测试用户",
                    metadata={"driver": "feishu_api", "dry_run": True, "receive_id_type": "open_id"},
                )
            ][:limit]

        receive_id = self._env(self.config.default_receive_id_env)
        receive_id_type = self._env(self.config.default_receive_id_type_env) or "open_id"
        if not receive_id:
            raise RuntimeError(f"真实模式下需要设置默认接收人: {self.config.default_receive_id_env}")
        return [
            ContactSession(
                contact_id=receive_id,
                app="feishu",
                app_user_id=receive_id,
                display_name=f"飞书接收人({receive_id_type})",
                metadata={"driver": "feishu_api", "receive_id_type": receive_id_type},
            )
        ]

    async def read_messages(self, contact_id: str, limit: int = 20) -> list[ChatMessage]:
        # 飞书消息读取通常需要事件订阅/回调或额外消息权限；MVP 先用 dry_run/占位。
        if self.dry_run:
            return [
                ChatMessage(
                    contact_id=contact_id,
                    direction=MessageDirection.INBOUND,
                    text="我想了解一下你们这个营销自动化服务怎么用。",
                    raw={"driver": "feishu_api", "dry_run": True},
                )
            ]
        return []

    async def send_message(self, contact_id: str, text: str) -> ExecutionResult:
        receive_id_type = "open_id"
        if self.dry_run:
            return ExecutionResult(
                task_id="send_message",
                status=ExecutionStatus.SUCCESS,
                observations={"dry_run": True, "contact_id": contact_id, "text": text, "driver": "feishu_api"},
            )

        receive_id_type = self._env(self.config.default_receive_id_type_env) or "open_id"
        token = await self._get_tenant_access_token()
        url = f"{self.config.base_url}/im/v1/messages"
        params = {"receive_id_type": receive_id_type}
        payload = {
            "receive_id": contact_id,
            "msg_type": "text",
            "content": json.dumps({"text": text}, ensure_ascii=False),
        }
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json; charset=utf-8"}

        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(url, params=params, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        ok = data.get("code") == 0
        return ExecutionResult(
            task_id="send_message",
            status=ExecutionStatus.SUCCESS if ok else ExecutionStatus.FAILED,
            observations={"response": data, "receive_id_type": receive_id_type},
            error=None if ok else str(data),
        )

    async def search_user(self, keyword: str) -> list[ContactSession]:
        # 后续可接通讯录搜索/邮箱/open_id 映射能力。
        return []

    async def add_friend(self, user_id: str, message: str = "") -> ExecutionResult:
        return ExecutionResult(
            task_id="add_friend",
            status=ExecutionStatus.SKIPPED,
            error="飞书 API driver 不实现陌生人加好友；请使用通讯录/组织内 open_id 或目标 app 专用策略",
            observations={"user_id": user_id, "message": message},
        )
