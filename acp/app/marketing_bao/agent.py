"""营销宝总编排入口。"""

from __future__ import annotations

from acp.app.marketing_bao.config_loader import CONFIG_DIR, load_runtime
from acp.app.marketing_bao.control_agent.control_agent import ControlAgent
from acp.app.marketing_bao.control_agent.knowledge import KnowledgeBase
from acp.app.marketing_bao.control_agent.state_machine import StageMachine
from acp.app.marketing_bao.execution_agent.app_drivers.feishu.feishu_driver import build_feishu_driver
from acp.app.marketing_bao.execution_agent.app_drivers.telegram.telegram_driver import build_telegram_driver
from acp.app.marketing_bao.execution_agent.execution_agent import ExecutionAgent
from acp.app.marketing_bao.schemas import ActionTask, DriverKind, TaskIntent
from acp.app.marketing_bao.control_agent.storage.sqlite_store import SQLiteStore


class MarketingBaoAgent:
    def __init__(self, store: SQLiteStore | None = None, runtime_path: str | None = None) -> None:
        self.runtime, self.driver_kind = load_runtime(runtime_path)
        self.store = store or SQLiteStore()
        self.state_machine = StageMachine.from_yaml(CONFIG_DIR / "stages.yaml")
        self.knowledge = KnowledgeBase.from_yaml(CONFIG_DIR / "playbooks.yaml", CONFIG_DIR / "product.yaml")
        self.control = ControlAgent(self.state_machine, self.knowledge, self.runtime)
        self.execution = ExecutionAgent(self._build_driver())

    def _build_driver(self):
        if self.driver_kind == DriverKind.FEISHU_API or self.runtime.app == "feishu":
            return build_feishu_driver(dry_run=self.runtime.dry_run)
        return build_telegram_driver(self.driver_kind, dry_run=self.runtime.dry_run)

    async def run_once(self) -> dict:
        """执行一轮：同步会话 → 读消息 → 判断状态 → 生成回复 → auto_send。"""
        sync_result = await self.execution.execute(
            ActionTask(intent=TaskIntent.SYNC_DIALOGS, payload={"limit": self.runtime.max_contacts_per_run})
        )
        self.store.add_execution_log(sync_result)
        dialogs = sync_result.observations.get("dialogs", [])

        summary: dict = {"dialogs": len(dialogs), "planned": 0, "sent": 0, "dry_run": self.runtime.dry_run, "results": []}
        from acp.app.marketing_bao.schemas import ContactSession

        for raw in dialogs:
            session = ContactSession.model_validate(raw)
            existing = self.store.get_contact(session.contact_id)
            if existing:
                # 保留已有状态/记忆，同时刷新基础信息
                existing.display_name = session.display_name or existing.display_name
                existing.username = session.username or existing.username
                session = existing

            read_result = await self.execution.execute(
                ActionTask(
                    contact_id=session.contact_id,
                    intent=TaskIntent.READ_MESSAGES,
                    payload={"limit": self.runtime.max_messages_per_contact},
                )
            )
            self.store.add_execution_log(read_result)
            self.store.add_messages(read_result.messages)

            session = self.control.observe_contact(session, read_result.messages)
            self.store.upsert_contact(session)

            task = self.control.decide_next_task(session)
            if not task:
                continue
            summary["planned"] += 1

            # human_review 模式下只规划不执行；auto_send 执行。dry_run 由 driver 控制。
            if task.requires_human_review:
                summary["results"].append({"contact_id": session.contact_id, "planned_text": task.payload.get("text")})
                continue

            result = await self.execution.execute(task)
            self.store.add_execution_log(result)
            if result.status.value == "success":
                summary["sent"] += 1
            summary["results"].append(
                {
                    "contact_id": session.contact_id,
                    "stage": session.current_stage,
                    "status": result.status.value,
                    "text": task.payload.get("text"),
                    "error": result.error,
                }
            )

        return summary
