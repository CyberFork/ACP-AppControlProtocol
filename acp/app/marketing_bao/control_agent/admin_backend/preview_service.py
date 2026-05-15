"""规划端预览服务。

用于后台在不真实发送消息的情况下，预览某个客户的：
- 阶段判断结果
- 记忆摘要更新
- 下一步 ActionTask
"""

from __future__ import annotations

from pathlib import Path

from acp.app.marketing_bao.config_loader import CONFIG_DIR, load_runtime
from acp.app.marketing_bao.control_agent.admin_backend.schemas import PreviewNextActionResult
from acp.app.marketing_bao.control_agent.control_agent import ControlAgent
from acp.app.marketing_bao.control_agent.knowledge import KnowledgeBase
from acp.app.marketing_bao.control_agent.state_machine import StageMachine
from acp.app.marketing_bao.control_agent.storage.sqlite_store import SQLiteStore


class PreviewService:
    def __init__(
        self,
        config_dir: str | Path = CONFIG_DIR,
        db_path: str | Path = "logs/marketing_bao/marketing_bao.sqlite3",
    ) -> None:
        self.config_dir = Path(config_dir)
        self.store = SQLiteStore(db_path)

    def _build_control_agent(self) -> ControlAgent:
        runtime, _driver = load_runtime(self.config_dir / "runtime.yaml")
        state_machine = StageMachine.from_yaml(self.config_dir / "stages.yaml")
        knowledge = KnowledgeBase.from_yaml(self.config_dir / "playbooks.yaml", self.config_dir / "product.yaml")
        return ControlAgent(state_machine, knowledge, runtime)

    def preview_next_action(self, contact_id: str, persist_observation: bool = False) -> PreviewNextActionResult:
        session = self.store.get_contact(contact_id)
        if session is None:
            raise KeyError(f"contact_id 不存在: {contact_id}")

        messages = self.store.list_messages(contact_id, limit=50)
        stage_before = session.current_stage
        control = self._build_control_agent()
        observed = control.observe_contact(session, messages)
        task = control.decide_next_task(observed)

        if persist_observation:
            self.store.upsert_contact(observed)

        return PreviewNextActionResult(
            contact_id=contact_id,
            stage_before=stage_before,
            stage_after=observed.current_stage,
            memory_summary=observed.memory_summary,
            task=task,
            messages_used=len(messages),
        )
