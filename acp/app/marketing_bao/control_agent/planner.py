"""ControlAgent 决策器。"""

from __future__ import annotations

from acp.app.marketing_bao.control_agent.knowledge import KnowledgeBase
from acp.app.marketing_bao.schemas import ActionTask, AgentRunMode, ContactSession, RuntimeConfig, TaskIntent


class ControlPlanner:
    def __init__(self, knowledge: KnowledgeBase, runtime: RuntimeConfig) -> None:
        self.knowledge = knowledge
        self.runtime = runtime

    def plan_reply(self, session: ContactSession) -> ActionTask | None:
        if session.status.value != "active":
            return None
        text = self.knowledge.render_message(session)
        return ActionTask(
            contact_id=session.contact_id,
            app=session.app,
            intent=TaskIntent.SEND_MESSAGE,
            payload={"text": text, "stage": session.current_stage},
            requires_human_review=self.runtime.run_mode == AgentRunMode.HUMAN_REVIEW,
        )
