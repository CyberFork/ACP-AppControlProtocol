"""营销宝 ControlAgent：业务状态、记忆、策略、任务编排。"""

from __future__ import annotations

from acp.app.marketing_bao.control_agent.knowledge import KnowledgeBase
from acp.app.marketing_bao.control_agent.memory import ChatMemory
from acp.app.marketing_bao.control_agent.planner import ControlPlanner
from acp.app.marketing_bao.control_agent.state_machine import StageMachine
from acp.app.marketing_bao.schemas import ActionTask, ChatMessage, ContactSession, RuntimeConfig


class ControlAgent:
    def __init__(self, state_machine: StageMachine, knowledge: KnowledgeBase, runtime: RuntimeConfig) -> None:
        self.state_machine = state_machine
        self.knowledge = knowledge
        self.runtime = runtime
        self.memory = ChatMemory()
        self.planner = ControlPlanner(knowledge, runtime)

    def observe_contact(self, session: ContactSession, messages: list[ChatMessage]) -> ContactSession:
        session = self.state_machine.ensure_initial(session)
        session = self.memory.update_summary(session, messages)
        session = self.state_machine.judge_stage(session, messages)
        return session

    def decide_next_task(self, session: ContactSession) -> ActionTask | None:
        return self.planner.plan_reply(session)
