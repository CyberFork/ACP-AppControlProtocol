"""聊天记忆 MVP。"""

from __future__ import annotations

from acp.app.marketing_bao.schemas import ChatMessage, ContactSession, now_iso


class ChatMemory:
    def update_summary(self, session: ContactSession, messages: list[ChatMessage]) -> ContactSession:
        """规则版记忆摘要。

        后续可替换为 LLM 总结：客户需求、异议、承诺、下一步。
        """
        inbound = [m.text for m in messages if m.direction.value == "inbound"]
        if inbound:
            latest = inbound[-3:]
            session.memory_summary = "最近客户消息：" + " / ".join(latest)
            session.last_interaction_at = messages[-1].timestamp
            session.updated_at = now_iso()
        return session
