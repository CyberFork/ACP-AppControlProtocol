"""营销宝状态链。

MVP 采用可配置 stages.yaml + 关键词规则：
- 不依赖 LLM 也能跑通自动化闭环。
- 后续可把 judge_stage 替换/增强为 LLM 分类器。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import yaml

from acp.app.marketing_bao.schemas import ChatMessage, ContactSession, StageDefinition, StageID, now_iso


class StageMachine:
    def __init__(self, stages: list[StageDefinition]) -> None:
        if not stages:
            raise ValueError("stages 不能为空")
        self.stages = stages
        self.by_id = {s.id: s for s in stages}
        self.order = [s.id for s in stages]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "StageMachine":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        stages = [StageDefinition(**item) for item in data.get("stages", [])]
        return cls(stages)

    def get(self, stage_id: str) -> StageDefinition:
        return self.by_id[stage_id]

    def next_stage_id(self, stage_id: str) -> str:
        if stage_id not in self.order:
            return self.order[0]
        idx = self.order.index(stage_id)
        return self.order[min(idx + 1, len(self.order) - 1)]

    def judge_stage(self, session: ContactSession, messages: Iterable[ChatMessage]) -> ContactSession:
        """根据最近入站消息关键词推进阶段。

        规则：如果当前阶段或后续阶段的 entry/exit 关键词命中，则推进到最高命中的阶段。
        """
        text = "\n".join(m.text for m in messages if m.direction.value == "inbound").lower()
        if not text:
            return session

        current_idx = self.order.index(session.current_stage) if session.current_stage in self.order else 0
        best_idx = current_idx

        for idx, stage_id in enumerate(self.order):
            if idx < current_idx:
                continue
            stage = self.by_id[stage_id]
            keywords = stage.entry_keywords + stage.exit_keywords
            if any(k.lower() in text for k in keywords):
                best_idx = max(best_idx, idx)

        if best_idx != current_idx:
            session.current_stage = self.order[best_idx]
            stage = self.by_id[session.current_stage]
            session.sub_status = stage.sub_statuses[0] if stage.sub_statuses else "active"
            session.updated_at = now_iso()

        # 简单意向分：阶段越后越高，后续可由 LLM/模型替换
        session.intent_score = round((best_idx + 1) / len(self.order), 2)
        return session

    def ensure_initial(self, session: ContactSession) -> ContactSession:
        if session.current_stage not in self.by_id:
            session.current_stage = StageID.BROAD_REACH.value
        stage = self.by_id[session.current_stage]
        if not session.sub_status and stage.sub_statuses:
            session.sub_status = stage.sub_statuses[0]
        return session
