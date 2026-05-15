"""轻量知识库/话术库。

MVP 使用 YAML playbook + product.yaml，不引入向量库。
"""

from __future__ import annotations

from pathlib import Path

import yaml

from acp.app.marketing_bao.schemas import ContactSession, PlaybookRule


class KnowledgeBase:
    def __init__(self, playbooks: list[PlaybookRule], product: dict) -> None:
        self.playbooks = {p.stage_id: p for p in playbooks}
        self.product = product

    @classmethod
    def from_yaml(cls, playbook_path: str | Path, product_path: str | Path) -> "KnowledgeBase":
        playbook_data = yaml.safe_load(Path(playbook_path).read_text(encoding="utf-8")) or {}
        product_data = yaml.safe_load(Path(product_path).read_text(encoding="utf-8")) or {}
        playbooks = [PlaybookRule(**item) for item in playbook_data.get("playbooks", [])]
        return cls(playbooks, product_data.get("product", {}))

    def get_playbook(self, stage_id: str) -> PlaybookRule | None:
        return self.playbooks.get(stage_id)

    def render_message(self, session: ContactSession) -> str:
        playbook = self.get_playbook(session.current_stage)
        product_name = self.product.get("name", "我们的服务")
        if not playbook or not playbook.templates:
            return f"你好，我这边主要做{product_name}，如果你有相关需求可以简单聊聊。"

        # MVP：取第一条模板。后续可改为 LLM 根据 memory/messages 生成。
        return playbook.templates[0].format(
            product_name=product_name,
            contact_name=session.display_name or "你",
        )
