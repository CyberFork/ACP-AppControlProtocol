"""飞书 driver 工厂。"""

from __future__ import annotations

from acp.app.marketing_bao.execution_agent.app_drivers.feishu.feishu_api_driver import FeishuAPIDriver
from acp.app.marketing_bao.execution_agent.app_drivers.feishu.schemas import FeishuConfig
from acp.app.marketing_bao.execution_agent.protocol import AppDriver


def build_feishu_driver(dry_run: bool = True) -> AppDriver:
    return FeishuAPIDriver(FeishuConfig(dry_run=dry_run))
