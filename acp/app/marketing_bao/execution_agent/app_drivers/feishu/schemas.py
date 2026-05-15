"""飞书 driver 配置。"""

from __future__ import annotations

from pydantic import BaseModel


class FeishuConfig(BaseModel):
    app_id_env: str = "FEISHU_APP_ID"
    app_secret_env: str = "FEISHU_APP_SECRET"
    default_receive_id_env: str = "FEISHU_DEFAULT_RECEIVE_ID"
    default_receive_id_type_env: str = "FEISHU_DEFAULT_RECEIVE_ID_TYPE"
    base_url: str = "https://open.feishu.cn/open-apis"
    dry_run: bool = True
