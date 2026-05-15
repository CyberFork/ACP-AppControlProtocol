"""营销宝规划端后台 FastAPI 应用。

启动：
    python3 -m acp.app.marketing_bao.cli admin-serve

或：
    uvicorn acp.app.marketing_bao.control_agent.admin_backend.app:app --reload --port 8787
"""

from __future__ import annotations

from pathlib import Path

from acp.app.marketing_bao.config_loader import CONFIG_DIR
from acp.app.marketing_bao.control_agent.admin_backend.routes import create_router


def create_app(config_dir: str | Path | None = CONFIG_DIR, db_path: str | Path = "logs/marketing_bao/marketing_bao.sqlite3"):
    try:
        from fastapi import FastAPI
    except ImportError as exc:  # pragma: no cover - optional dep missing
        raise RuntimeError("admin_backend 需要安装 FastAPI：pip install fastapi uvicorn") from exc

    app = FastAPI(
        title="MarketingBao ControlAgent Admin",
        description="营销宝规划端配置后台：阶段链、话术、产品知识、运行策略、客户状态预览。",
        version="0.1.0",
    )
    app.include_router(create_router(config_dir=config_dir or CONFIG_DIR, db_path=db_path))
    return app


app = create_app()
