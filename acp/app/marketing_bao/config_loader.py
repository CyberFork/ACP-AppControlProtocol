"""营销宝配置加载。"""

from __future__ import annotations

from pathlib import Path

import yaml

from acp.app.marketing_bao.schemas import DriverKind, RuntimeConfig


APP_DIR = Path(__file__).parent
CONFIG_DIR = APP_DIR / "config"


def load_runtime(path: str | Path | None = None) -> tuple[RuntimeConfig, DriverKind]:
    p = Path(path) if path else CONFIG_DIR / "runtime.yaml"
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    raw = data.get("runtime", data)
    driver = DriverKind(raw.pop("driver", DriverKind.TELEGRAM_API.value))
    return RuntimeConfig(**raw), driver
