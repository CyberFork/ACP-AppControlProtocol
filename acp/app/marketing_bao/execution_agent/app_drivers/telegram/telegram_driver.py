"""Telegram driver 工厂。"""

from __future__ import annotations

from acp.app.marketing_bao.execution_agent.app_drivers.telegram.schemas import TelegramConfig
from acp.app.marketing_bao.execution_agent.app_drivers.telegram.telegram_android_driver import TelegramAndroidDriver
from acp.app.marketing_bao.execution_agent.app_drivers.telegram.telegram_api_driver import TelegramAPIDriver
from acp.app.marketing_bao.execution_agent.protocol import AppDriver
from acp.app.marketing_bao.schemas import DriverKind


def build_telegram_driver(kind: DriverKind = DriverKind.TELEGRAM_API, dry_run: bool = True) -> AppDriver:
    if kind == DriverKind.TELEGRAM_ANDROID:
        return TelegramAndroidDriver()
    return TelegramAPIDriver(TelegramConfig(dry_run=dry_run))
