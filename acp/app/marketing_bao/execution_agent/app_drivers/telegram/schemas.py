"""Telegram driver 配置。"""

from __future__ import annotations

from pydantic import BaseModel


class TelegramConfig(BaseModel):
    api_id_env: str = "TELEGRAM_API_ID"
    api_hash_env: str = "TELEGRAM_API_HASH"
    phone_env: str = "TELEGRAM_PHONE"
    session_name: str = "marketing_bao_telegram"
    dry_run: bool = True
