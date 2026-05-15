"""Android 模拟器 M0 验证。"""

from __future__ import annotations

from acp.app.marketing_bao.execution_agent.platforms.android.adb_client import ADBClient


def check_android_environment() -> dict:
    adb = ADBClient()
    try:
        devices = adb.devices()
        return {"adb_ok": True, "devices": devices, "ready": bool(devices)}
    except Exception as exc:
        return {"adb_ok": False, "devices": [], "ready": False, "error": str(exc)}
