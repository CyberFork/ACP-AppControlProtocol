"""ADB 客户端最小封装，用于 M0 模拟器环境验证。"""

from __future__ import annotations

import subprocess


class ADBClient:
    def __init__(self, adb_path: str = "adb") -> None:
        self.adb_path = adb_path

    def run(self, *args: str, timeout: int = 15) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [self.adb_path, *args],
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )

    def devices(self) -> list[str]:
        result = self.run("devices")
        lines = result.stdout.strip().splitlines()[1:]
        return [line.split()[0] for line in lines if "device" in line]

    def shell(self, *args: str) -> str:
        result = self.run("shell", *args)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip())
        return result.stdout
