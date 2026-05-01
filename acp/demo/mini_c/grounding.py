"""
UITARSGrounding：本地 UI-TARS grounding 层。

截图 + target_description → 像素坐标 (x, y)。

设计：UI-TARS 没有专用 "find" action，用 click 指令作为 grounding 信号——
给模型一张截图和 "定位 X" 的指令，取它输出的第一个 click 坐标即可。
截图只发往本地 vLLM（192.168.50.129:8000），绝不出公网。

Mock 模式（UITARS_GROUNDING_MOCK=1）：
  根据 target_description 关键词返回 testenv popup-login 的实测坐标，
  用于在无 vLLM 连接时验证流程逻辑。
"""

from __future__ import annotations

import base64
import logging
import os
import time

import httpx

from acp.demo.mini_b.action_parser import parse as parse_uitars_action

logger = logging.getLogger(__name__)

VLLM_BASE_URL = "http://192.168.50.129:8000"
VLLM_MODEL = "/root/models/UI-TARS-7B-DPO"
DEFAULT_TIMEOUT = 30.0
VIEWPORT_WIDTH = 1280
VIEWPORT_HEIGHT = 800

# 基于 mini_b 实测数据（tc02b step 1-6 坐标均值）
_MOCK_COORDS: dict[str, tuple[int, int]] = {
    "X": (1223, 90),
    "关闭": (1223, 90),
    "弹窗": (1223, 90),
    "close": (1223, 90),
    "modal": (1223, 90),
    "用户名": (640, 430),
    "username": (640, 430),
    "账号": (640, 430),
    "密码": (640, 490),
    "password": (640, 490),
    "登录": (640, 550),
    "login": (640, 550),
    "提交": (640, 550),
    "submit": (640, 550),
    "按钮": (640, 550),
    "button": (640, 550),
}

_GROUNDING_PROMPT = """\
You are a GUI grounding assistant. Your ONLY task is to locate the target element on the screen and output ONE click action with its coordinates.

Target element: {target_description}

Output format (ONE action only, no extra text):
Action: click(start_box='(x,y)')

where x and y are normalized coordinates in range [0, 1000].
"""


class UITARSGrounding:
    """本地 UI-TARS grounding，截图永不出公网。"""

    def __init__(
        self,
        base_url: str = VLLM_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        mock: bool = False,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.chat_url = self.base_url + "/v1/chat/completions"
        self.timeout = timeout
        self.mock = mock or os.getenv("UITARS_GROUNDING_MOCK", "").lower() in ("1", "true")

    def locate(
        self,
        screenshot_bytes: bytes,
        target_description: str,
    ) -> tuple[int, int] | None:
        """定位目标元素，返回像素坐标 (x, y) 或 None（找不到）。

        截图只发往本地 vLLM，不进入任何云端 API。
        """
        if self.mock:
            return self._mock_locate(target_description)

        return self._real_locate(screenshot_bytes, target_description)

    def health_check(self) -> bool:
        if self.mock:
            return True
        try:
            r = httpx.get(self.base_url + "/v1/models", timeout=5.0)
            data = r.json()
            return any(
                "UI-TARS" in m["id"] or "uitars" in m["id"].lower()
                for m in data.get("data", [])
            )
        except Exception as exc:
            logger.warning("UI-TARS grounding health check 失败: %s", exc)
            return False

    def _real_locate(
        self,
        screenshot_bytes: bytes,
        target_description: str,
    ) -> tuple[int, int] | None:
        b64_img = base64.b64encode(screenshot_bytes).decode("utf-8")
        prompt = _GROUNDING_PROMPT.format(target_description=target_description)

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64_img}"},
                    }
                ],
            },
        ]

        payload = {
            "model": VLLM_MODEL,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 100,
        }

        t0 = time.time()
        try:
            resp = httpx.post(self.chat_url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            logger.warning("UI-TARS grounding 请求失败: %s", exc)
            return None

        elapsed = time.time() - t0
        logger.info("UI-TARS grounding 原始输出 (%.2fs): %r", elapsed, raw[:120])

        action = parse_uitars_action(raw)
        if action.action in ("click", "left_single", "left_double") and (action.x > 0 or action.y > 0):
            logger.info("grounding 成功: target=%r coord=(%d, %d)", target_description, action.x, action.y)
            return (action.x, action.y)

        logger.warning("grounding 未找到坐标: target=%r raw=%r", target_description, raw[:80])
        return None

    def _mock_locate(self, target_description: str) -> tuple[int, int] | None:
        desc_lower = target_description.lower()
        for keyword, coord in _MOCK_COORDS.items():
            if keyword.lower() in desc_lower:
                logger.info("[MOCK] grounding: target=%r → coord=%s", target_description, coord)
                return coord
        # fallback：返回页面中心
        logger.info("[MOCK] grounding fallback: target=%r → center (640, 400)", target_description)
        return (640, 400)
