"""
LLM 后端：通过 Ollama HTTP API 调用 qwen2.5:3b。

接口：predict(instruction, elements, history) -> Action
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import httpx

from acp.demo.mini_a.prompt_template import render_prompt, render_naive_prompt

if TYPE_CHECKING:
    from acp.demo.mini_a.perception import UIElement

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:3b"
DEFAULT_TIMEOUT = 60.0


@dataclass
class Action:
    action: str          # "click" | "type" | "done" | "fail"
    element_id: int = -1
    text: str = ""
    reason: str = ""
    raw_response: str = ""


class OllamaBackend:
    """Ollama HTTP 后端，调用本地 qwen2.5:3b。"""

    def __init__(
        self,
        base_url: str = OLLAMA_URL,
        model: str = MODEL_NAME,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.timeout = timeout

    def health_check(self) -> bool:
        """检查 Ollama 是否在运行且 model 已拉取。"""
        try:
            url = self.base_url.replace("/api/generate", "/api/tags")
            r = httpx.get(url, timeout=5.0)
            models = [m["name"] for m in r.json().get("models", [])]
            # 宽松匹配：qwen2.5:3b 或 qwen2.5:3b-instruct-q4_k_m 等
            return any(self.model.split(":")[0] in m for m in models)
        except Exception as exc:
            logger.warning("Ollama health check 失败: %s", exc)
            return False

    def predict(
        self,
        instruction: str,
        elements: list["UIElement"],
        history: list[str],
        naive: bool = False,
    ) -> Action:
        """发送 prompt，返回解析后的 Action。
        naive=True 时使用极简 prompt（无 few-shot/状态提示），用于 A2 对照实验。
        """
        if naive:
            system_prompt, user_content = render_naive_prompt(instruction, elements, history)
        else:
            system_prompt, user_content = render_prompt(instruction, elements, history)

        # Ollama generate API：system 通过 system 字段传入
        payload = {
            "model": self.model,
            "system": system_prompt,
            "prompt": user_content,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_predict": 128,
                "stop": ["\n\n", "---"],
            },
        }

        t0 = time.time()
        try:
            resp = httpx.post(self.base_url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.error("Ollama 请求失败: %s", exc)
            return Action(action="fail", reason=f"LLM 请求失败: {exc}")

        raw = data.get("response", "").strip()
        elapsed = time.time() - t0
        logger.debug("LLM 原始响应 (%.1fs): %s", elapsed, raw)

        return self._parse(raw)

    def _parse(self, raw: str) -> Action:
        """从原始文本中提取 JSON Action。"""
        # 尝试直接解析
        text = raw.strip()

        # 去除 markdown 代码块
        text = re.sub(r"```(?:json)?\s*", "", text)
        text = re.sub(r"```\s*", "", text)

        # 提取第一个 JSON 对象
        match = re.search(r"\{[^{}]+\}", text, re.DOTALL)
        if not match:
            logger.warning("无法从 LLM 响应中提取 JSON: %r", raw[:200])
            return Action(action="fail", reason=f"无法解析响应: {raw[:100]}", raw_response=raw)

        try:
            obj = json.loads(match.group())
        except json.JSONDecodeError as exc:
            logger.warning("JSON 解析失败: %s | 原文: %r", exc, match.group()[:200])
            return Action(action="fail", reason=f"JSON 解析错误: {exc}", raw_response=raw)

        action_type = obj.get("action", "fail")
        if action_type not in ("click", "type", "done", "fail"):
            action_type = "fail"

        return Action(
            action=action_type,
            element_id=int(obj.get("element_id", -1)),
            text=str(obj.get("text", "")),
            reason=str(obj.get("reason", "")),
            raw_response=raw,
        )
