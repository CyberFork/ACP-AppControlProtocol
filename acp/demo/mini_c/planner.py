"""
PlannerLLM：云端规划层。

state_text + history → 结构化 JSON 意图（下一步做什么 / 找什么）。

核心原则（D11）：
  - messages 中绝无 image 字段，由 _assert_no_image() 强制检查
  - base_url / api_key / model 全从环境变量读取
  - 截图永不传给此类

环境变量：
  PLANNER_LLM_BASE_URL  (default: https://api.deepseek.com)
  PLANNER_LLM_API_KEY
  PLANNER_LLM_MODEL     (default: deepseek-chat)

Mock 模式（PLANNER_LLM_MOCK=1）：
  为 popup-login 任务返回预定义动作序列，用于无 API key 时验证流程。
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an intelligent GUI automation planner. Your job is to decide the next single action to complete the user's task based on the current page state description and action history.

IMPORTANT:
- You will receive text descriptions of the page, NOT screenshots
- Output ONLY valid JSON, no markdown, no explanation
- Be precise in target_description so a grounding model can locate the element

Output format:
{"intent": "click|type|scroll|done|fail", "target_description": "natural language description of target element", "text": "input text (only for type intent)", "rationale": "brief reason", "is_done": false}

For "done": set is_done=true when you believe the task is fully complete.
For "fail": set is_done=true when you determine the task cannot be completed.
"""

_USER_PROMPT_TEMPLATE = """\
## Task
{instruction}

## Current Page State
{state_text}

## Action History
{history_text}

{grounding_feedback}Decide the next action. Output JSON only.
"""

# Mock 序列：popup-login 的标准 6 步流程
_MOCK_POPUP_LOGIN_SEQUENCE = [
    {"intent": "click", "target_description": "弹窗右上角的 X 关闭按钮", "rationale": "先关闭遮挡登录表单的弹窗", "is_done": False},
    {"intent": "click", "target_description": "用户名输入框", "rationale": "点击用户名输入框准备填写", "is_done": False},
    {"intent": "type", "target_description": "用户名输入框", "text": "demo", "rationale": "填写用户名 demo", "is_done": False},
    {"intent": "click", "target_description": "密码输入框", "rationale": "点击密码输入框准备填写", "is_done": False},
    {"intent": "type", "target_description": "密码输入框", "text": "123456", "rationale": "填写密码 123456", "is_done": False},
    {"intent": "click", "target_description": "登录按钮", "rationale": "点击登录按钮提交表单", "is_done": False},
    {"intent": "done", "target_description": "", "rationale": "已完成登录操作", "is_done": True},
]


@dataclass
class PlanIntent:
    intent: str               # "click" | "type" | "scroll" | "done" | "fail"
    target_description: str   # 自然语言定位描述
    text: str                 # type 动作的输入内容
    rationale: str            # 理由
    is_done: bool             # 任务完成标志


class PlannerLLM:
    """云端规划 LLM，只发文本，截图永不进入请求。"""

    def __init__(self, mock: bool = False) -> None:
        self.mock = mock or os.getenv("PLANNER_LLM_MOCK", "").lower() in ("1", "true")
        self._mock_step = 0

        if not self.mock:
            self.base_url = os.getenv("PLANNER_LLM_BASE_URL", "https://api.deepseek.com").rstrip("/")
            self.api_key = os.getenv("PLANNER_LLM_API_KEY", "")
            self.model = os.getenv("PLANNER_LLM_MODEL", "deepseek-chat")
            # base_url 可以是 https://api.deepseek.com/v1 或 https://open.bigmodel.cn/api/paas/v4
            # 统一追加 /chat/completions（不加 /v1，base_url 已包含版本段）
            self.chat_url = self.base_url + "/chat/completions"
            if not self.api_key:
                logger.warning(
                    "PLANNER_LLM_API_KEY 未设置，规划 LLM 将失败。"
                    "设置环境变量或使用 PLANNER_LLM_MOCK=1"
                )

    def plan(
        self,
        instruction: str,
        state_text: str,
        history: list[str],
        grounding_failed_desc: Optional[str] = None,
    ) -> PlanIntent:
        """根据页面状态文本决策下一步操作。

        截图绝不进入此方法——隐私边界在 state_text 处。
        """
        if self.mock:
            return self._mock_plan()

        history_text = "\n".join(history) if history else "（无历史操作）"
        grounding_feedback = ""
        if grounding_failed_desc:
            grounding_feedback = (
                f"IMPORTANT: 上一步 grounding 找不到元素 \"{grounding_failed_desc}\"，"
                "请换一个更准确的 target_description 重试，或换其他操作方式。\n\n"
            )

        user_prompt = _USER_PROMPT_TEMPLATE.format(
            instruction=instruction,
            state_text=state_text,
            history_text=history_text,
            grounding_feedback=grounding_feedback,
        )

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        # 隐私强制检查：messages 里绝无 image
        _assert_no_image(messages)

        t0 = time.time()
        try:
            resp = httpx.post(
                self.chat_url,
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0.0,
                    "max_tokens": 300,
                },
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=30.0,
            )
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            logger.error("Planner LLM 请求失败: %s", exc)
            return PlanIntent(intent="fail", target_description="", text="", rationale=str(exc), is_done=True)

        elapsed = time.time() - t0
        logger.info("Planner LLM 响应 (%.2fs): %s", elapsed, raw[:200])

        return _parse_intent(raw)

    def reset_mock(self) -> None:
        self._mock_step = 0

    def _mock_plan(self) -> PlanIntent:
        if self._mock_step >= len(_MOCK_POPUP_LOGIN_SEQUENCE):
            d = {"intent": "done", "target_description": "", "rationale": "序列已完成", "is_done": True}
        else:
            d = _MOCK_POPUP_LOGIN_SEQUENCE[self._mock_step]
            self._mock_step += 1
        logger.info("[MOCK] planner step %d: %s", self._mock_step, d)
        return PlanIntent(
            intent=d["intent"],
            target_description=d.get("target_description", ""),
            text=d.get("text", ""),
            rationale=d.get("rationale", ""),
            is_done=d.get("is_done", False),
        )


def _assert_no_image(messages: list[dict]) -> None:
    """D11 隐私强制检查：确保云 LLM 请求中不含任何图像数据。"""
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for item in content:
                if item.get("type") == "image_url":
                    raise AssertionError(
                        "D11 隐私违规：截图不得发往云端 LLM（PlannerLLM）"
                    )
        elif isinstance(content, str):
            # base64 字符串检测（PNG header 的 base64 编码以 iVBOR 开头）
            if "iVBOR" in content or (len(content) > 500 and "base64" in content):
                raise AssertionError(
                    "D11 隐私违规：疑似 base64 图像数据出现在云端 LLM 请求中"
                )


def _parse_intent(raw: str) -> PlanIntent:
    """解析 LLM 输出的 JSON，多层 fallback。"""
    # 去掉 markdown 代码块
    text = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()

    # 尝试直接解析
    try:
        d = json.loads(text)
        return PlanIntent(
            intent=d.get("intent", "fail"),
            target_description=d.get("target_description", ""),
            text=d.get("text", ""),
            rationale=d.get("rationale", ""),
            is_done=bool(d.get("is_done", False)),
        )
    except json.JSONDecodeError:
        pass

    # fallback：regex 提取关键字段
    intent_m = re.search(r'"intent"\s*:\s*"(\w+)"', text)
    desc_m = re.search(r'"target_description"\s*:\s*"([^"]*)"', text)
    text_m = re.search(r'"text"\s*:\s*"([^"]*)"', text)
    done_m = re.search(r'"is_done"\s*:\s*(true|false)', text)

    if intent_m:
        return PlanIntent(
            intent=intent_m.group(1),
            target_description=desc_m.group(1) if desc_m else "",
            text=text_m.group(1) if text_m else "",
            rationale="(regex parsed)",
            is_done=(done_m.group(1) == "true") if done_m else False,
        )

    logger.error("Planner 输出无法解析: %r", raw[:200])
    return PlanIntent(intent="fail", target_description="", text="", rationale=f"parse error: {raw[:100]}", is_done=True)
