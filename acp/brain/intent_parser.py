"""
意图识别器（Intent Parser）
将用户自然语言输入转化为结构化 Intent JSON。

两种工作模式：
  1. 简单模式（无 LLM）：正则/关键词匹配，处理常见简单指令
     - "打开 https://..."  → intent=navigate
     - "点击 X"           → intent=click
     - "在 X 输入 Y"      → intent=type
     - "截图"             → intent=screenshot
  2. LLM 模式：调用 OpenAI 兼容接口，Prompt 转换复杂意图

环境变量：
  ACP_LLM_API_KEY   — API Key
  ACP_LLM_BASE_URL  — Base URL（默认 https://api.openai.com/v1）
  ACP_LLM_MODEL     — 模型 ID（默认 gpt-4o）
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

from acp.schema.intent import Intent, SubTask

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 简单模式：正则规则表
# ---------------------------------------------------------------------------

# 每条规则：(pattern, handler_fn)
# handler_fn(match) -> (intent_str, app, params, sub_tasks)

_URL_RE = re.compile(
    r"^(?:打开|访问|navigate to|open|go to)\s+(https?://\S+)",
    re.IGNORECASE,
)
_CLICK_RE = re.compile(
    r"^(?:点击|单击|click)\s+(.+)$",
    re.IGNORECASE,
)
_TYPE_RE = re.compile(
    r"^(?:在|输入到|type in|在\s+(.+?)\s+(?:中|里)\s*)?(?:输入|键入|type)\s+(.+)$",
    re.IGNORECASE,
)
_TYPE_RE2 = re.compile(
    r"^在(.+?)(?:中|里)\s*(?:输入|键入|type)\s+(.+)$",
    re.IGNORECASE,
)
_SCREENSHOT_RE = re.compile(
    r"^(?:截图|截屏|screenshot|take screenshot)$",
    re.IGNORECASE,
)
_SCROLL_RE = re.compile(
    r"^(?:滚动|scroll)\s*(up|down|left|right|上|下|左|右)?",
    re.IGNORECASE,
)
_NAVIGATE_RE = re.compile(
    r"^(?:导航到|跳转到)\s+(.+)$",
    re.IGNORECASE,
)

_DIRECTION_MAP = {
    "上": "up", "下": "down", "左": "left", "右": "right",
    "up": "up", "down": "down", "left": "left", "right": "right",
}


def _try_simple_parse(user_input: str) -> Optional[Intent]:
    """尝试用简单规则解析输入，成功返回 Intent，否则返回 None。"""
    text = user_input.strip()

    # 1. 打开 URL
    m = _URL_RE.match(text)
    if m:
        url = m.group(1)
        return Intent(
            intent="navigate",
            app="browser",
            params={"url": url},
            sub_tasks=[SubTask(action="navigate", app="browser", params={"url": url})],
        )

    # 2. 导航到（非 URL）
    m = _NAVIGATE_RE.match(text)
    if m:
        target = m.group(1).strip()
        return Intent(
            intent="navigate",
            app="browser",
            params={"target": target},
            sub_tasks=[SubTask(action="navigate", app="browser", params={"target": target})],
        )

    # 3. 截图
    if _SCREENSHOT_RE.match(text):
        return Intent(
            intent="screenshot",
            app="browser",
            params={},
            sub_tasks=[SubTask(action="screenshot", app="browser")],
        )

    # 4. 滚动
    m = _SCROLL_RE.match(text)
    if m:
        raw_dir = (m.group(1) or "down").strip()
        direction = _DIRECTION_MAP.get(raw_dir.lower(), "down")
        return Intent(
            intent="scroll",
            app="browser",
            params={"direction": direction},
            sub_tasks=[SubTask(action="scroll", app="browser", params={"direction": direction})],
        )

    # 5. 点击 X
    m = _CLICK_RE.match(text)
    if m:
        target = m.group(1).strip()
        return Intent(
            intent="click",
            app="browser",
            params={"target": target},
            sub_tasks=[SubTask(action="click", app="browser", params={"target": target})],
        )

    # 6. 在 X 中输入 Y
    m = _TYPE_RE2.match(text)
    if m:
        element = m.group(1).strip()
        content = m.group(2).strip()
        return Intent(
            intent="type",
            app="browser",
            params={"element": element, "text": content},
            sub_tasks=[SubTask(action="type", app="browser", params={"element": element, "text": content})],
        )

    # 7. 输入 Y（无目标元素）
    m = _TYPE_RE.match(text)
    if m:
        # group(1) 可能是 target，group(2) 是文本
        grps = m.groups()
        element = (grps[0] or "").strip() or None
        content = (grps[1] or "").strip()
        p: dict = {"text": content}
        if element:
            p["element"] = element
        return Intent(
            intent="type",
            app="browser",
            params=p,
            sub_tasks=[SubTask(action="type", app="browser", params=p)],
        )

    return None


# ---------------------------------------------------------------------------
# LLM Provider
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
你是一个专业的任务意图解析器，负责将用户自然语言指令转化为结构化 JSON。

请严格按照以下 JSON Schema 输出，不要输出任何其他内容：
{
  "intent": "操作意图，如 navigate/click/type/screenshot/scroll/post_article/search/...",
  "app": "目标应用，如 browser/xiaohongshu/wechat/feishu，不知道填 null",
  "params": { "任意键值对": "描述操作参数" },
  "sub_tasks": [
    {
      "action": "具体操作，如 navigate/click/type/read_messages/search_images/compose/post",
      "app": "目标应用或 null",
      "params": { "键值对": "参数" },
      "filter": "过滤条件（可选）"
    }
  ]
}

规则：
1. 如果是简单单步操作（打开URL/点击/输入），sub_tasks 只有 1 个
2. 如果是复杂多步任务，分解为多个 sub_tasks
3. 不要包含任何 JSON 以外的文字
"""


async def _call_llm(user_input: str, api_key: str, base_url: str, model: str) -> Intent:
    """调用 OpenAI 兼容接口解析意图。"""
    try:
        import httpx
    except ImportError as e:
        raise RuntimeError("LLM 模式需要安装 httpx：pip install httpx") from e

    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()

    content = data["choices"][0]["message"]["content"]
    raw = json.loads(content)

    # 构造 sub_tasks
    sub_tasks = []
    for st in raw.get("sub_tasks", []):
        sub_tasks.append(SubTask(
            action=st.get("action", ""),
            app=st.get("app"),
            params=st.get("params", {}),
            filter=st.get("filter"),
        ))

    return Intent(
        intent=raw.get("intent", "unknown"),
        app=raw.get("app"),
        params=raw.get("params", {}),
        sub_tasks=sub_tasks,
    )


# ---------------------------------------------------------------------------
# IntentParser
# ---------------------------------------------------------------------------


class IntentParser:
    """意图识别器

    优先使用简单规则模式，复杂指令回退到 LLM 模式（需配置 API Key）。

    环境变量：
        ACP_LLM_API_KEY   — LLM API Key（为空时只用简单模式）
        ACP_LLM_BASE_URL  — LLM Base URL（默认 https://api.openai.com/v1）
        ACP_LLM_MODEL     — 模型 ID（默认 gpt-4o）

    使用示例：
        parser = IntentParser()
        intent = await parser.parse("打开 https://example.com")
        # Intent(intent='navigate', app='browser', params={'url': 'https://example.com'}, ...)
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        force_simple: bool = False,
    ) -> None:
        """初始化 IntentParser。

        Args:
            model:        LLM 模型 ID（优先级高于环境变量）
            api_key:      LLM API Key（优先级高于环境变量）
            base_url:     LLM Base URL（优先级高于环境变量）
            force_simple: 强制只用简单规则模式（测试时有用）
        """
        self._api_key = api_key or os.environ.get("ACP_LLM_API_KEY", "")
        self._base_url = base_url or os.environ.get("ACP_LLM_BASE_URL", "https://api.openai.com/v1")
        self._model = model or os.environ.get("ACP_LLM_MODEL", "gpt-4o")
        self._force_simple = force_simple

    @property
    def has_llm(self) -> bool:
        """是否已配置 LLM。"""
        return bool(self._api_key) and not self._force_simple

    async def parse(self, user_input: str) -> Intent:
        """解析用户自然语言，返回结构化意图。

        流程：
          1. 先尝试简单规则匹配
          2. 匹配失败 + 已配置 LLM → 调用 LLM API
          3. 匹配失败 + 无 LLM     → 返回 unknown 意图

        Args:
            user_input: 用户自然语言输入

        Returns:
            结构化意图对象

        Raises:
            RuntimeError: LLM 调用失败且无简单规则匹配时
        """
        text = user_input.strip()
        if not text:
            return Intent(intent="empty", params={}, sub_tasks=[])

        # 优先用简单规则
        simple = _try_simple_parse(text)
        if simple is not None:
            logger.debug("IntentParser: 简单规则匹配成功 → %s", simple.intent)
            return simple

        # 简单规则未匹配
        if self.has_llm:
            logger.debug("IntentParser: 调用 LLM 解析意图")
            try:
                return await _call_llm(text, self._api_key, self._base_url, self._model)
            except Exception as exc:
                logger.warning("IntentParser: LLM 调用失败 (%s)，返回 unknown", exc)
                raise

        # 无 LLM，返回 unknown（降级）
        logger.warning("IntentParser: 无法解析指令（未配置 LLM），返回 unknown")
        return Intent(
            intent="unknown",
            params={"raw_input": text},
            sub_tasks=[SubTask(action="unknown", params={"raw_input": text})],
        )
