"""
任务规划器（Task Planner）
将结构化意图拆解为有序的操作步骤，并为每个步骤选择最优执行通道。

工具选择优先级（三层分级）：
  Tier 1 专用 MCP → Tier 2 平台泛用 MCP → Tier 3 视觉兜底 MCP

两种规划模式：
  1. 直接映射模式：intent.sub_tasks → Step（简单指令，无需 LLM）
  2. LLM 辅助模式：复杂任务通过 LLM 生成更细粒度的 step 序列

默认平台映射：
  无法推断平台时，默认使用 web 平台（MVP 阶段）
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Optional

from acp.mcp.registry import MCPRegistry, MCPToolInfo
from acp.schema.intent import Intent, SubTask
from acp.schema.plan import Plan, Step

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 操作 → 平台推断表
# ---------------------------------------------------------------------------

# 常见操作到平台的粗粒度推断
_ACTION_PLATFORM_MAP: dict[str, str] = {
    "navigate": "web",
    "click": "web",
    "type": "web",
    "scroll": "web",
    "screenshot": "web",
    "get_elements": "web",
    "get_page_state": "web",
    "search_images": "web",
    "read_messages": "android",
    "send_message": "android",
    "extract_chat": "android",
}

# App → 平台推断
_APP_PLATFORM_MAP: dict[str, str] = {
    "browser": "web",
    "xiaohongshu": "web",
    "feishu": "web",
    "lark": "web",
    "wechat": "android",
    "weixin": "android",
}

# LLM 辅助规划的 system prompt
_PLANNER_SYSTEM_PROMPT = """\
你是一个 Web/App 自动化任务规划器。给定一个结构化意图，请将其分解为可执行的操作步骤序列。

可用工具（按优先级）：
  - Tier 1：专用 MCP（feishu-mcp、wechat-mcp 等）
  - Tier 2：平台泛用 MCP（web-mcp = Playwright、android-mcp）
  - Tier 3：视觉兜底（vision-mcp）

输出严格 JSON 格式，steps 数组：
[
  {
    "step_id": 1,
    "action": "navigate",
    "tool": "web-mcp",
    "tool_tier": 2,
    "params": {"url": "https://..."},
    "expected_output": "page_loaded",
    "fallback_tool": null
  }
]

规则：
1. 优先用专用 MCP（tier=1），其次泛用（tier=2），最后视觉（tier=3）
2. web 平台用 web-mcp，android 用 android-mcp
3. 每步必须有 action、tool、tool_tier、params
4. 不要输出 JSON 以外的内容
"""


def _infer_platform(sub_task: SubTask) -> str:
    """推断子任务的目标平台。"""
    if sub_task.app:
        plat = _APP_PLATFORM_MAP.get(sub_task.app.lower())
        if plat:
            return plat
    action = sub_task.action.lower()
    return _ACTION_PLATFORM_MAP.get(action, "web")


def _build_step(
    step_id: int,
    sub_task: SubTask,
    tool: MCPToolInfo,
    fallback_tool: Optional[MCPToolInfo] = None,
) -> Step:
    """根据子任务和选定工具构造 Step。"""
    return Step(
        step_id=step_id,
        action=sub_task.action,
        tool=tool.tool_id,
        tool_tier=tool.tier,
        params=dict(sub_task.params),
        expected_output=None,   # MVP 阶段由 feedback 自行推断
        fallback_tool=fallback_tool.tool_id if fallback_tool else None,
    )


# ---------------------------------------------------------------------------
# TaskPlanner
# ---------------------------------------------------------------------------


class TaskPlanner:
    """任务规划器

    依赖 MCPRegistry 进行工具选择；支持直接映射和 LLM 辅助两种规划模式。

    使用示例（无 LLM）：
        registry = MCPRegistry()
        registry.register(web_tool_info)
        planner = TaskPlanner(registry=registry)
        plan = await planner.plan(intent)
    """

    def __init__(
        self,
        registry: Optional[MCPRegistry] = None,
        llm_api_key: str = "",
        llm_base_url: str = "https://api.openai.com/v1",
        llm_model: str = "gpt-4o",
    ) -> None:
        """初始化任务规划器。

        Args:
            registry:     MCP 工具注册中心（为 None 时创建空注册表）
            llm_api_key:  LLM API Key（为空时只用直接映射模式）
            llm_base_url: LLM Base URL
            llm_model:    LLM 模型 ID
        """
        self._registry = registry or MCPRegistry()
        self._llm_api_key = llm_api_key
        self._llm_base_url = llm_base_url
        self._llm_model = llm_model

    @property
    def has_llm(self) -> bool:
        return bool(self._llm_api_key)

    # ---- 公共接口 ----

    async def plan(self, intent: Intent) -> Plan:
        """根据意图生成执行计划。

        Args:
            intent: 结构化意图（IntentParser 的输出）

        Returns:
            包含有序步骤的执行计划

        Raises:
            ValueError: 无法为任何 sub_task 选择工具时
        """
        plan_id = self._generate_plan_id()

        if not intent.sub_tasks:
            # 没有子任务：生成单步 unknown 计划
            return Plan(plan_id=plan_id, steps=[])

        # 检查是否适合 LLM 规划（复杂多步 or 无法直接映射）
        if self.has_llm and len(intent.sub_tasks) > 2:
            try:
                steps = await self._llm_plan(intent)
                if steps:
                    return Plan(plan_id=plan_id, steps=steps)
            except Exception as exc:
                logger.warning("TaskPlanner: LLM 规划失败 (%s)，回退直接映射", exc)

        # 直接映射模式
        steps = self._direct_plan(intent)
        return Plan(plan_id=plan_id, steps=steps)

    # ---- 直接映射模式 ----

    def _direct_plan(self, intent: Intent) -> list[Step]:
        """将 sub_tasks 直接映射为 Steps，工具选择依赖 registry。"""
        steps: list[Step] = []
        for idx, sub_task in enumerate(intent.sub_tasks, start=1):
            platform = _infer_platform(sub_task)
            app = sub_task.app or intent.app or ""
            action = sub_task.action

            # 选主工具
            tool = self._registry.select_tool(app, action, platform)
            if tool is None:
                # 注册表为空时，构造一个占位工具（MVP 保持可用）
                logger.warning(
                    "TaskPlanner: 无法为 app=%s action=%s platform=%s 选择工具，使用 web-mcp 兜底",
                    app, action, platform,
                )
                from acp.mcp.registry import MCPToolInfo as _TI
                tool = _TI(
                    tool_id="web-mcp",
                    tier=2,
                    name="web-mcp（占位）",
                    description="",
                    supported_apps=["*"],
                    capabilities=[],
                    platform="web",
                )

            # 选 fallback 工具（视觉兜底）
            fallback = None
            if tool.tier < 3:
                fallback = self._registry.find(tier=3)

            steps.append(_build_step(idx, sub_task, tool, fallback))

        return steps

    # ---- LLM 辅助规划 ----

    async def _llm_plan(self, intent: Intent) -> list[Step]:
        """调用 LLM 生成更细粒度的 step 序列。"""
        try:
            import httpx
        except ImportError as e:
            raise RuntimeError("LLM 规划需要安装 httpx：pip install httpx") from e

        intent_json = intent.model_dump_json(indent=2)
        url = self._llm_base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self._llm_model,
            "messages": [
                {"role": "system", "content": _PLANNER_SYSTEM_PROMPT},
                {"role": "user", "content": f"意图：\n{intent_json}\n\n请生成步骤序列："},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self._llm_api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        content = data["choices"][0]["message"]["content"]
        raw = json.loads(content)

        # 兼容 {"steps": [...]} 或直接 [...]
        raw_steps = raw.get("steps", raw) if isinstance(raw, dict) else raw

        steps: list[Step] = []
        for item in raw_steps:
            steps.append(Step(
                step_id=int(item.get("step_id", len(steps) + 1)),
                action=item.get("action", "unknown"),
                tool=item.get("tool", "web-mcp"),
                tool_tier=int(item.get("tool_tier", 2)),
                params=item.get("params", {}),
                expected_output=item.get("expected_output"),
                fallback_tool=item.get("fallback_tool"),
            ))

        return steps

    # ---- 工具方法 ----

    def _generate_plan_id(self) -> str:
        return f"plan_{uuid.uuid4().hex[:8]}"
