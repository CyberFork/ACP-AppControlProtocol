"""
执行调度器（Executor Dispatcher）
按 step 序列逐步执行，每步完成后触发反馈评估。

执行逻辑（每步）：
  1. 调用主工具 execute(method, params)
  2. FeedbackEvaluator 评估结果
     - MATCH          → 继续下一步
     - MINOR_DEVIATION → 重试（最多 MAX_RETRY 次）
     - MAJOR_DEVIATION → 中止计划（返回已收集结果）
     - FAILURE         → 尝试 fallback_tool，仍失败则中止
  3. 记录执行日志

日志格式：
  [STEP {id}] {action} via {tool} → {MATCH|FAILURE|...}
"""

from __future__ import annotations

import logging
from typing import Optional

from acp.brain.feedback import FeedbackEvaluator, FeedbackLevel, FeedbackResult
from acp.mcp.protocol import MCPTool
from acp.schema.elements import PageState
from acp.schema.plan import ActionResult, Plan, Step

logger = logging.getLogger(__name__)

MAX_RETRY = 3  # 轻微偏差最多重试次数


# ---------------------------------------------------------------------------
# ExecutorDispatcher
# ---------------------------------------------------------------------------


class ExecutorDispatcher:
    """执行调度器

    使用示例：
        tools = {"web-mcp": web_mcp_instance}
        executor = ExecutorDispatcher(tools=tools)
        results = await executor.execute(plan)
    """

    def __init__(
        self,
        tools: Optional[dict[str, MCPTool]] = None,
        evaluator: Optional[FeedbackEvaluator] = None,
    ) -> None:
        """初始化执行调度器。

        Args:
            tools:     工具字典 {tool_id: MCPTool 实例}（为 None 时使用空字典）
            evaluator: 反馈评估器（为 None 时自动创建）
        """
        self._tools: dict[str, MCPTool] = tools or {}
        self._evaluator = evaluator or FeedbackEvaluator()

    def register_tool(self, tool_id: str, tool: MCPTool) -> None:
        """动态注册 MCP 工具。"""
        self._tools[tool_id] = tool

    # ---- 主执行入口 ----

    async def execute(self, plan: Plan) -> list[ActionResult]:
        """按计划逐步执行，返回所有步骤的结果列表。

        Args:
            plan: 执行计划（TaskPlanner 输出）

        Returns:
            每个步骤的执行结果列表（仅成功执行的步骤）

        Note:
            遇到 MAJOR_DEVIATION 或彻底失败时，提前中止并返回已收集结果。
        """
        results: list[ActionResult] = []
        prev_page_state: Optional[PageState] = None

        for step in plan.steps:
            logger.info("[STEP %d] 执行 action=%s via tool=%s", step.step_id, step.action, step.tool)

            result = await self._execute_step_with_retry(step, prev_page_state)
            results.append(result)

            if not result.success:
                logger.warning(
                    "[STEP %d] 步骤最终失败，中止计划。error=%s",
                    step.step_id, result.error,
                )
                break

            # 更新页面状态供下一步使用
            if result.page_state:
                prev_page_state = result.page_state

        return results

    # ---- 带重试和 fallback 的步骤执行 ----

    async def _execute_step_with_retry(
        self,
        step: Step,
        prev_page_state: Optional[PageState],
    ) -> ActionResult:
        """执行单步（含重试和 fallback）。

        重试策略：
          - MINOR_DEVIATION → 同工具重试，最多 MAX_RETRY 次
          - 主工具 FAILURE  → 尝试 fallback_tool
          - fallback 也失败 → 返回失败结果
        """
        # ---- 主工具执行（含重试）----
        result: Optional[ActionResult] = None
        feedback: Optional[FeedbackResult] = None

        for attempt in range(1, MAX_RETRY + 1):
            result = await self._call_tool(step.tool, step.action, step.params)
            feedback = await self._evaluator.evaluate(result, step, prev_page_state)

            log_msg = "[STEP %d] attempt=%d tool=%s → %s"
            logger.debug(log_msg, step.step_id, attempt, step.tool, feedback.level.value)

            if feedback.ok:
                return result  # MATCH，直接返回

            if feedback.needs_retry and attempt < MAX_RETRY:
                logger.info(
                    "[STEP %d] MINOR_DEVIATION，重试 %d/%d…",
                    step.step_id, attempt, MAX_RETRY,
                )
                continue  # 重试

            # MAJOR_DEVIATION 或重试耗尽 → 退出重试循环
            break

        # ---- fallback 工具 ----
        assert result is not None
        assert feedback is not None

        if not result.success and step.fallback_tool:
            logger.info(
                "[STEP %d] 主工具失败（%s），尝试 fallback_tool=%s",
                step.step_id, step.tool, step.fallback_tool,
            )
            fallback_result = await self._call_tool(
                step.fallback_tool, step.action, step.params
            )
            fallback_feedback = await self._evaluator.evaluate(
                fallback_result, step, prev_page_state
            )
            logger.debug(
                "[STEP %d] fallback 结果 → %s",
                step.step_id, fallback_feedback.level.value,
            )
            if fallback_feedback.level != FeedbackLevel.FAILURE:
                return fallback_result

        if feedback.needs_replan:
            logger.warning(
                "[STEP %d] MAJOR_DEVIATION，中止当前步骤。message=%s",
                step.step_id, feedback.message,
            )
            # 返回失败结果（success=False），由 execute() 中止计划
            return ActionResult(
                success=False,
                error=f"MAJOR_DEVIATION: {feedback.message}",
                page_state=result.page_state,
            )

        return result  # 可能是成功也可能是失败

    # ---- 工具调用 ----

    async def _call_tool(
        self,
        tool_id: str,
        method: str,
        params: dict,
    ) -> ActionResult:
        """调用指定 MCP 工具。

        工具不存在时返回失败 ActionResult（不抛异常）。
        """
        tool = self._tools.get(tool_id)
        if tool is None:
            msg = f"工具 '{tool_id}' 未注册，可用工具: {list(self._tools.keys())}"
            logger.error(msg)
            return ActionResult(success=False, error=msg)

        try:
            return await tool.execute(method, params)
        except Exception as exc:
            msg = f"工具 {tool_id}.execute({method}) 抛出异常: {exc}"
            logger.exception(msg)
            return ActionResult(success=False, error=msg)
