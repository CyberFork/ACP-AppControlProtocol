"""
反馈评估器（Feedback Evaluator）
操作后验证结果是否符合预期（TVAE 框架）。

三级响应：
  MATCH           → 继续下一步
  MINOR_DEVIATION → 重试当前步骤（最多 3 次）
  MAJOR_DEVIATION → 回溯 + 重新规划
  FAILURE         → 上报用户

MVP 简化实现：
  1. 检查 ActionResult.success（False → FAILURE）
  2. 导航操作：检查 page_state URL/title 是否变化
  3. 关键词检查：expected_output 不为空时做字符串匹配
  4. 数据存在性检查：result.data 非空 → MATCH
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Optional

from acp.schema.elements import PageState
from acp.schema.plan import ActionResult, Step

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FeedbackLevel
# ---------------------------------------------------------------------------


class FeedbackLevel(str, Enum):
    MATCH = "MATCH"
    MINOR_DEVIATION = "MINOR_DEVIATION"
    MAJOR_DEVIATION = "MAJOR_DEVIATION"
    FAILURE = "FAILURE"


# ---------------------------------------------------------------------------
# FeedbackResult
# ---------------------------------------------------------------------------


class FeedbackResult:
    """反馈评估结果"""

    def __init__(self, level: FeedbackLevel, message: str = "") -> None:
        self.level = level
        self.message = message

    @property
    def ok(self) -> bool:
        """是否匹配（无需处理）。"""
        return self.level == FeedbackLevel.MATCH

    @property
    def needs_retry(self) -> bool:
        """是否需要重试（轻微偏差）。"""
        return self.level == FeedbackLevel.MINOR_DEVIATION

    @property
    def needs_replan(self) -> bool:
        """是否需要重规划（重大偏差）。"""
        return self.level == FeedbackLevel.MAJOR_DEVIATION

    @property
    def is_failure(self) -> bool:
        """是否彻底失败（需要上报）。"""
        return self.level == FeedbackLevel.FAILURE

    def __repr__(self) -> str:
        return f"FeedbackResult(level={self.level.value}, message={self.message!r})"


# ---------------------------------------------------------------------------
# 评估辅助函数
# ---------------------------------------------------------------------------


def _check_navigation(
    result: ActionResult,
    prev_page_state: Optional[PageState],
) -> Optional[FeedbackLevel]:
    """检查导航结果：URL/title 是否变化。

    Returns:
        FeedbackLevel 或 None（无法判断时）
    """
    new_state = result.page_state
    if new_state is None:
        return None  # 无页面状态，不能判断

    if prev_page_state is None:
        # 没有前置状态，只要有 page_state 就算成功
        return FeedbackLevel.MATCH

    old_url = prev_page_state.url or ""
    new_url = new_state.url or ""
    old_title = prev_page_state.title or ""
    new_title = new_state.title or ""

    # URL 或 title 任一变化 → 导航成功
    if new_url != old_url or new_title != old_title:
        return FeedbackLevel.MATCH

    # URL 和 title 均未变化 → 轻微偏差（可能页面内导航）
    return FeedbackLevel.MINOR_DEVIATION


def _check_expected_output(
    result: ActionResult,
    expected_output: Optional[str],
) -> Optional[FeedbackLevel]:
    """检查实际输出是否包含预期关键词。

    Returns:
        FeedbackLevel 或 None（无 expected_output 时）
    """
    if not expected_output:
        return None

    # 从 page_state 和 data 中提取可检查文本
    check_texts: list[str] = []
    if result.page_state:
        check_texts.append(result.page_state.title or "")
        check_texts.append(result.page_state.url or "")
    if result.data:
        for v in result.data.values():
            if isinstance(v, str):
                check_texts.append(v)
    if result.elements:
        for elem in result.elements:
            if elem.text:
                check_texts.append(elem.text)

    combined = " ".join(check_texts).lower()
    kw = expected_output.lower()

    if kw in combined:
        return FeedbackLevel.MATCH

    # 如果有 page_state 但不包含关键词，算轻微偏差
    if result.page_state:
        return FeedbackLevel.MINOR_DEVIATION

    return None


# ---------------------------------------------------------------------------
# FeedbackEvaluator
# ---------------------------------------------------------------------------


class FeedbackEvaluator:
    """反馈评估器（MVP 简化版）

    使用示例：
        evaluator = FeedbackEvaluator()
        result = ActionResult(success=True, data={"url": "https://example.com"}, ...)
        step = Step(step_id=1, action="navigate", ...)
        fb = await evaluator.evaluate(result, step)
        if fb.ok:
            # 继续执行
    """

    def __init__(self) -> None:
        # MVP 阶段不依赖 PTGManager
        pass

    async def evaluate(
        self,
        result: ActionResult,
        step: Step,
        prev_page_state: Optional[PageState] = None,
    ) -> FeedbackResult:
        """评估操作结果是否符合预期。

        评估流程（优先级由高到低）：
          1. ActionResult.success == False → FAILURE
          2. 导航操作（navigate）→ 检查 URL/title 变化
          3. expected_output 不为空 → 关键词检查
          4. result.data 非空 → MATCH（有数据即成功）
          5. 默认 MATCH（操作成功无明确预期）

        Args:
            result:          MCP 工具的执行结果
            step:            对应的执行步骤
            prev_page_state: 操作前的页面状态（供导航检查）

        Returns:
            FeedbackResult
        """
        # ---- 1. 基础失败检查 ----
        if not result.success:
            msg = result.error or "操作失败（ActionResult.success=False）"
            logger.debug("FeedbackEvaluator: FAILURE — %s", msg)
            return FeedbackResult(FeedbackLevel.FAILURE, msg)

        # ---- 2. 导航操作：检查页面跳转 ----
        if step.action in ("navigate", "go_to", "open"):
            nav_level = _check_navigation(result, prev_page_state)
            if nav_level is not None:
                logger.debug("FeedbackEvaluator: 导航结果 → %s", nav_level)
                return FeedbackResult(
                    nav_level,
                    f"导航检查：{'URL/title 已变化' if nav_level == FeedbackLevel.MATCH else 'URL/title 未变化'}",
                )

        # ---- 3. 预期关键词检查 ----
        kw_level = _check_expected_output(result, step.expected_output)
        if kw_level is not None:
            logger.debug("FeedbackEvaluator: 关键词检查 → %s", kw_level)
            return FeedbackResult(
                kw_level,
                f"关键词检查：{'找到' if kw_level == FeedbackLevel.MATCH else '未找到'} '{step.expected_output}'",
            )

        # ---- 4. 有数据则视为成功 ----
        if result.data or result.elements:
            return FeedbackResult(FeedbackLevel.MATCH, "操作成功，有返回数据")

        # ---- 5. 默认：操作成功无额外验证 ----
        return FeedbackResult(FeedbackLevel.MATCH, "操作成功")
