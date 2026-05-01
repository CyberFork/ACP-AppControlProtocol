"""
MiniCLoop：Demo C 混合架构 V-L-A 主循环。

流程：
  1. StateDescriber 提取页面文本状态（本地）
  2. PlannerLLM 决策下一步意图（云端，纯文本）
  3. UITARSGrounding 定位目标元素坐标（本地 vLLM）
  4. WebAdapter 执行动作
  5. JS DOM 检查真实状态 → 更新 history → 回到 1

D11 隐私不变量：截图只在步骤 3 使用，绝不进入步骤 2（PlannerLLM）。
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from acp.demo.mini_c.grounding import UITARSGrounding
from acp.demo.mini_c.planner import PlannerLLM
from acp.demo.mini_c.state_describer import StateDescriber

logger = logging.getLogger(__name__)

MAX_STEPS = 15
MAX_GROUNDING_RETRIES = 2   # grounding 连续失败 N 次后放弃
VIEWPORT_WIDTH = 1280
VIEWPORT_HEIGHT = 800


@dataclass
class CStepLog:
    step: int
    state_text: str           # 传给云 LLM 的文本（隐私审计用）
    planner_intent: dict      # PlannerLLM 输出的 JSON
    grounding_query: str      # target_description
    grounding_coord: Optional[tuple[int, int]]  # (x, y) 或 None
    exec_result: str          # "ok" | "error: ..." | "skipped"
    elapsed: float            # 总步耗时
    elapsed_planner: float    # 云 API RTT
    elapsed_grounding: float  # vLLM RTT


@dataclass
class LoopResult:
    success: bool
    steps: list[CStepLog] = field(default_factory=list)
    message: str = ""
    total_elapsed: float = 0.0


class MiniCLoop:
    """D11 混合架构主控：云端规划 + 本地 grounding + JS DOM 验证。"""

    def __init__(
        self,
        planner: PlannerLLM,
        grounder: UITARSGrounding,
        describer: Optional[StateDescriber] = None,
        log_dir: Optional[Path] = None,
    ) -> None:
        self.planner = planner
        self.grounder = grounder
        self.describer = describer or StateDescriber()
        self.log_dir = log_dir or Path("logs/mini_c")
        self.log_dir.mkdir(parents=True, exist_ok=True)

    async def run(
        self,
        instruction: str,
        start_url: str,
        success_check: str = "js:login-success.visible",
        run_id: str = "",
        max_steps: int = MAX_STEPS,
    ) -> LoopResult:
        """执行一次完整的混合架构 V-L-A 循环。"""
        from acp.adapters.web_adapter import WebAdapter
        from acp.eval.runner import check_success

        if not run_id:
            run_id = str(int(time.time()))

        run_log_dir = self.log_dir / run_id
        run_log_dir.mkdir(parents=True, exist_ok=True)

        # mock 模式下重置 planner 的步骤计数器
        if self.planner.mock:
            self.planner.reset_mock()

        result = LoopResult(success=False)
        history: list[str] = []
        t_start = time.time()
        grounding_fail_streak = 0
        last_grounding_failed_desc: Optional[str] = None

        async with WebAdapter(headless=True) as adapter:
            nav = await adapter.navigate(start_url)
            if not nav.success:
                result.message = f"导航失败: {nav.error}"
                return result

            for step_idx in range(max_steps):
                t_step = time.time()

                # 1. 提取页面状态文本
                state_text = await self.describer.describe(adapter._page)
                logger.info("[step %d] 页面状态:\n%s", step_idx + 1, state_text)

                # 2. 云端规划
                t_plan = time.time()
                intent = self.planner.plan(
                    instruction=instruction,
                    state_text=state_text,
                    history=history,
                    grounding_failed_desc=last_grounding_failed_desc,
                )
                elapsed_planner = time.time() - t_plan
                last_grounding_failed_desc = None

                logger.info(
                    "[step %d] Planner: intent=%s target=%r is_done=%s (%.2fs)",
                    step_idx + 1, intent.intent, intent.target_description, intent.is_done, elapsed_planner,
                )

                # 3. 判断失败（fail 优先于 done）
                if intent.intent == "fail":
                    result.message = f"planner:fail → {intent.rationale}"
                    result.steps.append(CStepLog(
                        step=step_idx, state_text=state_text,
                        planner_intent={"intent": "fail", "rationale": intent.rationale},
                        grounding_query="", grounding_coord=None,
                        exec_result="fail",
                        elapsed=round(time.time() - t_step, 2),
                        elapsed_planner=round(elapsed_planner, 2),
                        elapsed_grounding=0.0,
                    ))
                    break

                # 4. 判断完成
                if intent.is_done or intent.intent == "done":
                    js_ok = await check_success(adapter._page, success_check)
                    result.success = js_ok
                    result.message = f"planner:done → js_check:{js_ok}"
                    step_log = CStepLog(
                        step=step_idx,
                        state_text=state_text,
                        planner_intent={"intent": intent.intent, "rationale": intent.rationale, "is_done": True},
                        grounding_query="",
                        grounding_coord=None,
                        exec_result="done",
                        elapsed=round(time.time() - t_step, 2),
                        elapsed_planner=round(elapsed_planner, 2),
                        elapsed_grounding=0.0,
                    )
                    result.steps.append(step_log)
                    history.append(f"[step{step_idx+1}] done → js_success={js_ok}")
                    break

                # 5. type 动作：不需要 grounding（已有 target，直接 focus + type）
                if intent.intent == "type" and intent.text:
                    t_grounding = time.time()
                    screenshot_bytes = await adapter.screenshot()
                    coord = self.grounder.locate(screenshot_bytes, intent.target_description)
                    elapsed_grounding = time.time() - t_grounding

                    if coord is None:
                        grounding_fail_streak += 1
                        last_grounding_failed_desc = intent.target_description
                        logger.warning(
                            "[step %d] grounding 找不到 %r (streak=%d)",
                            step_idx + 1, intent.target_description, grounding_fail_streak,
                        )
                        if grounding_fail_streak >= MAX_GROUNDING_RETRIES:
                            result.message = f"grounding 连续 {MAX_GROUNDING_RETRIES} 次失败"
                            break
                        history.append(f"[step{step_idx+1}] grounding失败: {intent.target_description}")
                        result.steps.append(CStepLog(
                            step=step_idx, state_text=state_text,
                            planner_intent={"intent": intent.intent, "target_description": intent.target_description},
                            grounding_query=intent.target_description, grounding_coord=None,
                            exec_result="grounding_failed",
                            elapsed=round(time.time() - t_step, 2),
                            elapsed_planner=round(elapsed_planner, 2),
                            elapsed_grounding=round(elapsed_grounding, 2),
                        ))
                        continue

                    grounding_fail_streak = 0
                    # 先 click 定焦，再 type
                    exec_result = await self._execute_click(adapter, coord)
                    if not exec_result.startswith("error"):
                        exec_result = await self._execute_type(adapter, intent.text)
                    elapsed_total = time.time() - t_step

                    log_entry = f"[step{step_idx+1}] type@{coord} text={intent.text!r:20} → {exec_result}"
                    history.append(log_entry)
                    logger.info(log_entry)

                    result.steps.append(CStepLog(
                        step=step_idx, state_text=state_text,
                        planner_intent={"intent": "type", "target_description": intent.target_description, "text": intent.text},
                        grounding_query=intent.target_description, grounding_coord=coord,
                        exec_result=exec_result,
                        elapsed=round(elapsed_total, 2),
                        elapsed_planner=round(elapsed_planner, 2),
                        elapsed_grounding=round(elapsed_grounding, 2),
                    ))
                    continue

                # 6. click / scroll：需要 grounding
                t_grounding = time.time()
                screenshot_bytes = await adapter.screenshot()
                coord = self.grounder.locate(screenshot_bytes, intent.target_description)
                elapsed_grounding = time.time() - t_grounding

                if coord is None:
                    grounding_fail_streak += 1
                    last_grounding_failed_desc = intent.target_description
                    logger.warning(
                        "[step %d] grounding 找不到 %r (streak=%d)",
                        step_idx + 1, intent.target_description, grounding_fail_streak,
                    )
                    if grounding_fail_streak >= MAX_GROUNDING_RETRIES:
                        result.message = f"grounding 连续 {MAX_GROUNDING_RETRIES} 次失败"
                        break
                    history.append(f"[step{step_idx+1}] grounding失败: {intent.target_description}")
                    result.steps.append(CStepLog(
                        step=step_idx, state_text=state_text,
                        planner_intent={"intent": intent.intent, "target_description": intent.target_description},
                        grounding_query=intent.target_description, grounding_coord=None,
                        exec_result="grounding_failed",
                        elapsed=round(time.time() - t_step, 2),
                        elapsed_planner=round(elapsed_planner, 2),
                        elapsed_grounding=round(elapsed_grounding, 2),
                    ))
                    continue

                grounding_fail_streak = 0
                exec_result = await self._execute_click(adapter, coord)
                elapsed_total = time.time() - t_step

                log_entry = f"[step{step_idx+1}] {intent.intent}@{coord} target={intent.target_description!r:30} → {exec_result} ({elapsed_total:.2f}s)"
                history.append(log_entry)
                logger.info(log_entry)

                result.steps.append(CStepLog(
                    step=step_idx, state_text=state_text,
                    planner_intent={"intent": intent.intent, "target_description": intent.target_description},
                    grounding_query=intent.target_description, grounding_coord=coord,
                    exec_result=exec_result,
                    elapsed=round(elapsed_total, 2),
                    elapsed_planner=round(elapsed_planner, 2),
                    elapsed_grounding=round(elapsed_grounding, 2),
                ))

            else:
                # 超最大步数：做最终 JS 检查
                js_ok = await check_success(adapter._page, success_check)
                result.success = js_ok
                result.message = f"超过最大步数 {max_steps}，js_success={js_ok}"

        result.total_elapsed = time.time() - t_start
        self._save_run_log(run_log_dir, run_id, instruction, start_url, result)
        return result

    @staticmethod
    async def _execute_click(adapter, coord: tuple[int, int]) -> str:
        try:
            x, y = coord
            await adapter._page.mouse.click(x, y)
            await asyncio.sleep(0.5)
            return "ok"
        except Exception as exc:
            return f"error: {exc}"

    @staticmethod
    async def _execute_type(adapter, text: str) -> str:
        try:
            clean = text.replace("\\n", "\n")
            await adapter._page.keyboard.type(clean, delay=30)
            await asyncio.sleep(0.3)
            return "ok"
        except Exception as exc:
            return f"error: {exc}"

    def _save_run_log(
        self,
        run_log_dir: Path,
        run_id: str,
        instruction: str,
        url: str,
        result: LoopResult,
    ) -> None:
        log_path = run_log_dir / "run_summary.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "run_id": run_id,
                    "instruction": instruction,
                    "start_url": url,
                    "success": result.success,
                    "message": result.message,
                    "total_elapsed": round(result.total_elapsed, 2),
                    "steps": [
                        {
                            "step": s.step,
                            "state_text_preview": s.state_text[:200],
                            "planner_intent": s.planner_intent,
                            "grounding_query": s.grounding_query,
                            "grounding_coord": list(s.grounding_coord) if s.grounding_coord else None,
                            "exec_result": s.exec_result,
                            "elapsed": s.elapsed,
                            "elapsed_planner": s.elapsed_planner,
                            "elapsed_grounding": s.elapsed_grounding,
                        }
                        for s in result.steps
                    ],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        logger.info("Run 日志已保存: %s", log_path)
