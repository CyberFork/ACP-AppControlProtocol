"""
miniLoop：V → L → A 闭环主循环。

截图 → OmniParser 提元素 → 渲染 prompt → Qwen2.5-3B 选动作 → 执行 → 判断完成
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from PIL import Image

from acp.demo.mini_a.llm_backend import Action, OllamaBackend
from acp.demo.mini_a.perception import OmniPerception, UIElement

logger = logging.getLogger(__name__)

MAX_STEPS = 10         # 超过则强制终止
STEP_TIMEOUT = 30.0    # 单步最长等待（秒，含感知+LLM+执行）


@dataclass
class StepLog:
    step: int
    screenshot_path: str
    elements: list[dict]          # UIElement 序列化
    llm_prompt_user: str
    llm_response: str
    action: dict
    result: str                   # "ok" | "error: ..." | "done" | "fail"
    elapsed: float


@dataclass
class LoopResult:
    success: bool
    steps: list[StepLog] = field(default_factory=list)
    message: str = ""
    total_elapsed: float = 0.0


class MiniLoop:
    """V-L-A 闭环主控，管理 WebAdapter + Perception + LLM 的协作。"""

    def __init__(
        self,
        perception: OmniPerception,
        llm: OllamaBackend,
        log_dir: Optional[Path] = None,
    ) -> None:
        self.perception = perception
        self.llm = llm
        self.log_dir = log_dir or Path("logs/mini_a")
        self.log_dir.mkdir(parents=True, exist_ok=True)

    async def run_with_adapter(
        self,
        adapter,
        instruction: str,
        run_id: str = "",
        max_steps: int = MAX_STEPS,
        succeed_on_first_action: bool = True,
        forced_action: str = "",
        forced_text: str = "",
        label_keyword: str = "",
    ) -> LoopResult:
        """在已有的 adapter（已导航页面）上执行单一子任务。

        succeed_on_first_action=True：第一次成功执行 click/type 后即返回 success，
        适用于子任务分解模式（每个子任务只需完成一个操作）。
        """
        if not run_id:
            run_id = f"{int(time.time())}"
        run_log_dir = self.log_dir / run_id
        run_log_dir.mkdir(parents=True, exist_ok=True)

        result = LoopResult(success=False)
        history: list[str] = []
        t_start = time.time()

        for step_idx in range(max_steps):
            step_log, done, failed = await self._step(
                adapter, instruction, history, step_idx, run_log_dir,
                forced_action=forced_action, forced_text=forced_text,
                label_keyword=label_keyword,
            )
            result.steps.append(step_log)
            result.total_elapsed = time.time() - t_start

            action = step_log.action
            elem_id = action.get("element_id", -1)
            elem_label = ""
            if elem_id >= 0 and step_log.elements:
                found = next((e for e in step_log.elements if e.get("idx") == elem_id), None)
                if found:
                    elem_label = f'"{found.get("label","")[:20]}"'
            action_desc = (
                f"[step{step_idx+1}] {action.get('action','?')} "
                f"elem={elem_id}{elem_label} → {step_log.result}"
            )
            history.append(action_desc)

            if done:
                result.success = True
                result.message = action.get("reason", "done")
                break
            if failed:
                result.message = action.get("reason", "fail")
                break

            # 子任务模式：首次成功操作即视为完成
            if succeed_on_first_action and step_log.result == "ok":
                result.success = True
                result.message = f"操作成功: {action.get('action')} elem={elem_id}"
                break

        self._save_run_log(run_log_dir, run_id, instruction, "", result)
        return result

    async def run(
        self,
        instruction: str,
        start_url: str,
        run_id: str = "",
        naive: bool = False,
    ) -> LoopResult:
        """执行一次完整的 V-L-A 循环。

        Args:
            instruction: 用户自然语言指令
            start_url:   初始页面 URL
            run_id:      日志前缀（留空则用时间戳）
            naive:       True = A2 对照模式，使用极简 prompt，无 few-shot/状态提示
        """
        from acp.adapters.web_adapter import WebAdapter

        if not run_id:
            run_id = f"{int(time.time())}"

        run_log_dir = self.log_dir / run_id
        run_log_dir.mkdir(parents=True, exist_ok=True)

        result = LoopResult(success=False)
        history: list[str] = []
        t_start = time.time()

        async with WebAdapter(headless=True) as adapter:
            nav = await adapter.navigate(start_url)
            if not nav.success:
                result.message = f"导航失败: {nav.error}"
                return result

            for step_idx in range(MAX_STEPS):
                step_log, done, failed = await self._step(
                    adapter, instruction, history, step_idx, run_log_dir,
                    naive=naive,
                )
                result.steps.append(step_log)
                result.total_elapsed = time.time() - t_start

                # 更新历史
                # 历史描述里包含元素 label，帮助 LLM 追踪状态
                action = step_log.action
                elem_id = action.get("element_id", -1)
                elem_label = ""
                if elem_id >= 0 and step_log.elements:
                    found = next((e for e in step_log.elements if e.get("idx") == elem_id), None)
                    if found:
                        elem_label = f'"{found.get("label","")[:20]}"'
                action_desc = (
                    f"[step{step_idx+1}] {action.get('action','?')} "
                    f"elem={elem_id}{elem_label} "
                    f"→ {step_log.result}"
                )
                history.append(action_desc)

                if done:
                    result.success = True
                    result.message = step_log.action.get("reason", "done")
                    break
                if failed:
                    result.message = step_log.action.get("reason", "fail")
                    break

            else:
                result.message = f"超过最大步数 {MAX_STEPS}"

        self._save_run_log(run_log_dir, run_id, instruction, start_url, result)
        return result

    def _save_run_log(
        self, run_log_dir: Path, run_id: str, instruction: str, url: str, result: LoopResult
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
                    "steps": [self._step_log_to_dict(s) for s in result.steps],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        logger.info("Run 日志已保存: %s", log_path)

    async def _step(
        self,
        adapter,
        instruction: str,
        history: list[str],
        step_idx: int,
        run_log_dir: Path,
        forced_action: str = "",
        forced_text: str = "",
        label_keyword: str = "",
        naive: bool = False,
    ) -> tuple[StepLog, bool, bool]:
        """执行一步：截图 → 感知 → LLM → 执行，返回 (log, done, failed)。"""
        t0 = time.time()

        # ── 截图 ────────────────────────────────────────────────────────────
        screenshot_bytes = await adapter.screenshot()
        screenshot_path = run_log_dir / f"step_{step_idx:02d}.png"
        screenshot_path.write_bytes(screenshot_bytes)

        # ── 感知 ────────────────────────────────────────────────────────────
        image = Image.open(screenshot_path)
        elements = self.perception.detect(image)
        logger.info(
            "[step %d] 感知到 %d 个元素", step_idx + 1, len(elements)
        )

        # ── 页面状态描述（辅助模式才注入，naive 模式不注入）───────────────────
        if naive:
            history_with_state = list(history)
        else:
            page_state = await adapter.get_page_state()
            state_hint = f"[当前页面: {page_state.title} | 可见元素数={len(elements)}]"
            history_with_state = history + [state_hint] if history else [state_hint]

        # ── LLM ─────────────────────────────────────────────────────────────
        action: Action = self.llm.predict(
            instruction, elements, history_with_state, naive=naive
        )

        # 强制覆盖 action 类型和文本（子任务分解模式用）
        if forced_action:
            action.action = forced_action
            if forced_text:
                action.text = forced_text

        # label_keyword fallback：若 LLM 选的元素 label 不含关键词，用关键词重新匹配
        # 语法："==xxx" 精确匹配，"xxx" 包含匹配
        if label_keyword and elements:
            exact = label_keyword.startswith("==")
            kw = label_keyword[2:] if exact else label_keyword
            llm_elem = next((e for e in elements if e.idx == action.element_id), None)
            llm_label = (llm_elem.label or "") if llm_elem else ""
            llm_match = (llm_label.strip() == kw) if exact else (kw in llm_label)
            if not llm_match:
                def _matches(e):
                    lbl = (e.label or "").strip()
                    return (lbl == kw) if exact else (kw in lbl)
                matched = next((e for e in elements if _matches(e)), None)
                if matched:
                    logger.info(
                        "[step %d] label_keyword fallback: LLM选elem=%d(%r)，改选elem=%d(%r)",
                        step_idx + 1, action.element_id, llm_label[:20],
                        matched.idx, (matched.label or "")[:20]
                    )
                    action.element_id = matched.idx

        logger.info(
            "[step %d] LLM → action=%s elem=%d reason=%s",
            step_idx + 1, action.action, action.element_id, action.reason
        )

        # done / fail 直接返回
        if action.action == "done":
            log = StepLog(
                step=step_idx,
                screenshot_path=str(screenshot_path),
                elements=[self._elem_to_dict(e) for e in elements],
                llm_prompt_user="",
                llm_response=action.raw_response,
                action={"action": "done", "reason": action.reason},
                result="done",
                elapsed=round(time.time() - t0, 2),
            )
            return log, True, False

        if action.action == "fail":
            log = StepLog(
                step=step_idx,
                screenshot_path=str(screenshot_path),
                elements=[self._elem_to_dict(e) for e in elements],
                llm_prompt_user="",
                llm_response=action.raw_response,
                action={"action": "fail", "reason": action.reason},
                result=f"fail: {action.reason}",
                elapsed=round(time.time() - t0, 2),
            )
            return log, False, True

        # ── 执行 ────────────────────────────────────────────────────────────
        exec_result = await self._execute(adapter, action, elements)

        elapsed = round(time.time() - t0, 2)
        log = StepLog(
            step=step_idx,
            screenshot_path=str(screenshot_path),
            elements=[self._elem_to_dict(e) for e in elements],
            llm_prompt_user="",
            llm_response=action.raw_response,
            action={
                "action": action.action,
                "element_id": action.element_id,
                "text": action.text,
                "reason": action.reason,
            },
            result=exec_result,
            elapsed=elapsed,
        )

        logger.info("[step %d] 执行结果: %s (%.1fs)", step_idx + 1, exec_result, elapsed)
        return log, False, (exec_result.startswith("error") and "fail" in exec_result)

    async def _execute(
        self,
        adapter,
        action: Action,
        elements: list[UIElement],
    ) -> str:
        """把 Action 转换为 WebAdapter 调用，用坐标点击（OmniParser 输出 bbox）。"""
        # 找到对应元素
        elem = next((e for e in elements if e.idx == action.element_id), None)

        if action.action == "click":
            if elem is None:
                return f"error: element_id {action.element_id} not found"
            # 用坐标点击（OmniParser 不依赖 DOM selector）
            result = await self._click_at(adapter, elem.center_x, elem.center_y)
            return "ok" if result else "error: click failed"

        if action.action == "type":
            if elem is None:
                return f"error: element_id {action.element_id} not found"
            # 先点击聚焦，再用坐标 type
            await self._click_at(adapter, elem.center_x, elem.center_y)
            try:
                await adapter._page.keyboard.type(action.text, delay=30)
                return "ok"
            except Exception as exc:
                return f"error: type failed: {exc}"

        return f"error: unknown action {action.action}"

    @staticmethod
    async def _click_at(adapter, x: float, y: float) -> bool:
        """通过 Playwright page 直接坐标点击（绕过 selector）。"""
        try:
            await adapter._page.mouse.click(x, y)
            await asyncio.sleep(0.3)
            return True
        except Exception as exc:
            logger.warning("坐标点击失败 (%.0f, %.0f): %s", x, y, exc)
            return False

    @staticmethod
    def _elem_to_dict(e: UIElement) -> dict:
        return {
            "idx": e.idx,
            "label": e.label,
            "type": e.elem_type,
            "bbox": [round(v, 4) for v in e.bbox],
            "center": [round(e.center_x, 1), round(e.center_y, 1)],
            "interactivity": e.interactivity,
        }

    @staticmethod
    def _step_log_to_dict(s: StepLog) -> dict:
        return {
            "step": s.step,
            "screenshot": s.screenshot_path,
            "elements_count": len(s.elements),
            "action": s.action,
            "result": s.result,
            "elapsed": s.elapsed,
        }
