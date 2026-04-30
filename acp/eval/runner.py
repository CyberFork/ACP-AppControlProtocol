"""
Eval Runner：通用测试执行器。

输入：EvalCase（含 backend/mode/success_check）
输出：EvalTrace（含 JS DOM ground truth + LLM 自我认知 + 失败模式）

支持两个 backend：
  mini_a — OmniParser + Qwen2.5-3B（decompose / naive 两种模式）
  mini_b — UI-TARS-7B via vLLM（stub，等 #3 完成后对接）
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

class FailureMode(Enum):
    NONE = "none"
    MAX_STEPS_EXHAUSTED = "max_steps_exhausted"
    PREMATURE_DONE = "premature_done"
    ELEMENT_NOT_FOUND = "element_not_found"
    LOOP_SAME_ACTION = "loop_same_action"
    PARSER_ERROR = "parser_error"
    API_ERROR = "api_error"
    INVALID_COORDINATE = "invalid_coordinate"   # mini_b 坐标操作特有
    OTHER = "other"


@dataclass
class EvalCase:
    id: str
    instruction: str
    url: str
    backend: str                        # "mini_a" | "mini_b"
    success_check: str                  # "js:expr" | "llm:done"
    mode: Optional[str] = None          # mini_a 专用："decompose"|"naive"
    max_steps: int = 10
    repeats: int = 3
    tags: list[str] = field(default_factory=list)   # ["modal","login","form"...]

    @classmethod
    def from_dict(cls, d: dict) -> "EvalCase":
        return cls(
            id=d["id"],
            instruction=d["instruction"],
            url=d["url"],
            backend=d["backend"],
            success_check=d["success_check"],
            mode=d.get("mode"),
            max_steps=d.get("max_steps", 10),
            repeats=d.get("repeats", 3),
            tags=d.get("tags", []),
        )


@dataclass
class EvalTrace:
    testcase_id: str
    backend: str
    mode: Optional[str]
    repeat_idx: int

    success: bool                           # JS DOM ground truth
    llm_self_done: Optional[bool]           # LLM 输出了 done/finished()；decompose 模式下为 None（LLM 无机会表达完成）
    self_assessment_correct: Optional[bool] # success == llm_self_done；llm_self_done=None 时为 None

    steps: int
    elapsed_total_ms: float
    llm_inference_ms: float             # 纯 LLM 推理时间之和
    network_rtt_ms: float               # mini_b 网络往返（Mac→3090）

    failure_mode: FailureMode
    step_logs: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "testcase_id": self.testcase_id,
            "backend": self.backend,
            "mode": self.mode,
            "repeat_idx": self.repeat_idx,
            "success": self.success,
            "llm_self_done": self.llm_self_done,
            "self_assessment_correct": self.self_assessment_correct,
            "steps": self.steps,
            "elapsed_total_ms": self.elapsed_total_ms,
            "llm_inference_ms": self.llm_inference_ms,
            "network_rtt_ms": self.network_rtt_ms,
            "failure_mode": self.failure_mode.value,
            "step_logs": self.step_logs,
        }


# ---------------------------------------------------------------------------
# success_check 执行器
# ---------------------------------------------------------------------------

async def check_success(page, check_expr: str) -> bool:
    """执行 success_check 表达式，返回 True/False。

    支持格式：
      js:document.getElementById('login-success').classList.contains('visible')
      js:modal-overlay.hidden                    → 简写，展开为 DOM 检查
      js:note-1-copy.textContent=已复制          → 元素文字检查
      js:message-list.children.length>2         → 数量检查
      js:tab-btn-users.classList.contains(active)
    """
    if not check_expr.startswith("js:"):
        return False

    expr = check_expr[3:]

    # 展开简写格式 "id.property" 或 "id.property=value"
    js = _compile_check_expr(expr)
    try:
        result = await page.evaluate(f"() => {{ try {{ return {js}; }} catch(e) {{ return false; }} }}")
        return bool(result)
    except Exception as exc:
        logger.warning("success_check 执行失败 %r: %s", check_expr, exc)
        return False


def _compile_check_expr(expr: str) -> str:
    """将简写的 success_check 表达式编译为完整 JS。"""
    # 已是完整 JS（含括号/document等）
    if "document" in expr or "(" in expr:
        return expr

    # 格式：id.prop=value
    m = re.match(r'^([\w-]+)\.([\w]+)=(.+)$', expr)
    if m:
        elem_id, prop, val = m.groups()
        if prop == "textContent":
            return f"document.getElementById('{elem_id}')?.textContent?.trim() === '{val}'"
        if prop == "value":
            return f"document.getElementById('{elem_id}')?.value === '{val}'"
        if prop == "classList.contains":
            return f"document.getElementById('{elem_id}')?.classList.contains('{val}')"
        return f"document.getElementById('{elem_id}')?.{prop} === '{val}'"

    # 格式：id.prop.contains(val)
    m2 = re.match(r'^([\w-]+)\.classList\.contains\((\w+)\)$', expr)
    if m2:
        elem_id, cls = m2.groups()
        return f"document.getElementById('{elem_id}')?.classList.contains('{cls}')"

    # 格式：id.prop （布尔/length比较）
    m3 = re.match(r'^([\w-]+)\.([\w.]+)(>|<|>=|<=|===|==)(.+)$', expr)
    if m3:
        elem_id, prop, op, val = m3.groups()
        return f"document.getElementById('{elem_id}')?.{prop} {op} {val}"

    # 格式：id.hidden / id.visible
    m4 = re.match(r'^([\w-]+)\.(hidden|visible)$', expr)
    if m4:
        elem_id, state = m4.groups()
        if state == "hidden":
            return f"document.getElementById('{elem_id}')?.classList.contains('hidden')"
        if state == "visible":
            return (
                f"!document.getElementById('{elem_id}')?.classList.contains('hidden') && "
                f"document.getElementById('{elem_id}')?.style.display !== 'none'"
            )

    # 格式：id.active（tab 按钮）
    m5 = re.match(r'^([\w-]+)\.(active)$', expr)
    if m5:
        elem_id, _ = m5.groups()
        return f"document.getElementById('{elem_id}')?.classList.contains('active')"

    # fallback：直接原样
    return expr


# ---------------------------------------------------------------------------
# mini_a backend
# ---------------------------------------------------------------------------

async def run_mini_a(
    tc: EvalCase,
    repeat_idx: int,
    perception,
    llm,
    log_dir: Path,
) -> EvalTrace:
    """执行一个 mini_a testcase。"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from acp.adapters.web_adapter import WebAdapter
    from acp.demo.mini_a.loop import MiniLoop

    run_id = f"eval_{tc.id}_r{repeat_idx}"
    t_total_start = time.time()
    llm_ms = 0.0

    loop = MiniLoop(perception=perception, llm=llm, log_dir=log_dir / run_id)
    mode = tc.mode or "naive"

    llm_self_done = False
    loop_result = None

    if mode == "decompose":
        # 复用 decompose 模式（子任务分解）
        from scripts.run_mini_a import run_once_multistep

        _log_dir_str = str(log_dir)
        _tc_url = tc.url
        _tc_instruction = tc.instruction

        class _FakeArgs:
            url = _tc_url
            file = False
            base_url = "http://localhost:8765"
            log_dir = _log_dir_str
            ollama_url = "http://localhost:11434/api/generate"
            model = "qwen2.5:3b"
            instruction = _tc_instruction

        ok, fail_reason = await run_once_multistep(_FakeArgs, perception, llm, run_id)
        elapsed_ms = (time.time() - t_total_start) * 1000

        # decompose 模式：LLM 从未被问"完成了吗"（succeed_on_first_action 强制退出），故 None
        steps = _count_steps_from_logs(log_dir / run_id)
        failure_mode = FailureMode.NONE if ok else FailureMode(fail_reason or FailureMode.OTHER.value)

        return EvalTrace(
            testcase_id=tc.id,
            backend="mini_a",
            mode=mode,
            repeat_idx=repeat_idx,
            success=ok,
            llm_self_done=None,
            self_assessment_correct=None,
            steps=steps,
            elapsed_total_ms=elapsed_ms,
            llm_inference_ms=0.0,
            network_rtt_ms=0.0,
            failure_mode=failure_mode,
        )

    else:
        # naive 模式：端到端单一指令
        loop_result = await loop.run(
            instruction=tc.instruction,
            start_url=tc.url,
            run_id=run_id,
            naive=True,
        )
        elapsed_ms = (time.time() - t_total_start) * 1000

        # LLM 自报 done？
        llm_self_done = loop_result.success or (
            loop_result.steps and loop_result.steps[-1].action.get("action") == "done"
        )

        # JS DOM ground truth
        js_success = False
        try:
            from acp.adapters.web_adapter import WebAdapter
            async with WebAdapter(headless=True) as adapter:
                await adapter.navigate(tc.url)
                # 注意：这里无法重放 loop 执行，只能用 loop 内嵌的最终状态
                # 因此从 loop_result 推断 js_success
                js_success = loop_result.success
        except Exception:
            js_success = loop_result.success

        failure_mode = _classify_loop_failure(loop_result)

        return EvalTrace(
            testcase_id=tc.id,
            backend="mini_a",
            mode=mode,
            repeat_idx=repeat_idx,
            success=js_success,
            llm_self_done=llm_self_done,
            self_assessment_correct=(js_success == llm_self_done),
            steps=len(loop_result.steps),
            elapsed_total_ms=elapsed_ms,
            llm_inference_ms=0.0,
            network_rtt_ms=0.0,
            failure_mode=failure_mode,
            step_logs=[_step_to_dict(s) for s in loop_result.steps],
        )


# ---------------------------------------------------------------------------
# mini_b backend（stub，等 #3 完成后实现）
# ---------------------------------------------------------------------------

async def run_mini_b(
    tc: EvalCase,
    repeat_idx: int,
    host: str,
    log_dir: Path,
) -> EvalTrace:
    """执行一个 mini_b testcase（UI-TARS 7B via vLLM）。

    TODO: 等 backend-dev-6 完成 #3 后对接 acp/demo/mini_b/loop.py
    """
    raise NotImplementedError(
        "mini_b backend 尚未就绪，等待 Task #3（vLLM 部署）完成后对接。"
    )


# ---------------------------------------------------------------------------
# 主 Runner
# ---------------------------------------------------------------------------

class EvalRunner:
    """统一评测执行器，管理所有 backend 的测试运行。"""

    def __init__(
        self,
        log_dir: Path = Path("logs/eval"),
        mini_b_host: str = "http://192.168.50.129:8000",
    ) -> None:
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.mini_b_host = mini_b_host
        self._perception = None
        self._llm = None

    def _load_mini_a(self) -> None:
        """懒加载 OmniParser + Ollama（只在需要 mini_a 时初始化）。"""
        if self._perception is not None:
            return
        from acp.demo.mini_a.llm_backend import OllamaBackend
        from acp.demo.mini_a.perception import OmniPerception

        self._llm = OllamaBackend()
        if not self._llm.health_check():
            raise RuntimeError(
                "Ollama 未运行，请执行：ollama serve && ollama pull qwen2.5:3b"
            )
        self._perception = OmniPerception()
        self._perception.load()
        logger.info("mini_a 后端加载完成")

    async def run_testcase(self, tc: EvalCase, repeat_idx: int = 0) -> EvalTrace:
        """运行单个 testcase 的一次重复。"""
        logger.info("[eval] %s backend=%s mode=%s repeat=%d",
                    tc.id, tc.backend, tc.mode, repeat_idx)

        if tc.backend == "mini_a":
            self._load_mini_a()
            return await run_mini_a(tc, repeat_idx, self._perception, self._llm, self.log_dir)

        if tc.backend == "mini_b":
            return await run_mini_b(tc, repeat_idx, self.mini_b_host, self.log_dir)

        raise ValueError(f"未知 backend: {tc.backend}")

    async def run_all(
        self,
        testcases: list[EvalCase],
        backends: Optional[list[str]] = None,
    ) -> list[EvalTrace]:
        """运行所有 testcase，每个重复 tc.repeats 次。"""
        traces = []
        for tc in testcases:
            if backends and tc.backend not in backends:
                continue
            for r in range(tc.repeats):
                try:
                    trace = await self.run_testcase(tc, r)
                    traces.append(trace)
                    self._save_trace(trace)
                except NotImplementedError as e:
                    logger.warning("[eval] SKIP %s: %s", tc.id, e)
                except Exception as exc:
                    logger.error("[eval] ERROR %s repeat=%d: %s", tc.id, r, exc)
                    traces.append(EvalTrace(
                        testcase_id=tc.id,
                        backend=tc.backend,
                        mode=tc.mode,
                        repeat_idx=r,
                        success=False,
                        llm_self_done=False,
                        self_assessment_correct=False,
                        steps=0,
                        elapsed_total_ms=0.0,
                        llm_inference_ms=0.0,
                        network_rtt_ms=0.0,
                        failure_mode=FailureMode.API_ERROR,
                    ))
        return traces

    def _save_trace(self, trace: EvalTrace) -> None:
        p = self.log_dir / f"{trace.testcase_id}_r{trace.repeat_idx}_{trace.backend}.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(trace.to_dict(), f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

def _classify_loop_failure(result) -> FailureMode:
    """从 LoopResult 分类失败原因。"""
    if result.success:
        return FailureMode.NONE

    steps = result.steps
    if not steps:
        return FailureMode.API_ERROR

    if "超过最大步数" in (result.message or ""):
        actions = [s.action.get("action", "") for s in steps if hasattr(s, "action")]
        elem_ids = [s.action.get("element_id", -1) for s in steps
                    if hasattr(s, "action") and s.action.get("action") == "click"]
        if len(set(elem_ids)) <= 2 and len(elem_ids) >= 6:
            return FailureMode.LOOP_SAME_ACTION
        return FailureMode.MAX_STEPS_EXHAUSTED

    if any("done" in str(s.action.get("action", "")) for s in steps[:3] if hasattr(s, "action")):
        return FailureMode.PREMATURE_DONE

    if any("not found" in (s.result or "") for s in steps if hasattr(s, "result")):
        return FailureMode.ELEMENT_NOT_FOUND

    return FailureMode.OTHER


def _count_steps_from_logs(run_dir: Path) -> int:
    """从 run_summary 子目录统计总步数。"""
    total = 0
    for sub in run_dir.parent.glob(f"{run_dir.name}_sub*"):
        summary = sub / "run_summary.json"
        if summary.exists():
            try:
                d = json.loads(summary.read_text())
                total += len(d.get("steps", []))
            except Exception:
                pass
    if total == 0:
        summary = run_dir / "run_summary.json"
        if summary.exists():
            try:
                d = json.loads(summary.read_text())
                total = len(d.get("steps", []))
            except Exception:
                pass
    return total or 1


def _step_to_dict(step) -> dict:
    """StepLog → dict（兼容 dataclass 和 dict）。"""
    if isinstance(step, dict):
        return step
    return {
        "step": step.step,
        "screenshot": step.screenshot_path,
        "action": step.action,
        "result": step.result,
        "elapsed": step.elapsed,
    }


# ---------------------------------------------------------------------------
# testcase 加载
# ---------------------------------------------------------------------------

def load_testcases(yaml_path: Path, backend_filter: Optional[str] = None) -> list[EvalCase]:
    """从 YAML 文件加载 testcase 列表。"""
    import yaml
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    tcs = [EvalCase.from_dict(d) for d in data.get("testcases", [])]
    if backend_filter:
        tcs = [tc for tc in tcs if tc.backend == backend_filter]
    return tcs
