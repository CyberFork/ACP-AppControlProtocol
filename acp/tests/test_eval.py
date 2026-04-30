"""
Eval 框架单测（全 mock，不依赖 Ollama / OmniParser / 真实浏览器）。
"""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from acp.eval.runner import (
    EvalRunner,
    EvalTrace,
    FailureMode,
    EvalCase,
    _classify_loop_failure,
    _compile_check_expr,
    load_testcases,
)
from acp.eval.scoring import BackendMetrics, compare, compute_metrics, render_comparison_table


# ---------------------------------------------------------------------------
# EvalCase 和 EvalTrace 基础测试
# ---------------------------------------------------------------------------

class TestDataStructures:
    def test_testcase_from_dict(self):
        d = {
            "id": "tc01",
            "instruction": "关闭弹窗",
            "url": "http://localhost:8765/pages/popup-login.html",
            "backend": "mini_a",
            "success_check": "js:login-success.visible",
            "mode": "naive",
            "repeats": 3,
            "tags": ["modal"],
        }
        tc = EvalCase.from_dict(d)
        assert tc.id == "tc01"
        assert tc.mode == "naive"
        assert tc.repeats == 3
        assert "modal" in tc.tags

    def test_testcase_optional_mode(self):
        d = {
            "id": "tc_b",
            "instruction": "test",
            "url": "http://x",
            "backend": "mini_b",
            "success_check": "js:foo.bar",
        }
        tc = EvalCase.from_dict(d)
        assert tc.mode is None

    def test_eval_trace_to_dict(self):
        t = EvalTrace(
            testcase_id="tc01",
            backend="mini_a",
            mode="naive",
            repeat_idx=0,
            success=False,
            llm_self_done=True,
            self_assessment_correct=False,
            steps=10,
            elapsed_total_ms=25000,
            llm_inference_ms=0,
            network_rtt_ms=0,
            failure_mode=FailureMode.MAX_STEPS_EXHAUSTED,
        )
        d = t.to_dict()
        assert d["success"] is False
        assert d["failure_mode"] == "max_steps_exhausted"
        assert d["self_assessment_correct"] is False

    def test_failure_mode_enum_values(self):
        assert FailureMode.NONE.value == "none"
        assert FailureMode.LOOP_SAME_ACTION.value == "loop_same_action"
        assert FailureMode.INVALID_COORDINATE.value == "invalid_coordinate"


# ---------------------------------------------------------------------------
# success_check 编译器测试
# ---------------------------------------------------------------------------

class TestCompileCheckExpr:
    def test_visible(self):
        js = _compile_check_expr("login-success.visible")
        assert "login-success" in js
        assert "hidden" in js or "display" in js

    def test_hidden(self):
        js = _compile_check_expr("modal-overlay.hidden")
        assert "modal-overlay" in js
        assert "hidden" in js

    def test_value_eq(self):
        js = _compile_check_expr("login-username.value=demo")
        assert "login-username" in js
        assert "demo" in js

    def test_text_content(self):
        js = _compile_check_expr("note-1-copy.textContent=已复制")
        assert "note-1-copy" in js
        assert "已复制" in js

    def test_class_contains(self):
        js = _compile_check_expr("tab-btn-users.classList.contains(active)")
        assert "tab-btn-users" in js
        assert "active" in js

    def test_passthrough_full_js(self):
        expr = "document.getElementById('foo').classList.contains('bar')"
        js = _compile_check_expr(expr)
        assert js == expr

    def test_length_comparison(self):
        js = _compile_check_expr("message-list.children.length>2")
        assert "message-list" in js
        assert ">2" in js or "> 2" in js


# ---------------------------------------------------------------------------
# 失败模式分类测试
# ---------------------------------------------------------------------------

class TestClassifyFailure:
    def _make_result(self, success, message="", steps=None):
        r = MagicMock()
        r.success = success
        r.message = message
        steps = steps or []
        r.steps = steps
        return r

    def _make_step(self, action_type="click", elem_id=0, result="ok"):
        s = MagicMock()
        s.action = {"action": action_type, "element_id": elem_id, "reason": "test"}
        s.result = result
        return s

    def test_success_returns_none(self):
        r = self._make_result(True)
        assert _classify_loop_failure(r) == FailureMode.NONE

    def test_max_steps_loop(self):
        steps = [self._make_step("click", 4) for _ in range(10)]
        r = self._make_result(False, "超过最大步数 10", steps)
        mode = _classify_loop_failure(r)
        assert mode == FailureMode.LOOP_SAME_ACTION

    def test_max_steps_varied(self):
        steps = [self._make_step("click", i % 4) for i in range(10)]
        r = self._make_result(False, "超过最大步数 10", steps)
        mode = _classify_loop_failure(r)
        assert mode == FailureMode.MAX_STEPS_EXHAUSTED

    def test_premature_done(self):
        steps = [self._make_step("done")]
        r = self._make_result(False, "done after 1 step", steps)
        mode = _classify_loop_failure(r)
        assert mode == FailureMode.PREMATURE_DONE


# ---------------------------------------------------------------------------
# Scoring 测试
# ---------------------------------------------------------------------------

def _make_traces(
    backend="mini_a",
    mode="naive",
    success_flags=None,
    failure_modes=None,
    steps_list=None,
):
    success_flags = success_flags or [True, False, False]
    failure_modes = failure_modes or [FailureMode.NONE, FailureMode.MAX_STEPS_EXHAUSTED, FailureMode.LOOP_SAME_ACTION]
    steps_list = steps_list or [4, 10, 10]
    traces = []
    for i, (ok, fm, steps) in enumerate(zip(success_flags, failure_modes, steps_list)):
        # decompose 模式：llm_self_done/self_assessment_correct 为 None
        if mode == "decompose":
            lsd, sac = None, None
        else:
            lsd = ok
            sac = True
        traces.append(EvalTrace(
            testcase_id=f"tc{i:02d}",
            backend=backend,
            mode=mode,
            repeat_idx=0,
            success=ok,
            llm_self_done=lsd,
            self_assessment_correct=sac,
            steps=steps,
            elapsed_total_ms=steps * 2500.0,
            llm_inference_ms=steps * 500.0,
            network_rtt_ms=0.0,
            failure_mode=fm,
        ))
    return traces


class TestScoring:
    def test_compute_metrics_basic(self):
        traces = _make_traces(success_flags=[True, True, False])
        m = compute_metrics(traces, "mini_a", "naive", label="test")
        assert m.success_rate == pytest.approx(2/3, rel=0.01)
        assert m.n_total_runs == 3
        assert m.n_testcases == 3

    def test_compute_metrics_empty(self):
        m = compute_metrics([], "mini_a", "naive")
        assert m.success_rate == 0.0
        assert m.n_total_runs == 0

    def test_compute_metrics_all_fail(self):
        traces = _make_traces(success_flags=[False, False, False])
        m = compute_metrics(traces, "mini_a", "naive")
        assert m.success_rate == 0.0
        assert m.avg_steps_success == 0.0

    def test_failure_mode_top3(self):
        traces = _make_traces(
            success_flags=[False, False, False],
            failure_modes=[
                FailureMode.MAX_STEPS_EXHAUSTED,
                FailureMode.MAX_STEPS_EXHAUSTED,
                FailureMode.LOOP_SAME_ACTION,
            ],
        )
        m = compute_metrics(traces, "mini_a", "naive")
        assert len(m.failure_mode_top3) >= 1
        assert m.failure_mode_top3[0][0] == "max_steps_exhausted"

    def test_compare_returns_multiple(self):
        traces = (
            _make_traces("mini_a", "decompose", [True, True, True]) +
            _make_traces("mini_a", "naive", [False, False, False])
        )
        metrics = compare(traces)
        assert len(metrics) == 2
        labels = [m.label for m in metrics]
        assert any("decompose" in l for l in labels)
        assert any("naive" in l for l in labels)

    def test_render_table_not_empty(self):
        traces = _make_traces()
        metrics = compare(traces)
        table = render_comparison_table(metrics)
        assert "|" in table
        assert "成功率" in table

    def test_self_assessment_accuracy(self):
        traces = []
        # success=True, llm_self_done=True → correct
        traces.append(EvalTrace(
            testcase_id="t1", backend="mini_a", mode="naive", repeat_idx=0,
            success=True, llm_self_done=True, self_assessment_correct=True,
            steps=4, elapsed_total_ms=5000, llm_inference_ms=0, network_rtt_ms=0,
            failure_mode=FailureMode.NONE,
        ))
        # success=False, llm_self_done=True → incorrect
        traces.append(EvalTrace(
            testcase_id="t2", backend="mini_a", mode="naive", repeat_idx=0,
            success=False, llm_self_done=True, self_assessment_correct=False,
            steps=10, elapsed_total_ms=25000, llm_inference_ms=0, network_rtt_ms=0,
            failure_mode=FailureMode.PREMATURE_DONE,
        ))
        m = compute_metrics(traces, "mini_a", "naive")
        assert m.self_assessment_accuracy == pytest.approx(0.5, rel=0.01)

    def test_self_assessment_decompose_is_none(self):
        """decompose 模式下 self_assessment_accuracy 应为 None。"""
        traces = _make_traces("mini_a", "decompose", [True, True, True])
        m = compute_metrics(traces, "mini_a", "decompose")
        assert m.self_assessment_accuracy is None

    def test_self_assessment_mixed_none_ignored(self):
        """混合模式：None 的 trace 不计入分母。"""
        decompose_traces = _make_traces("mini_a", "decompose", [True])
        naive_traces = _make_traces("mini_a", "naive", [False])
        # 只取 naive 的指标
        m = compute_metrics(naive_traces, "mini_a", "naive")
        # naive 里 success=False, llm_self_done=False → correct=True → 100%
        assert m.self_assessment_accuracy == pytest.approx(1.0, rel=0.01)


# ---------------------------------------------------------------------------
# YAML 加载测试
# ---------------------------------------------------------------------------

class TestLoadTestcases:
    def test_load_yaml(self, tmp_path):
        yaml_content = """
testcases:
  - id: tc01
    instruction: test
    url: http://x/{TESTENV}/pages/foo.html
    backend: mini_a
    success_check: js:foo.bar
    mode: naive
    repeats: 2
    tags: [modal]
  - id: tc02
    instruction: test2
    url: http://x/{TESTENV}/pages/bar.html
    backend: mini_b
    success_check: js:baz.active
"""
        p = tmp_path / "testcases.yaml"
        p.write_text(yaml_content)
        tcs = load_testcases(p)
        assert len(tcs) == 2
        assert tcs[0].id == "tc01"
        assert tcs[1].backend == "mini_b"

    def test_load_yaml_filter(self, tmp_path):
        yaml_content = """
testcases:
  - id: a1
    instruction: x
    url: http://x
    backend: mini_a
    success_check: js:foo.bar
  - id: b1
    instruction: y
    url: http://y
    backend: mini_b
    success_check: js:baz.qux
"""
        p = tmp_path / "tc.yaml"
        p.write_text(yaml_content)
        tcs = load_testcases(p, backend_filter="mini_a")
        assert len(tcs) == 1
        assert tcs[0].id == "a1"


# ---------------------------------------------------------------------------
# compare_report 生成测试
# ---------------------------------------------------------------------------

class TestBuildReport:
    def test_report_contains_key_sections(self):
        from scripts.compare_report import build_report
        traces = (
            _make_traces("mini_a", "decompose", [True, True, True]) +
            _make_traces("mini_a", "naive", [False, False, False])
        )
        from acp.eval.scoring import compare
        metrics = compare(traces)
        report = build_report(metrics, traces, "2026-04-30")
        assert "总体对比" in report
        assert "成功率" in report
        assert "失败" in report or "结论" in report

    def test_report_has_recommendation(self):
        from scripts.compare_report import build_report
        from acp.eval.scoring import compare
        traces = _make_traces("mini_a", "naive", [False, False, False])
        metrics = compare(traces)
        report = build_report(metrics, traces, "2026-04-30")
        assert "Demo B 结果待补充" in report or "结论" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
