"""
Scoring：从 EvalTrace 列表计算指标，支持 Demo A vs B 对比。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Optional

from acp.eval.runner import EvalTrace, FailureMode


@dataclass
class BackendMetrics:
    """单个 backend（或 backend+mode 组合）的聚合指标。"""
    backend: str
    mode: Optional[str]
    label: str                          # 报告展示用的名称

    n_testcases: int
    n_total_runs: int

    success_rate: float                 # 成功率 0-1
    avg_steps_success: float            # 成功 case 平均步数
    avg_steps_all: float                # 全部 case 平均步数
    avg_elapsed_total_ms: float         # 平均总耗时
    avg_llm_inference_ms: float         # 平均 LLM 推理时间
    avg_network_rtt_ms: float           # 平均网络往返（mini_b）

    self_assessment_accuracy: Optional[float]  # LLM 自我认知准确率；decompose 模式下 None（N/A）
    failure_mode_top3: list[tuple[str, int]]   # [(mode_value, count), ...]

    by_tag: dict[str, float]            # 按 tag 的成功率 {"modal": 0.8, "form": 0.5}


def compute_metrics(
    traces: list[EvalTrace],
    backend: str,
    mode: Optional[str] = None,
    label: Optional[str] = None,
    testcase_tags: Optional[dict[str, list[str]]] = None,
) -> BackendMetrics:
    """计算指定 backend（和 mode）的指标。

    testcase_tags: {tc_id: [tag1, tag2]} 用于按场景分组统计。
    """
    subset = [t for t in traces if t.backend == backend]
    if mode is not None:
        subset = [t for t in subset if t.mode == mode]

    if not subset:
        return BackendMetrics(
            backend=backend, mode=mode,
            label=label or f"{backend}/{mode}",
            n_testcases=0, n_total_runs=0,
            success_rate=0.0, avg_steps_success=0.0, avg_steps_all=0.0,
            avg_elapsed_total_ms=0.0, avg_llm_inference_ms=0.0, avg_network_rtt_ms=0.0,
            self_assessment_accuracy=0.0, failure_mode_top3=[],
            by_tag={},
        )

    n_total = len(subset)
    successes = [t for t in subset if t.success]
    n_success = len(successes)

    success_rate = n_success / n_total

    steps_success = [t.steps for t in successes] if successes else [0]
    steps_all = [t.steps for t in subset]

    avg_steps_success = sum(steps_success) / len(steps_success)
    avg_steps_all = sum(steps_all) / n_total

    elapsed_vals = [t.elapsed_total_ms for t in subset]
    avg_elapsed = sum(elapsed_vals) / n_total

    llm_vals = [t.llm_inference_ms for t in subset]
    avg_llm = sum(llm_vals) / n_total

    rtt_vals = [t.network_rtt_ms for t in subset if t.network_rtt_ms > 0]
    avg_rtt = sum(rtt_vals) / len(rtt_vals) if rtt_vals else 0.0

    # 自我认知准确率：跳过 None（decompose 模式下未定义）
    assess_defined = [t for t in subset if t.self_assessment_correct is not None]
    if assess_defined:
        assess_correct = [t for t in assess_defined if t.self_assessment_correct]
        assess_acc = len(assess_correct) / len(assess_defined)
    else:
        assess_acc = None   # 全是 decompose，指标 N/A

    # 失败模式 Top3
    failures = [t for t in subset if not t.success]
    mode_counter = Counter(t.failure_mode.value for t in failures)
    top3 = mode_counter.most_common(3)

    # 按 tag 成功率
    by_tag: dict[str, list[bool]] = defaultdict(list)
    if testcase_tags:
        for t in subset:
            for tag in testcase_tags.get(t.testcase_id, []):
                by_tag[tag].append(t.success)
    by_tag_rates = {
        tag: sum(vals) / len(vals)
        for tag, vals in by_tag.items()
    }

    # testcase 数（去重）
    n_tcs = len(set(t.testcase_id for t in subset))

    return BackendMetrics(
        backend=backend,
        mode=mode,
        label=label or (f"{backend}/{mode}" if mode else backend),
        n_testcases=n_tcs,
        n_total_runs=n_total,
        success_rate=success_rate,
        avg_steps_success=avg_steps_success,
        avg_steps_all=avg_steps_all,
        avg_elapsed_total_ms=avg_elapsed,
        avg_llm_inference_ms=avg_llm,
        avg_network_rtt_ms=avg_rtt,
        self_assessment_accuracy=assess_acc,
        failure_mode_top3=top3,
        by_tag=by_tag_rates,
    )


def compare(
    traces: list[EvalTrace],
    testcase_tags: Optional[dict[str, list[str]]] = None,
) -> list[BackendMetrics]:
    """计算所有 backend+mode 组合的指标，便于并排对比。

    返回顺序：mini_a/decompose, mini_a/naive, mini_b（若有）
    """
    metrics = []

    # mini_a decompose（A1，工程辅助上限）
    decompose_traces = [t for t in traces if t.backend == "mini_a" and t.mode == "decompose"]
    if decompose_traces:
        metrics.append(compute_metrics(
            traces, "mini_a", "decompose",
            label="Demo A (A1-decompose)",
            testcase_tags=testcase_tags,
        ))

    # mini_a naive（A2，端到端基线）
    naive_traces = [t for t in traces if t.backend == "mini_a" and t.mode == "naive"]
    if naive_traces:
        metrics.append(compute_metrics(
            traces, "mini_a", "naive",
            label="Demo A (A2-naive)",
            testcase_tags=testcase_tags,
        ))

    # mini_b（UI-TARS 端到端）
    b_traces = [t for t in traces if t.backend == "mini_b"]
    if b_traces:
        metrics.append(compute_metrics(
            traces, "mini_b", None,
            label="Demo B (UI-TARS-7B)",
            testcase_tags=testcase_tags,
        ))

    return metrics


def render_comparison_table(metrics_list: list[BackendMetrics]) -> str:
    """渲染 Markdown 对比表格。"""
    if not metrics_list:
        return "_暂无数据_"

    headers = ["维度"] + [m.label for m in metrics_list]
    rows = [
        ["**总成功率**"] + [f"{m.success_rate*100:.0f}% ({int(m.success_rate*m.n_total_runs)}/{m.n_total_runs})" for m in metrics_list],
        ["**平均步数（成功 case）**"] + [f"{m.avg_steps_success:.1f}" for m in metrics_list],
        ["**平均步数（全部）**"] + [f"{m.avg_steps_all:.1f}" for m in metrics_list],
        ["**平均总耗时**"] + [f"{m.avg_elapsed_total_ms/1000:.1f}s" for m in metrics_list],
        ["**LLM 响应时间**"] + [
            f"{m.avg_llm_inference_ms:.0f}ms（含网络RTT，未拆分）" if m.backend == "mini_b" and m.avg_llm_inference_ms
            else (f"{m.avg_llm_inference_ms:.0f}ms" if m.avg_llm_inference_ms else "N/A")
            for m in metrics_list
        ],
        ["**网络 RTT（3090）**"] + [
            "见上（未拆分）" if m.backend == "mini_b" and m.avg_network_rtt_ms
            else (f"{m.avg_network_rtt_ms:.0f}ms" if m.avg_network_rtt_ms else "N/A")
            for m in metrics_list
        ],
        ["**LLM 自我认知准确率**"] + [
            "N/A (decompose 不适用)" if m.self_assessment_accuracy is None and m.mode == "decompose"
            else f"{m.self_assessment_accuracy*100:.0f}%" if m.self_assessment_accuracy is not None
            else "N/A"
            for m in metrics_list
        ],
        ["**失败模式 Top1**"] + [
            f"{m.failure_mode_top3[0][0]}（工程bug）" if m.failure_mode_top3 and m.failure_mode_top3[0][0] == "other" and m.backend == "mini_b"
            else (m.failure_mode_top3[0][0] if m.failure_mode_top3 else "—")
            for m in metrics_list
        ],
    ]

    # 按场景成功率（取所有 tag 的并集）
    all_tags = sorted(set(tag for m in metrics_list for tag in m.by_tag))
    for tag in all_tags:
        rows.append(
            [f"**成功率：{tag}**"] + [
                f"{m.by_tag.get(tag, float('nan'))*100:.0f}%" if tag in m.by_tag else "N/A"
                for m in metrics_list
            ]
        )

    # 构造 Markdown
    sep = "|" + "|".join(["---"] * len(headers)) + "|"
    header_line = "|" + "|".join(headers) + "|"
    body = "\n".join("|" + "|".join(row) + "|" for row in rows)
    return f"{header_line}\n{sep}\n{body}"
