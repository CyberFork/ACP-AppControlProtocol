"""
compare_report.py — 生成 Demo A vs B 对比 Markdown 报告

用法：
    # 从已有 trace 文件生成报告
    python scripts/compare_report.py \\
        --traces logs/eval/traces_mini_a_all.json logs/eval/traces_mini_b_all.json

    # 生成报告到指定路径
    python scripts/compare_report.py --traces ... --output .plans/acp/docs/report.md
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))


def build_report(metrics_list, traces, date: str) -> str:
    """构建完整 Markdown 报告。"""
    from acp.eval.runner import EvalTrace, FailureMode
    from acp.eval.scoring import render_comparison_table

    # 提取 testcase tags（从 trace 列表推断）
    tags_map: dict[str, list[str]] = {}

    lines = [
        f"# ACP miniDemo 对比评测报告",
        f"",
        f"> 日期：{date}",
        f"> 场景：testenv/pages/popup-login.html + 9 个其他页面",
        f"> 重复次数：每 testcase 3 次",
        f"",
        f"---",
        f"",
        f"## 一、总体对比",
        f"",
    ]

    lines.append(render_comparison_table(metrics_list))
    lines.append("")

    # 柱状图替代（ASCII 版本）
    lines += [
        "### 成功率一览（ASCII 柱状图）",
        "",
        "```",
    ]
    for m in metrics_list:
        bar_len = int(m.success_rate * 40)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        lines.append(f"{m.label:<28} {bar} {m.success_rate*100:.0f}%")
    lines += ["```", ""]

    # 各 backend 详情
    for section_idx, m in enumerate(metrics_list, start=2):
        section_num = "二三四五六七八九"[section_idx - 2] if section_idx - 2 < 8 else str(section_idx)
        lines += [
            f"---",
            f"",
            f"## {section_num}、{m.label} 详情",
            f"",
            f"- 测试用例数：{m.n_testcases}",
            f"- 总运行次数：{m.n_total_runs}",
            f"- **成功率：{m.success_rate*100:.0f}%**（{int(m.success_rate*m.n_total_runs)}/{m.n_total_runs}）",
            f"- 平均步数（成功）：{m.avg_steps_success:.1f}",
            f"- 平均步数（全部）：{m.avg_steps_all:.1f}",
            f"- 平均总耗时：{m.avg_elapsed_total_ms/1000:.1f}s",
            f"- LLM 推理时间：{m.avg_llm_inference_ms:.0f}ms" if m.avg_llm_inference_ms else "- LLM 推理时间：N/A",
            f"- 网络 RTT：{m.avg_network_rtt_ms:.0f}ms" if m.avg_network_rtt_ms else "- 网络 RTT：N/A（本地）",
            (f"- LLM 自我认知准确率：{m.self_assessment_accuracy*100:.0f}%"
             if m.self_assessment_accuracy is not None
             else "- LLM 自我认知准确率：N/A（decompose 模式，LLM 无机会表达完成）"),
            f"",
        ]

        if m.failure_mode_top3:
            lines += ["**失败模式 Top3：**", ""]
            for mode_val, cnt in m.failure_mode_top3:
                lines.append(f"- `{mode_val}`：{cnt} 次")
            lines.append("")

        if m.by_tag:
            lines += ["**按场景成功率：**", ""]
            for tag, rate in sorted(m.by_tag.items()):
                lines.append(f"- {tag}：{rate*100:.0f}%")
            lines.append("")

    # 失败 trace 样例
    failed_traces = [t for t in traces if not t.success]
    if failed_traces:
        lines += [
            "---",
            "",
            f"## {'三四五六七'[len(metrics_list)-1] if len(metrics_list)-1 < 5 else str(len(metrics_list)+1)}、典型失败案例",
            "",
        ]
        seen_modes = set()
        for t in failed_traces[:6]:
            key = (t.backend, t.mode, t.failure_mode.value)
            if key in seen_modes:
                continue
            seen_modes.add(key)
            lines += [
                f"### {t.testcase_id} ({t.backend}/{t.mode})",
                f"",
                f"- 失败模式：`{t.failure_mode.value}`",
                f"- 步数：{t.steps}",
                f"- LLM 自报 done：{t.llm_self_done}",
                f"",
            ]
            if t.step_logs:
                lines += ["最后 3 步：", "```"]
                for s in t.step_logs[-3:]:
                    action = s.get("action", {})
                    lines.append(
                        f"  step{s.get('step','?')}: {action.get('action','?')} "
                        f"elem={action.get('element_id','?')} "
                        f"→ {s.get('result','?')}"
                    )
                lines += ["```", ""]

    # 推荐结论
    lines += [
        "---",
        "",
        f"## {'四五六七八'[len(metrics_list)-1] if len(metrics_list)-1 < 5 else str(len(metrics_list)+2)}、推荐结论",
        "",
    ]

    # 自我认知注释
    lines += [
        "> **关于「LLM 自我认知准确率」的说明：**",
        "> - A1 (decompose) 模式下，系统在第一次 action 成功后强制退出（`succeed_on_first_action=True`），",
        ">   LLM 没有机会判断完成与否，故此维度标为 N/A。",
        "> - A2 (naive) 的 100% 是「正确知道自己失败」——LLM 从未输出 done，与 ground truth（失败）一致，",
        ">   含义是「不误报」而非「推理准确」。",
        "> - 此指标的真正对比价值在 Demo B：若 UI-TARS 在失败时仍输出 `finished()`，",
        ">   则 self_assessment_correct=False，与 A2 的 100% 形成对比，揭示端到端模型的自信度校准问题。",
        "",
    ]

    a1 = next((m for m in metrics_list if m.mode == "decompose"), None)
    a2 = next((m for m in metrics_list if m.mode == "naive" and m.backend == "mini_a"), None)
    b = next((m for m in metrics_list if m.backend == "mini_b"), None)

    if a2 and b:
        if b.success_rate >= 0.8:
            lines += [
                f"**推荐：优先使用 Demo B（UI-TARS-7B）**",
                f"",
                f"UI-TARS 端到端成功率 {b.success_rate*100:.0f}% ≥ 80%，优于 Demo A naive 基线（{a2.success_rate*100:.0f}%）。",
                f"大模型端到端推理在这个场景有本质优势，ACP Brain 应优先考虑专用 VLA 模型。",
                f"Demo A 的工程辅助方案（A1 {a1.success_rate*100:.0f}%）可作为降级备份。",
            ]
        elif b.success_rate > a2.success_rate:
            lines += [
                f"**推荐：Demo B 有优势但尚需工程辅助**",
                f"",
                f"UI-TARS 端到端成功率 {b.success_rate*100:.0f}%，优于 Qwen2.5-3B naive（{a2.success_rate*100:.0f}%），",
                f"但未达 80% 验收标准。建议对 Demo B 也引入子任务分解或 few-shot 辅助后重测。",
            ]
        else:
            lines += [
                f"**结论：Demo B 与 Demo A naive 基线相当**",
                f"",
                f"UI-TARS 端到端成功率 {b.success_rate*100:.0f}%，Demo A naive {a2.success_rate*100:.0f}%。",
                f"在当前场景下 7B 端到端模型未体现出相对于 3B+工程辅助的优势，建议评估更大规模模型或针对性微调。",
            ]
    elif a2:
        lines += [
            f"**Demo B 结果待补充（Task #3 完成后）**",
            f"",
            f"当前仅有 Demo A 数据：",
            f"- A1 (decompose)：{a1.success_rate*100:.0f}%" if a1 else "",
            f"- A2 (naive)：{a2.success_rate*100:.0f}%",
            f"",
            f"Qwen2.5-3B 端到端能力接近零，子任务分解是当前唯一可行路径。",
            f"Demo B 结果补充后可做最终判断。",
        ]

    lines += [""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="生成 Demo A/B 对比报告")
    parser.add_argument("--traces", nargs="+", required=True, help="trace JSON 文件路径")
    parser.add_argument("--output", default=None, help="报告输出路径（默认打印到 stdout）")
    parser.add_argument("--date", default=None)
    args = parser.parse_args()

    import datetime
    from acp.eval.runner import EvalTrace, FailureMode
    from acp.eval.scoring import compare

    date = args.date or datetime.date.today().isoformat()

    # 加载所有 trace
    all_traces = []
    for p in args.traces:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        for d in data:
            t = EvalTrace(
                testcase_id=d["testcase_id"],
                backend=d["backend"],
                mode=d.get("mode"),
                repeat_idx=d.get("repeat_idx", 0),
                success=d["success"],
                llm_self_done=d.get("llm_self_done", False),
                self_assessment_correct=d.get("self_assessment_correct", False),
                steps=d["steps"],
                elapsed_total_ms=d.get("elapsed_total_ms", 0),
                llm_inference_ms=d.get("llm_inference_ms", 0),
                network_rtt_ms=d.get("network_rtt_ms", 0),
                failure_mode=FailureMode(d.get("failure_mode", "other")),
                step_logs=d.get("step_logs", []),
            )
            all_traces.append(t)

    metrics = compare(all_traces)
    report = build_report(metrics, all_traces, date)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(report, encoding="utf-8")
        print(f"报告已写入: {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
