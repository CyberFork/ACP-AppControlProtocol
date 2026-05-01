"""
run_eval.py — 评测 CLI

用法：
    # 跑 mini_a 的所有 testcase
    python scripts/run_eval.py --backend mini_a

    # 只跑 naive 模式
    python scripts/run_eval.py --backend mini_a --mode naive

    # 指定特定 testcase
    python scripts/run_eval.py --backend mini_a --ids tc01_popup_decompose tc02_popup_naive

    # 指定 testenv server
    python scripts/run_eval.py --backend mini_a --base-url http://localhost:8765
    python scripts/run_eval.py --backend mini_a --file  # 用 file:// 协议

    # 跑完立刻生成报告
    python scripts/run_eval.py --backend mini_a --report
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_eval")


def parse_args():
    parser = argparse.ArgumentParser(description="ACP Demo Evaluator")
    parser.add_argument("--backend", choices=["mini_a", "mini_b", "mini_c", "all"], default="mini_a")
    parser.add_argument("--mode", choices=["decompose", "naive", "all"], default="all",
                        help="mini_a 模式过滤（all = 不过滤）")
    parser.add_argument("--ids", nargs="*", help="只跑指定 testcase ID")
    parser.add_argument("--base-url", default="http://localhost:8765")
    parser.add_argument("--file", action="store_true", help="用 file:// 协议（不需要 server）")
    parser.add_argument("--testcases", default=str(ROOT / "acp/eval/testcases.yaml"))
    parser.add_argument("--log-dir", default="logs/eval")
    parser.add_argument("--mini-b-host", default="http://192.168.50.129:8000")
    parser.add_argument("--report", action="store_true", help="完成后生成对比报告")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def resolve_url(url_template: str, base_url: str, use_file: bool) -> str:
    """将 testcase URL 中的 {TESTENV} 替换为实际 base_url。"""
    if use_file:
        testenv_dir = ROOT / "acp" / "testenv"
        base = testenv_dir.as_uri()
    else:
        base = base_url.rstrip("/")
    return url_template.replace("{TESTENV}", base)


async def main():
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    from acp.eval.runner import EvalRunner, load_testcases

    # 加载 testcases
    tc_path = Path(args.testcases)
    backend_filter = None if args.backend == "all" else args.backend
    testcases = load_testcases(tc_path, backend_filter=backend_filter)

    # 过滤 mode
    if args.mode != "all" and backend_filter == "mini_a":
        testcases = [tc for tc in testcases if tc.mode == args.mode or tc.mode is None]

    # 过滤 ids
    if args.ids:
        testcases = [tc for tc in testcases if tc.id in args.ids]

    if not testcases:
        logger.error("没有找到符合条件的 testcase（backend=%s mode=%s ids=%s）",
                     args.backend, args.mode, args.ids)
        sys.exit(1)

    # 替换 URL 占位符
    for tc in testcases:
        tc.url = resolve_url(tc.url, args.base_url, args.file)

    logger.info("将运行 %d 个 testcase，每个 %d 次重复",
                len(testcases), testcases[0].repeats if testcases else 0)

    runner = EvalRunner(
        log_dir=Path(args.log_dir),
        mini_b_host=args.mini_b_host,
    )

    t0 = time.time()
    traces = await runner.run_all(testcases)
    elapsed = time.time() - t0

    # 汇总
    successes = sum(1 for t in traces if t.success)
    total = len(traces)
    print(f"\n{'='*55}")
    print(f"Backend: {args.backend}  Mode: {args.mode}")
    print(f"成功率: {successes}/{total}  ({successes/total*100:.0f}%)" if total else "无结果")
    print(f"总耗时: {elapsed:.1f}s")
    print(f"日志目录: {Path(args.log_dir).resolve()}")
    print(f"{'='*55}")

    # 保存全量 trace
    trace_path = Path(args.log_dir) / f"traces_{args.backend}_{args.mode}.json"
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump([t.to_dict() for t in traces], f, ensure_ascii=False, indent=2)
    logger.info("Trace 已保存: %s", trace_path)

    # 自动生成报告
    if args.report:
        report_path = await generate_report(traces, Path(args.log_dir))
        print(f"报告: {report_path}")


async def generate_report(traces, log_dir: Path) -> Path:
    from acp.eval.scoring import compare
    from scripts.compare_report import build_report

    metrics = compare(traces)
    today = __import__("datetime").date.today().isoformat()
    report_path = ROOT / ".plans/acp/docs" / f"miniDemo-eval-{today}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report_text = build_report(metrics, traces, today)
    report_path.write_text(report_text, encoding="utf-8")
    logger.info("报告已保存: %s", report_path)
    return report_path


if __name__ == "__main__":
    asyncio.run(main())
