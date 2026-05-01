#!/usr/bin/env python3
"""
Demo C CLI — 混合架构：本地 UI-TARS grounding + 云端 LLM 规划。

用法：
  # Mock 模式（无需 API key 和 vLLM）
  python scripts/run_mini_c.py --mock

  # 真实 API（需配置 .env）
  python scripts/run_mini_c.py \
    --url http://localhost:8765/pages/popup-login.html \
    --instruction "关闭弹窗，然后用用户名 demo 密码 123456 登录"

  # 多次重复跑
  python scripts/run_mini_c.py --mock --repeats 5

环境变量（真实模式）：
  PLANNER_LLM_BASE_URL   (default: https://api.deepseek.com)
  PLANNER_LLM_API_KEY    (必须)
  PLANNER_LLM_MODEL      (default: deepseek-chat)
  UITARS_GROUNDING_MOCK  (=1 时 grounding 也用 mock)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

# 加载 .env（如果存在）
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

sys.path.insert(0, str(Path(__file__).parent.parent))

from acp.demo.mini_c.grounding import UITARSGrounding
from acp.demo.mini_c.loop import MiniCLoop
from acp.demo.mini_c.planner import PlannerLLM
from acp.demo.mini_c.state_describer import StateDescriber

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_mini_c")

DEFAULT_URL = "http://localhost:8765/pages/popup-login.html"
DEFAULT_INSTRUCTION = "关闭弹窗，然后用用户名 demo 密码 123456 登录"
DEFAULT_SUCCESS_CHECK = "js:login-success.visible"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Demo C — 混合架构 V-L-A")
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    p.add_argument("--success-check", default=DEFAULT_SUCCESS_CHECK)
    p.add_argument("--mock", action="store_true", help="使用 mock planner + grounding（无需 API key / vLLM）")
    p.add_argument("--mock-planner", action="store_true", help="仅 mock planner，grounding 用真实 UI-TARS")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--log-dir", default="logs/mini_c")
    p.add_argument("--max-steps", type=int, default=15)
    return p


async def run_once(
    args: argparse.Namespace,
    run_id: str,
    planner: PlannerLLM,
    grounder: UITARSGrounding,
) -> tuple[bool, str]:
    describer = StateDescriber()
    loop = MiniCLoop(
        planner=planner,
        grounder=grounder,
        describer=describer,
        log_dir=Path(args.log_dir),
    )

    result = await loop.run(
        instruction=args.instruction,
        start_url=args.url,
        success_check=args.success_check,
        run_id=run_id,
        max_steps=args.max_steps,
    )

    return result.success, result.message


def main() -> None:
    args = build_parser().parse_args()

    use_mock = args.mock
    use_mock_planner = args.mock or args.mock_planner
    use_mock_grounder = args.mock

    planner = PlannerLLM(mock=use_mock_planner)
    grounder = UITARSGrounding(mock=use_mock_grounder)

    if not use_mock and not use_mock_planner:
        api_key = os.getenv("PLANNER_LLM_API_KEY", "")
        if not api_key:
            logger.error(
                "未设置 PLANNER_LLM_API_KEY。"
                "使用 --mock 进行流程验证，或设置 .env 后运行真实 API。"
            )
            sys.exit(1)

    if not use_mock_grounder:
        if not grounder.health_check():
            logger.warning("UI-TARS grounding health check 失败，继续运行（可能报错）")

    logger.info("=== Demo C 开始 ===")
    logger.info("URL: %s", args.url)
    logger.info("指令: %s", args.instruction)
    logger.info("Mock模式: planner=%s grounder=%s", use_mock_planner, use_mock_grounder)
    logger.info("重复次数: %d", args.repeats)

    results: list[tuple[bool, str]] = []
    t_total = time.time()

    for i in range(args.repeats):
        run_id = f"run{i+1:02d}_{int(time.time())}"
        logger.info("--- Run %d/%d (run_id=%s) ---", i + 1, args.repeats, run_id)
        t_run = time.time()

        try:
            ok, msg = asyncio.run(run_once(args, run_id, planner, grounder))
        except Exception as exc:
            logger.error("Run %d 异常: %s", i + 1, exc, exc_info=True)
            ok, msg = False, f"exception: {exc}"

        elapsed = time.time() - t_run
        status = "SUCCESS" if ok else "FAIL"
        logger.info("Run %d: %s | %s | %.1fs", i + 1, status, msg, elapsed)
        results.append((ok, msg))

    # 汇总
    total = time.time() - t_total
    successes = sum(1 for ok, _ in results if ok)
    rate = successes / len(results) * 100 if results else 0

    print()
    print("=" * 60)
    print(f"Demo C 结果汇总  ({args.repeats} 次)")
    print("=" * 60)
    for i, (ok, msg) in enumerate(results, 1):
        status = "SUCCESS" if ok else "FAIL   "
        print(f"  Run {i:2d}: {status} | {msg}")
    print("-" * 60)
    print(f"  成功率: {successes}/{len(results)} = {rate:.0f}%")
    print(f"  总耗时: {total:.1f}s")
    print("=" * 60)

    # 机器可读的 JSON 输出
    summary = {
        "runs": args.repeats,
        "successes": successes,
        "success_rate": round(rate / 100, 2),
        "results": [{"run": i + 1, "success": ok, "message": msg} for i, (ok, msg) in enumerate(results)],
    }
    summary_path = Path(args.log_dir) / "run_summary_all.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    logger.info("汇总已保存: %s", summary_path)


if __name__ == "__main__":
    main()
