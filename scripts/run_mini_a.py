"""
Demo A CLI：OmniParser + Qwen2.5-3B miniLoop

用法：
    python scripts/run_mini_a.py \\
        --instruction "关闭弹窗，进入登录页面" \\
        --url "http://localhost:8765/pages/popup-login.html"

    # 使用 file:// 协议（不需要 server）：
    python scripts/run_mini_a.py \\
        --instruction "关闭弹窗，进入登录页面" \\
        --file

    # 重复跑 5 次统计成功率：
    python scripts/run_mini_a.py --file --runs 5

前置依赖：
    1. ollama 已安装并运行：`ollama serve`
    2. 模型已拉取：`ollama pull qwen2.5:3b`
    3. Python 依赖：`pip install transformers huggingface_hub easyocr supervision opencv-python-headless torchvision accelerate`
"""

import argparse
import asyncio
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
logger = logging.getLogger("run_mini_a")


def parse_args():
    parser = argparse.ArgumentParser(description="Demo A: OmniParser + Qwen2.5-3B miniLoop")
    parser.add_argument(
        "--instruction", "-i",
        default="关闭弹窗，然后用用户名 demo 密码 123456 登录",
        help="任务指令",
    )
    parser.add_argument(
        "--url", "-u",
        default=None,
        help="起始页面 URL（默认使用 --file 模式的本地路径）",
    )
    parser.add_argument(
        "--file", action="store_true",
        help="使用 file:// 协议直接加载本地 popup-login.html",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8765",
        help="testenv server 地址（--url 未指定时使用）",
    )
    parser.add_argument(
        "--runs", type=int, default=1,
        help="重复运行次数（用于统计成功率）",
    )
    parser.add_argument(
        "--log-dir",
        default="logs/mini_a",
        help="日志目录",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434/api/generate",
    )
    parser.add_argument(
        "--model",
        default="qwen2.5:3b",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--mode",
        choices=["decompose", "naive"],
        default="decompose",
        help=(
            "decompose（A1）: 子任务分解 + forced_action + label_keyword fallback\n"
            "naive（A2）: 纯 LLM 端到端，无任何辅助，单一复合指令"
        ),
    )
    return parser.parse_args()


async def run_once(args, perception, llm, run_id: str) -> bool:
    """执行一次完整的 miniDemo：弹窗关闭 → 登录。

    将复合指令拆分为两个子任务顺序执行：
      1. 关闭弹窗（单步）
      2. 登录（填用户名 + 填密码 + 点登录）
    每个子任务独立调用 loop.run()，共享同一个 WebAdapter 实例。
    """
    from acp.adapters.web_adapter import WebAdapter
    from acp.demo.mini_a.loop import MiniLoop

    # 决定 URL
    if args.url:
        url = args.url
    elif args.file:
        testenv = ROOT / "acp" / "testenv"
        url = (testenv / "pages" / "popup-login.html").as_uri()
    else:
        url = f"{args.base_url.rstrip('/')}/pages/popup-login.html"

    loop = MiniLoop(
        perception=perception,
        llm=llm,
        log_dir=Path(args.log_dir),
    )

    logger.info("=== Run %s 开始 | URL: %s ===", run_id, url)
    result = await loop.run(
        instruction=args.instruction,
        start_url=url,
        run_id=run_id,
    )

    status = "SUCCESS" if result.success else "FAIL"
    logger.info(
        "=== Run %s %s | %d steps | %.1fs | %s ===",
        run_id, status, len(result.steps), result.total_elapsed, result.message
    )
    return result.success, ("" if result.success else result.message)


async def run_once_multistep(args, perception, llm, run_id: str) -> tuple[bool, str]:
    """多子任务模式：拆解复合指令为顺序子任务，每步单一目标。

    这是针对 Qwen2.5-3B 推理能力受限的适配方案：
    3B 模型难以完成需要状态追踪的多步复合任务，
    但对单步简单任务（"找 X 按钮点击"）表现良好。
    """
    from acp.adapters.web_adapter import WebAdapter
    from acp.demo.mini_a.loop import MiniLoop, LoopResult

    if args.url:
        url = args.url
    elif args.file:
        testenv = ROOT / "acp" / "testenv"
        url = (testenv / "pages" / "popup-login.html").as_uri()
    else:
        url = f"{args.base_url.rstrip('/')}/pages/popup-login.html"

    # 子任务序列：(描述, 强制 action 类型, 强制 text, label 关键词 for fallback 匹配)
    # label_keyword: 若 LLM 选错元素，用此关键词在元素列表里做 fallback 匹配
    sub_tasks = [
        ("关闭弹窗：找 label 为空且坐标 x>700 的图标，点击", "click", "", ""),
        ('填用户名：找 label 含"请输入用户名"的元素，type "demo"', "type", "demo", "请输入用户名"),
        ('填密码：找 label 含"请输入密码"的元素，type "123456"', "type", "123456", "请输入密码"),
        ('点登录：找 label 精确为"登录"的短 label 按钮（不是"请登录"），click', "click", "", "==登录"),
    ]

    logger.info("=== Run %s 多步模式开始 | URL: %s ===", run_id, url)
    total_steps = 0
    t_start = __import__("time").time()

    login_success = False

    async with WebAdapter(headless=True) as adapter:
        nav = await adapter.navigate(url)
        if not nav.success:
            logger.error("导航失败: %s", nav.error)
            return False

        for sub_idx, (sub_instruction, forced_action, forced_text, label_kw) in enumerate(sub_tasks):
            sub_run_id = f"{run_id}_sub{sub_idx+1}"
            sub_loop = MiniLoop(
                perception=perception,
                llm=llm,
                log_dir=Path(args.log_dir),
            )
            logger.info("--- 子任务 %d [%s]: %s ---", sub_idx + 1, forced_action, sub_instruction)

            result = await sub_loop.run_with_adapter(
                adapter=adapter,
                instruction=sub_instruction,
                run_id=sub_run_id,
                max_steps=3,
                forced_action=forced_action,
                forced_text=forced_text,
                label_keyword=label_kw,
            )
            total_steps += len(result.steps)
            if not result.success:
                logger.warning("子任务 %d 未成功: %s，继续", sub_idx + 1, result.message)

        # 验证最终页面状态
        try:
            login_success = await adapter._page.evaluate(
                '() => document.getElementById("login-success") ? '
                'document.getElementById("login-success").classList.contains("visible") : false'
            )
        except Exception:
            login_success = False

    elapsed = __import__("time").time() - t_start
    status = "SUCCESS" if login_success else "FAIL"
    logger.info(
        "=== Run %s %s | %d steps | %.1fs ===",
        run_id, status, total_steps, elapsed
    )
    fail_reason = "" if login_success else "最终 login-success 未出现"
    return login_success, fail_reason


async def run_once_naive(args, perception, llm, run_id: str) -> tuple[bool, str]:
    """A2 模式：纯 LLM 端到端，无子任务分解、无 forced_action、无 label_keyword。

    LLM 读单一复合指令，自己判断每步动作和 done 条件。
    完全反映 Qwen2.5-3B 的真实端到端能力上限。
    """
    from acp.demo.mini_a.loop import MiniLoop

    if args.url:
        url = args.url
    elif args.file:
        testenv = ROOT / "acp" / "testenv"
        url = (testenv / "pages" / "popup-login.html").as_uri()
    else:
        url = f"{args.base_url.rstrip('/')}/pages/popup-login.html"

    loop = MiniLoop(
        perception=perception,
        llm=llm,
        log_dir=Path(args.log_dir),
    )

    logger.info("=== Run %s [naive] 开始 | URL: %s ===", run_id, url)
    result = await loop.run(
        instruction=args.instruction,
        start_url=url,
        run_id=run_id,
        naive=True,
    )

    status = "SUCCESS" if result.success else "FAIL"
    logger.info(
        "=== Run %s [naive] %s | %d steps | %.1fs | %s ===",
        run_id, status, len(result.steps), result.total_elapsed, result.message,
    )

    # 分析失败模式
    fail_reason = ""
    if not result.success:
        fail_reason = _classify_failure(result)

    return result.success, fail_reason


def _classify_failure(result) -> str:
    """从 LoopResult 判断失败原因类型。"""
    steps = result.steps
    if not steps:
        return "no_steps"

    actions = [s.action.get("action", "") for s in steps]
    reasons = [s.action.get("reason", "") for s in steps]

    # 提前判 done（步数少但没真正完成）
    if "done" in actions and len(steps) <= 3:
        return "premature_done"

    # 超步（MAX_STEPS 耗尽）
    if "超过最大步数" in result.message:
        # 细分：一直点同一个元素
        elem_ids = [s.action.get("element_id", -1) for s in steps if s.action.get("action") == "click"]
        if len(set(elem_ids)) <= 2 and len(elem_ids) >= 6:
            return "loop_same_element"
        return "max_steps_exhausted"

    # LLM 输出 fail
    if "fail" in actions:
        return "llm_output_fail"

    return f"other: {result.message[:60]}"


async def main():
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    from acp.demo.mini_a.llm_backend import OllamaBackend
    from acp.demo.mini_a.perception import OmniPerception

    # 检查 Ollama
    llm = OllamaBackend(base_url=args.ollama_url, model=args.model)
    if not llm.health_check():
        logger.error(
            "Ollama 未运行或模型 %s 未拉取。\n"
            "请执行：ollama serve  &&  ollama pull %s",
            args.model, args.model,
        )
        sys.exit(1)
    logger.info("Ollama 就绪，模型: %s", args.model)

    # 加载 OmniParser（一次性，所有 run 共享）
    perception = OmniPerception()
    perception.load()

    mode = args.mode
    runner = run_once_multistep if mode == "decompose" else run_once_naive
    logger.info("运行模式: %s", mode)

    successes = 0
    failure_modes: list[str] = []
    t_total = time.time()
    for i in range(args.runs):
        run_id = f"{mode}_{i+1:02d}_{int(time.time())}"
        ok, fail_reason = await runner(args, perception, llm, run_id)
        if ok:
            successes += 1
        else:
            failure_modes.append(fail_reason or "unknown")

    elapsed = time.time() - t_total
    print(f"\n{'='*50}")
    print(f"模式: {mode}")
    print(f"成功率: {successes}/{args.runs}  ({successes/args.runs*100:.0f}%)")
    print(f"总耗时: {elapsed:.1f}s  | 平均: {elapsed/args.runs:.1f}s/run")
    if failure_modes:
        print(f"失败模式:")
        for fm in failure_modes:
            print(f"  - {fm}")
    print(f"日志目录: {Path(args.log_dir).resolve()}")
    print(f"{'='*50}")


if __name__ == "__main__":
    asyncio.run(main())
