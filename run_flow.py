"""ACP Flow Runner 入口。

用法：
    # 执行单个流程
    python3 run_flow.py ai6666 login

    # 登录 + 评论（一步到位）
    python3 run_flow.py ai6666 login_and_comment --comment "ACP测试"

    # 只评论（已登录状态）
    python3 run_flow.py ai6666 comment --comment "ACP测试"

    # headless 模式
    python3 run_flow.py ai6666 login --headless

    # 执行完保持浏览器 10 秒
    python3 run_flow.py ai6666 login_and_comment --comment "ACP测试" --keep 15

    # 录制执行过程（保存为新 flow）
    python3 run_flow.py ai6666 login --record

    # 开启 DEBUG 日志
    python3 run_flow.py ai6666 login --verbose
"""
import asyncio
import logging
import sys

from acp.brain.flow_runner import FlowRunner


async def main():
    args = sys.argv[1:]

    if len(args) < 2:
        print("用法: python3 run_flow.py <站点> <流程名> [选项]")
        print()
        print("示例:")
        print('  python3 run_flow.py ai6666 login')
        print('  python3 run_flow.py ai6666 login_and_comment --comment "ACP测试"')
        print('  python3 run_flow.py ai6666 login_and_comment --comment "ACP测试" --keep 15')
        print('  python3 run_flow.py ai6666 login --record')
        print('  python3 run_flow.py ai6666 login --verbose')
        print()
        print("选项:")
        print("  --comment <文本>    评论内容")
        print("  --keep <秒数>       执行完保持浏览器打开（默认 10）")
        print("  --headless          无头模式（不显示浏览器）")
        print("  --record            录制执行过程，保存为新 flow")
        print("  --record-name <名称> 录制的 flow 名称（配合 --record）")
        print("  --verbose           开启 DEBUG 日志")
        return

    site = args[0]
    flow = args[1]

    # 解析选项
    extra_vars = {}
    headless = False
    keep_open = 10  # 默认保持 10 秒
    record = False
    record_name = ""
    verbose = False

    i = 2
    while i < len(args):
        if args[i] == "--comment" and i + 1 < len(args):
            extra_vars["comment"] = args[i + 1]
            i += 2
        elif args[i] == "--keep" and i + 1 < len(args):
            keep_open = int(args[i + 1])
            i += 2
        elif args[i] == "--headless":
            headless = True
            i += 1
        elif args[i] == "--record":
            record = True
            i += 1
        elif args[i] == "--record-name" and i + 1 < len(args):
            record_name = args[i + 1]
            i += 2
        elif args[i] == "--verbose":
            verbose = True
            i += 1
        else:
            # 其他 key=value 格式的变量
            if "=" in args[i]:
                k, v = args[i].split("=", 1)
                extra_vars[k] = v
            i += 1

    # 配置日志
    if verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
        logging.getLogger("acp").setLevel(logging.DEBUG)
        print("[verbose] DEBUG 日志已开启")
    else:
        logging.basicConfig(
            level=logging.WARNING,
            format="%(levelname)s: %(message)s",
        )

    site_dir = f"acp/config/sites/{site}"
    runner = FlowRunner(site_dir=site_dir, headless=headless)

    ok = await runner.run(
        flow,
        extra_vars=extra_vars,
        keep_open=keep_open,
        record=record,
        record_name=record_name,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    asyncio.run(main())
