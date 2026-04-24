"""
ACP - Application Control Protocol
主入口

用法：
  python -m acp.main "打开 https://example.com"
  python acp/main.py "打开 https://example.com"
  python acp/main.py  # 交互式模式
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

# 确保无论从哪个目录运行，都能找到 acp 包
_REPO_ROOT = Path(__file__).parent.parent  # acp/ 的上一级
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# 日志配置
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("acp.main")


# ---------------------------------------------------------------------------
# ACP 主类
# ---------------------------------------------------------------------------


class ACP:
    """ACP 主控制器 — 组装所有模块，提供完整执行流程。

    使用示例：
        async with ACP() as acp:
            result = await acp.run("打开 https://example.com")
            print(result)
    """

    def __init__(
        self,
        headless: bool = True,
        tools_yaml: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        llm_model: Optional[str] = None,
    ) -> None:
        """初始化 ACP 主控制器。

        Args:
            headless:     浏览器是否无头模式
            tools_yaml:   MCP 工具配置文件路径（为 None 时自动查找）
            llm_api_key:  LLM API Key（优先级高于环境变量）
            llm_base_url: LLM Base URL
            llm_model:    LLM 模型 ID
        """
        # ---- 延迟导入（避免顶层 import 触发 playwright 初始化）----
        from acp.brain.executor import ExecutorDispatcher
        from acp.brain.feedback import FeedbackEvaluator
        from acp.brain.intent_parser import IntentParser
        from acp.brain.ptg_manager import PTGManager
        from acp.brain.task_planner import TaskPlanner
        from acp.mcp.registry import MCPRegistry
        from acp.mcp.tools.web_mcp import WebMCP

        # ---- 工具配置 ----
        if tools_yaml is None:
            # 自动查找：相对于本文件的 config/tools.yaml
            _here = Path(__file__).parent
            tools_yaml = str(_here / "config" / "tools.yaml")

        # ---- MCP Registry ----
        self.registry = MCPRegistry()
        if Path(tools_yaml).exists():
            self.registry.load_from_yaml(tools_yaml)
            logger.info("MCPRegistry 已加载: %s", tools_yaml)
        else:
            logger.warning("未找到 tools.yaml: %s，使用空注册表", tools_yaml)

        # ---- WebMCP ----
        self.web_mcp = WebMCP(headless=headless)

        # ---- Brain 模块 ----
        _api_key = llm_api_key or os.environ.get("ACP_LLM_API_KEY", "")
        _base_url = llm_base_url or os.environ.get("ACP_LLM_BASE_URL", "https://api.openai.com/v1")
        _model = llm_model or os.environ.get("ACP_LLM_MODEL", "gpt-4o")

        self.intent_parser = IntentParser(
            api_key=_api_key or None,
            base_url=_base_url,
            model=_model,
        )
        self.task_planner = TaskPlanner(
            registry=self.registry,
            llm_api_key=_api_key,
            llm_base_url=_base_url,
            llm_model=_model,
        )
        self.executor = ExecutorDispatcher(
            tools={"web-mcp": self.web_mcp},
            evaluator=FeedbackEvaluator(),
        )
        self.ptg_manager = PTGManager()
        self._started = False

    # ---- 生命周期 ----

    async def start(self) -> None:
        """启动 WebMCP（打开浏览器）。"""
        if not self._started:
            await self.web_mcp.start()
            self._started = True
            logger.info("ACP 启动完成（浏览器已就绪）")

    async def close(self) -> None:
        """关闭 WebMCP（关闭浏览器）。"""
        if self._started:
            await self.web_mcp.close()
            self._started = False
            logger.info("ACP 已关闭")

    async def __aenter__(self) -> "ACP":
        await self.start()
        return self

    async def __aexit__(self, *_) -> None:
        await self.close()

    # ---- 核心接口 ----

    async def run(self, user_input: str) -> dict[str, Any]:
        """完整流程：意图解析 → 任务规划 → 执行 → 返回结果摘要。

        Args:
            user_input: 用户自然语言指令

        Returns:
            结果摘要字典，包含：
              - input:    原始输入
              - intent:   解析出的意图
              - plan_id:  执行计划 ID
              - steps:    步骤数
              - results:  每步结果列表
              - success:  整体是否成功
        """
        if not self._started:
            await self.start()

        print(f"\n{'='*60}")
        print(f"[ACP] 指令: {user_input}")
        print(f"{'='*60}")

        # 1. 意图解析
        print("\n[步骤 1/3] 意图解析...")
        intent = await self.intent_parser.parse(user_input)
        print(f"  ✓ 意图: {intent.intent}")
        if intent.app:
            print(f"  ✓ 应用: {intent.app}")
        if intent.params:
            print(f"  ✓ 参数: {intent.params}")
        if intent.sub_tasks:
            print(f"  ✓ 子任务: {[st.action for st in intent.sub_tasks]}")

        # 2. 任务规划
        print("\n[步骤 2/3] 任务规划...")
        plan = await self.task_planner.plan(intent)
        print(f"  ✓ 计划 ID: {plan.plan_id}")
        print(f"  ✓ 步骤数: {len(plan.steps)}")
        for step in plan.steps:
            print(f"    步骤 {step.step_id}: {step.action} → {step.tool} | 参数: {step.params}")

        # 3. 执行
        print("\n[步骤 3/3] 执行...")
        results = await self.executor.execute(plan)

        # 4. 汇总结果
        success_count = sum(1 for r in results if r.success)
        overall_success = success_count == len(plan.steps) and len(plan.steps) > 0

        # 记录 PTG 状态转换
        prev_state = None
        for i, (step, result) in enumerate(zip(plan.steps, results)):
            if result.page_state:
                if prev_state is not None:
                    self.ptg_manager.record_transition(
                        from_state=prev_state,
                        action=step.action,
                        to_state=result.page_state,
                        params=step.params,
                    )
                prev_state = result.page_state

        print(f"\n{'='*60}")
        print(f"[ACP] 执行完成: {success_count}/{len(plan.steps)} 步成功")
        print(f"[ACP] PTG 状态图: {self.ptg_manager.node_count()} 节点, {self.ptg_manager.edge_count()} 条边")
        print(f"{'='*60}\n")

        return {
            "input": user_input,
            "intent": intent.intent,
            "plan_id": plan.plan_id,
            "steps": len(plan.steps),
            "results": [
                {
                    "step": i + 1,
                    "success": r.success,
                    "error": r.error,
                    "data": r.data,
                    "url": r.page_state.url if r.page_state else None,
                    "title": r.page_state.title if r.page_state else None,
                    "elements": len(r.elements) if r.elements else 0,
                }
                for i, r in enumerate(results)
            ],
            "success": overall_success,
            "ptg_nodes": self.ptg_manager.node_count(),
            "ptg_edges": self.ptg_manager.edge_count(),
        }

    # ---- 便捷工具方法 ----

    async def get_elements(self) -> list:
        """获取当前页面元素列表（快捷方式）。"""
        from acp.schema.plan import ActionResult
        result: ActionResult = await self.web_mcp.execute("get_elements", {})
        return result.elements or []

    async def get_page_state(self):
        """获取当前页面状态（快捷方式）。"""
        result = await self.web_mcp.execute("get_page_state", {})
        return result.page_state

    async def screenshot(self) -> bytes:
        """截图（返回 PNG bytes）。"""
        import base64
        result = await self.web_mcp.execute("screenshot", {})
        if result.success and result.data:
            return base64.b64decode(result.data["base64"])
        raise RuntimeError(f"截图失败: {result.error}")

    # ---- 交互模式 ----

    async def interactive(self) -> None:
        """交互式命令行模式。

        支持的命令：
          quit / exit     — 退出
          elements        — 查看当前页面元素
          screenshot      — 截图并保存
          state           — 查看当前页面状态
          ptg             — 查看 PTG 状态图
          <其他>          — 作为用户指令执行
        """
        if not self._started:
            await self.start()

        print("\n" + "="*60)
        print("  ACP 交互模式")
        print("  命令: quit/exit | elements | screenshot | state | ptg")
        print("  其他输入视为 ACP 指令执行")
        print("="*60 + "\n")

        while True:
            try:
                user_input = input(">>> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[ACP] 已退出")
                break

            if not user_input:
                continue

            cmd = user_input.lower()

            if cmd in ("quit", "exit"):
                print("[ACP] 再见！")
                break

            elif cmd == "elements":
                print("[ACP] 正在获取页面元素...")
                elements = await self.get_elements()
                if not elements:
                    print("  (无元素或页面未加载)")
                else:
                    print(f"  共 {len(elements)} 个元素：")
                    for elem in elements[:20]:  # 最多显示 20 个
                        text = elem.text or elem.label or elem.placeholder or ""
                        print(f"    [{elem.type.value:12s}] {elem.id} | {text[:60]}")
                    if len(elements) > 20:
                        print(f"  ... 还有 {len(elements) - 20} 个元素")

            elif cmd == "screenshot":
                print("[ACP] 截图中...")
                try:
                    png_bytes = await self.screenshot()
                    save_path = "/tmp/acp_screenshot.png"
                    Path(save_path).write_bytes(png_bytes)
                    print(f"  截图已保存至: {save_path} ({len(png_bytes)} bytes)")
                except Exception as e:
                    print(f"  截图失败: {e}")

            elif cmd == "state":
                print("[ACP] 当前页面状态：")
                state = await self.get_page_state()
                if state:
                    print(f"  平台:  {state.platform}")
                    print(f"  应用:  {state.app}")
                    print(f"  标题:  {state.title}")
                    print(f"  URL:   {state.url}")
                else:
                    print("  (无页面状态)")

            elif cmd == "ptg":
                print("[ACP] PTG 状态图：")
                print(f"  节点数: {self.ptg_manager.node_count()}")
                print(f"  边数:   {self.ptg_manager.edge_count()}")
                graph = self.ptg_manager.get_graph()
                for node_id, node in graph.nodes.items():
                    print(f"    节点: {node_id[:16]}... | {node.description}")
                for edge in graph.edges:
                    print(f"    边:   {edge.from_node[:12]}... -[{edge.action}]-> {edge.to_node[:12]}...")

            else:
                # 作为 ACP 指令执行
                try:
                    result = await self.run(user_input)
                    if result["success"]:
                        print(f"[ACP] 执行成功")
                    else:
                        print(f"[ACP] 执行失败或部分失败")
                        for r in result["results"]:
                            if not r["success"]:
                                print(f"  步骤 {r['step']} 失败: {r['error']}")
                except Exception as e:
                    print(f"[ACP] 执行异常: {e}")


# ---------------------------------------------------------------------------
# 模块入口
# ---------------------------------------------------------------------------


async def _main() -> None:
    args = sys.argv[1:]

    if not args:
        # 无参数：交互式模式
        async with ACP(headless=True) as acp:
            await acp.interactive()
    else:
        # 有参数：执行单条指令
        user_input = " ".join(args)
        async with ACP(headless=True) as acp:
            result = await acp.run(user_input)
            if not result["success"]:
                sys.exit(1)


if __name__ == "__main__":
    asyncio.run(_main())
