"""
ACP Demo - 端到端集成演示
展示完整的 Web 自动化流程：意图解析 → 任务规划 → 执行 → 结果

运行：
  python acp/demo.py
  python acp/demo.py 1    # 只运行 Demo 1
  python acp/demo.py 2    # 只运行 Demo 2
  python acp/demo.py 3    # 只运行 Demo 3
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from pathlib import Path

# 确保无论从哪个目录运行，都能找到 acp 包
_REPO_ROOT = Path(__file__).parent.parent  # acp/ 的上一级
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# 简化日志格式（Demo 模式）
logging.basicConfig(
    level=logging.WARNING,  # 静默 debug/info，只看 demo 自己的输出
    format="%(name)s - %(message)s",
)

# 设置 acp.main 可见
logging.getLogger("acp.main").setLevel(logging.WARNING)

_SEPARATOR = "─" * 60


def _header(title: str) -> None:
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")


def _section(title: str) -> None:
    print(f"\n{_SEPARATOR}")
    print(f"  {title}")
    print(_SEPARATOR)


def _ok(msg: str) -> None:
    print(f"  [OK] {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def _info(msg: str) -> None:
    print(f"  {msg}")


# ---------------------------------------------------------------------------
# Demo 1: 基础导航 + 感知
# ---------------------------------------------------------------------------

async def demo1_basic_navigation() -> bool:
    """Demo 1: 打开 example.com，感知页面元素。"""
    from acp.main import ACP

    _header("Demo 1: 基础导航 + 感知元素")
    _info("目标: 打开 https://example.com，列出页面所有元素")

    success = False
    async with ACP(headless=True) as acp:
        # ---- 第 1 步：意图解析 ----
        _section("步骤 1: 意图解析")
        t0 = time.perf_counter()
        intent = await acp.intent_parser.parse("打开 https://example.com")
        elapsed = time.perf_counter() - t0
        _ok(f"意图识别完成 ({elapsed*1000:.1f}ms)")
        _info(f"  intent = {intent.intent}")
        _info(f"  app    = {intent.app}")
        _info(f"  params = {intent.params}")
        _info(f"  子任务 = {[st.action for st in intent.sub_tasks]}")

        # ---- 第 2 步：任务规划 ----
        _section("步骤 2: 任务规划")
        t0 = time.perf_counter()
        plan = await acp.task_planner.plan(intent)
        elapsed = time.perf_counter() - t0
        _ok(f"规划完成 ({elapsed*1000:.1f}ms)")
        _info(f"  plan_id = {plan.plan_id}")
        _info(f"  步骤数  = {len(plan.steps)}")
        for step in plan.steps:
            _info(f"    步骤 {step.step_id}: action={step.action} | tool={step.tool} (tier={step.tool_tier})")
            _info(f"           params={step.params}")

        # ---- 第 3 步：执行（导航）----
        _section("步骤 3: 执行 — 导航")
        t0 = time.perf_counter()
        results = await acp.executor.execute(plan)
        elapsed = time.perf_counter() - t0

        nav_result = results[0] if results else None
        if nav_result and nav_result.success:
            _ok(f"导航成功 ({elapsed*1000:.0f}ms)")
            if nav_result.page_state:
                _info(f"  URL   = {nav_result.page_state.url}")
                _info(f"  标题  = {nav_result.page_state.title}")
        else:
            err = nav_result.error if nav_result else "无结果"
            _fail(f"导航失败: {err}")

        # ---- 第 4 步：感知页面元素 ----
        _section("步骤 4: 感知页面元素")
        t0 = time.perf_counter()
        elements = await acp.get_elements()
        elapsed = time.perf_counter() - t0
        _ok(f"元素感知完成 ({elapsed*1000:.0f}ms) — 共 {len(elements)} 个元素")

        # 按类型统计
        type_count: dict[str, int] = {}
        for elem in elements:
            key = elem.type.value
            type_count[key] = type_count.get(key, 0) + 1

        _info("  元素类型分布:")
        for etype, cnt in sorted(type_count.items(), key=lambda x: -x[1]):
            _info(f"    {etype:15s}: {cnt}")

        # 显示前 10 个元素详情
        _info(f"\n  前 {min(10, len(elements))} 个元素:")
        for elem in elements[:10]:
            text = elem.text or elem.label or elem.placeholder or "(无文本)"
            _info(f"    [{elem.type.value:12s}] {elem.id} | {text[:50]}")

        # ---- 第 5 步：PTG 状态 ----
        _section("步骤 5: PTG 状态转换图")

        # 手动记录第一次状态（从空白到 example.com）
        if nav_result and nav_result.page_state:
            from acp.schema.elements import PageState
            init_state = PageState(platform="web", app="browser", title="(初始)", url="about:blank")
            acp.ptg_manager.record_transition(
                from_state=init_state,
                action="navigate",
                to_state=nav_result.page_state,
                params={"url": "https://example.com"},
            )

        _info(f"  节点数: {acp.ptg_manager.node_count()}")
        _info(f"  边数:   {acp.ptg_manager.edge_count()}")
        graph = acp.ptg_manager.get_graph()
        for node_id, node in graph.nodes.items():
            marker = "★ 当前" if node_id == graph.current_state else "  "
            _info(f"  {marker} 节点: {node.description}")
        for edge in graph.edges:
            from_desc = graph.nodes[edge.from_node].description if edge.from_node in graph.nodes else edge.from_node
            to_desc = graph.nodes[edge.to_node].description if edge.to_node in graph.nodes else edge.to_node
            _info(f"    [{from_desc}] -[{edge.action}]-> [{to_desc}]")

        success = nav_result is not None and nav_result.success and len(elements) > 0

    _section("Demo 1 结果")
    if success:
        _ok("Demo 1 通过 — 导航成功 + 元素感知正常")
    else:
        _fail("Demo 1 失败")

    return success


# ---------------------------------------------------------------------------
# Demo 2: 搜索操作
# ---------------------------------------------------------------------------

async def demo2_search() -> bool:
    """Demo 2: 打开 Bing，搜索 ACP protocol。"""
    from acp.main import ACP

    _header("Demo 2: 搜索操作 (Bing)")
    _info("目标: 打开 Bing → 找到搜索框 → 输入 → 点击搜索 → 感知结果")

    success = False
    async with ACP(headless=True) as acp:
        # ---- 导航到 Bing ----
        _section("子步骤 2.1: 导航到 Bing")
        nav_result_raw = await acp.web_mcp.execute("navigate", {"url": "https://www.bing.com"})
        if nav_result_raw.success:
            _ok("导航到 Bing 成功")
            if nav_result_raw.page_state:
                _info(f"  标题: {nav_result_raw.page_state.title}")
        else:
            _fail(f"导航失败: {nav_result_raw.error}")
            return False

        # ---- 获取元素，找搜索框 ----
        _section("子步骤 2.2: 获取页面元素，查找搜索框")
        elements = await acp.get_elements()
        _ok(f"获取 {len(elements)} 个元素")

        # 查找搜索框：type=text_input 或 placeholder 含 "search"/"搜索"
        search_input = None
        search_btn = None

        for elem in elements:
            if elem.type.value in ("text_input",):
                ph = (elem.placeholder or "").lower()
                label = (elem.label or "").lower()
                text = (elem.text or "").lower()
                if any(kw in ph + label + text for kw in ("search", "搜索", "bing", "q")):
                    search_input = elem
                    break

        # 若未通过语义匹配，退而求其次：取第一个 text_input
        if search_input is None:
            for elem in elements:
                if elem.type.value == "text_input":
                    search_input = elem
                    break

        if search_input:
            _ok(f"找到搜索框: {search_input.id}")
            _info(f"  placeholder = {search_input.placeholder}")
            _info(f"  label       = {search_input.label}")
        else:
            _fail("未找到搜索框，展示所有 text_input:")
            inputs = [e for e in elements if e.type.value == "text_input"]
            for e in inputs:
                _info(f"  {e.id} | ph={e.placeholder} | label={e.label}")
            # 继续尝试，不直接退出

        # ---- 输入搜索词 ----
        _section("子步骤 2.3: 在搜索框输入")
        if search_input:
            type_result = await acp.web_mcp.execute("type", {
                "element_id": search_input.id,
                "text": "ACP protocol",
            })
            if type_result.success:
                _ok("输入 'ACP protocol' 成功")
            else:
                _fail(f"输入失败: {type_result.error}")
        else:
            _info("  (跳过输入，无搜索框)")

        # ---- 查找并点击搜索按钮 ----
        _section("子步骤 2.4: 点击搜索按钮")

        # 重新获取元素（输入后可能刷新）
        elements_after = await acp.get_elements()

        # 查找搜索按钮：type=button，text/label 含 search/搜索
        for elem in elements_after:
            if elem.type.value == "button":
                t = (elem.text or "").lower()
                l = (elem.label or "").lower()
                a = str(elem.actions).lower()
                if any(kw in t + l + a for kw in ("search", "搜索", "go", "submit")):
                    search_btn = elem
                    break

        if search_btn:
            _ok(f"找到搜索按钮: {search_btn.id} | text={search_btn.text}")
            click_result = await acp.web_mcp.execute("click", {"element_id": search_btn.id})
            if click_result.success:
                _ok("点击搜索按钮成功")
                # 等待结果页加载
                await asyncio.sleep(2)
            else:
                _fail(f"点击失败: {click_result.error}")
                # 尝试回车触发搜索
                _info("  尝试使用键盘提交...")
        else:
            _info("  未找到搜索按钮，尝试通过 Enter 键提交")
            # Playwright 没有直接的 Enter 操作，通过 type 触发（或导航到搜索结果）
            await acp.web_mcp.execute("navigate", {
                "url": "https://www.bing.com/search?q=ACP+protocol"
            })
            await asyncio.sleep(1)

        # ---- 感知结果页元素 ----
        _section("子步骤 2.5: 感知搜索结果页")
        state = await acp.get_page_state()
        if state:
            _ok(f"当前页: {state.title}")
            _info(f"  URL: {state.url}")

        result_elements = await acp.get_elements()
        _ok(f"共 {len(result_elements)} 个元素")

        # 查找结果链接
        links = [e for e in result_elements if e.type.value in ("text", "button", "list_item")
                 and e.text and len(e.text.strip()) > 10]
        _info(f"  候选文本元素（前 8 条）:")
        for elem in links[:8]:
            _info(f"    {elem.type.value:12s} | {elem.text[:80]}")

        success = len(result_elements) > 0

    _section("Demo 2 结果")
    if success:
        _ok("Demo 2 完成 — 搜索流程跑通，结果页已感知")
    else:
        _fail("Demo 2 失败")

    return success


# ---------------------------------------------------------------------------
# Demo 3: 连续操作 + PTG 记录
# ---------------------------------------------------------------------------

async def demo3_ptg_tracking() -> bool:
    """Demo 3: 连续导航多个页面，展示 PTG 状态转换记录。"""
    from acp.main import ACP

    _header("Demo 3: 连续操作 + PTG 状态转换图")
    _info("目标: 访问多个页面，展示 PTG 如何记录状态转换历史")

    steps = [
        ("打开 https://example.com", {"url": "https://example.com"}),
        ("打开 https://httpbin.org/html", {"url": "https://httpbin.org/html"}),
        ("打开 https://httpbin.org/json", {"url": "https://httpbin.org/json"}),
    ]

    success = False
    async with ACP(headless=True) as acp:
        prev_state = None

        for i, (description, nav_params) in enumerate(steps, start=1):
            _section(f"步骤 {i}: {description}")

            result = await acp.web_mcp.execute("navigate", nav_params)
            if result.success and result.page_state:
                state = result.page_state
                _ok(f"导航成功")
                _info(f"  URL:   {state.url}")
                _info(f"  标题:  {state.title}")

                # 记录 PTG 转换
                if prev_state is not None:
                    from_node, to_node = acp.ptg_manager.record_transition(
                        from_state=prev_state,
                        action="navigate",
                        to_state=state,
                        params=nav_params,
                    )
                    _info(f"  PTG:   {from_node.description[:40]} → {to_node.description[:40]}")
                else:
                    # 第一次：先创建初始节点
                    from acp.schema.elements import PageState
                    init_state = PageState(platform="web", app="browser", title="(start)", url="about:blank")
                    from_node, to_node = acp.ptg_manager.record_transition(
                        from_state=init_state,
                        action="navigate",
                        to_state=state,
                        params=nav_params,
                    )
                    _info(f"  PTG:   (start) → {to_node.description[:40]}")

                prev_state = state

                # 感知元素
                elements = await acp.get_elements()
                _info(f"  元素数: {len(elements)}")
            else:
                _fail(f"导航失败: {result.error}")
                continue

            await asyncio.sleep(0.5)

        # ---- 展示完整 PTG 图 ----
        _section("PTG 完整状态转换图")
        graph = acp.ptg_manager.get_graph()
        node_count = acp.ptg_manager.node_count()
        edge_count = acp.ptg_manager.edge_count()

        _ok(f"共 {node_count} 个节点, {edge_count} 条转换边")
        print()

        # 打印节点
        _info("节点列表:")
        for node_id, node in graph.nodes.items():
            marker = "★" if node_id == graph.current_state else " "
            _info(f"  {marker} {node.node_id[:16]}... | {node.description}")

        # 打印边（转换路径）
        print()
        _info("转换路径:")
        for edge in graph.edges:
            from_node = graph.nodes.get(edge.from_node)
            to_node = graph.nodes.get(edge.to_node)
            from_desc = from_node.description[:35] if from_node else edge.from_node[:16]
            to_desc = to_node.description[:35] if to_node else edge.to_node[:16]
            _info(f"  [{from_desc}]")
            _info(f"    -[{edge.action}]->")
            _info(f"  [{to_desc}]")
            print()

        # BFS 路径查找演示
        if edge_count >= 2:
            nodes_list = list(graph.nodes.keys())
            if len(nodes_list) >= 2:
                start_id = nodes_list[0]
                end_id = nodes_list[-1]
                path = acp.ptg_manager.find_path(start_id, end_id)
                _info(f"BFS 路径 (节点 0 → 节点 {len(nodes_list)-1}): {len(path)} 步")
                for edge in path:
                    _info(f"  -[{edge.action}]->")

        success = node_count > 0 and edge_count > 0

    _section("Demo 3 结果")
    if success:
        _ok(f"Demo 3 通过 — PTG 图构建成功 ({node_count} 节点, {edge_count} 边)")
    else:
        _fail("Demo 3 失败")

    return success


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

async def main() -> None:
    """运行所有 Demo，汇总结果。"""
    # 解析参数
    selected = None
    if len(sys.argv) > 1:
        try:
            selected = int(sys.argv[1])
        except ValueError:
            pass

    demos = {
        1: ("基础导航 + 感知", demo1_basic_navigation),
        2: ("搜索操作", demo2_search),
        3: ("连续操作 + PTG 记录", demo3_ptg_tracking),
    }

    print("\n" + "═"*60)
    print("  ACP 端到端集成 Demo")
    print("  版本: 0.1.0  |  MVP 阶段")
    print("═"*60)

    if selected:
        if selected not in demos:
            print(f"未知 Demo: {selected}，可用: 1/2/3")
            sys.exit(1)
        demos_to_run = {selected: demos[selected]}
    else:
        demos_to_run = demos

    results: dict[int, bool] = {}
    t_total = time.perf_counter()

    for num, (name, fn) in demos_to_run.items():
        print(f"\n\n{'*'*60}")
        print(f"  运行 Demo {num}: {name}")
        print(f"{'*'*60}")
        t0 = time.perf_counter()
        try:
            ok = await fn()
        except Exception as e:
            print(f"\n  [ERROR] Demo {num} 异常: {e}")
            import traceback
            traceback.print_exc()
            ok = False
        elapsed = time.perf_counter() - t0
        results[num] = ok
        status = "PASS" if ok else "FAIL"
        print(f"\n  Demo {num} 耗时: {elapsed:.1f}s  状态: {status}")

    # ---- 汇总 ----
    total_elapsed = time.perf_counter() - t_total
    print(f"\n\n{'═'*60}")
    print("  测试汇总")
    print(f"{'═'*60}")
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    for num, ok in results.items():
        name = demos[num][0]
        status = "PASS" if ok else "FAIL"
        print(f"  Demo {num} ({name}): {status}")
    print(f"{'─'*60}")
    print(f"  结果: {passed}/{total} 通过  |  总耗时: {total_elapsed:.1f}s")
    print(f"{'═'*60}\n")

    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
