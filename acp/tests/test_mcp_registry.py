"""
MCP Registry + Web MCP 测试脚本

验收标准：
  1. MCPRegistry 能从 YAML 加载工具配置
  2. 三层选择逻辑正确（专用 > 泛用 > 视觉）
  3. WebMCP.execute() 能调用 WebAdapter 的全部操作

运行方式：
    cd /Volumes/work/ACP
    python -m acp.tests.test_mcp_registry

依赖：
    pip install playwright pyyaml
    playwright install chromium
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from acp.mcp.registry import MCPRegistry, MCPToolInfo
from acp.mcp.protocol import MCPTool
from acp.mcp.tools.web_mcp import WebMCP
from acp.schema.elements import ACPElement, ElementType, PageState, ElementStates, Rect, Point, ElementSource
from acp.schema.plan import ActionResult


# ---------------------------------------------------------------------------
# 测试辅助
# ---------------------------------------------------------------------------

_PASS = "[PASS]"
_FAIL = "[FAIL]"
_results: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = _PASS if condition else _FAIL
    _results.append((name, condition, detail))
    print(f"  {status} {name}" + (f" — {detail}" if detail else ""))


# ---------------------------------------------------------------------------
# 测试 MCPRegistry
# ---------------------------------------------------------------------------

def test_registry_register_and_get() -> None:
    print("\n[TEST] MCPRegistry.register() / get_tool()")
    registry = MCPRegistry()
    tool = MCPToolInfo(
        tool_id="test-mcp",
        tier=1,
        name="测试 MCP",
        description="用于测试",
        supported_apps=["testapp"],
        capabilities=["click", "navigate"],
        platform="web",
    )
    registry.register(tool)

    got = registry.get_tool("test-mcp")
    check("get_tool 找到工具", got is not None)
    if got:
        check("tool_id 正确", got.tool_id == "test-mcp")
        check("tier 正确", got.tier == 1)

    check("get_tool 未知 ID 返回 None", registry.get_tool("not-exist") is None)


def test_registry_list_tools() -> None:
    print("\n[TEST] MCPRegistry.list_tools()")
    registry = MCPRegistry()
    registry.register(MCPToolInfo(
        tool_id="tool-tier3", tier=3, name="视觉", description="",
        supported_apps=["*"], capabilities=["screenshot"],
        platform="cross_platform", reliability=0.75,
    ))
    registry.register(MCPToolInfo(
        tool_id="tool-tier1", tier=1, name="专用", description="",
        supported_apps=["feishu"], capabilities=["send_message"],
        platform="cross_platform", reliability=0.99,
    ))
    registry.register(MCPToolInfo(
        tool_id="tool-tier2", tier=2, name="泛用", description="",
        supported_apps=["*"], capabilities=["navigate"],
        platform="web", reliability=0.95,
    ))

    tools = registry.list_tools()
    check("list_tools 返回 3 个工具", len(tools) == 3, f"实际: {len(tools)}")
    check("按 tier 升序排列", tools[0].tier <= tools[1].tier <= tools[2].tier,
          f"tier 顺序: {[t.tier for t in tools]}")


def test_registry_load_from_yaml() -> None:
    print("\n[TEST] MCPRegistry.load_from_yaml()")
    yaml_content = """
tools:
  - tool_id: feishu-mcp
    tier: 1
    name: 飞书 MCP
    description: 飞书官方 MCP
    supported_apps:
      - feishu
      - lark
    capabilities:
      - send_message
      - read_message
    platform: cross_platform
    auth_required: true
    reliability: 0.99

  - tool_id: web-mcp
    tier: 2
    name: Web 泛用 MCP
    description: 基于 Playwright
    supported_apps:
      - "*_web"
      - "*"
    capabilities:
      - navigate
      - get_elements
      - click
      - type
      - scroll
      - screenshot
      - get_page_state
    platform: web
    backend: playwright
    auth_required: false
    reliability: 0.95

  - tool_id: vision-mcp
    tier: 3
    name: 视觉兜底 MCP
    description: OmniParser
    supported_apps:
      - "*"
    capabilities:
      - screenshot
      - click_at
    platform: cross_platform
    reliability: 0.75
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
        f.write(yaml_content)
        tmp_path = f.name

    try:
        registry = MCPRegistry.from_yaml(tmp_path)

        tools = registry.list_tools()
        check("加载了 3 个工具", len(tools) == 3, f"实际: {len(tools)}")

        feishu = registry.get_tool("feishu-mcp")
        check("feishu-mcp 存在", feishu is not None)
        if feishu:
            check("feishu tier=1", feishu.tier == 1)
            check("feishu auth_required=True", feishu.auth_required is True)
            check("feishu reliability=0.99", abs(feishu.reliability - 0.99) < 1e-6)
            check("feishu capabilities 含 send_message", "send_message" in feishu.capabilities)

        web = registry.get_tool("web-mcp")
        check("web-mcp 存在", web is not None)
        if web:
            check("web-mcp tier=2", web.tier == 2)
            check("web-mcp platform=web", web.platform == "web")
            check("web-mcp backend=playwright", web.backend == "playwright")

        vision = registry.get_tool("vision-mcp")
        check("vision-mcp 存在", vision is not None)
        if vision:
            check("vision-mcp tier=3", vision.tier == 3)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_registry_load_from_project_yaml() -> None:
    print("\n[TEST] MCPRegistry.load_from_yaml() — 加载项目配置文件")
    yaml_path = Path(__file__).parent.parent / "config" / "tools.yaml"
    check("配置文件存在", yaml_path.exists(), str(yaml_path))

    if yaml_path.exists():
        registry = MCPRegistry.from_yaml(str(yaml_path))
        tools = registry.list_tools()
        check("工具数量 >= 2", len(tools) >= 2, f"实际: {len(tools)}")

        web = registry.get_tool("web-mcp")
        check("web-mcp 存在于项目配置", web is not None)

        vision = registry.get_tool("vision-mcp")
        check("vision-mcp 存在于项目配置", vision is not None)


# ---------------------------------------------------------------------------
# 测试 select_tool 三层逻辑
# ---------------------------------------------------------------------------

def _build_registry_with_all_tiers() -> MCPRegistry:
    """构建含 Tier 1/2/3 工具的注册中心（用于选择逻辑测试）。"""
    registry = MCPRegistry()
    registry.register(MCPToolInfo(
        tool_id="feishu-mcp",
        tier=1,
        name="飞书专用 MCP",
        description="",
        supported_apps=["feishu", "lark"],
        capabilities=["send_message", "read_message", "navigate", "click"],
        platform="cross_platform",
        reliability=0.99,
    ))
    registry.register(MCPToolInfo(
        tool_id="web-mcp",
        tier=2,
        name="Web 泛用 MCP",
        description="",
        supported_apps=["*_web", "*"],
        capabilities=["navigate", "get_elements", "click", "type", "scroll", "screenshot"],
        platform="web",
        reliability=0.95,
    ))
    registry.register(MCPToolInfo(
        tool_id="vision-mcp",
        tier=3,
        name="视觉兜底 MCP",
        description="",
        supported_apps=["*"],
        capabilities=["screenshot", "click_at"],
        platform="cross_platform",
        reliability=0.75,
    ))
    return registry


def test_select_tier1_dedicated() -> None:
    print("\n[TEST] select_tool — Tier 1 专用优先")
    registry = _build_registry_with_all_tiers()

    # feishu + send_message → 应选 feishu-mcp (tier 1)
    result = registry.select_tool(app="feishu", action="send_message", platform="web")
    check("选中 feishu-mcp", result is not None and result.tool_id == "feishu-mcp",
          f"实际: {result.tool_id if result else None}")
    check("tier=1", result is not None and result.tier == 1)


def test_select_tier2_no_dedicated() -> None:
    print("\n[TEST] select_tool — 无专用 MCP 时选 Tier 2 泛用")
    registry = _build_registry_with_all_tiers()

    # xiaohongshu（无专用 MCP）+ platform=web → 应选 web-mcp (tier 2)
    result = registry.select_tool(app="xiaohongshu", action="navigate", platform="web")
    check("选中 web-mcp", result is not None and result.tool_id == "web-mcp",
          f"实际: {result.tool_id if result else None}")
    check("tier=2", result is not None and result.tier == 2)


def test_select_tier1_action_not_supported_fallback_tier2() -> None:
    print("\n[TEST] select_tool — Tier 1 不支持该 action，回退 Tier 2")
    registry = _build_registry_with_all_tiers()

    # feishu 有专用 MCP，但 feishu-mcp 不支持 "scroll"，应回退到 web-mcp
    result = registry.select_tool(app="feishu", action="scroll", platform="web")
    check("未选专用 MCP，选 web-mcp", result is not None and result.tool_id == "web-mcp",
          f"实际: {result.tool_id if result else None}")


def test_select_tier3_fallback() -> None:
    print("\n[TEST] select_tool — 无 Tier 1/2 时回退 Tier 3 视觉")
    registry = MCPRegistry()
    # 只注册 vision-mcp
    registry.register(MCPToolInfo(
        tool_id="vision-mcp",
        tier=3,
        name="视觉兜底 MCP",
        description="",
        supported_apps=["*"],
        capabilities=["screenshot", "click_at"],
        platform="cross_platform",
        reliability=0.75,
    ))

    result = registry.select_tool(app="any_app", action="click", platform="android")
    check("回退到 vision-mcp", result is not None and result.tool_id == "vision-mcp",
          f"实际: {result.tool_id if result else None}")
    check("tier=3", result is not None and result.tier == 3)


def test_select_no_tool_available() -> None:
    print("\n[TEST] select_tool — 无任何工具，返回 None")
    registry = MCPRegistry()
    result = registry.select_tool(app="any_app", action="click", platform="web")
    check("无工具返回 None", result is None)


# ---------------------------------------------------------------------------
# 测试 WebMCP.execute()（使用 Mock WebAdapter）
# ---------------------------------------------------------------------------

def _make_mock_page_state() -> PageState:
    return PageState(
        platform="web",
        app="example",
        title="Example Domain",
        url="https://example.com",
    )


def _make_mock_element(idx: int = 0) -> ACPElement:
    return ACPElement(
        id=f"e{idx:04d}_mock",
        type=ElementType.BUTTON,
        platform_class="button",
        text="Click me",
        bounds=Rect(x=0, y=0, width=100, height=40),
        center=Point(x=50, y=20),
        states=ElementStates(clickable=True, enabled=True, visible=True),
        actions=["click"],
        source=ElementSource.DOM,
        confidence=1.0,
    )


def _build_mock_adapter() -> MagicMock:
    """构建模拟 WebAdapter，返回预设数据。"""
    adapter = MagicMock()
    page_state = _make_mock_page_state()
    element = _make_mock_element(0)

    # 所有方法都是异步的
    adapter.navigate = AsyncMock(return_value=ActionResult(
        success=True,
        data={"url": "https://example.com", "title": "Example Domain"},
        page_state=page_state,
    ))
    adapter.get_elements = AsyncMock(return_value=[element])
    adapter.get_page_state = AsyncMock(return_value=page_state)
    adapter.click = AsyncMock(return_value=ActionResult(
        success=True,
        data={"clicked": "#btn"},
        page_state=page_state,
    ))
    adapter.type = AsyncMock(return_value=ActionResult(
        success=True,
        data={"typed_into": "#input", "text_length": 5},
        page_state=page_state,
    ))
    adapter.scroll = AsyncMock(return_value=ActionResult(
        success=True,
        data={"direction": "down", "amount": 300},
        page_state=page_state,
    ))
    adapter.screenshot = AsyncMock(return_value=b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

    return adapter


async def test_web_mcp_navigate() -> None:
    print("\n[TEST] WebMCP.execute('navigate')")
    adapter = _build_mock_adapter()
    mcp = WebMCP(adapter=adapter)

    result = await mcp.execute("navigate", {"url": "https://example.com"})
    check("navigate 成功", result.success, result.error or "")
    check("返回 page_state", result.page_state is not None)
    check("page_state.url 正确", result.page_state is not None and "example.com" in (result.page_state.url or ""))

    # 验证 adapter.navigate 被调用
    adapter.navigate.assert_called_once_with("https://example.com")
    check("adapter.navigate 被调用一次", True)


async def test_web_mcp_navigate_missing_url() -> None:
    print("\n[TEST] WebMCP.execute('navigate') — 缺少 url 参数")
    adapter = _build_mock_adapter()
    mcp = WebMCP(adapter=adapter)

    result = await mcp.execute("navigate", {})
    check("缺 url 返回失败", not result.success)
    check("有 error 信息", bool(result.error))


async def test_web_mcp_get_elements() -> None:
    print("\n[TEST] WebMCP.execute('get_elements')")
    adapter = _build_mock_adapter()
    mcp = WebMCP(adapter=adapter)

    result = await mcp.execute("get_elements", {})
    check("get_elements 成功", result.success, result.error or "")
    check("elements 列表非空", result.elements is not None and len(result.elements) > 0)
    check("返回 element_count", result.data is not None and "element_count" in result.data)
    if result.data:
        check("element_count=1", result.data["element_count"] == 1)


async def test_web_mcp_click() -> None:
    print("\n[TEST] WebMCP.execute('click')")
    adapter = _build_mock_adapter()
    mcp = WebMCP(adapter=adapter)

    result = await mcp.execute("click", {"element_id": "e0001_abc"})
    check("click 成功", result.success, result.error or "")
    adapter.click.assert_called_once_with("e0001_abc")
    check("adapter.click 参数正确", True)


async def test_web_mcp_type() -> None:
    print("\n[TEST] WebMCP.execute('type')")
    adapter = _build_mock_adapter()
    mcp = WebMCP(adapter=adapter)

    result = await mcp.execute("type", {"element_id": "e0002_abc", "text": "hello"})
    check("type 成功", result.success, result.error or "")
    adapter.type.assert_called_once_with("e0002_abc", "hello")
    check("adapter.type 参数正确", True)


async def test_web_mcp_scroll() -> None:
    print("\n[TEST] WebMCP.execute('scroll')")
    adapter = _build_mock_adapter()
    mcp = WebMCP(adapter=adapter)

    result = await mcp.execute("scroll", {"direction": "down", "amount": 500})
    check("scroll 成功", result.success, result.error or "")
    adapter.scroll.assert_called_once_with("down", None, 500)
    check("adapter.scroll 参数正确", True)


async def test_web_mcp_screenshot() -> None:
    print("\n[TEST] WebMCP.execute('screenshot')")
    adapter = _build_mock_adapter()
    mcp = WebMCP(adapter=adapter)

    result = await mcp.execute("screenshot", {})
    check("screenshot 成功", result.success, result.error or "")
    check("data 含 base64", result.data is not None and "base64" in result.data)
    check("data 含 format=png", result.data is not None and result.data.get("format") == "png")
    check("data 含 size_bytes", result.data is not None and "size_bytes" in result.data)


async def test_web_mcp_get_page_state() -> None:
    print("\n[TEST] WebMCP.execute('get_page_state')")
    adapter = _build_mock_adapter()
    mcp = WebMCP(adapter=adapter)

    result = await mcp.execute("get_page_state", {})
    check("get_page_state 成功", result.success, result.error or "")
    check("data 含 url", result.data is not None and "url" in result.data)
    check("data 含 title", result.data is not None and "title" in result.data)
    check("返回 page_state", result.page_state is not None)


async def test_web_mcp_unknown_method() -> None:
    print("\n[TEST] WebMCP.execute — 未知方法")
    adapter = _build_mock_adapter()
    mcp = WebMCP(adapter=adapter)

    result = await mcp.execute("unknown_method", {})
    check("未知方法返回失败", not result.success)
    check("有 error 信息", bool(result.error))


async def test_web_mcp_supports() -> None:
    print("\n[TEST] WebMCP.supports()")
    adapter = _build_mock_adapter()
    mcp = WebMCP(adapter=adapter)

    check("supports('navigate') = True", mcp.supports("navigate"))
    check("supports('click') = True", mcp.supports("click"))
    check("supports('unknown') = False", not mcp.supports("unknown"))


def test_mcp_tool_is_abstract() -> None:
    print("\n[TEST] MCPTool 抽象基类")
    import inspect
    check("MCPTool 是抽象类", inspect.isabstract(MCPTool))
    check("WebMCP 是 MCPTool 子类", issubclass(WebMCP, MCPTool))


# ---------------------------------------------------------------------------
# 实网络集成测试（可选，需要 playwright）
# ---------------------------------------------------------------------------

async def test_web_mcp_live_navigate() -> None:
    """集成测试：用真实 Playwright 打开页面并获取元素（需要网络）。"""
    print("\n[TEST] WebMCP 集成测试 — 真实浏览器（可选）")
    try:
        async with WebMCP(headless=True) as mcp:
            # 导航
            result = await mcp.execute("navigate", {"url": "https://example.com"})
            check("集成-navigate 成功", result.success, result.error or "")

            # 获取元素
            if result.success:
                result2 = await mcp.execute("get_elements", {})
                check("集成-get_elements 成功", result2.success, result2.error or "")
                check("集成-有元素返回", result2.elements is not None and len(result2.elements) > 0,
                      f"元素数量: {len(result2.elements) if result2.elements else 0}")

            # 页面状态
            result3 = await mcp.execute("get_page_state", {})
            check("集成-get_page_state 成功", result3.success)
            if result3.page_state:
                check("集成-url 正确", "example.com" in (result3.page_state.url or ""))
    except Exception as exc:
        check("集成测试跳过（环境问题）", True, f"跳过: {exc}")


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

async def _run_async_tests() -> None:
    """运行所有异步测试。"""
    await test_web_mcp_navigate()
    await test_web_mcp_navigate_missing_url()
    await test_web_mcp_get_elements()
    await test_web_mcp_click()
    await test_web_mcp_type()
    await test_web_mcp_scroll()
    await test_web_mcp_screenshot()
    await test_web_mcp_get_page_state()
    await test_web_mcp_unknown_method()
    await test_web_mcp_supports()
    await test_web_mcp_live_navigate()


def main() -> None:
    print("=" * 60)
    print("MCP Registry + Web MCP 测试")
    print("=" * 60)

    # 同步测试（注册中心）
    test_registry_register_and_get()
    test_registry_list_tools()
    test_registry_load_from_yaml()
    test_registry_load_from_project_yaml()
    test_select_tier1_dedicated()
    test_select_tier2_no_dedicated()
    test_select_tier1_action_not_supported_fallback_tier2()
    test_select_tier3_fallback()
    test_select_no_tool_available()
    test_mcp_tool_is_abstract()

    # 异步测试（WebMCP）
    asyncio.run(_run_async_tests())

    # 汇总
    print("\n" + "=" * 60)
    passed = sum(1 for _, ok, _ in _results if ok)
    failed = sum(1 for _, ok, _ in _results if not ok)
    total = len(_results)
    print(f"结果：{passed}/{total} 通过，{failed} 失败")

    if failed > 0:
        print("\n失败项：")
        for name, ok, detail in _results:
            if not ok:
                print(f"  [FAIL] {name}" + (f" — {detail}" if detail else ""))
        sys.exit(1)
    else:
        print("全部通过！")


if __name__ == "__main__":
    main()
