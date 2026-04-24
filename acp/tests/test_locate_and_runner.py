"""
测试 _locate() 5 级定位策略 + FlowRunner MCPToolCall 分层

覆盖：
  - _locate() 策略1-5 mock 测试
  - FlowRunner 通过 MCPToolCall → WebMCP → mock adapter 路径
  - MCPToolCall 数据结构
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from acp.adapters.web_adapter import WebAdapter
from acp.brain.flow_runner import FlowRunner, MCPToolCall
from acp.mcp.tools.web_mcp import WebMCP
from acp.schema.elements import (
    ACPElement,
    ElementSource,
    ElementStates,
    ElementType,
    PageState,
    Point,
    Rect,
)
from acp.schema.plan import ActionResult


# ---------------------------------------------------------------------------
# Mock 工具
# ---------------------------------------------------------------------------

def make_element(
    eid: str = "e0001_ab",
    etype: ElementType = ElementType.BUTTON,
    text: str = "提交",
    placeholder: str = "",
    role: str = "button",
    selector: str = "#submit",
) -> ACPElement:
    """构造测试用 ACPElement。"""
    return ACPElement(
        id=eid,
        type=etype,
        platform_class="button",
        text=text,
        placeholder=placeholder or None,
        bounds=Rect(x=10, y=20, width=100, height=40),
        center=Point(x=60, y=40),
        states=ElementStates(clickable=True, enabled=True, visible=True),
        selector=selector,
        actions=["click"],
        source=ElementSource.DOM,
        confidence=1.0,
    )


def make_mock_page() -> MagicMock:
    """构造 mock Page 对象，模拟 Playwright Page API。"""
    page = MagicMock()
    page.url = "https://example.com/login"

    # Locator mock
    def make_locator(count=1, is_visible=True):
        loc = MagicMock()
        loc.count = AsyncMock(return_value=count)
        loc.first = loc
        loc.filter = MagicMock(return_value=loc)
        loc.nth = MagicMock(return_value=loc)
        loc.is_visible = AsyncMock(return_value=is_visible)
        loc.get_attribute = AsyncMock(return_value=None)
        loc.evaluate = AsyncMock(return_value=0)
        loc.click = AsyncMock()
        loc.fill = AsyncMock()
        return loc

    page.get_by_placeholder = MagicMock(return_value=make_locator(1))
    page.get_by_role = MagicMock(return_value=make_locator(1))
    page.get_by_text = MagicMock(return_value=make_locator(1))
    page.locator = MagicMock(return_value=make_locator(1))
    page.evaluate = AsyncMock(return_value=None)
    page.wait_for_load_state = AsyncMock()
    page.title = AsyncMock(return_value="测试页面")
    return page


def make_mock_adapter(page=None) -> WebAdapter:
    """构造 mock WebAdapter。"""
    adapter = MagicMock(spec=WebAdapter)
    adapter._page = page or make_mock_page()
    adapter._element_cache = {}
    adapter._element_text_cache = {}
    adapter._element_semantic_cache = {}
    return adapter


# ---------------------------------------------------------------------------
# TestMCPToolCall
# ---------------------------------------------------------------------------

class TestMCPToolCall:
    """MCPToolCall 数据结构测试。"""

    def test_basic_fields(self):
        call = MCPToolCall(tool_id="web-mcp", method="navigate", params={"url": "https://x.com"})
        assert call.tool_id == "web-mcp"
        assert call.method == "navigate"
        assert call.params == {"url": "https://x.com"}

    def test_repr(self):
        call = MCPToolCall(tool_id="web-mcp", method="click", params={"element_id": "e001"})
        r = repr(call)
        assert "web-mcp" in r
        assert "click" in r

    def test_different_methods(self):
        methods = ["navigate", "click", "type", "scroll", "get_elements", "screenshot"]
        for m in methods:
            call = MCPToolCall(tool_id="web-mcp", method=m, params={})
            assert call.method == m


# ---------------------------------------------------------------------------
# TestLocate — _locate() 5 级定位策略
# ---------------------------------------------------------------------------

class TestLocate:
    """_locate() 方法的 5 级定位策略 mock 测试。"""

    def setup_method(self):
        """每个测试前构造 adapter + mock page。"""
        self.page = make_mock_page()
        self.adapter = WebAdapter.__new__(WebAdapter)
        self.adapter._page = self.page
        self.adapter._element_cache = {}
        self.adapter._element_semantic_cache = {}

    def _setup_element(
        self,
        eid: str,
        placeholder: str = "",
        role: str = "",
        text: str = "",
        tag: str = "button",
        selector: str = "#btn",
    ):
        """注册元素到缓存。"""
        self.adapter._element_cache[eid] = selector
        self.adapter._element_semantic_cache[eid] = {
            "placeholder": placeholder,
            "role": role,
            "text": text,
            "tag": tag,
        }

    @pytest.mark.asyncio
    async def test_strategy1_placeholder(self):
        """策略1: placeholder 精确定位。"""
        eid = "e0001"
        self._setup_element(eid, placeholder="请输入邮箱", tag="input")

        mock_loc = MagicMock()
        mock_loc.count = AsyncMock(return_value=1)
        self.page.get_by_placeholder = MagicMock(return_value=mock_loc)

        result = await self.adapter._locate(eid)
        self.page.get_by_placeholder.assert_called_once_with("请输入邮箱", exact=True)
        assert result == mock_loc

    @pytest.mark.asyncio
    async def test_strategy1b_placeholder_not_unique(self):
        """策略1b: placeholder 不唯一时回退 CSS selector。"""
        eid = "e0002"
        self._setup_element(eid, placeholder="请输入", tag="input", selector="input[name=email]")

        # placeholder 返回 2 个匹配
        ph_loc = MagicMock()
        ph_loc.count = AsyncMock(return_value=2)
        self.page.get_by_placeholder = MagicMock(return_value=ph_loc)

        # CSS selector 返回 1 个
        css_loc = MagicMock()
        css_loc.count = AsyncMock(return_value=1)
        css_loc.first = css_loc
        self.page.locator = MagicMock(return_value=css_loc)

        result = await self.adapter._locate(eid)
        # 应该回退到 CSS selector
        self.page.locator.assert_called_with("input[name=email]")

    @pytest.mark.asyncio
    async def test_strategy2_role_and_text(self):
        """策略2: role + name 定位。"""
        eid = "e0003"
        self._setup_element(eid, role="button", text="登录", tag="button")

        mock_loc = MagicMock()
        mock_loc.count = AsyncMock(return_value=1)
        self.page.get_by_role = MagicMock(return_value=mock_loc)

        result = await self.adapter._locate(eid)
        self.page.get_by_role.assert_called_once_with("button", name="登录", exact=True)
        assert result == mock_loc

    @pytest.mark.asyncio
    async def test_strategy3_text_only(self):
        """策略3: 纯文本匹配（无 placeholder 和 role）。"""
        eid = "e0004"
        self._setup_element(eid, text="立即注册", tag="a")

        # get_by_role 返回 0（无匹配，进入策略3）
        role_loc = MagicMock()
        role_loc.count = AsyncMock(return_value=0)
        self.page.get_by_role = MagicMock(return_value=role_loc)

        text_loc = MagicMock()
        text_loc.count = AsyncMock(return_value=1)
        self.page.get_by_text = MagicMock(return_value=text_loc)

        result = await self.adapter._locate(eid)
        self.page.get_by_text.assert_called_once_with("立即注册", exact=True)
        assert result == text_loc

    @pytest.mark.asyncio
    async def test_strategy4_css_selector(self):
        """策略4: CSS selector fallback。"""
        eid = "e0005"
        self._setup_element(eid, selector="#login-btn")

        # 没有 placeholder、role、text，直接走 CSS selector
        css_loc = MagicMock()
        css_loc.count = AsyncMock(return_value=1)
        self.page.locator = MagicMock(return_value=css_loc)

        result = await self.adapter._locate(eid)
        # 应该调用 CSS selector
        self.page.locator.assert_called_with("#login-btn")

    @pytest.mark.asyncio
    async def test_strategy5_no_cache(self):
        """策略5: 无语义缓存且无 selector，返回 None。"""
        eid = "e9999_nonexistent"
        # 不添加任何缓存

        result = await self.adapter._locate(eid)
        assert result is None

    @pytest.mark.asyncio
    async def test_locate_with_only_selector_cache(self):
        """只有 selector 缓存（无语义缓存）时直接用 CSS selector。"""
        eid = "e0006"
        self.adapter._element_cache[eid] = ".my-button"
        # 不设置 semantic cache

        css_loc = MagicMock()
        css_loc.count = AsyncMock(return_value=1)
        css_loc.first = css_loc
        self.page.locator = MagicMock(return_value=css_loc)

        result = await self.adapter._locate(eid)
        self.page.locator.assert_called_with(".my-button")
        assert result is not None


# ---------------------------------------------------------------------------
# TestFlowRunnerMCP — FlowRunner 通过 MCPToolCall 路径测试
# ---------------------------------------------------------------------------

class TestFlowRunnerMCP:
    """FlowRunner MCPToolCall 分层测试（mock WebMCP）。"""

    def _make_runner(self, tmp_path) -> "tuple[FlowRunner, MagicMock]":
        """构造带 mock WebMCP 的 FlowRunner。"""
        # 写一个最小的 flows.yaml
        flows_content = """
flows:
  test_flow:
    description: 测试流程
    steps:
      - action: navigate
        url: https://example.com
        wait: 0
      - action: wait
        wait: 0
"""
        (tmp_path / "flows.yaml").write_text(flows_content)
        (tmp_path / "site.yaml").write_text("name: test\nbase_url: https://example.com\n")

        mock_mcp = MagicMock(spec=WebMCP)
        mock_mcp.execute = AsyncMock(return_value=ActionResult(
            success=True,
            data={"url": "https://example.com"},
            page_state=PageState(platform="web", app="example", url="https://example.com", title="Test"),
        ))
        mock_mcp.start = AsyncMock()
        mock_mcp.close = AsyncMock()
        mock_mcp._adapter = MagicMock()
        mock_mcp._adapter._page = make_mock_page()

        runner = FlowRunner(site_dir=str(tmp_path), mcp=mock_mcp)
        return runner, mock_mcp

    @pytest.mark.asyncio
    async def test_call_mcp_generates_tool_call(self, tmp_path):
        """_call_mcp() 应通过 WebMCP.execute 调用，而非直接调 adapter。"""
        runner, mock_mcp = self._make_runner(tmp_path)

        result = await runner._call_mcp("navigate", {"url": "https://example.com"})

        mock_mcp.execute.assert_called_once_with("navigate", {"url": "https://example.com"})
        assert result.success is True

    @pytest.mark.asyncio
    async def test_run_flow_uses_mcp_layer(self, tmp_path):
        """run() 执行 navigate 步骤时应通过 MCP 层，而非直接调 adapter.navigate。"""
        runner, mock_mcp = self._make_runner(tmp_path)

        ok = await runner.run("test_flow", keep_open=0)

        assert ok is True
        # 验证通过了 MCP execute（而非直接 adapter.navigate）
        assert mock_mcp.execute.called
        calls = [c.args[0] for c in mock_mcp.execute.call_args_list]
        assert "navigate" in calls

    @pytest.mark.asyncio
    async def test_flow_not_found(self, tmp_path):
        """请求不存在的 flow 应返回 False。"""
        runner, _ = self._make_runner(tmp_path)
        ok = await runner.run("nonexistent_flow", keep_open=0)
        assert ok is False

    @pytest.mark.asyncio
    async def test_get_elements_via_mcp(self, tmp_path):
        """_get_elements_via_mcp() 应通过 MCP get_elements 方法获取元素。"""
        runner, mock_mcp = self._make_runner(tmp_path)
        el = make_element()
        mock_mcp.execute = AsyncMock(return_value=ActionResult(
            success=True,
            elements=[el],
            data={"element_count": 1},
        ))

        elements = await runner._get_elements_via_mcp()
        assert len(elements) == 1
        assert elements[0].id == el.id
        mock_mcp.execute.assert_called_once_with("get_elements", {})

    @pytest.mark.asyncio
    async def test_mcp_tool_call_struct(self, tmp_path):
        """MCPToolCall 在 _call_mcp 内部正确构造。"""
        runner, mock_mcp = self._make_runner(tmp_path)

        # 验证多种方法
        for method, params in [
            ("navigate", {"url": "https://x.com"}),
            ("click", {"element_id": "e001"}),
            ("type", {"element_id": "e002", "text": "hello"}),
            ("scroll", {"direction": "down"}),
        ]:
            mock_mcp.execute.reset_mock()
            await runner._call_mcp(method, params)
            mock_mcp.execute.assert_called_once_with(method, params)


# ---------------------------------------------------------------------------
# TestFlowRunnerOwnsLifecycle — FlowRunner 自己管理 MCP 生命周期
# ---------------------------------------------------------------------------

class TestFlowRunnerOwnsLifecycle:
    """FlowRunner 自己启动/关闭 WebMCP 的生命周期测试。"""

    @pytest.mark.asyncio
    async def test_runner_starts_and_closes_mcp(self, tmp_path):
        """当 mcp=None 时，FlowRunner 应自行启动并关闭 WebMCP。"""
        flows_content = """
flows:
  minimal:
    description: 最小流程
    steps:
      - action: wait
        wait: 0
"""
        (tmp_path / "flows.yaml").write_text(flows_content)
        (tmp_path / "site.yaml").write_text("name: test\n")

        with patch("acp.brain.flow_runner.WebMCP") as MockWebMCP:
            mock_instance = MagicMock()
            mock_instance.start = AsyncMock()
            mock_instance.close = AsyncMock()
            mock_instance._adapter = MagicMock()
            mock_instance._adapter._page = make_mock_page()
            mock_instance.execute = AsyncMock(return_value=ActionResult(success=True))
            MockWebMCP.return_value = mock_instance

            runner = FlowRunner(site_dir=str(tmp_path), mcp=None, headless=True)
            await runner.run("minimal", keep_open=0)

            # 应调用 start 和 close
            mock_instance.start.assert_called_once()
            mock_instance.close.assert_called_once()
            # 构造时应传入 headless=True
            MockWebMCP.assert_called_once_with(headless=True)
