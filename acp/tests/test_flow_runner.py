"""
FlowRunner 单元测试

覆盖：
  - 流程执行逻辑（mock WebMCP）
  - 变量替换（${auth.email} / extra_vars）
  - 步骤失败后的重试（fallback 逻辑）
  - verify 步骤验证
  - MCP 分层路径（不直接调 adapter）
  - MCPRegistry 集成（传入 registry 时路径）
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest

from acp.brain.flow_runner import FlowRunner, MCPToolCall
from acp.mcp.tools.web_mcp import WebMCP
from acp.schema.elements import PageState
from acp.schema.plan import ActionResult


# ---------------------------------------------------------------------------
# 测试工具
# ---------------------------------------------------------------------------

def make_page_state(url: str = "https://example.com", title: str = "Test") -> PageState:
    return PageState(platform="web", app="example", url=url, title=title)


def make_action_result(success: bool = True, url: str = "https://example.com", title: str = "Test") -> ActionResult:
    return ActionResult(
        success=success,
        data={"url": url},
        page_state=make_page_state(url=url, title=title),
    )


def make_mock_mcp(default_success: bool = True) -> MagicMock:
    """构造默认成功的 mock WebMCP。"""
    mock_mcp = MagicMock(spec=WebMCP)
    mock_mcp.execute = AsyncMock(return_value=make_action_result(success=default_success))
    mock_mcp.start = AsyncMock()
    mock_mcp.close = AsyncMock()
    mock_mcp._adapter = MagicMock()
    mock_mcp._adapter._page = MagicMock()
    return mock_mcp


def make_runner(tmp_path, flows_content: str, credentials: dict = None, mcp=None) -> FlowRunner:
    """创建测试用 FlowRunner，注入 mock MCP。"""
    (tmp_path / "flows.yaml").write_text(flows_content)
    (tmp_path / "site.yaml").write_text("name: test_site\nbase_url: https://example.com\n")

    if credentials:
        import yaml
        (tmp_path / "credentials.yaml").write_text(yaml.dump(credentials))

    if mcp is None:
        mcp = make_mock_mcp()

    runner = FlowRunner(site_dir=str(tmp_path), mcp=mcp)
    return runner


# ---------------------------------------------------------------------------
# TestVariableResolution — 变量替换
# ---------------------------------------------------------------------------

class TestVariableResolution:
    """测试 ${xxx.yyy} 变量替换逻辑。"""

    def setup_method(self):
        self.credentials = {
            "auth": {
                "email": "user@example.com",
                "password": "Secret123",
            }
        }

    def _make_runner(self, tmp_path) -> FlowRunner:
        flows = "flows:\n  dummy:\n    steps: []\n"
        return make_runner(tmp_path, flows, credentials=self.credentials)

    def test_resolve_from_credentials(self, tmp_path):
        """${auth.email} 应从 credentials 中替换。"""
        runner = self._make_runner(tmp_path)
        result = runner._resolve_vars("${auth.email}")
        assert result == "user@example.com"

    def test_resolve_password_from_credentials(self, tmp_path):
        """${auth.password} 应从 credentials 中替换。"""
        runner = self._make_runner(tmp_path)
        result = runner._resolve_vars("${auth.password}")
        assert result == "Secret123"

    def test_resolve_from_extra_vars(self, tmp_path):
        """extra_vars 优先于 credentials。"""
        runner = self._make_runner(tmp_path)
        result = runner._resolve_vars("${comment}", extra_vars={"comment": "ACP测试"})
        assert result == "ACP测试"

    def test_extra_vars_override_credentials(self, tmp_path):
        """extra_vars 中的键应覆盖 credentials 中的同名键。"""
        runner = self._make_runner(tmp_path)
        result = runner._resolve_vars(
            "${auth.email}",
            extra_vars={"auth": {"email": "override@test.com"}},
        )
        assert result == "override@test.com"

    def test_unknown_var_preserved(self, tmp_path):
        """未知变量不应被替换，保留原始占位符。"""
        runner = self._make_runner(tmp_path)
        result = runner._resolve_vars("${unknown.var}")
        assert result == "${unknown.var}"

    def test_mixed_text_and_vars(self, tmp_path):
        """混合文本和变量的字符串应正确替换。"""
        runner = self._make_runner(tmp_path)
        result = runner._resolve_vars("邮箱: ${auth.email}，密码: ${auth.password}")
        assert result == "邮箱: user@example.com，密码: Secret123"

    def test_no_vars(self, tmp_path):
        """不含变量的文本应原样返回。"""
        runner = self._make_runner(tmp_path)
        result = runner._resolve_vars("https://example.com/login")
        assert result == "https://example.com/login"

    def test_empty_string(self, tmp_path):
        """空字符串应原样返回。"""
        runner = self._make_runner(tmp_path)
        result = runner._resolve_vars("")
        assert result == ""


# ---------------------------------------------------------------------------
# TestStepExecution — 步骤执行逻辑
# ---------------------------------------------------------------------------

class TestStepExecution:
    """测试各类步骤的执行路径（通过 mock MCP）。"""

    @pytest.mark.asyncio
    async def test_navigate_step_calls_mcp(self, tmp_path):
        """navigate 步骤应通过 MCP execute('navigate') 调用。"""
        flows = """
flows:
  login:
    description: 导航测试
    steps:
      - action: navigate
        url: https://example.com/login
        wait: 0
"""
        mock_mcp = make_mock_mcp()
        runner = make_runner(tmp_path, flows, mcp=mock_mcp)

        ok = await runner.run("login")

        assert ok is True
        calls = [c.args[0] for c in mock_mcp.execute.call_args_list]
        assert "navigate" in calls
        # 验证参数
        nav_call = next(c for c in mock_mcp.execute.call_args_list if c.args[0] == "navigate")
        assert nav_call.args[1]["url"] == "https://example.com/login"

    @pytest.mark.asyncio
    async def test_wait_step_succeeds(self, tmp_path):
        """wait 步骤应直接返回成功，不调用 MCP。"""
        flows = """
flows:
  wait_flow:
    description: 等待测试
    steps:
      - action: wait
        wait: 0
"""
        mock_mcp = make_mock_mcp()
        runner = make_runner(tmp_path, flows, mcp=mock_mcp)

        ok = await runner.run("wait_flow")

        assert ok is True
        # wait 步骤不应调用 MCP execute
        mock_mcp.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_scroll_step_calls_mcp(self, tmp_path):
        """scroll 步骤应通过 MCP execute('scroll') 调用。"""
        flows = """
flows:
  scroll_flow:
    description: 滚动测试
    steps:
      - action: scroll
        direction: down
        wait: 0
"""
        mock_mcp = make_mock_mcp()
        runner = make_runner(tmp_path, flows, mcp=mock_mcp)

        ok = await runner.run("scroll_flow")

        assert ok is True
        calls = [c.args[0] for c in mock_mcp.execute.call_args_list]
        assert "scroll" in calls

    @pytest.mark.asyncio
    async def test_unknown_action_returns_false(self, tmp_path):
        """未知 action 应返回 False 且不中断其他步骤。"""
        flows = """
flows:
  unknown_flow:
    description: 未知操作测试
    steps:
      - action: unknown_xyz
        wait: 0
"""
        mock_mcp = make_mock_mcp()
        runner = make_runner(tmp_path, flows, mcp=mock_mcp)

        ok = await runner.run("unknown_flow")

        # 未知 action 步骤失败，但非 verify，不中断
        assert ok is False

    @pytest.mark.asyncio
    async def test_multiple_steps_all_succeed(self, tmp_path):
        """多个步骤全部成功时应返回 True。"""
        flows = """
flows:
  multi_flow:
    description: 多步骤测试
    steps:
      - action: navigate
        url: https://example.com
        wait: 0
      - action: wait
        wait: 0
      - action: scroll
        direction: down
        wait: 0
"""
        mock_mcp = make_mock_mcp()
        runner = make_runner(tmp_path, flows, mcp=mock_mcp)

        ok = await runner.run("multi_flow")

        assert ok is True
        assert len(runner.log) == 3
        assert all(l["success"] for l in runner.log)

    @pytest.mark.asyncio
    async def test_step_failure_logged(self, tmp_path):
        """步骤失败应记录到 log 中。"""
        flows = """
flows:
  fail_flow:
    description: 失败测试
    steps:
      - action: navigate
        url: https://example.com
        wait: 0
      - action: wait
        wait: 0
"""
        mock_mcp = make_mock_mcp()
        # navigate 步骤：先调用 navigate（失败），再调用 get_page_state（成功，显示当前页）
        mock_mcp.execute = AsyncMock(side_effect=[
            make_action_result(success=False),   # navigate 失败
            make_action_result(success=True),    # get_page_state（navigate 失败后依然调用）
        ])
        runner = make_runner(tmp_path, flows, mcp=mock_mcp)

        ok = await runner.run("fail_flow")

        assert ok is False
        assert runner.log[0]["success"] is False


# ---------------------------------------------------------------------------
# TestVerifyStep — verify 步骤
# ---------------------------------------------------------------------------

class TestVerifyStep:
    """测试 verify 步骤的验证逻辑。"""

    @pytest.mark.asyncio
    async def test_verify_login_success(self, tmp_path):
        """verify 步骤：URL 不包含 /login 时应验证成功。"""
        flows = """
flows:
  verify_flow:
    description: 验证登录
    steps:
      - action: verify
        expected: 登录成功
        wait: 0
"""
        mock_mcp = make_mock_mcp()
        # get_page_state 返回非登录页 URL
        mock_mcp.execute = AsyncMock(return_value=ActionResult(
            success=True,
            page_state=make_page_state(url="https://example.com/dashboard", title="首页"),
        ))
        runner = make_runner(tmp_path, flows, mcp=mock_mcp)

        ok = await runner.run("verify_flow")

        assert ok is True

    @pytest.mark.asyncio
    async def test_verify_login_failure(self, tmp_path):
        """verify 步骤：URL 包含 /login 时应验证失败并中断流程。"""
        flows = """
flows:
  verify_fail_flow:
    description: 验证登录失败
    steps:
      - action: verify
        expected: 登录成功
        wait: 0
      - action: wait
        wait: 0
"""
        mock_mcp = make_mock_mcp()
        # 仍在登录页
        mock_mcp.execute = AsyncMock(return_value=ActionResult(
            success=True,
            page_state=make_page_state(url="https://example.com/login", title="登录"),
        ))
        runner = make_runner(tmp_path, flows, mcp=mock_mcp)

        ok = await runner.run("verify_fail_flow")

        # verify 失败后应中断，返回 False
        assert ok is False
        # verify 步骤之后的 wait 不应被执行
        assert len(runner.log) == 1

    @pytest.mark.asyncio
    async def test_verify_without_login_keyword(self, tmp_path):
        """verify 步骤：expected 不含登录关键字时应默认通过。"""
        flows = """
flows:
  verify_generic:
    description: 通用验证
    steps:
      - action: verify
        expected: 页面正常显示
        wait: 0
"""
        mock_mcp = make_mock_mcp()
        mock_mcp.execute = AsyncMock(return_value=ActionResult(
            success=True,
            page_state=make_page_state(url="https://example.com", title="首页"),
        ))
        runner = make_runner(tmp_path, flows, mcp=mock_mcp)

        ok = await runner.run("verify_generic")

        assert ok is True


# ---------------------------------------------------------------------------
# TestClickRetry — click 步骤失败后重试逻辑
# ---------------------------------------------------------------------------

class TestClickRetry:
    """测试 click 步骤失败后的重试逻辑。"""

    @pytest.mark.asyncio
    async def test_click_retry_on_failure(self, tmp_path):
        """click 步骤第一次失败后应重试（通过 _find_element + _call_mcp 再次调用）。"""
        flows = """
flows:
  click_flow:
    description: 点击重试测试
    steps:
      - action: click
        target: 登录按钮
        wait: 0
"""
        mock_mcp = make_mock_mcp()
        # get_elements 返回空列表（简化：让 _find_element 走 None）
        from acp.schema.plan import ActionResult as AR
        mock_mcp.execute = AsyncMock(return_value=AR(
            success=True,
            elements=[],
            page_state=make_page_state(),
        ))

        runner = make_runner(tmp_path, flows, mcp=mock_mcp)

        # mock _find_element 让它返回 None（目标元素找不到）
        runner._find_element = AsyncMock(return_value=None)

        ok = await runner.run("click_flow")

        # 元素找不到时步骤失败
        assert ok is False
        assert runner.log[0]["success"] is False

    @pytest.mark.asyncio
    async def test_click_retries_on_mcp_failure(self, tmp_path):
        """click 步骤 MCP 调用失败后应重试一次（两次调用 _call_mcp）。"""
        flows = """
flows:
  click_retry_flow:
    description: 点击 MCP 失败重试
    steps:
      - action: click
        target: 提交按钮
        wait: 0
"""
        mock_mcp = make_mock_mcp()
        # click 先失败后成功
        click_responses = [
            ActionResult(success=False, error="元素不可见"),  # 第一次 click 失败
            ActionResult(success=True),                       # 第二次 click 成功
        ]
        # get_elements/get_page_state/click 的顺序
        # 需要 get_page_state 和 get_elements 调用成功，只有 click 先失败
        from acp.schema.elements import ACPElement, ElementSource, ElementStates, ElementType, Rect, Point

        def make_el(eid="e001"):
            return ACPElement(
                id=eid, type=ElementType.BUTTON, platform_class="button",
                text="提交", bounds=Rect(x=0, y=0, width=100, height=40),
                center=Point(x=50, y=20),
                states=ElementStates(clickable=True, enabled=True, visible=True),
                selector="#submit", actions=["click"], source=ElementSource.DOM,
                confidence=1.0,
            )

        # mock _find_element 始终返回固定 element_id
        runner = make_runner(tmp_path, flows, mcp=mock_mcp)
        runner._find_element = AsyncMock(return_value="e001")

        # 实际调用顺序：
        # 1. get_page_state（click 前取 before_state）
        # 2. click → 失败
        # 3. click → 重试成功
        # 4. get_page_state（_verify_click 内取 after_state）
        page_state_result = make_action_result(success=True)
        mock_mcp.execute = AsyncMock(side_effect=[
            page_state_result,                              # get_page_state (before)
            ActionResult(success=False, error="点击失败"),  # 第一次 click 失败
            ActionResult(success=True),                     # 重试 click 成功
            page_state_result,                              # get_page_state (_verify_click)
        ])

        ok = await runner.run("click_retry_flow")

        # 重试成功，整体应成功
        assert ok is True
        assert mock_mcp.execute.call_count == 4


# ---------------------------------------------------------------------------
# TestFlowRunnerLifecycle — FlowRunner 生命周期管理
# ---------------------------------------------------------------------------

class TestFlowRunnerLifecycle:
    """测试 FlowRunner 自管理 MCP 生命周期（mcp=None 时）。"""

    @pytest.mark.asyncio
    async def test_runner_creates_and_closes_mcp(self, tmp_path):
        """mcp=None 时，FlowRunner 应自动创建和关闭 WebMCP。"""
        flows = """
flows:
  minimal:
    description: 最小流程
    steps:
      - action: wait
        wait: 0
"""
        (tmp_path / "flows.yaml").write_text(flows)
        (tmp_path / "site.yaml").write_text("name: test\n")

        with patch("acp.brain.flow_runner.WebMCP") as MockWebMCP:
            mock_instance = MagicMock()
            mock_instance.start = AsyncMock()
            mock_instance.close = AsyncMock()
            mock_instance._adapter = MagicMock()
            mock_instance._adapter._page = MagicMock()
            mock_instance.execute = AsyncMock(return_value=ActionResult(success=True))
            MockWebMCP.return_value = mock_instance

            runner = FlowRunner(site_dir=str(tmp_path), mcp=None, headless=True)
            ok = await runner.run("minimal")

        assert ok is True
        mock_instance.start.assert_called_once()
        mock_instance.close.assert_called_once()
        MockWebMCP.assert_called_once_with(headless=True, cookie_file=ANY)

    @pytest.mark.asyncio
    async def test_mcp_closed_on_exception(self, tmp_path):
        """即使执行过程中出现异常，MCP 也应被关闭。"""
        flows = """
flows:
  err_flow:
    description: 异常测试
    steps:
      - action: navigate
        url: https://example.com
        wait: 0
"""
        (tmp_path / "flows.yaml").write_text(flows)
        (tmp_path / "site.yaml").write_text("name: test\n")

        with patch("acp.brain.flow_runner.WebMCP") as MockWebMCP:
            mock_instance = MagicMock()
            mock_instance.start = AsyncMock()
            mock_instance.close = AsyncMock()
            mock_instance._adapter = MagicMock()
            mock_instance._adapter._page = MagicMock()
            # execute 抛出异常
            mock_instance.execute = AsyncMock(side_effect=RuntimeError("MCP 异常"))
            MockWebMCP.return_value = mock_instance

            runner = FlowRunner(site_dir=str(tmp_path), mcp=None, headless=True)
            with pytest.raises(RuntimeError):
                await runner.run("err_flow")

        # 即使出现异常，close 也应被调用
        mock_instance.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_external_mcp_not_closed(self, tmp_path):
        """传入外部 MCP 时，FlowRunner 不应关闭它。"""
        flows = """
flows:
  external_mcp:
    description: 外部 MCP 测试
    steps:
      - action: wait
        wait: 0
"""
        mock_mcp = make_mock_mcp()
        runner = make_runner(tmp_path, flows, mcp=mock_mcp)

        await runner.run("external_mcp")

        # 外部传入的 MCP 不应被 FlowRunner 关闭
        mock_mcp.close.assert_not_called()


# ---------------------------------------------------------------------------
# TestFlowRunnerMCPLayer — MCP 分层验证
# ---------------------------------------------------------------------------

class TestFlowRunnerMCPLayer:
    """验证 FlowRunner 通过 MCP 层操作，不直接调用 adapter。"""

    @pytest.mark.asyncio
    async def test_navigate_does_not_call_adapter_directly(self, tmp_path):
        """navigate 步骤不应直接调用 adapter.navigate，只通过 MCP.execute。"""
        flows = """
flows:
  mcp_layer:
    description: MCP 分层验证
    steps:
      - action: navigate
        url: https://example.com
        wait: 0
"""
        mock_mcp = make_mock_mcp()
        runner = make_runner(tmp_path, flows, mcp=mock_mcp)

        await runner.run("mcp_layer")

        # adapter.navigate 不应被直接调用
        adapter = mock_mcp._adapter
        if hasattr(adapter, "navigate"):
            adapter.navigate.assert_not_called()

        # MCP.execute 应被调用
        mock_mcp.execute.assert_called()

    @pytest.mark.asyncio
    async def test_mcp_call_passes_correct_method(self, tmp_path):
        """_call_mcp 应将正确的 method 和 params 传给 WebMCP.execute。"""
        flows = "flows:\n  dummy:\n    steps: []\n"
        mock_mcp = make_mock_mcp()
        runner = make_runner(tmp_path, flows, mcp=mock_mcp)

        await runner._call_mcp("navigate", {"url": "https://test.com"})
        mock_mcp.execute.assert_called_once_with("navigate", {"url": "https://test.com"})

    @pytest.mark.asyncio
    async def test_scroll_step_passes_direction(self, tmp_path):
        """scroll 步骤应将 direction 参数传给 MCP.execute。"""
        flows = """
flows:
  scroll_test:
    description: scroll 方向测试
    steps:
      - action: scroll
        direction: up
        wait: 0
"""
        mock_mcp = make_mock_mcp()
        runner = make_runner(tmp_path, flows, mcp=mock_mcp)
        await runner.run("scroll_test")

        scroll_calls = [
            c for c in mock_mcp.execute.call_args_list
            if c.args[0] == "scroll"
        ]
        assert len(scroll_calls) == 1
        assert scroll_calls[0].args[1]["direction"] == "up"

    @pytest.mark.asyncio
    async def test_flow_not_found(self, tmp_path):
        """请求不存在的流程应返回 False。"""
        flows = "flows:\n  existing: {steps: []}\n"
        mock_mcp = make_mock_mcp()
        runner = make_runner(tmp_path, flows, mcp=mock_mcp)

        ok = await runner.run("nonexistent")

        assert ok is False

    @pytest.mark.asyncio
    async def test_run_multiple_flows(self, tmp_path):
        """run_multiple 应顺序执行多个流程。"""
        flows = """
flows:
  flow_a:
    description: 流程A
    steps:
      - action: wait
        wait: 0
  flow_b:
    description: 流程B
    steps:
      - action: wait
        wait: 0
"""
        with patch("acp.brain.flow_runner.WebMCP") as MockWebMCP:
            mock_instance = MagicMock()
            mock_instance.start = AsyncMock()
            mock_instance.close = AsyncMock()
            mock_instance._adapter = MagicMock()
            mock_instance._adapter._page = MagicMock()
            mock_instance.execute = AsyncMock(return_value=ActionResult(success=True))
            MockWebMCP.return_value = mock_instance

            (tmp_path / "flows.yaml").write_text(flows)
            (tmp_path / "site.yaml").write_text("name: test\n")

            runner = FlowRunner(site_dir=str(tmp_path), mcp=None, headless=True)
            ok = await runner.run_multiple(["flow_a", "flow_b"])

        assert ok is True
