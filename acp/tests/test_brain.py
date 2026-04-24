"""
Brain 模块测试套件

覆盖：
  - IntentParser  ：简单规则模式（不调 LLM）
  - TaskPlanner   ：step 生成和工具选择
  - ExecutorDispatcher：逐步执行（mock MCP tool）
  - FeedbackEvaluator ：MATCH/MINOR_DEVIATION/MAJOR_DEVIATION/FAILURE
  - PTGManager    ：状态记录和路径查找
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional
from unittest.mock import AsyncMock

import pytest

from acp.brain.executor import ExecutorDispatcher
from acp.brain.feedback import FeedbackEvaluator, FeedbackLevel
from acp.brain.intent_parser import IntentParser, _try_simple_parse
from acp.brain.ptg_manager import PTGManager
from acp.brain.task_planner import TaskPlanner
from acp.mcp.protocol import MCPTool
from acp.mcp.registry import MCPRegistry, MCPToolInfo
from acp.schema.elements import PageState
from acp.schema.intent import Intent, SubTask
from acp.schema.plan import ActionResult, Plan, Step
from acp.schema.ptg import PTGEdge, PTGNode, PTGNodeType


# ===========================================================================
# Fixtures & helpers
# ===========================================================================


def make_web_tool_info() -> MCPToolInfo:
    return MCPToolInfo(
        tool_id="web-mcp",
        tier=2,
        name="Web MCP",
        description="Playwright",
        supported_apps=["*_web", "browser", "xiaohongshu"],
        capabilities=["navigate", "click", "type", "scroll", "screenshot", "get_elements", "get_page_state"],
        platform="web",
        reliability=0.95,
    )


def make_vision_tool_info() -> MCPToolInfo:
    return MCPToolInfo(
        tool_id="vision-mcp",
        tier=3,
        name="Vision MCP",
        description="OmniParser",
        supported_apps=["*"],
        capabilities=["screenshot", "detect"],
        platform="cross_platform",
        reliability=0.80,
    )


def make_feishu_tool_info() -> MCPToolInfo:
    return MCPToolInfo(
        tool_id="feishu-mcp",
        tier=1,
        name="Feishu MCP",
        description="飞书专用",
        supported_apps=["feishu", "lark"],
        capabilities=["send_message", "read_document", "create_calendar"],
        platform="cross_platform",
        reliability=0.99,
    )


def make_registry() -> MCPRegistry:
    reg = MCPRegistry()
    reg.register(make_web_tool_info())
    reg.register(make_vision_tool_info())
    reg.register(make_feishu_tool_info())
    return reg


def make_page_state(url: str = "https://example.com", title: str = "Example") -> PageState:
    return PageState(platform="web", app="browser", url=url, title=title)


class MockMCPTool(MCPTool):
    """可配置的 mock MCP 工具"""

    tool_id = "mock-tool"
    capabilities = ["navigate", "click", "type", "screenshot"]

    def __init__(self, results: Optional[list[ActionResult]] = None) -> None:
        """
        Args:
            results: 按调用顺序返回的结果列表；耗尽后返回最后一个；为 None 时返回成功
        """
        self._results = results or []
        self._call_count = 0

    async def execute(self, method: str, params: dict[str, Any]) -> ActionResult:
        if not self._results:
            return ActionResult(success=True, data={"method": method})
        idx = min(self._call_count, len(self._results) - 1)
        self._call_count += 1
        return self._results[idx]


# ===========================================================================
# 1. IntentParser — 简单规则模式
# ===========================================================================


class TestIntentParserSimple:

    def test_parse_navigate_url(self):
        result = _try_simple_parse("打开 https://example.com")
        assert result is not None
        assert result.intent == "navigate"
        assert result.params["url"] == "https://example.com"
        assert len(result.sub_tasks) == 1
        assert result.sub_tasks[0].action == "navigate"

    def test_parse_navigate_url_english(self):
        result = _try_simple_parse("open https://google.com")
        assert result is not None
        assert result.intent == "navigate"
        assert result.params["url"] == "https://google.com"

    def test_parse_navigate_go_to(self):
        result = _try_simple_parse("go to https://github.com")
        assert result is not None
        assert result.intent == "navigate"
        assert result.params["url"] == "https://github.com"

    def test_parse_screenshot(self):
        result = _try_simple_parse("截图")
        assert result is not None
        assert result.intent == "screenshot"
        assert result.sub_tasks[0].action == "screenshot"

    def test_parse_screenshot_english(self):
        result = _try_simple_parse("screenshot")
        assert result is not None
        assert result.intent == "screenshot"

    def test_parse_click(self):
        result = _try_simple_parse("点击 登录按钮")
        assert result is not None
        assert result.intent == "click"
        assert result.params["target"] == "登录按钮"

    def test_parse_click_english(self):
        result = _try_simple_parse("click submit button")
        assert result is not None
        assert result.intent == "click"
        assert result.params["target"] == "submit button"

    def test_parse_scroll_down(self):
        result = _try_simple_parse("滚动下")
        assert result is not None
        assert result.intent == "scroll"
        assert result.params["direction"] == "down"

    def test_parse_scroll_up(self):
        result = _try_simple_parse("scroll up")
        assert result is not None
        assert result.intent == "scroll"
        assert result.params["direction"] == "up"

    def test_parse_type_with_element(self):
        result = _try_simple_parse("在搜索框中输入 Python")
        assert result is not None
        assert result.intent == "type"
        assert "text" in result.params
        assert result.params["text"] == "Python"

    def test_unrecognized_returns_none(self):
        result = _try_simple_parse("帮我写一篇关于外星人的文章并发布到小红书")
        assert result is None

    @pytest.mark.asyncio
    async def test_parser_force_simple_returns_unknown_for_complex(self):
        parser = IntentParser(force_simple=True)
        intent = await parser.parse("帮我写一篇文章")
        assert intent.intent == "unknown"
        assert intent.sub_tasks[0].action == "unknown"

    @pytest.mark.asyncio
    async def test_parser_empty_input(self):
        parser = IntentParser(force_simple=True)
        intent = await parser.parse("")
        assert intent.intent == "empty"

    @pytest.mark.asyncio
    async def test_parser_simple_navigate(self):
        parser = IntentParser(force_simple=True)
        intent = await parser.parse("打开 https://example.com")
        assert intent.intent == "navigate"
        assert intent.params["url"] == "https://example.com"

    def test_has_llm_false_without_key(self):
        parser = IntentParser(force_simple=True)
        assert parser.has_llm is False

    def test_has_llm_false_when_force_simple(self):
        parser = IntentParser(api_key="fake-key", force_simple=True)
        assert parser.has_llm is False

    def test_has_llm_true_with_key(self):
        parser = IntentParser(api_key="fake-key")
        assert parser.has_llm is True


# ===========================================================================
# 2. TaskPlanner — step 生成和工具选择
# ===========================================================================


class TestTaskPlanner:

    def _make_planner(self) -> TaskPlanner:
        return TaskPlanner(registry=make_registry())

    @pytest.mark.asyncio
    async def test_plan_navigate(self):
        planner = self._make_planner()
        intent = Intent(
            intent="navigate",
            app="browser",
            sub_tasks=[SubTask(action="navigate", app="browser", params={"url": "https://example.com"})],
        )
        plan = await planner.plan(intent)
        assert plan.plan_id.startswith("plan_")
        assert len(plan.steps) == 1
        step = plan.steps[0]
        assert step.action == "navigate"
        assert step.tool == "web-mcp"
        assert step.tool_tier == 2
        assert step.params["url"] == "https://example.com"

    @pytest.mark.asyncio
    async def test_plan_selects_tier1_for_feishu(self):
        planner = self._make_planner()
        intent = Intent(
            intent="send_message",
            app="feishu",
            sub_tasks=[SubTask(action="send_message", app="feishu", params={"text": "hello"})],
        )
        plan = await planner.plan(intent)
        step = plan.steps[0]
        assert step.tool == "feishu-mcp"
        assert step.tool_tier == 1

    @pytest.mark.asyncio
    async def test_plan_fallback_to_vision(self):
        """web-mcp 存在时，fallback 应为 vision-mcp（tier=3）"""
        planner = self._make_planner()
        intent = Intent(
            intent="click",
            app="browser",
            sub_tasks=[SubTask(action="click", app="browser", params={"target": "btn"})],
        )
        plan = await planner.plan(intent)
        step = plan.steps[0]
        assert step.tool == "web-mcp"
        assert step.fallback_tool == "vision-mcp"

    @pytest.mark.asyncio
    async def test_plan_no_subtasks(self):
        planner = self._make_planner()
        intent = Intent(intent="empty", sub_tasks=[])
        plan = await planner.plan(intent)
        assert len(plan.steps) == 0

    @pytest.mark.asyncio
    async def test_plan_multi_steps(self):
        planner = self._make_planner()
        intent = Intent(
            intent="composite",
            app="browser",
            sub_tasks=[
                SubTask(action="navigate", app="browser", params={"url": "https://a.com"}),
                SubTask(action="click", app="browser", params={"target": "btn"}),
                SubTask(action="screenshot", app="browser"),
            ],
        )
        plan = await planner.plan(intent)
        assert len(plan.steps) == 3
        for i, step in enumerate(plan.steps, start=1):
            assert step.step_id == i

    @pytest.mark.asyncio
    async def test_plan_empty_registry_uses_fallback(self):
        """注册表为空时，TaskPlanner 应使用 web-mcp 占位，而不是崩溃"""
        planner = TaskPlanner(registry=MCPRegistry())
        intent = Intent(
            intent="navigate",
            app="browser",
            sub_tasks=[SubTask(action="navigate", app="browser", params={"url": "https://x.com"})],
        )
        plan = await planner.plan(intent)
        assert len(plan.steps) == 1
        assert plan.steps[0].tool == "web-mcp"

    def test_plan_id_unique(self):
        planner = TaskPlanner()
        ids = {planner._generate_plan_id() for _ in range(100)}
        assert len(ids) == 100


# ===========================================================================
# 3. ExecutorDispatcher — 逐步执行（mock）
# ===========================================================================


class TestExecutorDispatcher:

    def _make_plan(self, steps: list[Step]) -> Plan:
        return Plan(plan_id="plan_test", steps=steps)

    def _make_step(
        self,
        step_id: int = 1,
        action: str = "navigate",
        tool: str = "mock-tool",
        fallback_tool: Optional[str] = None,
        expected_output: Optional[str] = None,
    ) -> Step:
        return Step(
            step_id=step_id,
            action=action,
            tool=tool,
            tool_tier=2,
            params={"url": "https://example.com"},
            expected_output=expected_output,
            fallback_tool=fallback_tool,
        )

    @pytest.mark.asyncio
    async def test_execute_success(self):
        mock_tool = MockMCPTool([ActionResult(success=True, data={"result": "ok"})])
        executor = ExecutorDispatcher(tools={"mock-tool": mock_tool})
        plan = self._make_plan([self._make_step()])
        results = await executor.execute(plan)
        assert len(results) == 1
        assert results[0].success is True

    @pytest.mark.asyncio
    async def test_execute_multiple_steps(self):
        mock_tool = MockMCPTool([
            ActionResult(success=True, data={"step": 1}),
            ActionResult(success=True, data={"step": 2}),
            ActionResult(success=True, data={"step": 3}),
        ])
        executor = ExecutorDispatcher(tools={"mock-tool": mock_tool})
        steps = [self._make_step(i) for i in range(1, 4)]
        results = await executor.execute(self._make_plan(steps))
        assert len(results) == 3
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_execute_stops_on_failure(self):
        """步骤 1 失败后，步骤 2 不应被执行"""
        mock_tool = MockMCPTool([ActionResult(success=False, error="网络错误")])
        executor = ExecutorDispatcher(tools={"mock-tool": mock_tool})
        plan = self._make_plan([
            self._make_step(1),
            self._make_step(2),
        ])
        results = await executor.execute(plan)
        assert len(results) == 1
        assert results[0].success is False

    @pytest.mark.asyncio
    async def test_execute_uses_fallback_on_failure(self):
        """主工具失败 → fallback_tool 成功"""
        main_tool = MockMCPTool([ActionResult(success=False, error="主工具失败")])
        fallback_tool = MockMCPTool([ActionResult(success=True, data={"fallback": True})])

        executor = ExecutorDispatcher(tools={
            "main-tool": main_tool,
            "fallback-tool": fallback_tool,
        })
        step = Step(
            step_id=1,
            action="navigate",
            tool="main-tool",
            tool_tier=2,
            params={"url": "https://x.com"},
            fallback_tool="fallback-tool",
        )
        results = await executor.execute(self._make_plan([step]))
        assert len(results) == 1
        assert results[0].success is True
        assert results[0].data.get("fallback") is True

    @pytest.mark.asyncio
    async def test_execute_unregistered_tool_returns_failure(self):
        executor = ExecutorDispatcher(tools={})
        plan = self._make_plan([self._make_step(tool="nonexistent-tool")])
        results = await executor.execute(plan)
        assert results[0].success is False
        assert "未注册" in results[0].error

    @pytest.mark.asyncio
    async def test_execute_register_tool(self):
        executor = ExecutorDispatcher()
        executor.register_tool("mock-tool", MockMCPTool())
        plan = self._make_plan([self._make_step()])
        results = await executor.execute(plan)
        assert results[0].success is True

    @pytest.mark.asyncio
    async def test_execute_empty_plan(self):
        executor = ExecutorDispatcher()
        results = await executor.execute(Plan(plan_id="empty", steps=[]))
        assert results == []


# ===========================================================================
# 4. FeedbackEvaluator — 三种状态判定
# ===========================================================================


class TestFeedbackEvaluator:

    def _make_step(self, action: str = "click", expected_output: Optional[str] = None) -> Step:
        return Step(
            step_id=1,
            action=action,
            tool="web-mcp",
            tool_tier=2,
            params={},
            expected_output=expected_output,
        )

    @pytest.mark.asyncio
    async def test_failure_on_success_false(self):
        evaluator = FeedbackEvaluator()
        result = ActionResult(success=False, error="工具错误")
        fb = await evaluator.evaluate(result, self._make_step())
        assert fb.level == FeedbackLevel.FAILURE
        assert fb.is_failure is True
        assert not fb.ok

    @pytest.mark.asyncio
    async def test_match_on_success_true_no_expectation(self):
        evaluator = FeedbackEvaluator()
        result = ActionResult(success=True)
        fb = await evaluator.evaluate(result, self._make_step())
        assert fb.level == FeedbackLevel.MATCH
        assert fb.ok is True

    @pytest.mark.asyncio
    async def test_match_on_data_present(self):
        evaluator = FeedbackEvaluator()
        result = ActionResult(success=True, data={"element_count": 5})
        fb = await evaluator.evaluate(result, self._make_step())
        assert fb.ok is True

    @pytest.mark.asyncio
    async def test_navigate_match_url_changed(self):
        evaluator = FeedbackEvaluator()
        prev = make_page_state("https://a.com", "A")
        new_state = make_page_state("https://b.com", "B")
        result = ActionResult(success=True, page_state=new_state)
        step = self._make_step(action="navigate")
        fb = await evaluator.evaluate(result, step, prev_page_state=prev)
        assert fb.level == FeedbackLevel.MATCH

    @pytest.mark.asyncio
    async def test_navigate_minor_deviation_url_unchanged(self):
        evaluator = FeedbackEvaluator()
        prev = make_page_state("https://a.com", "A")
        same_state = make_page_state("https://a.com", "A")
        result = ActionResult(success=True, page_state=same_state)
        step = self._make_step(action="navigate")
        fb = await evaluator.evaluate(result, step, prev_page_state=prev)
        assert fb.level == FeedbackLevel.MINOR_DEVIATION
        assert fb.needs_retry is True

    @pytest.mark.asyncio
    async def test_expected_output_keyword_match(self):
        evaluator = FeedbackEvaluator()
        state = make_page_state("https://example.com/success", "操作成功页面")
        result = ActionResult(success=True, page_state=state)
        step = self._make_step(expected_output="操作成功")
        fb = await evaluator.evaluate(result, step)
        assert fb.level == FeedbackLevel.MATCH

    @pytest.mark.asyncio
    async def test_expected_output_keyword_not_found(self):
        evaluator = FeedbackEvaluator()
        state = make_page_state("https://example.com/error", "错误页")
        result = ActionResult(success=True, page_state=state)
        step = self._make_step(expected_output="操作成功")
        fb = await evaluator.evaluate(result, step)
        assert fb.level == FeedbackLevel.MINOR_DEVIATION

    @pytest.mark.asyncio
    async def test_feedback_result_properties(self):
        from acp.brain.feedback import FeedbackResult

        assert FeedbackResult(FeedbackLevel.MATCH).ok is True
        assert FeedbackResult(FeedbackLevel.MINOR_DEVIATION).needs_retry is True
        assert FeedbackResult(FeedbackLevel.MAJOR_DEVIATION).needs_replan is True
        assert FeedbackResult(FeedbackLevel.FAILURE).is_failure is True


# ===========================================================================
# 5. PTGManager — 状态记录和路径查找
# ===========================================================================


class TestPTGManager:

    def _make_state(self, url: str, title: str = "") -> PageState:
        return PageState(platform="web", app="browser", url=url, title=title or url)

    def test_add_and_get_node(self):
        manager = PTGManager()
        node = PTGNode(node_id="n1", app="browser", description="首页")
        manager.add_node(node)
        assert manager.get_node("n1") is not None
        assert manager.node_count() == 1

    def test_record_transition_creates_nodes_and_edge(self):
        manager = PTGManager()
        state_a = self._make_state("https://a.com", "A")
        state_b = self._make_state("https://b.com", "B")
        from_node, to_node = manager.record_transition(state_a, "click_link", state_b)
        assert manager.node_count() == 2
        assert manager.edge_count() == 1

    def test_record_transition_updates_current_state(self):
        manager = PTGManager()
        state_a = self._make_state("https://a.com")
        state_b = self._make_state("https://b.com")
        manager.record_transition(state_a, "navigate", state_b)
        current = manager.get_current_state()
        assert current is not None
        assert current.app == "browser"

    def test_record_transition_deduplicates_edges(self):
        manager = PTGManager()
        state_a = self._make_state("https://a.com")
        state_b = self._make_state("https://b.com")
        # 记录两次相同转换
        manager.record_transition(state_a, "click", state_b)
        manager.record_transition(state_a, "click", state_b)
        assert manager.edge_count() == 1  # 只应有一条边

    def test_find_path_direct(self):
        manager = PTGManager()
        state_a = self._make_state("https://a.com")
        state_b = self._make_state("https://b.com")
        from_node, to_node = manager.record_transition(state_a, "navigate", state_b)
        path = manager.find_path(from_node.node_id, to_node.node_id)
        assert len(path) == 1
        assert path[0].action == "navigate"

    def test_find_path_multi_hop(self):
        """A → B → C，从 A 到 C 的路径应有 2 个边"""
        manager = PTGManager()
        state_a = self._make_state("https://a.com")
        state_b = self._make_state("https://b.com")
        state_c = self._make_state("https://c.com")
        node_a, node_b = manager.record_transition(state_a, "step1", state_b)
        _node_b2, node_c = manager.record_transition(state_b, "step2", state_c)
        path = manager.find_path(node_a.node_id, node_c.node_id)
        assert len(path) == 2
        assert path[0].action == "step1"
        assert path[1].action == "step2"

    def test_find_path_same_node(self):
        manager = PTGManager()
        state_a = self._make_state("https://a.com")
        node = manager._ensure_node(state_a)
        path = manager.find_path(node.node_id, node.node_id)
        assert path == []

    def test_find_path_no_route(self):
        manager = PTGManager()
        state_a = self._make_state("https://a.com")
        state_b = self._make_state("https://b.com")
        manager.add_node(PTGNode(node_id="n_a", app="browser"))
        manager.add_node(PTGNode(node_id="n_b", app="browser"))
        # 没有边
        path = manager.find_path("n_a", "n_b")
        assert path == []

    def test_set_current_state(self):
        manager = PTGManager()
        node = PTGNode(node_id="page_home", app="browser", description="首页")
        manager.add_node(node)
        manager.set_current_state("page_home")
        assert manager.get_current_state().node_id == "page_home"

    def test_set_current_state_nonexistent_raises(self):
        manager = PTGManager()
        with pytest.raises(KeyError):
            manager.set_current_state("nonexistent")

    def test_match_page_state(self):
        manager = PTGManager()
        state = self._make_state("https://example.com", "Example")
        node = manager._ensure_node(state)
        matched = manager.match_page_state(state)
        assert matched is not None
        assert matched.node_id == node.node_id

    def test_reset_clears_graph(self):
        manager = PTGManager()
        state_a = self._make_state("https://a.com")
        state_b = self._make_state("https://b.com")
        manager.record_transition(state_a, "navigate", state_b)
        assert manager.node_count() > 0
        manager.reset()
        assert manager.node_count() == 0
        assert manager.edge_count() == 0
        assert manager.get_current_state() is None

    def test_get_graph_returns_graph(self):
        from acp.schema.ptg import PTGGraph
        manager = PTGManager()
        graph = manager.get_graph()
        assert isinstance(graph, PTGGraph)


# ===========================================================================
# 6. 集成测试：IntentParser → TaskPlanner → Executor
# ===========================================================================


class TestBrainIntegration:

    @pytest.mark.asyncio
    async def test_parse_plan_execute_navigate(self):
        """端到端：解析 "打开 URL" → 规划 → 执行（mock）"""
        # 解析
        parser = IntentParser(force_simple=True)
        intent = await parser.parse("打开 https://example.com")
        assert intent.intent == "navigate"

        # 规划
        registry = make_registry()
        planner = TaskPlanner(registry=registry)
        plan = await planner.plan(intent)
        assert len(plan.steps) == 1
        assert plan.steps[0].tool == "web-mcp"

        # 执行（mock web-mcp）
        new_state = make_page_state("https://example.com", "Example Domain")
        mock_tool = MockMCPTool([ActionResult(success=True, page_state=new_state)])
        executor = ExecutorDispatcher(tools={"web-mcp": mock_tool})
        results = await executor.execute(plan)
        assert len(results) == 1
        assert results[0].success is True

    @pytest.mark.asyncio
    async def test_parse_plan_execute_click(self):
        """端到端：解析 "点击 X" → 规划 → 执行（mock）"""
        parser = IntentParser(force_simple=True)
        intent = await parser.parse("点击 提交按钮")
        assert intent.intent == "click"

        registry = make_registry()
        planner = TaskPlanner(registry=registry)
        plan = await planner.plan(intent)
        assert plan.steps[0].action == "click"

        mock_tool = MockMCPTool([ActionResult(success=True)])
        executor = ExecutorDispatcher(tools={"web-mcp": mock_tool})
        results = await executor.execute(plan)
        assert results[0].success is True
