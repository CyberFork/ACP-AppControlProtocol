"""
_locate() 5 级定位策略单元测试

覆盖：
  - 策略1：placeholder 精确匹配
  - 策略1b：placeholder 不唯一，回退 CSS selector
  - 策略2：role + name 匹配
  - 策略2b：role+text 不唯一，缩小 parent 范围
  - 策略3：纯文本精确匹配
  - 策略4：CSS selector + text 筛选
  - 策略4b：多匹配时 z-index 筛选
  - 策略5：全失败返回 None
  - _pick_visible_top：z-index 最高且可见的元素选取
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from acp.adapters.web_adapter import WebAdapter
from acp.schema.elements import ElementSource, ElementStates, ElementType


# ---------------------------------------------------------------------------
# 测试工具
# ---------------------------------------------------------------------------

def make_locator(count: int = 1, is_visible: bool = True, z_index: int = 0) -> MagicMock:
    """构造 Playwright Locator mock。"""
    loc = MagicMock()
    loc.count = AsyncMock(return_value=count)
    loc.is_visible = AsyncMock(return_value=is_visible)
    loc.get_attribute = AsyncMock(return_value=None)  # aria-hidden=None 默认可见
    loc.evaluate = AsyncMock(return_value=z_index)
    loc.first = loc
    loc.filter = MagicMock(return_value=loc)
    loc.nth = MagicMock(return_value=loc)
    return loc


def make_mock_page() -> MagicMock:
    """构造带基本 mock 方法的 Playwright Page。"""
    page = MagicMock()
    page.url = "https://example.com/login"

    default_loc = make_locator(count=0)  # 默认无匹配
    page.get_by_placeholder = MagicMock(return_value=default_loc)
    page.get_by_role = MagicMock(return_value=default_loc)
    page.get_by_text = MagicMock(return_value=default_loc)
    page.locator = MagicMock(return_value=default_loc)
    page.evaluate = AsyncMock(return_value=None)
    return page


class LocateTestBase:
    """_locate() 测试基类，提供 adapter 和 page 初始化。"""

    def setup_method(self):
        self.page = make_mock_page()
        # 用 __new__ 绕过 __init__，手动初始化需要的属性
        self.adapter = WebAdapter.__new__(WebAdapter)
        self.adapter._page = self.page
        self.adapter._element_cache = {}
        self.adapter._element_semantic_cache = {}

    def register_element(
        self,
        eid: str,
        placeholder: str = "",
        role: str = "",
        text: str = "",
        tag: str = "button",
        selector: str = "#btn",
    ) -> None:
        """向 adapter 缓存注册元素信息。"""
        self.adapter._element_cache[eid] = selector
        self.adapter._element_semantic_cache[eid] = {
            "placeholder": placeholder,
            "role": role,
            "text": text,
            "tag": tag,
        }


# ---------------------------------------------------------------------------
# 策略1：placeholder 精确匹配
# ---------------------------------------------------------------------------

class TestStrategy1Placeholder(LocateTestBase):
    """策略1：placeholder 精确匹配（输入框优先）。"""

    @pytest.mark.asyncio
    async def test_input_with_unique_placeholder(self):
        """input 元素有唯一 placeholder 时，应通过 get_by_placeholder 定位。"""
        eid = "e_email"
        self.register_element(eid, placeholder="请输入邮箱", tag="input")

        unique_loc = make_locator(count=1)
        self.page.get_by_placeholder = MagicMock(return_value=unique_loc)

        result = await self.adapter._locate(eid)

        self.page.get_by_placeholder.assert_called_once_with("请输入邮箱", exact=True)
        assert result is unique_loc

    @pytest.mark.asyncio
    async def test_textarea_with_unique_placeholder(self):
        """textarea 元素有唯一 placeholder 时，应通过 get_by_placeholder 定位。"""
        eid = "e_textarea"
        self.register_element(eid, placeholder="请输入内容", tag="textarea")

        unique_loc = make_locator(count=1)
        self.page.get_by_placeholder = MagicMock(return_value=unique_loc)

        result = await self.adapter._locate(eid)

        self.page.get_by_placeholder.assert_called_once_with("请输入内容", exact=True)
        assert result is unique_loc

    @pytest.mark.asyncio
    async def test_non_input_tag_skips_placeholder_strategy(self):
        """非 input/textarea 元素即使有 placeholder 字段也不走策略1。"""
        eid = "e_btn"
        # button 有 placeholder 字段，但不是 input/textarea
        self.register_element(eid, placeholder="不应匹配", tag="button", text="提交", role="button")

        btn_loc = make_locator(count=1)
        self.page.get_by_role = MagicMock(return_value=btn_loc)
        self.page.get_by_placeholder = MagicMock(return_value=make_locator(count=0))

        result = await self.adapter._locate(eid)

        # 应走策略2（role+name），不走策略1
        self.page.get_by_role.assert_called()
        assert result is btn_loc


# ---------------------------------------------------------------------------
# 策略1b：placeholder 不唯一，回退 CSS selector
# ---------------------------------------------------------------------------

class TestStrategy1bPlaceholderNotUnique(LocateTestBase):
    """策略1b：placeholder 不唯一时，使用 CSS selector 精确定位。"""

    @pytest.mark.asyncio
    async def test_non_unique_placeholder_falls_back_to_css(self):
        """placeholder 匹配多个元素时，应回退到 CSS selector。"""
        eid = "e_input"
        self.register_element(eid, placeholder="请输入", tag="input", selector="input[name=email]")

        # placeholder 匹配 2 个
        ph_loc = make_locator(count=2)
        self.page.get_by_placeholder = MagicMock(return_value=ph_loc)

        # CSS selector 匹配 1 个
        css_loc = make_locator(count=1)
        css_loc.first = css_loc
        self.page.locator = MagicMock(return_value=css_loc)

        result = await self.adapter._locate(eid)

        # 应调用 CSS selector
        self.page.locator.assert_called_with("input[name=email]")

    @pytest.mark.asyncio
    async def test_zero_placeholder_match_continues_to_next_strategy(self):
        """placeholder 匹配 0 个时，应继续到下一策略（role+name）。"""
        eid = "e_zero_ph"
        self.register_element(eid, placeholder="不存在的placeholder", tag="input", role="textbox", text="user@example.com")

        ph_loc = make_locator(count=0)
        self.page.get_by_placeholder = MagicMock(return_value=ph_loc)

        role_loc = make_locator(count=1)
        self.page.get_by_role = MagicMock(return_value=role_loc)

        result = await self.adapter._locate(eid)

        # placeholder 无匹配，走策略2 role+text
        self.page.get_by_role.assert_called()


# ---------------------------------------------------------------------------
# 策略2：role + name 匹配
# ---------------------------------------------------------------------------

class TestStrategy2RoleName(LocateTestBase):
    """策略2：role + text 语义化定位。"""

    @pytest.mark.asyncio
    async def test_button_role_with_text(self):
        """button role 有唯一 text 时，应通过 get_by_role(button, name=text) 定位。"""
        eid = "e_btn"
        self.register_element(eid, role="button", text="登录", tag="button")

        btn_loc = make_locator(count=1)
        self.page.get_by_role = MagicMock(return_value=btn_loc)
        self.page.get_by_placeholder = MagicMock(return_value=make_locator(count=0))

        result = await self.adapter._locate(eid)

        self.page.get_by_role.assert_called_once_with("button", name="登录", exact=True)
        assert result is btn_loc

    @pytest.mark.asyncio
    async def test_link_role_with_text(self):
        """link role（a 标签）有唯一 text 时，应通过 get_by_role(link) 定位。"""
        eid = "e_link"
        self.register_element(eid, role="a", text="立即注册", tag="a")

        link_loc = make_locator(count=1)
        self.page.get_by_role = MagicMock(return_value=link_loc)
        self.page.get_by_placeholder = MagicMock(return_value=make_locator(count=0))

        result = await self.adapter._locate(eid)

        self.page.get_by_role.assert_called_with("link", name="立即注册", exact=True)

    @pytest.mark.asyncio
    async def test_no_text_skips_strategy2(self):
        """有 role 但无 text 时，跳过策略2，走策略4（CSS selector）。"""
        eid = "e_no_text"
        self.register_element(eid, role="button", text="", tag="button", selector="#btn")

        # 无 text，不应调用 get_by_role（text 为 ""）
        css_loc = make_locator(count=1)
        self.page.locator = MagicMock(return_value=css_loc)
        self.page.get_by_placeholder = MagicMock(return_value=make_locator(count=0))
        self.page.get_by_role = MagicMock(return_value=make_locator(count=0))

        result = await self.adapter._locate(eid)

        # playwright_role 存在但 text 为空，跳过策略2
        # 走 CSS selector（策略4）
        self.page.locator.assert_called_with("#btn")


# ---------------------------------------------------------------------------
# 策略3：纯文本匹配
# ---------------------------------------------------------------------------

class TestStrategy3TextOnly(LocateTestBase):
    """策略3：get_by_text 精确匹配（无 placeholder 和有效 role）。"""

    @pytest.mark.asyncio
    async def test_short_text_gets_by_text(self):
        """短文本（<50 字符）且 role 不匹配时，应走 get_by_text。"""
        eid = "e_text"
        self.register_element(eid, text="立即注册", tag="span")

        # get_by_placeholder → 0，get_by_role → 0（span 无映射 role）
        self.page.get_by_placeholder = MagicMock(return_value=make_locator(count=0))
        self.page.get_by_role = MagicMock(return_value=make_locator(count=0))

        text_loc = make_locator(count=1)
        self.page.get_by_text = MagicMock(return_value=text_loc)

        result = await self.adapter._locate(eid)

        self.page.get_by_text.assert_called_once_with("立即注册", exact=True)
        assert result is text_loc

    @pytest.mark.asyncio
    async def test_long_text_skips_strategy3(self):
        """过长文本（≥50 字符）不走策略3，直接走策略4（CSS selector）。"""
        eid = "e_long"
        long_text = "A" * 50  # 恰好 50 个字符（len(text) < 50 条件不满足）
        assert len(long_text) >= 50, "确保测试文本确实足够长"
        self.register_element(eid, text=long_text, tag="p", selector=".long-text")

        self.page.get_by_placeholder = MagicMock(return_value=make_locator(count=0))
        self.page.get_by_role = MagicMock(return_value=make_locator(count=0))

        css_loc = make_locator(count=1)
        self.page.locator = MagicMock(return_value=css_loc)

        result = await self.adapter._locate(eid)

        # 长文本跳过策略3，调用 CSS selector
        self.page.locator.assert_called_with(".long-text")
        self.page.get_by_text.assert_not_called()


# ---------------------------------------------------------------------------
# 策略4：CSS selector + text 筛选
# ---------------------------------------------------------------------------

class TestStrategy4CSSSelector(LocateTestBase):
    """策略4：CSS selector fallback，多匹配时用 text 筛选。"""

    @pytest.mark.asyncio
    async def test_unique_css_selector(self):
        """CSS selector 唯一匹配时直接返回。"""
        eid = "e_css"
        self.register_element(eid, selector="#login-btn")

        self.page.get_by_placeholder = MagicMock(return_value=make_locator(count=0))
        self.page.get_by_role = MagicMock(return_value=make_locator(count=0))

        css_loc = make_locator(count=1)
        self.page.locator = MagicMock(return_value=css_loc)

        result = await self.adapter._locate(eid)

        self.page.locator.assert_called_with("#login-btn")
        assert result is css_loc

    @pytest.mark.asyncio
    async def test_multiple_css_matches_filtered_by_text(self):
        """CSS selector 多匹配时，用 text 过滤精确选中。"""
        eid = "e_multi"
        self.register_element(eid, text="确认", tag="button", selector=".btn")

        self.page.get_by_placeholder = MagicMock(return_value=make_locator(count=0))
        self.page.get_by_role = MagicMock(return_value=make_locator(count=0))
        self.page.get_by_text = MagicMock(return_value=make_locator(count=0))

        # CSS 有 2 个匹配
        filtered_loc = make_locator(count=1)
        filtered_loc.first = filtered_loc
        multi_loc = MagicMock()
        multi_loc.count = AsyncMock(return_value=2)
        multi_loc.filter = MagicMock(return_value=filtered_loc)
        multi_loc.first = multi_loc
        multi_loc.nth = MagicMock(return_value=multi_loc)
        self.page.locator = MagicMock(return_value=multi_loc)

        result = await self.adapter._locate(eid)

        # 应调用 filter(has_text="确认")
        multi_loc.filter.assert_called_once_with(has_text="确认")

    @pytest.mark.asyncio
    async def test_only_selector_cache_no_semantic_cache(self):
        """只有 selector 缓存（无语义缓存）时，直接用 CSS selector 的 .first。"""
        eid = "e_no_semantic"
        # 只注册 selector，不注册 semantic
        self.adapter._element_cache[eid] = ".my-button"

        css_loc = make_locator(count=1)
        css_loc.first = css_loc
        self.page.locator = MagicMock(return_value=css_loc)

        result = await self.adapter._locate(eid)

        self.page.locator.assert_called_with(".my-button")
        assert result is not None


# ---------------------------------------------------------------------------
# 策略5：全失败返回 None
# ---------------------------------------------------------------------------

class TestStrategy5AllFail(LocateTestBase):
    """策略5：所有定位策略失败时返回 None。"""

    @pytest.mark.asyncio
    async def test_no_cache_returns_none(self):
        """元素 ID 既无语义缓存也无 selector 缓存时，返回 None。"""
        eid = "e_nonexistent"
        # 不注册任何缓存
        result = await self.adapter._locate(eid)
        assert result is None

    @pytest.mark.asyncio
    async def test_all_strategies_fail_returns_none(self):
        """所有定位策略均无匹配（count=0）时，返回 None。"""
        eid = "e_all_fail"
        self.register_element(eid, text="不存在的按钮", role="button", tag="button", selector=".missing")

        # 所有 locator 返回 count=0
        self.page.get_by_placeholder = MagicMock(return_value=make_locator(count=0))
        self.page.get_by_role = MagicMock(return_value=make_locator(count=0))
        self.page.get_by_text = MagicMock(return_value=make_locator(count=0))

        zero_css_loc = MagicMock()
        zero_css_loc.count = AsyncMock(return_value=0)
        zero_css_loc.first = zero_css_loc
        zero_css_loc.nth = MagicMock(return_value=zero_css_loc)
        self.page.locator = MagicMock(return_value=zero_css_loc)

        result = await self.adapter._locate(eid)

        # CSS count=0 时，策略4 不返回，最终返回 None
        assert result is None


# ---------------------------------------------------------------------------
# z-index 筛选测试（_pick_visible_top）
# ---------------------------------------------------------------------------

class TestPickVisibleTop(LocateTestBase):
    """_pick_visible_top：多匹配时选 z-index 最高且可见的元素。"""

    @pytest.mark.asyncio
    async def test_picks_highest_zindex_element(self):
        """多个元素中应选 z-index 最高的可见元素。"""
        count = 3

        # 三个 locator：z-index 分别为 10, 100, 50
        z_values = [10, 100, 50]
        items = []
        for z in z_values:
            item = MagicMock()
            item.is_visible = AsyncMock(return_value=True)
            item.get_attribute = AsyncMock(return_value=None)
            item.evaluate = AsyncMock(return_value=z)
            items.append(item)

        loc = MagicMock()
        loc.nth = MagicMock(side_effect=lambda i: items[i])

        result = await self.adapter._pick_visible_top(loc, count)

        # z=100 是索引 1，应选 items[1]
        assert result is items[1]

    @pytest.mark.asyncio
    async def test_skips_invisible_elements(self):
        """不可见的元素应被跳过，选可见且 z-index 最高的。"""
        count = 3

        # item0: 不可见，z=999
        # item1: 可见，z=50
        # item2: 可见，z=30
        configs = [
            {"visible": False, "z": 999},
            {"visible": True, "z": 50},
            {"visible": True, "z": 30},
        ]
        items = []
        for cfg in configs:
            item = MagicMock()
            item.is_visible = AsyncMock(return_value=cfg["visible"])
            item.get_attribute = AsyncMock(return_value=None)
            item.evaluate = AsyncMock(return_value=cfg["z"])
            items.append(item)

        loc = MagicMock()
        loc.nth = MagicMock(side_effect=lambda i: items[i])

        result = await self.adapter._pick_visible_top(loc, count)

        # item0 不可见被跳过，item1 可见且 z=50 > item2 的 z=30
        assert result is items[1]

    @pytest.mark.asyncio
    async def test_skips_aria_hidden_elements(self):
        """aria-hidden=true 的元素应被跳过。"""
        count = 2

        # item0: aria-hidden=true，z=999
        # item1: 可见，aria-hidden=None，z=10
        item0 = MagicMock()
        item0.is_visible = AsyncMock(return_value=True)
        item0.get_attribute = AsyncMock(return_value="true")  # aria-hidden
        item0.evaluate = AsyncMock(return_value=999)

        item1 = MagicMock()
        item1.is_visible = AsyncMock(return_value=True)
        item1.get_attribute = AsyncMock(return_value=None)
        item1.evaluate = AsyncMock(return_value=10)

        items = [item0, item1]
        loc = MagicMock()
        loc.nth = MagicMock(side_effect=lambda i: items[i])

        result = await self.adapter._pick_visible_top(loc, count)

        # aria-hidden 被跳过，选 item1
        assert result is items[1]

    @pytest.mark.asyncio
    async def test_all_invisible_returns_first(self):
        """所有元素都不可见时，默认返回第一个（索引 0）。"""
        count = 2

        items = []
        for i in range(count):
            item = MagicMock()
            item.is_visible = AsyncMock(return_value=False)
            item.get_attribute = AsyncMock(return_value=None)
            item.evaluate = AsyncMock(return_value=0)
            items.append(item)

        loc = MagicMock()
        loc.nth = MagicMock(side_effect=lambda i: items[i])

        result = await self.adapter._pick_visible_top(loc, count)

        # 所有不可见，返回默认的 idx=0
        assert result is items[0]

    @pytest.mark.asyncio
    async def test_equal_zindex_keeps_first_found(self):
        """z-index 相同时，先找到的那个（索引较小）被选中。"""
        count = 2

        items = []
        for _ in range(count):
            item = MagicMock()
            item.is_visible = AsyncMock(return_value=True)
            item.get_attribute = AsyncMock(return_value=None)
            item.evaluate = AsyncMock(return_value=5)  # 相同 z-index
            items.append(item)

        loc = MagicMock()
        loc.nth = MagicMock(side_effect=lambda i: items[i])

        result = await self.adapter._pick_visible_top(loc, count)

        # z 相等时，第一个可见的 item（索引0）先更新 best，索引1 z 不大于 best
        # 所以仍然是索引 0
        assert result is items[0]

    @pytest.mark.asyncio
    async def test_locate_strategy4b_uses_pick_visible_top(self, monkeypatch):
        """策略4b：CSS 多匹配且无 text 时，调用 _pick_visible_top 筛选。"""
        eid = "e_multi_no_text"
        # 无 text，有 selector，会触发策略4b
        self.register_element(eid, text="", tag="div", selector=".card")

        self.page.get_by_placeholder = MagicMock(return_value=make_locator(count=0))
        self.page.get_by_role = MagicMock(return_value=make_locator(count=0))

        # CSS 有 3 个匹配，无 text
        best_item = MagicMock()
        best_item.is_visible = AsyncMock(return_value=True)
        best_item.get_attribute = AsyncMock(return_value=None)
        best_item.evaluate = AsyncMock(return_value=10)

        multi_loc = MagicMock()
        multi_loc.count = AsyncMock(return_value=3)
        multi_loc.first = multi_loc
        multi_loc.filter = MagicMock(return_value=make_locator(count=0))
        multi_loc.nth = MagicMock(return_value=best_item)
        self.page.locator = MagicMock(return_value=multi_loc)

        # mock _pick_visible_top 以便追踪调用
        called = []

        async def mock_pick(loc, count):
            called.append((loc, count))
            return best_item

        monkeypatch.setattr(self.adapter, "_pick_visible_top", mock_pick)

        result = await self.adapter._locate(eid)

        # 应调用 _pick_visible_top（策略4b）
        assert len(called) == 1
        assert called[0][1] == 3
        assert result is best_item
