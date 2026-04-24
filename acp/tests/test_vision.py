"""
视觉兜底模块测试

覆盖：
  - UIDetector: Mock 后端、detect() 输出格式、detect_to_acp_elements() 转换
  - VisionMCP: execute() 方法路由、element_id 点击、mock adapter 集成
  - FallbackDetector: get_fallback_mcp() 自动切换、selector 失败计数

运行：
    python3 -m pytest acp/tests/test_vision.py -v

    # 如果 ultralytics/PIL 未安装，使用 Mock 后端（默认已 mock，无需真实模型）
    python3 -m pytest acp/tests/test_vision.py -v --tb=short
"""

from __future__ import annotations

import struct
import zlib
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from acp.adapters.fallback import FallbackDetector
from acp.mcp.tools.vision_mcp import VisionMCP
from acp.schema.elements import ACPElement, ElementSource, ElementType, PageState
from acp.schema.plan import ActionResult
from acp.vision.detector import UIDetector, YOLO_TO_ELEMENT_TYPE


# ---------------------------------------------------------------------------
# 辅助工具
# ---------------------------------------------------------------------------

def make_tiny_png() -> bytes:
    """生成最小有效 PNG（1×1 白色像素），用于 mock 截图，不触发真实推理。"""
    def png_chunk(name: bytes, data: bytes) -> bytes:
        chunk = name + data
        crc = zlib.crc32(chunk) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + chunk + struct.pack(">I", crc)

    header = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    ihdr = png_chunk(b"IHDR", ihdr_data)
    idat_data = zlib.compress(b"\x00\x00\x00\x00")
    idat = png_chunk(b"IDAT", idat_data)
    iend = png_chunk(b"IEND", b"")
    return header + ihdr + idat + iend


def make_mock_adapter(screenshot_bytes: bytes = None) -> MagicMock:
    """构造 mock WebAdapter（不连接真实浏览器）。"""
    adapter = MagicMock()
    adapter.screenshot = AsyncMock(return_value=screenshot_bytes or make_tiny_png())
    adapter.get_page_state = AsyncMock(return_value=PageState(
        platform="web",
        app="example",
        url="https://example.com",
        title="Test Page",
    ))
    # 模拟 Playwright Page.mouse.click
    adapter._page = MagicMock()
    adapter._page.mouse = MagicMock()
    adapter._page.mouse.click = AsyncMock()
    return adapter


def make_mock_detections() -> list[dict[str, Any]]:
    """返回标准格式的 mock 检测结果。"""
    return [
        {"class": "Button",   "bbox": [10.0, 20.0, 110.0, 60.0],  "confidence": 0.92},
        {"class": "EditText", "bbox": [10.0, 80.0, 300.0, 120.0], "confidence": 0.87},
        {"class": "TextView", "bbox": [10.0, 140.0, 200.0, 170.0], "confidence": 0.75},
    ]


# ---------------------------------------------------------------------------
# TestUIDetector — 使用内置 Mock 后端
# ---------------------------------------------------------------------------

class TestUIDetector:
    """UIDetector 单元测试，使用 use_mock=True 不需要真实模型。"""

    def test_init_mock_backend(self):
        """use_mock=True 时应成功初始化，不需要安装 ultralytics。"""
        det = UIDetector(use_mock=True)
        assert det.confidence == 0.3  # 默认值

    def test_init_custom_confidence(self):
        """自定义置信度阈值应正确保存。"""
        det = UIDetector(confidence=0.5, use_mock=True)
        assert det.confidence == 0.5

    def test_detect_returns_list_of_dicts(self):
        """detect() 应返回正确格式的字典列表。"""
        det = UIDetector(use_mock=True)
        screenshot = make_tiny_png()
        results = det.detect(screenshot)

        assert isinstance(results, list)
        assert len(results) > 0
        for item in results:
            assert "class" in item
            assert "bbox" in item
            assert "confidence" in item
            assert len(item["bbox"]) == 4
            assert isinstance(item["confidence"], float)
            assert 0.0 <= item["confidence"] <= 1.0

    def test_detect_custom_mock_detections(self):
        """自定义 mock_detections 应被正确返回。"""
        custom = [
            {"class": "Switch", "bbox": [0.0, 0.0, 50.0, 30.0], "confidence": 0.95},
        ]
        det = UIDetector(use_mock=True, mock_detections=custom)
        results = det.detect(make_tiny_png())
        assert len(results) == 1
        assert results[0]["class"] == "Switch"

    def test_detect_to_acp_elements_types(self):
        """detect_to_acp_elements() 应正确映射类别到 ElementType。"""
        custom_detections = [
            {"class": "Button",   "bbox": [0.0, 0.0, 100.0, 40.0], "confidence": 0.9},
            {"class": "EditText", "bbox": [0.0, 50.0, 200.0, 80.0], "confidence": 0.85},
            {"class": "TextView", "bbox": [0.0, 90.0, 150.0, 110.0], "confidence": 0.75},
            {"class": "CheckBox", "bbox": [0.0, 120.0, 30.0, 150.0], "confidence": 0.8},
            {"class": "Switch",   "bbox": [0.0, 160.0, 60.0, 185.0], "confidence": 0.7},
        ]
        det = UIDetector(use_mock=True, mock_detections=custom_detections)
        elements = det.detect_to_acp_elements(make_tiny_png(), page_url="https://test.com")

        assert len(elements) == 5
        types = [e.type for e in elements]
        assert ElementType.BUTTON in types
        assert ElementType.TEXT_INPUT in types
        assert ElementType.TEXT in types
        assert ElementType.CHECKBOX in types
        assert ElementType.SWITCH in types

    def test_detect_to_acp_elements_source(self):
        """所有元素 source 应为 VISUAL_MODEL。"""
        det = UIDetector(use_mock=True)
        elements = det.detect_to_acp_elements(make_tiny_png())
        for el in elements:
            assert el.source == ElementSource.VISUAL_MODEL, (
                f"元素 {el.id} 的 source 应为 VISUAL_MODEL，实际为 {el.source}"
            )

    def test_detect_to_acp_elements_bounds(self):
        """元素边界框应从 bbox 正确转换。"""
        custom = [{"class": "Button", "bbox": [10.0, 20.0, 110.0, 70.0], "confidence": 0.9}]
        det = UIDetector(use_mock=True, mock_detections=custom)
        elements = det.detect_to_acp_elements(make_tiny_png())

        assert len(elements) == 1
        el = elements[0]
        assert el.bounds.x == 10.0
        assert el.bounds.y == 20.0
        assert el.bounds.width == 100.0   # 110 - 10
        assert el.bounds.height == 50.0   # 70 - 20

    def test_detect_to_acp_elements_center(self):
        """元素中心点应正确计算。"""
        custom = [{"class": "Button", "bbox": [10.0, 20.0, 110.0, 70.0], "confidence": 0.9}]
        det = UIDetector(use_mock=True, mock_detections=custom)
        elements = det.detect_to_acp_elements(make_tiny_png())

        el = elements[0]
        assert el.center.x == 60.0   # 10 + 100/2
        assert el.center.y == 45.0   # 20 + 50/2

    def test_detect_to_acp_elements_confidence(self):
        """元素 confidence 应与检测结果一致。"""
        custom = [{"class": "Button", "bbox": [0.0, 0.0, 50.0, 30.0], "confidence": 0.92}]
        det = UIDetector(use_mock=True, mock_detections=custom)
        elements = det.detect_to_acp_elements(make_tiny_png())
        assert elements[0].confidence == 0.92

    def test_detect_to_acp_elements_unique_ids(self):
        """多个元素应有唯一 ID。"""
        custom = [
            {"class": "Button",   "bbox": [0.0, 0.0, 100.0, 40.0], "confidence": 0.9},
            {"class": "Button",   "bbox": [200.0, 0.0, 300.0, 40.0], "confidence": 0.85},
            {"class": "EditText", "bbox": [0.0, 50.0, 200.0, 80.0], "confidence": 0.8},
        ]
        det = UIDetector(use_mock=True, mock_detections=custom)
        elements = det.detect_to_acp_elements(make_tiny_png())
        ids = [e.id for e in elements]
        assert len(ids) == len(set(ids)), f"发现重复 ID: {ids}"

    def test_detect_to_acp_elements_id_format(self):
        """元素 ID 应以 'vis_' 开头。"""
        det = UIDetector(use_mock=True)
        elements = det.detect_to_acp_elements(make_tiny_png())
        for el in elements:
            assert el.id.startswith("vis_"), f"ID 格式错误: {el.id}"

    def test_detect_to_acp_elements_empty(self):
        """空检测结果应返回空列表。"""
        det = UIDetector(use_mock=True, mock_detections=[])
        elements = det.detect_to_acp_elements(make_tiny_png())
        assert elements == []

    def test_detect_to_acp_elements_unknown_class(self):
        """未知类别应映射到 UNKNOWN 类型（不抛异常）。"""
        custom = [{"class": "WeirdUnknownWidget", "bbox": [0.0, 0.0, 50.0, 50.0], "confidence": 0.6}]
        det = UIDetector(use_mock=True, mock_detections=custom)
        elements = det.detect_to_acp_elements(make_tiny_png())
        assert len(elements) == 1
        assert elements[0].type == ElementType.UNKNOWN

    def test_element_states_button(self):
        """Button 元素应 clickable=True。"""
        custom = [{"class": "Button", "bbox": [0.0, 0.0, 100.0, 40.0], "confidence": 0.9}]
        det = UIDetector(use_mock=True, mock_detections=custom)
        elements = det.detect_to_acp_elements(make_tiny_png())
        assert elements[0].states.clickable is True
        assert elements[0].states.enabled is True
        assert elements[0].states.visible is True

    def test_element_actions_text_input(self):
        """TEXT_INPUT 元素应包含 click 和 type 动作。"""
        custom = [{"class": "EditText", "bbox": [0.0, 0.0, 200.0, 40.0], "confidence": 0.9}]
        det = UIDetector(use_mock=True, mock_detections=custom)
        elements = det.detect_to_acp_elements(make_tiny_png())
        assert "click" in elements[0].actions
        assert "type" in elements[0].actions

    def test_element_selector_format(self):
        """元素 selector 应包含类别和坐标信息。"""
        custom = [{"class": "Button", "bbox": [10.0, 20.0, 110.0, 60.0], "confidence": 0.9}]
        det = UIDetector(use_mock=True, mock_detections=custom)
        elements = det.detect_to_acp_elements(make_tiny_png())
        assert "Button" in elements[0].selector
        assert "10" in elements[0].selector  # x1 坐标

    def test_yolo_to_element_type_mapping_coverage(self):
        """映射表应覆盖基本 UI 类别。"""
        required = ["Button", "EditText", "CheckBox", "Switch", "TextView", "Toolbar"]
        for cls in required:
            assert cls in YOLO_TO_ELEMENT_TYPE, f"缺少映射: {cls}"


# ---------------------------------------------------------------------------
# TestVisionMCP — 使用 mock adapter + mock detector
# ---------------------------------------------------------------------------

class TestVisionMCP:
    """VisionMCP 单元测试（mock adapter + use_mock=True detector）。"""

    def _make_mcp(
        self,
        screenshot_bytes: bytes = None,
        mock_detections: list[dict] = None,
    ) -> tuple[VisionMCP, MagicMock]:
        """创建带 mock 后端的 VisionMCP 实例。"""
        adapter = make_mock_adapter(screenshot_bytes)
        mcp = VisionMCP(adapter=adapter, use_mock=True)
        # 注入自定义 mock_detections 到 detector 内部的 _backend
        if mock_detections is not None:
            mcp._detector._backend._detections = mock_detections
        return mcp, adapter

    # ---- get_elements ----

    @pytest.mark.asyncio
    async def test_get_elements_success(self):
        """get_elements 应返回 ACPElement 列表，success=True。"""
        mcp, adapter = self._make_mcp()
        result = await mcp.execute("get_elements", {})

        assert result.success is True
        assert result.data["source"] == "vision"
        assert result.data["element_count"] > 0
        assert result.elements is not None
        assert len(result.elements) > 0
        adapter.screenshot.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_elements_returns_acp_elements(self):
        """get_elements 返回的每个元素都是 ACPElement 格式。"""
        mcp, _ = self._make_mcp()
        result = await mcp.execute("get_elements", {})

        for el in result.elements:
            assert isinstance(el, ACPElement)
            assert el.source == ElementSource.VISUAL_MODEL
            assert isinstance(el.type, ElementType)
            assert el.bounds is not None
            assert el.center is not None

    @pytest.mark.asyncio
    async def test_get_elements_no_adapter(self):
        """没有 adapter 时 get_elements 应返回 success=False。"""
        mcp = VisionMCP(adapter=None, use_mock=True)
        result = await mcp.execute("get_elements", {})
        assert result.success is False
        assert "adapter" in result.error.lower()

    @pytest.mark.asyncio
    async def test_get_elements_updates_element_cache(self):
        """get_elements 后 element_cache 应被更新，可用于 click by element_id。"""
        mcp, _ = self._make_mcp()
        result = await mcp.execute("get_elements", {})
        assert result.success is True

        # 缓存应已填充
        assert len(mcp._element_cache) > 0
        for element_id in mcp._element_cache:
            assert element_id.startswith("vis_")

    # ---- click ----

    @pytest.mark.asyncio
    async def test_click_with_coords(self):
        """直接坐标点击应调用 page.mouse.click 并返回成功。"""
        mcp, adapter = self._make_mcp()
        result = await mcp.execute("click", {"x": 150, "y": 200})

        assert result.success is True
        adapter._page.mouse.click.assert_called_once_with(150.0, 200.0)
        assert result.data["clicked_at"]["x"] == 150.0
        assert result.data["clicked_at"]["y"] == 200.0

    @pytest.mark.asyncio
    async def test_click_with_element_id(self):
        """通过 element_id 点击应查找缓存并使用元素中心坐标。"""
        custom = [{"class": "Button", "bbox": [100.0, 200.0, 200.0, 250.0], "confidence": 0.9}]
        mcp, adapter = self._make_mcp(mock_detections=custom)

        # 先 get_elements 填充缓存
        get_result = await mcp.execute("get_elements", {})
        assert get_result.success is True

        element_id = get_result.elements[0].id
        click_result = await mcp.execute("click", {"element_id": element_id})

        assert click_result.success is True
        # 中心坐标：x = 100 + 50 = 150, y = 200 + 25 = 225
        adapter._page.mouse.click.assert_called_once_with(150.0, 225.0)

    @pytest.mark.asyncio
    async def test_click_element_id_not_found(self):
        """元素 ID 不在缓存中时应返回失败并提示重新调用 get_elements。"""
        mcp, _ = self._make_mcp()
        result = await mcp.execute("click", {"element_id": "vis_nonexistent"})

        assert result.success is False
        assert "vis_nonexistent" in result.error

    @pytest.mark.asyncio
    async def test_click_missing_params(self):
        """缺少坐标和 element_id 时应返回失败。"""
        mcp, _ = self._make_mcp()
        result = await mcp.execute("click", {})

        assert result.success is False

    @pytest.mark.asyncio
    async def test_click_no_adapter(self):
        """没有 adapter 时 click 应返回失败。"""
        mcp = VisionMCP(adapter=None, use_mock=True)
        result = await mcp.execute("click", {"x": 10, "y": 20})
        assert result.success is False

    # ---- screenshot ----

    @pytest.mark.asyncio
    async def test_screenshot_success(self):
        """screenshot 方法应返回成功且包含 base64 数据。"""
        import base64
        png_bytes = make_tiny_png()
        mcp, adapter = self._make_mcp(screenshot_bytes=png_bytes)
        result = await mcp.execute("screenshot", {})

        assert result.success is True
        assert result.data["format"] == "png"
        decoded = base64.b64decode(result.data["base64"])
        assert decoded == png_bytes

    @pytest.mark.asyncio
    async def test_screenshot_no_adapter(self):
        """没有 adapter 时 screenshot 应返回失败。"""
        mcp = VisionMCP(adapter=None, use_mock=True)
        result = await mcp.execute("screenshot", {})
        assert result.success is False

    # ---- 不支持的方法 ----

    @pytest.mark.asyncio
    async def test_unsupported_method(self):
        """不支持的方法应返回 success=False。"""
        mcp, _ = self._make_mcp()
        result = await mcp.execute("navigate", {"url": "https://x.com"})
        assert result.success is False

    # ---- capabilities 和继承 ----

    def test_tool_id(self):
        """VisionMCP 的 tool_id 应为 'vision-mcp'。"""
        mcp, _ = self._make_mcp()
        assert mcp.tool_id == "vision-mcp"

    def test_capabilities(self):
        """VisionMCP 应声明支持 get_elements, click, screenshot。"""
        mcp, _ = self._make_mcp()
        assert "get_elements" in mcp.capabilities
        assert "click" in mcp.capabilities
        assert "screenshot" in mcp.capabilities

    def test_supports(self):
        """MCPTool.supports() 应根据 capabilities 返回正确结果。"""
        mcp, _ = self._make_mcp()
        assert mcp.supports("get_elements") is True
        assert mcp.supports("type") is False

    # ---- 兼容旧接口 ----

    @pytest.mark.asyncio
    async def test_capture_and_recognize_compat(self):
        """向后兼容接口 capture_and_recognize() 应返回 PageSnapshot。"""
        from acp.schema.elements import PageSnapshot
        mcp, _ = self._make_mcp()
        snapshot = await mcp.capture_and_recognize()
        assert isinstance(snapshot, PageSnapshot)
        assert snapshot.page is not None
        assert len(snapshot.elements) > 0

    @pytest.mark.asyncio
    async def test_click_at_compat(self):
        """向后兼容接口 click_at(x, y) 应正常调用坐标点击。"""
        mcp, adapter = self._make_mcp()
        result = await mcp.click_at(100.0, 150.0)
        assert result.success is True
        adapter._page.mouse.click.assert_called_once_with(100.0, 150.0)


# ---------------------------------------------------------------------------
# TestFallbackDetector — Fallback 规则和自动切换
# ---------------------------------------------------------------------------

class TestFallbackDetector:
    """FallbackDetector 单元测试。"""

    def test_check_elements_count_triggers(self):
        """元素数 < 5 时应触发降级。"""
        should_fb, reason = FallbackDetector.check_elements_count(3)
        assert should_fb is True
        assert "3" in reason

    def test_check_elements_count_no_trigger(self):
        """元素数 >= 5 时不应触发降级。"""
        should_fb, reason = FallbackDetector.check_elements_count(5)
        assert should_fb is False
        assert reason == ""

    def test_check_elements_count_zero(self):
        """0 个元素时应触发降级。"""
        should_fb, _ = FallbackDetector.check_elements_count(0)
        assert should_fb is True

    def test_selector_failure_tracking(self):
        """record_selector_failure() 应累计失败次数。"""
        det = FallbackDetector()
        sel = "input[name=q]"

        det.record_selector_failure(sel)
        det.record_selector_failure(sel)
        assert det.get_selector_fail_count(sel) == 2

        det.record_selector_failure(sel)
        assert det.get_selector_fail_count(sel) == 3

    def test_selector_success_resets_count(self):
        """record_selector_success() 应重置失败计数为 0。"""
        det = FallbackDetector()
        sel = "button.submit"

        det.record_selector_failure(sel)
        det.record_selector_failure(sel)
        det.record_selector_success(sel)
        assert det.get_selector_fail_count(sel) == 0

    def test_selector_unknown_returns_zero(self):
        """未记录过的 selector 应返回 0。"""
        det = FallbackDetector()
        assert det.get_selector_fail_count("never_seen_selector") == 0

    def test_mark_mcp_auth_failure(self):
        """mark_mcp_auth_failure() 应返回 True。"""
        det = FallbackDetector()
        should_fb, reason = det.mark_mcp_auth_failure()
        assert should_fb is True
        assert "认证失败" in reason or "auth" in reason.lower() or "规则7" in reason

    @pytest.mark.asyncio
    async def test_get_fallback_mcp_returns_vision_mcp(self):
        """get_fallback_mcp() 应返回绑定 adapter 的 VisionMCP。"""
        adapter = make_mock_adapter()
        vision_mcp = await FallbackDetector.get_fallback_mcp(adapter)

        assert isinstance(vision_mcp, VisionMCP)
        assert vision_mcp._adapter is adapter

    @pytest.mark.asyncio
    async def test_fallback_flow_end_to_end(self):
        """完整 fallback 流程：触发条件检测 → 切换 VisionMCP → 执行 get_elements。"""
        adapter = make_mock_adapter()

        # 模拟元素不足（规则 1 触发）
        should_fb, reason = FallbackDetector.check_elements_count(2)
        assert should_fb is True

        # 自动切换到视觉 MCP（use_mock 通过 adapter 传入，这里不传 use_mock 会使用 auto 后端）
        vision_mcp = await FallbackDetector.get_fallback_mcp(adapter)
        # 注入 mock backend 避免需要真实模型
        from acp.vision.detector import _MockBackend
        vision_mcp._detector._backend = _MockBackend()

        result = await vision_mcp.execute("get_elements", {})
        assert result.success is True
        assert result.elements is not None
