"""
视觉兜底 MCP (vision-mcp)
基于 YOLOv8n 的截图识别 + 坐标操作。

适用场景：控件树不可用时（Canvas、Closed Shadow DOM、验证码等）。

流程：截图 → YOLOv8n 检测 → 坐标操作

Phase 2 预留（已接口化）：
  - VLM 接口（ShowUI-2B / UGround-2B）
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from acp.mcp.protocol import MCPTool
from acp.schema.elements import ACPElement, PageSnapshot, PageState
from acp.schema.plan import ActionResult
from acp.vision.detector import UIDetector

logger = logging.getLogger(__name__)


class VisionMCP(MCPTool):
    """视觉兜底 MCP 工具（Tier-3）

    使用 YOLOv8n 对截图进行 UI 元素检测，
    将检测结果转换为 ACP Element Schema 供 Brain 层使用。

    使用示例：
        vision_mcp = VisionMCP(adapter=web_adapter)
        result = await vision_mcp.execute("get_elements", {})
        result = await vision_mcp.execute("click", {"element_id": "vis_abc123"})
        result = await vision_mcp.execute("click", {"x": 100, "y": 200})
    """

    tool_id = "vision-mcp"
    capabilities = ["get_elements", "click", "screenshot"]
    TIER = 3

    def __init__(
        self,
        adapter=None,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.3,
        backend: str = "auto",
        use_mock: bool = False,
        mock_detections: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        """初始化视觉 MCP。

        Args:
            adapter: WebAdapter 实例（用于截图和坐标点击）
            model_path: YOLO 权重路径（ultralytics: .pt；ONNX: .onnx）
            confidence: 检测置信度阈值
            backend: 后端选择（"auto" / "ultralytics" / "onnx" / "mock"）
            use_mock: 为 True 时使用 Mock 后端（测试模式）
            mock_detections: 自定义 Mock 检测结果（仅 use_mock=True 时有效）
        """
        self._adapter = adapter
        self._detector = UIDetector(
            model_path=model_path,
            confidence=confidence,
            backend=backend,
            use_mock=use_mock,
            mock_detections=mock_detections,
        )
        # 元素缓存：get_elements 后用 element_id 查找坐标
        self._element_cache: dict[str, ACPElement] = {}

    # ── 核心接口 ────────────────────────────────────────────────────────────

    async def execute(self, method: str, params: dict[str, Any]) -> ActionResult:
        """执行指定方法。

        支持的 method：
            get_elements   — 截图 + YOLO 检测 → ACP Element Schema
            click          — 坐标点击（params: {x, y}）
            screenshot     — 截图返回 base64
        """
        dispatch = {
            "get_elements": self._get_elements,
            "click": self._click,
            "screenshot": self._screenshot,
        }
        handler = dispatch.get(method)
        if handler is None:
            return ActionResult(
                success=False,
                error=f"VisionMCP 不支持方法: {method}，可用: {list(dispatch.keys())}",
            )
        try:
            return await handler(params)
        except Exception as exc:
            return ActionResult(
                success=False,
                error=f"VisionMCP.execute({method}) 异常: {exc}",
            )

    # ── 方法实现 ────────────────────────────────────────────────────────────

    async def _get_elements(self, params: dict[str, Any]) -> ActionResult:
        """截图 + YOLO 检测 → 转 ACP Element Schema。"""
        if self._adapter is None:
            return ActionResult(success=False, error="VisionMCP 未绑定 adapter，无法截图")

        # 1. 截图
        try:
            screenshot_bytes = await self._adapter.screenshot()
        except Exception as exc:
            return ActionResult(success=False, error=f"截图失败: {exc}")

        # 2. 获取页面 URL
        page_url = ""
        try:
            page_state = await self._adapter.get_page_state()
            page_url = page_state.url or ""
        except Exception:
            pass

        # 3. YOLO 检测 → ACP Elements
        try:
            elements = self._detector.detect_to_acp_elements(
                screenshot=screenshot_bytes,
                page_url=page_url,
            )
        except Exception as exc:
            return ActionResult(success=False, error=f"YOLO 检测失败: {exc}")

        # 4. 更新元素缓存（供后续 click by element_id 使用）
        self._element_cache = {elem.id: elem for elem in elements}

        logger.info("VisionMCP 检测到 %d 个元素（conf >= %.2f）",
                    len(elements), self._detector.confidence)

        page_state_obj = None
        try:
            page_state_obj = await self._adapter.get_page_state()
        except Exception:
            pass

        return ActionResult(
            success=True,
            data={"element_count": len(elements), "source": "vision"},
            elements=elements,
            page_state=page_state_obj,
        )

    async def _click(self, params: dict[str, Any]) -> ActionResult:
        """坐标点击。

        params 支持：
          {"element_id": "vis_abc123"}    — 从缓存查找元素中心坐标（推荐）
          {"x": 150, "y": 200}           — 直接坐标（像素）
          两者同时提供时，优先使用直接坐标。
        """
        if self._adapter is None:
            return ActionResult(success=False, error="VisionMCP 未绑定 adapter，无法点击")

        # 解析坐标：优先直接坐标，否则从 element_id 缓存查找
        x = params.get("x")
        y = params.get("y")

        if x is None or y is None:
            element_id = params.get("element_id")
            if element_id is None:
                return ActionResult(
                    success=False,
                    error="click 需要 'element_id' 或 ('x', 'y') 坐标参数",
                )
            element = self._element_cache.get(element_id)
            if element is None:
                return ActionResult(
                    success=False,
                    error=(
                        f"未找到 element_id={element_id!r}。"
                        "请先调用 get_elements 刷新元素缓存。"
                    ),
                )
            x = element.center.x
            y = element.center.y
            logger.info(
                "VisionMCP 点击元素 %s（%s）坐标: (%.1f, %.1f)",
                element_id, element.type, x, y,
            )
        else:
            logger.info("VisionMCP 直接坐标点击: (%.1f, %.1f)", x, y)

        try:
            # 优先使用 adapter._page.mouse.click（Playwright/Patchright）
            page = getattr(self._adapter, "_page", None)
            if page is not None and hasattr(page, "mouse"):
                await page.mouse.click(float(x), float(y))
            else:
                # 回退：通过 adapter 的 click_at_coords（如有）
                click_at = getattr(self._adapter, "click_at_coords", None)
                if click_at is not None:
                    await click_at(float(x), float(y))
                else:
                    raise RuntimeError(
                        "adapter 没有 _page.mouse 或 click_at_coords，无法执行坐标点击。"
                    )

            page_state = await self._adapter.get_page_state()
            return ActionResult(
                success=True,
                data={"clicked_at": {"x": x, "y": y}},
                page_state=page_state,
            )
        except Exception as exc:
            return ActionResult(success=False, error=f"坐标点击失败: {exc}")

    async def _screenshot(self, params: dict[str, Any]) -> ActionResult:
        """截图并返回 base64 编码的 PNG。"""
        if self._adapter is None:
            return ActionResult(success=False, error="VisionMCP 未绑定 adapter，无法截图")

        import base64
        try:
            png_bytes = await self._adapter.screenshot()
            b64 = base64.b64encode(png_bytes).decode("ascii")
            return ActionResult(
                success=True,
                data={"format": "png", "base64": b64, "size_bytes": len(png_bytes)},
            )
        except Exception as exc:
            return ActionResult(success=False, error=f"截图失败: {exc}")

    # ── 兼容旧接口 ────────────────────────────────────────────────────────────

    async def capture_and_recognize(self) -> "PageSnapshot":
        """截图并通过视觉模型识别元素（向后兼容接口）。"""
        result = await self._get_elements({})
        if not result.success:
            raise RuntimeError(f"capture_and_recognize 失败: {result.error}")

        page_state = result.page_state or PageState(
            platform="web", app="unknown", url="", title=""
        )
        return PageSnapshot(page=page_state, elements=result.elements or [])

    async def click_at(self, x: float, y: float) -> ActionResult:
        """在指定坐标点击（向后兼容接口）。"""
        return await self._click({"x": x, "y": y})
