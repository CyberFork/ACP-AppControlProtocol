"""
Web 泛用 MCP (web-mcp)
基于 Playwright 的 Web/H5 自动化工具。

Tier 2：平台泛用 MCP，覆盖无专用 API 的 Web/H5 应用。
底层委托给 WebAdapter 执行，统一返回 ActionResult。

支持的 method（对应 execute() 的 method 参数）：
  - "navigate"       → adapter.navigate(url)
  - "get_elements"   → adapter.get_elements()
  - "click"          → adapter.click(element_id)
  - "type"           → adapter.type(element_id, text)
  - "scroll"         → adapter.scroll(direction, element_id?, amount?)
  - "screenshot"     → adapter.screenshot()
  - "get_page_state" → adapter.get_page_state()
"""

from __future__ import annotations

import base64
from typing import Any, Optional

from acp.adapters.web_adapter import WebAdapter
from acp.mcp.protocol import MCPTool
from acp.schema.plan import ActionResult


class WebMCP(MCPTool):
    """Web 泛用 MCP 工具（Playwright 实现）

    使用示例：
        async with WebMCP() as mcp:
            result = await mcp.execute("navigate", {"url": "https://example.com"})
            result = await mcp.execute("get_elements", {})
            result = await mcp.execute("click", {"element_id": "e0001_abc"})
    """

    tool_id: str = "web-mcp"
    capabilities: list[str] = [
        "navigate",
        "get_elements",
        "click",
        "type",
        "scroll",
        "screenshot",
        "get_page_state",
    ]

    def __init__(
        self,
        adapter: Optional[WebAdapter] = None,
        headless: bool = True,
        browser_type: str = "chromium",
        slow_mo: int = 0,
        cookie_file: str = None,
    ) -> None:
        """初始化 WebMCP。

        Args:
            adapter:      传入已有的 WebAdapter 实例（测试时注入 mock）
            headless:     是否无头模式（adapter 为 None 时生效）
            browser_type: 浏览器类型，"chromium" / "firefox" / "webkit"
            slow_mo:      操作延迟毫秒数，便于调试
            cookie_file:  持久化 cookie 的文件路径（None 则不持久化）
        """
        if adapter is not None:
            self._adapter = adapter
            self._owns_adapter = False
        else:
            self._adapter = WebAdapter(
                headless=headless,
                browser_type=browser_type,
                slow_mo=slow_mo,
                cookie_file=cookie_file,
            )
            self._owns_adapter = True

    # ---- 生命周期 ----

    async def start(self) -> None:
        """启动底层 WebAdapter（若由本实例持有）。"""
        if self._owns_adapter:
            await self._adapter.start()

    async def close(self) -> None:
        """关闭底层 WebAdapter（若由本实例持有）。"""
        if self._owns_adapter:
            await self._adapter.close()

    async def __aenter__(self) -> "WebMCP":
        await self.start()
        return self

    async def __aexit__(self, *_) -> None:
        await self.close()

    # ---- 核心接口 ----

    async def execute(self, method: str, params: dict[str, Any]) -> ActionResult:
        """执行指定方法，将 method 映射到 WebAdapter 对应操作。

        Args:
            method: 操作名称
            params: 操作参数

        Returns:
            ActionResult：统一结果封装

        支持的 method：
            navigate       — params: {url: str}
            get_elements   — params: {}
            click          — params: {element_id: str}
            type           — params: {element_id: str, text: str}
            scroll         — params: {direction: str, element_id?: str, amount?: int}
            screenshot     — params: {}
            get_page_state — params: {}
        """
        dispatch = {
            "navigate":       self._navigate,
            "get_elements":   self._get_elements,
            "click":          self._click,
            "type":           self._type_text,
            "scroll":         self._scroll,
            "screenshot":     self._screenshot,
            "get_page_state": self._get_page_state,
        }

        handler = dispatch.get(method)
        if handler is None:
            return ActionResult(
                success=False,
                error=f"WebMCP 不支持方法: {method}，可用方法: {list(dispatch.keys())}",
            )

        try:
            return await handler(params)
        except Exception as exc:
            return ActionResult(success=False, error=f"WebMCP.execute({method}) 异常: {exc}")

    # ---- 方法映射 ----

    async def _navigate(self, params: dict[str, Any]) -> ActionResult:
        url = params.get("url")
        if not url:
            return ActionResult(success=False, error="navigate 需要 'url' 参数")
        return await self._adapter.navigate(url)

    async def _get_elements(self, params: dict[str, Any]) -> ActionResult:
        elements = await self._adapter.get_elements()
        page_state = await self._adapter.get_page_state()
        return ActionResult(
            success=True,
            data={"element_count": len(elements)},
            page_state=page_state,
            elements=elements,
        )

    async def _click(self, params: dict[str, Any]) -> ActionResult:
        element_id = params.get("element_id")
        if not element_id:
            return ActionResult(success=False, error="click 需要 'element_id' 参数")
        return await self._adapter.click(element_id)

    async def _type_text(self, params: dict[str, Any]) -> ActionResult:
        element_id = params.get("element_id")
        text = params.get("text")
        if not element_id:
            return ActionResult(success=False, error="type 需要 'element_id' 参数")
        if text is None:
            return ActionResult(success=False, error="type 需要 'text' 参数")
        return await self._adapter.type(element_id, text)

    async def _scroll(self, params: dict[str, Any]) -> ActionResult:
        direction = params.get("direction", "down")
        element_id = params.get("element_id")
        amount = int(params.get("amount", 300))
        return await self._adapter.scroll(direction, element_id, amount)

    async def _screenshot(self, params: dict[str, Any]) -> ActionResult:
        png_bytes = await self._adapter.screenshot()
        page_state = await self._adapter.get_page_state()
        # 将 bytes 转为 base64 字符串，方便在 ActionResult.data 中序列化
        b64 = base64.b64encode(png_bytes).decode("ascii")
        return ActionResult(
            success=True,
            data={"format": "png", "base64": b64, "size_bytes": len(png_bytes)},
            page_state=page_state,
        )

    async def _get_page_state(self, params: dict[str, Any]) -> ActionResult:
        page_state = await self._adapter.get_page_state()
        return ActionResult(
            success=True,
            data={
                "url": page_state.url,
                "title": page_state.title,
                "platform": page_state.platform,
                "app": page_state.app,
            },
            page_state=page_state,
        )
