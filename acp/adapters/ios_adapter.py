"""
iOS 平台适配器（预留占位）
底层工具：Appium + XCUITest Driver
感知方式：XCUITest Accessibility Tree
操作方式：Appium WebDriver API

TODO（阶段 4 - iOS 适配器实现）：
  - 实现 Appium 连接管理
  - 实现 XCUITest Accessibility Tree 解析
  - 实现 iOS 元素 → ACP Element Schema 转换
  - 实现 tap/type/swipe/key 操作
"""

from __future__ import annotations

from acp.adapters.base import BaseAdapter
from acp.schema.elements import ACPElement, PageState
from acp.schema.plan import ActionResult


class IOSAdapter(BaseAdapter):
    """iOS 平台适配器（预留未实现）"""

    @property
    def platform(self) -> str:
        return "ios"

    async def get_elements(self) -> list[ACPElement]:
        raise NotImplementedError("IOSAdapter 预留未实现，见阶段 4")

    async def get_page_state(self) -> PageState:
        raise NotImplementedError("IOSAdapter 预留未实现，见阶段 4")

    async def screenshot(self) -> bytes:
        raise NotImplementedError("IOSAdapter 预留未实现，见阶段 4")

    async def click(self, element_id: str) -> ActionResult:
        raise NotImplementedError("IOSAdapter 预留未实现，见阶段 4")

    async def type(self, element_id: str, text: str) -> ActionResult:
        raise NotImplementedError("IOSAdapter 预留未实现，见阶段 4")

    async def scroll(
        self,
        direction: str,
        element_id: str | None = None,
        amount: int = 300,
    ) -> ActionResult:
        raise NotImplementedError("IOSAdapter 预留未实现，见阶段 4")

    async def wait_for_element(
        self,
        selector: str,
        timeout: int = 10,
    ) -> ACPElement | None:
        raise NotImplementedError("IOSAdapter 预留未实现，见阶段 4")

    async def wait_for_navigation(self, timeout: int = 30) -> PageState:
        raise NotImplementedError("IOSAdapter 预留未实现，见阶段 4")
