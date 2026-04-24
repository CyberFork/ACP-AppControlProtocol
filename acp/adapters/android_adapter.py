"""
Android 平台适配器（预留占位）
底层工具：UI Automator 2 直连（通过 ADB）
感知方式：Accessibility Tree
操作方式：UIAutomator API / AccessibilityService

TODO（阶段 4 - Android 适配器实现）：
  - 实现 ADB 连接管理
  - 实现 UIAutomator2 Accessibility Tree 解析
  - 实现 Android 元素 → ACP Element Schema 转换
  - 实现 click/type/swipe/key 操作
"""

from __future__ import annotations

from acp.adapters.base import BaseAdapter
from acp.schema.elements import ACPElement, PageState
from acp.schema.plan import ActionResult


class AndroidAdapter(BaseAdapter):
    """Android 平台适配器（预留未实现）"""

    @property
    def platform(self) -> str:
        return "android"

    async def get_elements(self) -> list[ACPElement]:
        raise NotImplementedError("AndroidAdapter 预留未实现，见阶段 4")

    async def get_page_state(self) -> PageState:
        raise NotImplementedError("AndroidAdapter 预留未实现，见阶段 4")

    async def screenshot(self) -> bytes:
        raise NotImplementedError("AndroidAdapter 预留未实现，见阶段 4")

    async def click(self, element_id: str) -> ActionResult:
        raise NotImplementedError("AndroidAdapter 预留未实现，见阶段 4")

    async def type(self, element_id: str, text: str) -> ActionResult:
        raise NotImplementedError("AndroidAdapter 预留未实现，见阶段 4")

    async def scroll(
        self,
        direction: str,
        element_id: str | None = None,
        amount: int = 300,
    ) -> ActionResult:
        raise NotImplementedError("AndroidAdapter 预留未实现，见阶段 4")

    async def wait_for_element(
        self,
        selector: str,
        timeout: int = 10,
    ) -> ACPElement | None:
        raise NotImplementedError("AndroidAdapter 预留未实现，见阶段 4")

    async def wait_for_navigation(self, timeout: int = 30) -> PageState:
        raise NotImplementedError("AndroidAdapter 预留未实现，见阶段 4")
