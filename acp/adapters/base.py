"""
平台适配器基类
定义所有平台适配器必须实现的接口。

适配器职责：
  1. 获取当前屏幕的 UI 元素（控件树/DOM）
  2. 转换为 ACP Element Schema（统一 JSON）
  3. 执行操作（click/type/scroll 等）
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from acp.schema.elements import ACPElement, PageSnapshot, PageState
from acp.schema.plan import ActionResult


class BaseAdapter(ABC):
    """平台适配器抽象基类"""

    @property
    @abstractmethod
    def platform(self) -> str:
        """平台标识，如 'web', 'android', 'ios'"""
        ...

    # ---- 感知接口 ----

    @abstractmethod
    async def get_elements(self) -> list[ACPElement]:
        """获取当前页面所有 UI 元素（转为 ACP 统一格式）。"""
        ...

    @abstractmethod
    async def get_page_state(self) -> PageState:
        """获取当前页面状态（URL/标题/Activity 等）。"""
        ...

    @abstractmethod
    async def screenshot(self) -> bytes:
        """截取当前屏幕截图。"""
        ...

    async def get_snapshot(self) -> PageSnapshot:
        """获取完整页面快照（状态 + 元素列表）。"""
        page = await self.get_page_state()
        elements = await self.get_elements()
        return PageSnapshot(page=page, elements=elements)

    # ---- 操作接口 ----

    @abstractmethod
    async def click(self, element_id: str) -> ActionResult:
        """点击指定元素。"""
        ...

    @abstractmethod
    async def type(self, element_id: str, text: str) -> ActionResult:
        """向指定元素输入文本。"""
        ...

    @abstractmethod
    async def scroll(
        self,
        direction: str,
        element_id: str | None = None,
        amount: int = 300,
    ) -> ActionResult:
        """滚动页面或指定容器。direction: 'up'|'down'|'left'|'right'"""
        ...

    # ---- 等待接口 ----

    @abstractmethod
    async def wait_for_element(
        self,
        selector: str,
        timeout: int = 10,
    ) -> ACPElement | None:
        """等待指定选择器的元素出现。"""
        ...

    @abstractmethod
    async def navigate(self, url: str) -> ActionResult:
        """导航到指定 URL。"""
        ...

    @abstractmethod
    async def wait_for_navigation(self, timeout: int = 30) -> PageState:
        """等待页面导航完成。"""
        ...

    # ---- 生命周期 ----

    async def close(self) -> None:
        """释放适配器资源（子类按需覆写）。"""
        ...
