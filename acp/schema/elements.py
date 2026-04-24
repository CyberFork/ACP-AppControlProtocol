"""
ACP Element Schema - 统一元素格式定义
所有平台适配器的输出统一为此格式，供大脑模块消费。
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ElementType(str, Enum):
    """统一语义类型枚举（跨平台）"""
    BUTTON = "button"
    TEXT_INPUT = "text_input"
    TEXT = "text"
    IMAGE = "image"
    LIST = "list"
    LIST_ITEM = "list_item"
    CHECKBOX = "checkbox"
    SWITCH = "switch"
    TAB = "tab"
    NAV_BAR = "nav_bar"
    SCROLL_VIEW = "scroll_view"
    CONTAINER = "container"
    UNKNOWN = "unknown"


class ElementSource(str, Enum):
    """元素来源"""
    DOM = "dom"
    ACCESSIBILITY_TREE = "accessibility_tree"
    VISUAL_MODEL = "visual_model"


class Rect(BaseModel):
    """矩形边界"""
    x: float
    y: float
    width: float
    height: float


class Point(BaseModel):
    """二维坐标点"""
    x: float
    y: float


class ElementStates(BaseModel):
    """元素状态"""
    clickable: bool = False
    enabled: bool = True
    visible: bool = True
    checked: bool = False
    scrollable: bool = False
    focused: bool = False


class ACPElement(BaseModel):
    """ACP 统一元素格式"""
    id: str
    type: ElementType
    platform_class: str = ""
    text: Optional[str] = None
    label: Optional[str] = None
    placeholder: Optional[str] = None
    bounds: Rect
    center: Point
    states: ElementStates = Field(default_factory=ElementStates)
    selector: Optional[str] = None
    parent_id: Optional[str] = None
    child_ids: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    source: ElementSource = ElementSource.DOM
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class PageState(BaseModel):
    """页面状态"""
    platform: str
    app: str
    title: str = ""
    url: Optional[str] = None          # Web 专用
    activity: Optional[str] = None     # Android 专用
    ptg_node_id: Optional[str] = None  # PTG 中的节点 ID
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z"
    )


class PageSnapshot(BaseModel):
    """页面快照：页面状态 + 元素列表"""
    page: PageState
    elements: list[ACPElement] = Field(default_factory=list)
