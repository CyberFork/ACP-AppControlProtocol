"""
PTG Schema - 页面转换图数据模型
Page Transition Graph 的数据结构定义。
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from acp.schema.elements import ACPElement


class PTGNodeType(str, Enum):
    """PTG 节点类型"""
    PAGE = "page"
    DIALOG = "dialog"
    BOTTOM_SHEET = "bottom_sheet"
    LOADING = "loading"


class PTGNode(BaseModel):
    """PTG 节点：代表一个页面状态"""
    node_id: str
    type: PTGNodeType = PTGNodeType.PAGE
    app: str
    description: str = ""
    elements_snapshot: list[ACPElement] = Field(default_factory=list)


class PTGEdge(BaseModel):
    """PTG 边：代表页面间的转换"""
    from_node: str
    to_node: str
    action: str
    params: dict[str, Any] = Field(default_factory=dict)


class PTGGraph(BaseModel):
    """完整的页面转换图"""
    nodes: dict[str, PTGNode] = Field(default_factory=dict)
    edges: list[PTGEdge] = Field(default_factory=list)
    current_state: Optional[str] = None  # 当前所在节点 ID
