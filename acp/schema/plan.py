"""
Plan Schema - 任务规划数据模型
Task Planner 的输出结构定义。
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from acp.schema.elements import ACPElement, PageState


class Step(BaseModel):
    """执行步骤"""
    step_id: int
    action: str
    tool: str                           # MCP 工具 ID，如 "web-mcp"
    tool_tier: int = Field(ge=1, le=3)  # 1=专用, 2=泛用, 3=视觉
    params: dict[str, Any] = Field(default_factory=dict)
    expected_output: Optional[str] = None
    fallback_tool: Optional[str] = None


class Plan(BaseModel):
    """执行计划：包含有序的步骤序列"""
    plan_id: str
    steps: list[Step] = Field(default_factory=list)


class ActionResult(BaseModel):
    """MCP 工具调用结果"""
    success: bool
    data: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    page_state: Optional[PageState] = None
    elements: Optional[list[ACPElement]] = None


class MCPToolCall(BaseModel):
    """MCP 工具调用请求"""
    tool_id: str
    method: str
    params: dict[str, Any] = Field(default_factory=dict)
    timeout: int = 30  # 秒
