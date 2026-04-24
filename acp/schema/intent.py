"""
Intent Schema - 意图和子任务数据模型
Intent Parser 的输出结构定义。
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class SubTask(BaseModel):
    """子任务：单个操作单元"""
    action: str
    app: Optional[str] = None
    params: dict[str, Any] = Field(default_factory=dict)
    filter: Optional[str] = None


class Intent(BaseModel):
    """结构化意图：Intent Parser 的完整输出"""
    intent: str
    app: Optional[str] = None
    params: dict[str, Any] = Field(default_factory=dict)
    sub_tasks: list[SubTask] = Field(default_factory=list)
