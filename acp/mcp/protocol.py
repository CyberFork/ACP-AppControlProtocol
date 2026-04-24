"""
MCP Protocol - MCP 协议基础定义
ACP 与 MCP 工具之间的通信协议层。

协议：JSON-RPC 2.0 over stdio（MCP 标准）

包含：
  - MCPTool：所有具体 MCP Tool 的抽象基类
  - JSONRPCRequest / JSONRPCResponse：JSON-RPC 2.0 消息封装
  - MCPProtocolClient：协议客户端（占位，供后续 stdio 实现扩展）
"""

from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from typing import Any, Optional

from acp.schema.plan import ActionResult


# ---------------------------------------------------------------------------
# MCPTool 抽象基类
# ---------------------------------------------------------------------------


class MCPTool(ABC):
    """所有 MCP 工具的抽象基类。

    每个具体 MCP 工具（WebMCP、AndroidMCP 等）继承此类并实现 execute()。

    属性：
        tool_id:      工具唯一标识，如 "web-mcp"
        capabilities: 工具支持的操作列表，如 ["navigate", "click", "type"]
    """

    tool_id: str
    capabilities: list[str]

    @abstractmethod
    async def execute(self, method: str, params: dict[str, Any]) -> ActionResult:
        """执行指定方法。

        Args:
            method: 操作名称，如 "navigate"、"click"
            params: 操作参数字典

        Returns:
            ActionResult：统一结果封装
        """

    def supports(self, method: str) -> bool:
        """检查工具是否支持指定方法。"""
        return method in self.capabilities


# ---------------------------------------------------------------------------
# JSON-RPC 2.0 消息封装
# ---------------------------------------------------------------------------


class JSONRPCRequest:
    """JSON-RPC 2.0 请求"""

    def __init__(self, method: str, params: dict[str, Any], id: Optional[str] = None) -> None:
        self.jsonrpc = "2.0"
        self.method = method
        self.params = params
        self.id = id or str(uuid.uuid4())

    def to_dict(self) -> dict:
        return {
            "jsonrpc": self.jsonrpc,
            "method": self.method,
            "params": self.params,
            "id": self.id,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class JSONRPCResponse:
    """JSON-RPC 2.0 响应"""

    def __init__(
        self,
        id: str,
        result: Optional[Any] = None,
        error: Optional[dict] = None,
    ) -> None:
        self.jsonrpc = "2.0"
        self.id = id
        self.result = result
        self.error = error

    @property
    def success(self) -> bool:
        return self.error is None

    @classmethod
    def from_dict(cls, data: dict) -> "JSONRPCResponse":
        return cls(
            id=data.get("id", ""),
            result=data.get("result"),
            error=data.get("error"),
        )

    @classmethod
    def from_json(cls, raw: str) -> "JSONRPCResponse":
        return cls.from_dict(json.loads(raw))


# ---------------------------------------------------------------------------
# MCPProtocolClient（占位，供后续 stdio 实现扩展）
# ---------------------------------------------------------------------------


class MCPProtocolClient:
    """MCP 协议客户端（占位实现）

    后续实现：
      - stdio 进程管理（启动 MCP 服务器子进程）
      - 异步读写循环
      - 请求队列和响应匹配
    """

    async def call(
        self,
        tool_id: str,
        method: str,
        params: dict[str, Any],
        timeout: int = 30,
    ) -> JSONRPCResponse:
        """调用 MCP 工具方法。"""
        raise NotImplementedError("MCPProtocolClient 尚未实现 stdio 传输层")
