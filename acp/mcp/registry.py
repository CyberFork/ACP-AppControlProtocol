"""
MCP Tool Registry - 工具注册中心
管理所有可用的 MCP 工具，提供发现、选择和调用能力。

工具选择优先级：
  Tier 1：第三方专用 MCP（最优先，feishu-mcp, wechat-mcp 等）
  Tier 2：平台泛用 MCP（web-mcp, android-mcp, ios-mcp）
  Tier 3：视觉兜底 MCP（vision-mcp）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class MCPToolInfo:
    """MCP 工具注册信息"""
    tool_id: str
    tier: int                             # 1=专用, 2=泛用, 3=视觉
    name: str
    description: str
    supported_apps: list[str]
    capabilities: list[str]
    platform: str
    auth_required: bool = False
    reliability: float = 0.9
    backend: Optional[str] = None        # e.g. "playwright"


class MCPRegistry:
    """MCP 工具注册中心"""

    def __init__(self) -> None:
        self._tools: dict[str, MCPToolInfo] = {}

    # ---- 注册 ----

    def register(self, tool: MCPToolInfo) -> None:
        """注册一个 MCP 工具。"""
        self._tools[tool.tool_id] = tool

    # ---- 加载 ----

    def load_from_yaml(self, path: str) -> None:
        """从 YAML 配置文件加载并注册工具。

        YAML 结构：
            tools:
              - tool_id: web-mcp
                tier: 2
                name: ...
                ...

        Args:
            path: YAML 文件路径
        """
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        tools_list = data.get("tools", [])
        for item in tools_list:
            tool = MCPToolInfo(
                tool_id=item["tool_id"],
                tier=int(item["tier"]),
                name=item.get("name", item["tool_id"]),
                description=item.get("description", ""),
                supported_apps=item.get("supported_apps", []),
                capabilities=item.get("capabilities", []),
                platform=item.get("platform", "cross_platform"),
                auth_required=bool(item.get("auth_required", False)),
                reliability=float(item.get("reliability", 0.9)),
                backend=item.get("backend"),
            )
            self.register(tool)

    @classmethod
    def from_yaml(cls, path: str) -> "MCPRegistry":
        """从 YAML 文件创建注册中心实例。

        Args:
            path: YAML 文件路径

        Returns:
            加载完毕的 MCPRegistry 实例
        """
        registry = cls()
        registry.load_from_yaml(path)
        return registry

    # ---- 查询 ----

    def get_tool(self, tool_id: str) -> Optional[MCPToolInfo]:
        """按 tool_id 获取工具信息。

        Args:
            tool_id: 工具唯一标识

        Returns:
            MCPToolInfo 或 None（未找到时）
        """
        return self._tools.get(tool_id)

    def list_tools(self) -> list[MCPToolInfo]:
        """列出所有已注册工具（按 tier 升序、reliability 降序）。"""
        tools = list(self._tools.values())
        tools.sort(key=lambda t: (t.tier, -t.reliability))
        return tools

    def find(
        self,
        app: Optional[str] = None,
        tier: Optional[int] = None,
        platform: Optional[str] = None,
    ) -> Optional[MCPToolInfo]:
        """查找符合条件的工具（返回最优匹配）。

        Args:
            app:      目标 App 名称
            tier:     工具层级 (1/2/3)
            platform: 平台 ("web"/"android"/"ios"/"cross_platform")

        Returns:
            匹配的工具信息，无匹配返回 None
        """
        candidates = list(self._tools.values())

        if tier is not None:
            candidates = [t for t in candidates if t.tier == tier]

        if platform is not None:
            candidates = [
                t for t in candidates
                if t.platform == platform or t.platform == "cross_platform"
            ]

        if app is not None:
            candidates = [
                t for t in candidates
                if (
                    app in t.supported_apps
                    or "*" in t.supported_apps
                    or any(
                        app.endswith(suffix.lstrip("*"))
                        for suffix in t.supported_apps
                        if suffix.startswith("*") and suffix != "*"
                    )
                )
            ]

        # 按层级升序、可靠性降序排序，返回最优候选
        candidates.sort(key=lambda t: (t.tier, -t.reliability))
        return candidates[0] if candidates else None

    # ---- 工具选择（三层分级逻辑）----

    def select_tool(self, app: str, action: str, platform: str) -> Optional[MCPToolInfo]:
        """按三层优先级选择最优工具。

        选择逻辑：
          1. Tier 1：查找专用 MCP（匹配 app），且支持该 action
          2. Tier 2：查找平台泛用 MCP（匹配 platform）
          3. Tier 3：回退视觉 MCP

        Args:
            app:      目标 App 名称（如 "feishu"、"xiaohongshu"）
            action:   要执行的操作（如 "click"、"navigate"）
            platform: 当前平台（如 "web"、"android"）

        Returns:
            最优 MCP 工具信息，无可用工具返回 None
        """
        # Tier 1：专用 MCP（匹配 app + 支持 action）
        dedicated = self.find(app=app, tier=1)
        if dedicated and action in dedicated.capabilities:
            return dedicated

        # Tier 2：平台泛用 MCP（匹配 platform）
        generic = self.find(platform=platform, tier=2)
        if generic:
            return generic

        # Tier 3：视觉兜底
        return self.find(tier=3)

    # ---- 兼容旧接口 ----

    def list_all(self) -> list[MCPToolInfo]:
        """列出所有已注册工具（list_tools 的别名）。"""
        return self.list_tools()
