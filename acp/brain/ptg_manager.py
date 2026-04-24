"""
PTG 管理器（Page Transition Graph Manager）
管理页面状态转换，确保多步操作的逻辑一致性。

MVP 实现：
  - 内存中维护有向图（dict + list）
  - 支持节点注册、边记录、当前状态追踪
  - BFS 路径查找

数据结构：
  nodes: {node_id → PTGNode}
  edges: [PTGEdge, ...]
  current_state: node_id | None
"""

from __future__ import annotations

import hashlib
import logging
from collections import deque
from typing import Optional

from acp.schema.elements import PageState
from acp.schema.ptg import PTGEdge, PTGGraph, PTGNode, PTGNodeType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PTGManager
# ---------------------------------------------------------------------------


class PTGManager:
    """页面转换图管理器（内存版，MVP 简化实现）

    使用示例：
        manager = PTGManager()

        # 记录转换
        manager.record_transition(
            from_state=page_state_a,
            action="click_login_btn",
            to_state=page_state_b,
        )

        # 获取当前状态
        node = manager.get_current_state()

        # 查找路径
        path = manager.find_path("node_a", "node_b")
    """

    def __init__(self) -> None:
        self._graph = PTGGraph()

    # ---- 节点管理 ----

    def add_node(self, node: PTGNode) -> None:
        """添加或覆盖节点。"""
        self._graph.nodes[node.node_id] = node
        logger.debug("PTGManager: 添加节点 %s (%s)", node.node_id, node.description)

    def get_node(self, node_id: str) -> Optional[PTGNode]:
        """按 node_id 获取节点。"""
        return self._graph.nodes.get(node_id)

    # ---- 边管理 ----

    def add_edge(self, edge: PTGEdge) -> None:
        """添加转换边（已存在相同 from/to/action 时跳过）。"""
        for existing in self._graph.edges:
            if (
                existing.from_node == edge.from_node
                and existing.to_node == edge.to_node
                and existing.action == edge.action
            ):
                return  # 已存在，不重复添加
        self._graph.edges.append(edge)
        logger.debug(
            "PTGManager: 添加边 %s -[%s]-> %s",
            edge.from_node, edge.action, edge.to_node,
        )

    # ---- 状态转换记录 ----

    def record_transition(
        self,
        from_state: PageState,
        action: str,
        to_state: PageState,
        params: Optional[dict] = None,
    ) -> tuple[PTGNode, PTGNode]:
        """记录一次页面转换（自动创建或复用节点）。

        Args:
            from_state: 操作前页面状态
            action:     触发转换的操作名
            to_state:   操作后页面状态
            params:     操作参数（可选）

        Returns:
            (from_node, to_node) 元组
        """
        from_node = self._ensure_node(from_state)
        to_node = self._ensure_node(to_state)

        edge = PTGEdge(
            from_node=from_node.node_id,
            to_node=to_node.node_id,
            action=action,
            params=params or {},
        )
        self.add_edge(edge)

        # 更新当前状态
        self._graph.current_state = to_node.node_id
        logger.info(
            "PTGManager: 记录转换 %s -[%s]-> %s",
            from_node.node_id, action, to_node.node_id,
        )
        return from_node, to_node

    def set_current_state(self, node_id: str) -> None:
        """手动设置当前节点（节点必须已注册）。"""
        if node_id not in self._graph.nodes:
            raise KeyError(f"节点 '{node_id}' 不存在")
        self._graph.current_state = node_id

    # ---- 查询 ----

    def get_current_state(self) -> Optional[PTGNode]:
        """获取当前所在节点。"""
        if self._graph.current_state is None:
            return None
        return self._graph.nodes.get(self._graph.current_state)

    def match_page_state(self, page_state: PageState) -> Optional[PTGNode]:
        """将页面状态匹配到 PTG 中的节点（URL/title 精确匹配）。

        Args:
            page_state: 当前页面状态

        Returns:
            匹配的 PTGNode，无匹配返回 None
        """
        node_id = self._page_state_to_node_id(page_state)
        return self._graph.nodes.get(node_id)

    def find_path(self, from_node_id: str, to_node_id: str) -> list[PTGEdge]:
        """BFS 查找从 from_node 到 to_node 的最短路径。

        Args:
            from_node_id: 起始节点 ID
            to_node_id:   目标节点 ID

        Returns:
            边列表（按顺序），未找到时返回空列表
        """
        if from_node_id == to_node_id:
            return []

        if from_node_id not in self._graph.nodes or to_node_id not in self._graph.nodes:
            return []

        # BFS
        queue: deque[tuple[str, list[PTGEdge]]] = deque()
        queue.append((from_node_id, []))
        visited: set[str] = {from_node_id}

        while queue:
            current, path = queue.popleft()

            for edge in self._graph.edges:
                if edge.from_node != current:
                    continue
                next_node = edge.to_node
                new_path = path + [edge]

                if next_node == to_node_id:
                    return new_path

                if next_node not in visited:
                    visited.add(next_node)
                    queue.append((next_node, new_path))

        return []  # 未找到路径

    # ---- 图信息 ----

    def node_count(self) -> int:
        return len(self._graph.nodes)

    def edge_count(self) -> int:
        return len(self._graph.edges)

    def get_graph(self) -> PTGGraph:
        """获取完整图结构（只读）。"""
        return self._graph

    def reset(self) -> None:
        """清空图（测试时使用）。"""
        self._graph = PTGGraph()

    # ---- 兼容旧占位接口 ----

    def get_current_node(self) -> Optional[PTGNode]:
        """get_current_state 的别名（兼容旧接口）。"""
        return self.get_current_state()

    # ---- 内部辅助 ----

    def _page_state_to_node_id(self, state: PageState) -> str:
        """将 PageState 转换为稳定的 node_id（基于 url/activity/app/title）。"""
        raw = f"{state.platform}::{state.app}::{state.url or state.activity or state.title}"
        return "node_" + hashlib.md5(raw.encode()).hexdigest()[:12]

    def _ensure_node(self, state: PageState) -> PTGNode:
        """获取或创建对应 PageState 的 PTGNode。"""
        node_id = (
            state.ptg_node_id
            if state.ptg_node_id
            else self._page_state_to_node_id(state)
        )

        if node_id not in self._graph.nodes:
            node = PTGNode(
                node_id=node_id,
                type=PTGNodeType.PAGE,
                app=state.app,
                description=f"{state.app} | {state.title or state.url or ''}",
            )
            self.add_node(node)

        return self._graph.nodes[node_id]
