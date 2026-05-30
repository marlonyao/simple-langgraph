"""
Simple LangGraph — 一个简化版 LangGraph 实现

核心概念：
- StateGraph：图构建器，用于注册节点和边
- CompiledGraph：编译后的可执行图
- START / END：虚拟节点，标识图的入口和出口
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

# ─── 虚拟节点常量 ───────────────────────────────────────────────
START = "__start__"
END = "__end__"


# ─── 数据结构 ───────────────────────────────────────────────────
@dataclass
class Node:
    """图中一个节点的定义"""
    name: str
    action: Callable[[dict], dict]


@dataclass
class Edge:
    """一条固定边：从 src 到 dst"""
    src: str
    dst: str


# ─── 编译后的图（执行器） ──────────────────────────────────────────
class CompiledGraph:
    """
    编译后的可执行图。

    由 StateGraph.compile() 产生，支持 invoke() 执行。
    内部维护一个邻接表 (adjacency)，存储每个节点的出边列表。
    """

    def __init__(self, nodes: dict[str, Node], adjacency: dict[str, list[str]]) -> None:
        self._nodes = nodes
        self._adjacency = adjacency

    def invoke(self, input: dict[str, Any]) -> dict[str, Any]:
        """
        执行图：从 START 开始，沿边依次执行节点，直到到达 END。

        核心逻辑：
        1. 初始 state = input 的副本
        2. 当前节点 = START 的下一个节点
        3. 循环：执行当前节点，合并返回值到 state，找下一个节点
        4. 遇到 END 停止
        """
        state: dict[str, Any] = dict(input)

        # 从 START 出发，找到第一个要执行的节点
        current_nodes = self._adjacency.get(START, [])

        while current_nodes:
            # 线性执行：当前只有一个后继节点
            next_node_name = current_nodes[0]

            if next_node_name == END:
                break

            node = self._nodes[next_node_name]
            # 执行节点函数，获取返回的部分状态
            updates = node.action(state)
            # 合并到总状态（last-writer-wins：新值覆盖旧值）
            state.update(updates)

            # 移动到下一个节点
            current_nodes = self._adjacency.get(next_node_name, [])

        return state


# ─── 图构建器 ────────────────────────────────────────────────────
class StateGraph:
    """
    状态图构建器。

    用法：
        graph = StateGraph()
        graph.add_node("name", fn)
        graph.add_edge(START, "name")
        graph.add_edge("name", END)
        app = graph.compile()
        result = app.invoke({"key": "value"})
    """

    def __init__(self) -> None:
        self._nodes: dict[str, Node] = {}
        self._edges: list[Edge] = []

    def add_node(self, name: str, action: Callable[[dict], dict]) -> None:
        """注册一个节点。name 是唯一标识，action 是接收 state 返回部分 state 的函数。"""
        if name in self._nodes:
            raise ValueError(f"Node '{name}' already exists")
        if name in (START, END):
            raise ValueError(f"Cannot use reserved name '{name}' for a node")
        self._nodes[name] = Node(name=name, action=action)

    def add_edge(self, src: str, dst: str) -> None:
        """
        添加一条固定边。

        src/dst 可以是节点名、START、或 END。
        如果引用了不存在的节点名（非 START/END），立即报错。
        """
        # 验证 src
        if src not in (START, END) and src not in self._nodes:
            raise ValueError(f"Source node '{src}' does not exist")
        # 验证 dst
        if dst not in (START, END) and dst not in self._nodes:
            raise ValueError(f"Destination node '{dst}' does not exist")

        self._edges.append(Edge(src=src, dst=dst))

    def compile(self) -> CompiledGraph:
        """
        编译图：验证结构 + 构建邻接表。

        验证规则：
        1. 必须有从 START 出发的边
        2. 所有注册的节点必须可达（有入边）
        """
        # 检查 1：必须有 START 出边
        start_edges = [e for e in self._edges if e.src == START]
        if not start_edges:
            raise ValueError(
                "Graph must have at least one edge from START. "
                "Use add_edge(START, 'your_node') to set the entry point."
            )

        # 构建邻接表
        adjacency: dict[str, list[str]] = {}
        for edge in self._edges:
            adjacency.setdefault(edge.src, []).append(edge.dst)

        # 检查 2：所有节点必须可达（从 START 可达）
        reachable = set()
        queue = [START]
        while queue:
            current = queue.pop(0)
            for next_node in adjacency.get(current, []):
                if next_node not in reachable and next_node != END:
                    reachable.add(next_node)
                    queue.append(next_node)

        orphan_nodes = set(self._nodes.keys()) - reachable
        if orphan_nodes:
            raise ValueError(
                f"Node(s) {orphan_nodes} are unreachable from START. "
                "Every node must have an incoming edge."
            )

        return CompiledGraph(nodes=self._nodes, adjacency=adjacency)
