"""
Simple LangGraph — 一个简化版 LangGraph 实现

核心概念：
- StateGraph：图构建器，用于注册节点和边
- CompiledGraph：编译后的可执行图
- START / END：虚拟节点，标识图的入口和出口
- 固定边：无条件从 A 走到 B
- 条件边：根据路由函数和 state 动态选择下一个节点
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

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


@dataclass
class ConditionalEdge:
    """一条条件边：从 src 出发，由 path_fn 决定下一个节点"""
    src: str
    path_fn: Callable[[dict], str]       # 路由函数：state → 下一节点名
    path_map: Optional[dict[str, str]] = None  # 可选映射：返回值 → 节点名


# ─── 编译后的图（执行器） ──────────────────────────────────────────
class CompiledGraph:
    """
    编译后的可执行图。

    由 StateGraph.compile() 产生，支持 invoke() 执行。
    内部维护：
    - _nodes: 节点注册表
    - _adjacency: 固定边邻接表 {节点名: [目标节点名列表]}
    - _conditional: 条件边 {源节点名: ConditionalEdge}
    """

    def __init__(
        self,
        nodes: dict[str, Node],
        adjacency: dict[str, list[str]],
        conditional: dict[str, ConditionalEdge],
    ) -> None:
        self._nodes = nodes
        self._adjacency = adjacency
        self._conditional = conditional

    def invoke(self, input: dict[str, Any]) -> dict[str, Any]:
        """
        执行图：从 START 开始，沿边依次执行节点，直到到达 END。

        对于每个节点，先检查有没有条件边：
        - 有条件边 → 调用路由函数决定下一个节点
        - 没有条件边 → 走固定边
        """
        state: dict[str, Any] = dict(input)
        current_nodes = self._adjacency.get(START, [])

        while current_nodes:
            next_node_name = current_nodes[0]

            if next_node_name == END:
                break

            node = self._nodes[next_node_name]
            updates = node.action(state)
            state.update(updates)

            # 查找下一个节点：条件边优先，否则走固定边
            current_nodes = self._resolve_next(next_node_name, state)

        return state

    def _resolve_next(self, node_name: str, state: dict) -> list[str]:
        """
        决定 node_name 执行完后去哪。

        ① 如果有条件边 → 调用路由函数，可选 path_map 翻译，返回结果
        ② 否则走固定边邻接表
        """
        cond = self._conditional.get(node_name)
        if cond:
            raw = cond.path_fn(state)
            # 如果有 path_map，翻译返回值
            if cond.path_map:
                if raw in cond.path_map:
                    next_name = cond.path_map[raw]
                else:
                    raise ValueError(
                        f"Router for '{node_name}' returned '{raw}', "
                        f"which is not in path_map. "
                        f"Available keys: {list(cond.path_map.keys())}"
                    )
            else:
                next_name = raw

            # 验证路由结果
            if next_name != END and next_name not in self._nodes:
                raise ValueError(
                    f"Router for '{node_name}' returned '{next_name}', "
                    f"which is not a valid node name. "
                    f"Available nodes: {list(self._nodes.keys()) + [END]}"
                )
            return [next_name]

        return self._adjacency.get(node_name, [])


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
        self._conditional_edges: list[ConditionalEdge] = []

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
        if src not in (START, END) and src not in self._nodes:
            raise ValueError(f"Source node '{src}' does not exist")
        if dst not in (START, END) and dst not in self._nodes:
            raise ValueError(f"Destination node '{dst}' does not exist")

        self._edges.append(Edge(src=src, dst=dst))

    def add_conditional_edges(
        self,
        source: str,
        path_fn: Callable[[dict], str],
        path_map: Optional[dict[str, str]] = None,
    ) -> None:
        """
        添加条件边。

        参数：
          source: 源节点名（条件边从哪个节点出发）
          path_fn: 路由函数，接收 state，返回下一个节点名（或 path_map 的 key）
          path_map: 可选映射表，把 path_fn 的返回值翻译成实际节点名
        """
        if source not in (START, END) and source not in self._nodes:
            raise ValueError(f"Source node '{source}' does not exist")

        self._conditional_edges.append(ConditionalEdge(
            src=source, path_fn=path_fn, path_map=path_map
        ))

    def compile(self) -> CompiledGraph:
        """
        编译图：验证结构 + 构建邻接表 + 注册条件边。

        验证规则：
        1. 必须有从 START 出发的边
        2. 所有注册的节点必须可达
        3. 一个节点不能同时有固定出边和条件出边
        """
        # 检查 1：必须有 START 出边
        start_edges = [e for e in self._edges if e.src == START]
        if not start_edges:
            raise ValueError(
                "Graph must have at least one edge from START. "
                "Use add_edge(START, 'your_node') to set the entry point."
            )

        # 检查 2：同一节点不能同时有固定出边和条件出边
        fixed_sources = {e.src for e in self._edges if e.src != START}
        cond_sources = {ce.src for ce in self._conditional_edges}
        conflict = fixed_sources & cond_sources
        if conflict:
            raise ValueError(
                f"Node(s) {conflict} have both fixed edges and conditional edges. "
                "Use one or the other, not both."
            )

        # 构建固定边邻接表
        adjacency: dict[str, list[str]] = {}
        for edge in self._edges:
            adjacency.setdefault(edge.src, []).append(edge.dst)

        # 构建条件边字典
        conditional: dict[str, ConditionalEdge] = {}
        for ce in self._conditional_edges:
            conditional[ce.src] = ce

        # 检查 3：所有节点必须可达（BFS）
        # 从 START 出发，沿固定边 + 有 path_map 的条件边做 BFS
        reachability: dict[str, list[str]] = {}
        for edge in self._edges:
            reachability.setdefault(edge.src, []).append(edge.dst)
        for ce in self._conditional_edges:
            if ce.path_map:
                reachability.setdefault(ce.src, []).extend(ce.path_map.values())

        reachable = set()
        queue = [START]
        while queue:
            current = queue.pop(0)
            for next_node in reachability.get(current, []):
                if next_node not in reachable and next_node != END:
                    reachable.add(next_node)
                    queue.append(next_node)

        # 没有 path_map 的条件边无法静态分析目标，跳过孤立检查
        has_dynamic_cond = any(
            not ce.path_map and ce.src in (reachable | {START})
            for ce in self._conditional_edges
        )

        if not has_dynamic_cond:
            orphan_nodes = set(self._nodes.keys()) - reachable
            if orphan_nodes:
                raise ValueError(
                    f"Node(s) {orphan_nodes} are unreachable from START. "
                    "Every node must have an incoming edge."
                )

        return CompiledGraph(nodes=self._nodes, adjacency=adjacency, conditional=conditional)
