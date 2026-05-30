"""
Simple LangGraph — 一个简化版 LangGraph 实现

核心概念：
- StateGraph：图构建器，用于注册节点和边
- CompiledGraph：编译后的可执行图
- START / END：虚拟节点，标识图的入口和出口
- 固定边：无条件从 A 走到 B
- 条件边：根据路由函数和 state 动态选择下一个节点
- Reducer：自定义状态合并策略（如列表追加、取最大值）
- Checkpointer：状态持久化，支持时间旅行（历史回溯）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from simple_langgraph.checkpoint import BaseCheckpointSaver

# ─── 虚拟节点常量 ───────────────────────────────────────────────
START = "__start__"
END = "__end__"

# ─── 默认最大迭代次数 ──────────────────────────────────────────
DEFAULT_MAX_ITERATIONS = 100


# ─── 状态合并辅助 ───────────────────────────────────────────────
def _merge_state(
    state: dict[str, Any],
    updates: dict[str, Any],
    reducers: dict[str, Callable],
) -> None:
    """
    把 updates 合并到 state。

    - 有 reducer 的 key：调用 reducer(old, new) 合并
    - 没有 reducer 的 key：直接覆盖（last-writer-wins）
    """
    for key, new_val in updates.items():
        if key in reducers:
            old_val = state.get(key)
            state[key] = reducers[key](old_val, new_val)
        else:
            state[key] = new_val


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
    """

    def __init__(
        self,
        nodes: dict[str, Node],
        adjacency: dict[str, list[str]],
        conditional: dict[str, ConditionalEdge],
        reducers: dict[str, Callable],
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        checkpointer: Optional[BaseCheckpointSaver] = None,
    ) -> None:
        self._nodes = nodes
        self._adjacency = adjacency
        self._conditional = conditional
        self._reducers = reducers
        self._max_iterations = max_iterations
        self._checkpointer = checkpointer

    def invoke(
        self,
        input: dict[str, Any],
        config: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        执行图：从 START 开始，沿边依次执行节点，直到到达 END。

        如果配置了 checkpointer，每执行一个节点后保存 checkpoint。
        config 必须包含 thread_id（当有 checkpointer 时）。
        """
        state: dict[str, Any] = dict(input)

        # 有 checkpointer 时必须提供 thread_id
        thread_id: Optional[str] = None
        if self._checkpointer:
            if not config or "thread_id" not in config:
                raise ValueError(
                    "A checkpointer is configured, but no thread_id was provided. "
                    "Pass config={'thread_id': 'your-id'} to invoke()."
                )
            thread_id = config["thread_id"]

        current_nodes = self._adjacency.get(START, [])

        iterations = 0
        while current_nodes:
            if iterations >= self._max_iterations:
                raise RuntimeError(
                    f"Graph execution exceeded maximum iterations ({self._max_iterations}). "
                    "This likely indicates an infinite loop in your graph."
                )

            next_node_name = current_nodes[0]

            if next_node_name == END:
                break

            node = self._nodes[next_node_name]
            updates = node.action(state)
            _merge_state(state, updates, self._reducers)

            # 保存 checkpoint
            if self._checkpointer and thread_id:
                self._checkpointer.save(
                    thread_id, state, metadata={"node": next_node_name, "step": iterations}
                )

            current_nodes = self._resolve_next(next_node_name, state)
            iterations += 1

        return state

    def get_state(self, config: dict[str, Any]) -> Optional[dict[str, Any]]:
        """
        获取最新的 checkpoint state。

        返回 state 字典，如果没有 checkpointer 或没有数据则返回 None。
        """
        if not self._checkpointer:
            return None

        thread_id = config.get("thread_id")
        if not thread_id:
            return None

        cp = self._checkpointer.load(thread_id)
        if cp is None:
            return None
        return cp["state"]

    def get_state_history(self, config: dict[str, Any]) -> list[dict[str, Any]]:
        """
        获取所有历史 state（最新在前）。

        返回 state 字典列表，用于时间旅行。
        """
        if not self._checkpointer:
            return []

        thread_id = config.get("thread_id")
        if not thread_id:
            return []

        history = self._checkpointer.list_history(thread_id)
        return [h["state"] for h in history]

    def _resolve_next(self, node_name: str, state: dict) -> list[str]:
        """
        决定 node_name 执行完后去哪。

        ① 如果有条件边 → 调用路由函数，可选 path_map 翻译，返回结果
        ② 否则走固定边邻接表
        """
        cond = self._conditional.get(node_name)
        if cond:
            raw = cond.path_fn(state)
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

    带 reducer：
        from operator import add
        graph = StateGraph(schema={"items": (list, add)})

    带 checkpointer：
        from simple_langgraph.checkpoint import MemorySaver
        app = graph.compile(checkpointer=MemorySaver())
        result = app.invoke({"key": "value"}, config={"thread_id": "t1"})
    """

    def __init__(self, schema: Optional[dict] = None) -> None:
        self._nodes: dict[str, Node] = {}
        self._edges: list[Edge] = []
        self._conditional_edges: list[ConditionalEdge] = []
        self._reducers: dict[str, Callable] = {}

        # 解析 schema 中的 reducer
        if schema is not None:
            for key, spec in schema.items():
                if isinstance(spec, tuple) and len(spec) == 2:
                    # {"key": (type, reducer_func)}
                    self._reducers[key] = spec[1]

    def add_node(self, name: str, action: Callable[[dict], dict]) -> None:
        """注册一个节点。name 是唯一标识，action 是接收 state 返回部分 state 的函数。"""
        if name in self._nodes:
            raise ValueError(f"Node '{name}' already exists")
        if name in (START, END):
            raise ValueError(f"Cannot use reserved name '{name}' for a node")
        self._nodes[name] = Node(name=name, action=action)

    def add_edge(self, src: str, dst: str) -> None:
        """添加一条固定边。"""
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
        """添加条件边。"""
        if source not in (START, END) and source not in self._nodes:
            raise ValueError(f"Source node '{source}' does not exist")

        self._conditional_edges.append(ConditionalEdge(
            src=source, path_fn=path_fn, path_map=path_map
        ))

    def compile(
        self,
        checkpointer: Optional[BaseCheckpointSaver] = None,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
    ) -> CompiledGraph:
        """
        编译图：验证结构 + 构建邻接表 + 注册条件边。

        参数：
          checkpointer: 可选的状态持久化后端（如 MemorySaver）
          max_iterations: 最大执行步数，防止无限循环（默认 100）
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

        return CompiledGraph(
            nodes=self._nodes,
            adjacency=adjacency,
            conditional=conditional,
            reducers=self._reducers,
            max_iterations=max_iterations,
            checkpointer=checkpointer,
        )
