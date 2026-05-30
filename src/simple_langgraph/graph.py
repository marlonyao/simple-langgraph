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
- Human-in-the-loop：断点暂停 + 人工修改 + 恢复执行
- Streaming：流式输出，支持 values 和 updates 两种模式
"""

from __future__ import annotations

from collections.abc import Generator
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

    由 StateGraph.compile() 产生。
    - invoke(): 一次性执行完，返回最终 state
    - stream(): 生成器，每执行一个节点 yield 一次
    支持断点暂停、人工修改状态、流式输出。
    """

    def __init__(
        self,
        nodes: dict[str, Node],
        adjacency: dict[str, list[str]],
        conditional: dict[str, ConditionalEdge],
        reducers: dict[str, Callable],
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        checkpointer: Optional[BaseCheckpointSaver] = None,
        interrupt_before: Optional[list[str]] = None,
        interrupt_after: Optional[list[str]] = None,
    ) -> None:
        self._nodes = nodes
        self._adjacency = adjacency
        self._conditional = conditional
        self._reducers = reducers
        self._max_iterations = max_iterations
        self._checkpointer = checkpointer
        self._interrupt_before = set(interrupt_before or [])
        self._interrupt_after = set(interrupt_after or [])

    def invoke(
        self,
        input: Optional[dict[str, Any]] = None,
        config: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        执行图：从 START 开始，沿边依次执行节点，直到到达 END 或断点。
        返回最终（或断点处的）state。
        """
        final_state: dict[str, Any] = {}
        for event in self._run(input, config):
            if event[0] == "value":
                final_state = event[1]
        return final_state

    def stream(
        self,
        input: Optional[dict[str, Any]] = None,
        config: Optional[dict[str, Any]] = None,
        *,
        mode: str = "values",
    ) -> Generator[dict[str, Any], None, None]:
        """
        流式执行图：每执行一个节点 yield 一次。

        mode 参数：
        - "values": yield 完整 state 快照（默认）
        - "updates": yield {节点名: 节点返回的 updates}
        """
        if mode not in ("values", "updates"):
            raise ValueError(
                f"Unsupported stream mode: '{mode}'. "
                "Use 'values' or 'updates'."
            )

        for event in self._run(input, config, yield_updates=(mode == "updates")):
            if mode == "values" and event[0] == "value":
                yield event[1]
            elif mode == "updates" and event[0] == "update":
                yield {event[1]: event[2]}

    def _run(
        self,
        input: Optional[dict[str, Any]],
        config: Optional[dict[str, Any]],
        yield_updates: bool = False,
    ) -> Generator[tuple, dict[str, Any], None]:
        """
        内部执行引擎：invoke 和 stream 的共享逻辑。

        yield 元组：
        - ("value", state_dict) — 节点执行后的完整 state
        - ("update", node_name, updates_dict) — 节点返回的增量（仅 yield_updates=True 时）

        返回值通过 Generator 的 send 不使用（统一用 yield）。
        """
        thread_id = self._get_thread_id(config)

        # ── 校验：有 checkpointer 且有 input 时必须提供 thread_id ──
        if self._checkpointer and input is not None and not thread_id:
            raise ValueError(
                "A checkpointer is configured, but no thread_id was provided. "
                "Pass config={'thread_id': 'your-id'} to invoke()."
            )

        # ── 判断是否为断点恢复 ──
        is_resume = False
        saved_resume = None
        saved_resume_from: Optional[str] = None
        if self._checkpointer and thread_id:
            cp = self._checkpointer.load(thread_id)
            if cp is not None:
                is_resume = True
                saved_resume = cp["metadata"].get("resume_node")
                saved_resume_from = cp["metadata"].get("resume_from")

        # ── 确定初始 state ──
        if is_resume:
            cp = self._checkpointer.load(thread_id)
            state: dict[str, Any] = dict(cp["state"])
        elif input is not None:
            state = dict(input)
        else:
            state = {}

        # ── 决定从哪个节点开始 ──
        skip_interrupt_before_once: Optional[str] = None
        if is_resume and saved_resume:
            if saved_resume == END:
                yield ("value", state)
                return

            if saved_resume_from:
                current_nodes = self._resolve_next(saved_resume_from, state)
                if not current_nodes or current_nodes[0] == END:
                    yield ("value", state)
                    return
                skip_interrupt_before_once = current_nodes[0]
            else:
                current_nodes = [saved_resume]
                skip_interrupt_before_once = saved_resume
        elif is_resume and not saved_resume:
            yield ("value", state)
            return
        else:
            current_nodes = self._adjacency.get(START, [])

        # ── 执行循环 ──
        iterations = 0
        prev_node_name: Optional[str] = None
        while current_nodes:
            if iterations >= self._max_iterations:
                raise RuntimeError(
                    f"Graph execution exceeded maximum iterations ({self._max_iterations}). "
                    "This likely indicates an infinite loop in your graph."
                )

            next_node_name = current_nodes[0]

            # 到达 END → 结束
            if next_node_name == END:
                break

            # interrupt_before：在这个节点执行前暂停
            if next_node_name in self._interrupt_before and next_node_name != skip_interrupt_before_once:
                if self._checkpointer and thread_id:
                    self._checkpointer.save(
                        thread_id, state,
                        metadata={
                            "node": next_node_name,
                            "step": iterations,
                            "interrupt": "before",
                            "resume_node": next_node_name,
                            "resume_from": prev_node_name,
                        },
                    )
                yield ("value", state)
                return

            # 清除跳过标志
            skip_interrupt_before_once = None

            # 执行节点
            node = self._nodes[next_node_name]
            updates = node.action(state)

            # yield updates（stream updates 模式需要）
            if yield_updates:
                yield ("update", next_node_name, dict(updates))

            _merge_state(state, updates, self._reducers)

            # interrupt_after：在这个节点执行后暂停
            if next_node_name in self._interrupt_after:
                if self._checkpointer and thread_id:
                    successors = self._resolve_next(next_node_name, state)
                    resume_node = successors[0] if successors else END
                    self._checkpointer.save(
                        thread_id, state,
                        metadata={
                            "node": next_node_name,
                            "step": iterations,
                            "interrupt": "after",
                            "resume_node": resume_node,
                        },
                    )
                yield ("value", state)
                return

            # 保存 checkpoint（非断点的常规保存）
            if self._checkpointer and thread_id:
                self._checkpointer.save(
                    thread_id, state,
                    metadata={"node": next_node_name, "step": iterations},
                )

            # yield value（stream values 模式 + invoke 都需要）
            yield ("value", dict(state))

            current_nodes = self._resolve_next(next_node_name, state)
            prev_node_name = next_node_name
            iterations += 1

        # 循环正常结束（到达 END 或没有后续节点）
        # 不需要额外 yield，上面循环中已经 yield 了每个节点的 value
        # invoke() 会用空 dict 作为默认值

    def update_state(
        self,
        config: dict[str, Any],
        values: dict[str, Any],
    ) -> None:
        """
        人工修改当前 checkpoint 的 state。

        只能在有 checkpointer 时使用。
        修改后，下次 invoke(None, config) 会用修改后的 state 继续。
        """
        if not self._checkpointer:
            raise ValueError(
                "Cannot update state: no checkpointer is configured. "
                "Pass a checkpointer to compile() first."
            )

        thread_id = config.get("thread_id")
        if not thread_id:
            raise ValueError(
                "Cannot update state: no thread_id provided in config."
            )

        cp = self._checkpointer.load(thread_id)
        if cp is None:
            raise ValueError(
                f"No checkpoint found for thread_id '{thread_id}'. "
                "Run invoke() first to create an initial checkpoint."
            )

        state = dict(cp["state"])
        _merge_state(state, values, self._reducers)
        metadata = dict(cp["metadata"])
        self._checkpointer.save(thread_id, state, metadata=metadata)

    def get_state(self, config: dict[str, Any]) -> Optional[dict[str, Any]]:
        """获取最新的 checkpoint state。"""
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
        """获取所有历史 state（最新在前）。"""
        if not self._checkpointer:
            return []

        thread_id = config.get("thread_id")
        if not thread_id:
            return []

        history = self._checkpointer.list_history(thread_id)
        return [h["state"] for h in history]

    # ─── 内部辅助方法 ─────────────────────────────────────────────

    def _get_thread_id(self, config: Optional[dict[str, Any]]) -> Optional[str]:
        """从 config 中提取 thread_id。"""
        if not config:
            return None
        return config.get("thread_id")

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

    带 checkpointer + 断点：
        from simple_langgraph.checkpoint import MemorySaver
        app = graph.compile(
            checkpointer=MemorySaver(),
            interrupt_before=["approve"],
        )
        result = app.invoke({"key": "value"}, config={"thread_id": "t1"})

    流式输出：
        for chunk in app.stream({"key": "value"}):
            print(chunk)  # 每个节点执行后的 state
    """

    def __init__(self, schema: Optional[dict] = None) -> None:
        self._nodes: dict[str, Node] = {}
        self._edges: list[Edge] = []
        self._conditional_edges: list[ConditionalEdge] = []
        self._reducers: dict[str, Callable] = {}

        if schema is not None:
            for key, spec in schema.items():
                if isinstance(spec, tuple) and len(spec) == 2:
                    self._reducers[key] = spec[1]

    def add_node(self, name: str, action: Callable[[dict], dict]) -> None:
        """注册一个节点。"""
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
        interrupt_before: Optional[list[str]] = None,
        interrupt_after: Optional[list[str]] = None,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
    ) -> CompiledGraph:
        """
        编译图：验证结构 + 构建邻接表 + 注册条件边 + 配置断点。

        参数：
          checkpointer: 可选的状态持久化后端
          interrupt_before: 在这些节点执行前暂停
          interrupt_after: 在这些节点执行后暂停
          max_iterations: 最大执行步数，防止无限循环
        """
        start_edges = [e for e in self._edges if e.src == START]
        if not start_edges:
            raise ValueError(
                "Graph must have at least one edge from START. "
                "Use add_edge(START, 'your_node') to set the entry point."
            )

        fixed_sources = {e.src for e in self._edges if e.src != START}
        cond_sources = {ce.src for ce in self._conditional_edges}
        conflict = fixed_sources & cond_sources
        if conflict:
            raise ValueError(
                f"Node(s) {conflict} have both fixed edges and conditional edges. "
                "Use one or the other, not both."
            )

        adjacency: dict[str, list[str]] = {}
        for edge in self._edges:
            adjacency.setdefault(edge.src, []).append(edge.dst)

        conditional: dict[str, ConditionalEdge] = {}
        for ce in self._conditional_edges:
            conditional[ce.src] = ce

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
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
        )
