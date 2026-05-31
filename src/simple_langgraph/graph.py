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
- 并行执行：扇出/扇入，DAG 波次执行引擎
"""

from __future__ import annotations

import contextvars
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


# ─── 动态 Interrupt 相关 ──────────────────────────────────────

class GraphInterrupt(Exception):
    """节点调用 interrupt() 时抛出的异常，用于暂停图执行。"""

    def __init__(self, value: Any, node_name: str):
        self.value = value
        self.node_name = node_name
        super().__init__(
            f"Graph interrupted in node '{node_name}': {value!r}"
        )


class Command:
    """恢复指令：invoke(Command(resume=value)) 传值给 interrupt()。"""

    def __init__(self, *, resume: Any = None):
        self.resume = resume


# 执行上下文：interrupt() 通过 ContextVar 获取当前节点信息和 resume 值
# ContextVar 天然线程隔离，避免并发场景下上下文互相污染
_exec_ctx: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar("_exec_ctx")

_SENTINEL = object()  # 标记"还没有 resume 值"


def interrupt(value: Any) -> Any:
    """
    在节点内部调用，暂停图执行并传值给客户端。

    首次调用：抛出 GraphInterrupt，value 存入 checkpoint。
    恢复后重新执行节点时：返回 Command(resume=...) 传入的值。
    """
    try:
        ctx = _exec_ctx.get()
    except LookupError:
        raise RuntimeError(
            "interrupt() can only be called inside a graph node"
        )

    # 检查是否有 resume 值（恢复模式）
    resume_list = ctx.get("resume", _SENTINEL)
    if resume_list is not _SENTINEL:
        # 有 resume 列表 → 按顺序消费
        idx = ctx.get("interrupt_counter", 0)
        if idx < len(resume_list):
            ctx["interrupt_counter"] = idx + 1
            return resume_list[idx]
        # resume 值已全部消费，后续 interrupt 正常抛出

    # 没有 resume → 抛出中断
    raise GraphInterrupt(value, ctx["node"])


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


@dataclass
class _Task:
    """内部任务：任务队列执行模型（类似官方 LangGraph）"""
    node: str
    state: str = "pending"   # pending / interrupted / completed
    sources: list[str] | None = None  # 条件边来源列表（用于恢复时精确重路由）
    extra: dict | None = None  # 额外 checkpoint 数据（如 interrupt_value）

    def to_dict(self) -> dict:
        """序列化为 checkpoint pending_tasks 条目。"""
        d: dict = {"node": self.node, "state": self.state}
        if self.sources:
            d["sources"] = self.sources
        if self.extra:
            d.update(self.extra)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "_Task":
        """从 checkpoint pending_tasks 条目反序列化。"""
        extra = {k: v for k, v in d.items()
                 if k not in ("node", "state", "sources")}
        return cls(
            node=d["node"],
            state=d.get("state", "pending"),
            sources=d.get("sources"),
            extra=extra if extra else None,
        )


# ─── 编译后的图（执行器） ──────────────────────────────────────────
class CompiledGraph:
    """
    编译后的可执行图。

    由 StateGraph.compile() 产生。
    - invoke(): 一次性执行完，返回最终 state
    - stream(): 生成器，每执行一个节点 yield 一次
    支持并行扇出/扇入、断点暂停、人工修改状态、流式输出。
    """

    def __init__(
        self,
        nodes: dict[str, Node],
        adjacency: dict[str, list[str]],
        conditional: dict[str, ConditionalEdge],
        reducers: dict[str, Callable],
        incoming_count: dict[str, int],
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        checkpointer: Optional[BaseCheckpointSaver] = None,
        interrupt_before: Optional[list[str]] = None,
        interrupt_after: Optional[list[str]] = None,
    ) -> None:
        self._nodes = nodes
        self._adjacency = adjacency
        self._conditional = conditional
        self._reducers = reducers
        self._incoming_count = incoming_count
        self._max_iterations = max_iterations
        self._checkpointer = checkpointer
        self._interrupt_before = set(interrupt_before or [])
        self._interrupt_after = set(interrupt_after or [])

    def invoke(
        self,
        input: Optional[dict[str, Any] | Command] = None,
        config: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        执行图：从 START 开始，沿边依次执行节点，直到到达 END 或断点。
        返回最终（或断点处的）state。

        input 可以是 dict（正常输入）或 Command(resume=value)（恢复中断）。
        """
        final_state: dict[str, Any] = {}
        for event in self._run(input, config):
            if event[0] == "value":
                final_state = event[1]
        return final_state

    def stream(
        self,
        input: Optional[dict[str, Any] | Command] = None,
        config: Optional[dict[str, Any]] = None,
        *,
        mode: str = "values",
    ) -> Generator[dict[str, Any], None, None]:
        """
        流式执行图：每执行一个节点 yield 一次。

        mode 参数：
        - "values": yield 完整 state 快照（默认）
        - "updates": yield {节点名: 节点返回的 updates}

        input 可以是 dict 或 Command(resume=value)。
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

    def _run_init(
        self,
        input: Optional[dict[str, Any] | Command],
        config: Optional[dict[str, Any]],
    ) -> tuple[dict[str, Any], Optional[list[_Task]], bool, Any, list[Any]]:
        """
        解析输入、加载 checkpoint、准备初始 state 和任务队列。

        返回 (state, ready_tasks, is_resuming, resume_value, dynamic_resume_prefix)。
        ready_tasks=None 表示图已执行完毕，由调用者 yield state 后 return。
        """
        thread_id = self._get_thread_id(config)

        # 解析 input
        resume_value: Any = _SENTINEL
        if isinstance(input, Command):
            resume_value = input.resume
            input = None

        # 校验
        if self._checkpointer and input is not None and not thread_id:
            raise ValueError(
                "A checkpointer is configured, but no thread_id was provided. "
                "Pass config={'thread_id': 'your-id'} to invoke()."
            )

        # 加载 checkpoint
        loaded_cp: Optional[dict[str, Any]] = None
        if self._checkpointer and thread_id:
            loaded_cp = self._checkpointer.load(thread_id)

        # 确定初始 state 和任务队列
        is_resuming = False
        if loaded_cp is not None:
            state = dict(loaded_cp["state"])
            raw_pending = loaded_cp["metadata"].get("pending_tasks")
            if raw_pending:
                is_resuming = True
                ready_tasks = self._restore_tasks(raw_pending, state)
            else:
                # 已执行完毕（无 pending_tasks）
                return state, None, False, resume_value, []
        else:
            state = dict(input) if input is not None else {}
            raw_start = list(self._adjacency.get(START, []))
            ready_tasks = [_Task(node=n) for n in raw_start if n != END]

        # 动态 interrupt 恢复准备
        dynamic_resume_prefix: list[Any] = []
        if is_resuming and loaded_cp:
            cp_meta = loaded_cp["metadata"]
            if cp_meta.get("interrupt") == "dynamic":
                dynamic_resume_prefix = cp_meta.get("consumed_resumes", [])

        return state, ready_tasks, is_resuming, resume_value, dynamic_resume_prefix

    def _run(
        self,
        input: Optional[dict[str, Any] | Command],
        config: Optional[dict[str, Any]],
        yield_updates: bool = False,
    ) -> Generator[tuple, dict[str, Any], None]:
        """
        内部执行引擎：基于任务队列的波次 DAG 执行（类似官方 LangGraph）。

        三阶段结构：
        - Phase 1: interrupt_before 检查（恢复时跳过）
        - Phase 2: 执行所有 pending 任务（含动态中断、interrupt_after）
        - Phase 3: 计算下一波次就绪节点

        yield 元组：
        - ("value", state_dict) — 节点执行后的完整 state
        - ("update", node_name, updates_dict) — 节点返回的增量
        """
        state, ready_tasks, is_resuming, resume_value, dynamic_resume_prefix = \
            self._run_init(input, config)
        if ready_tasks is None:
            yield ("value", state)
            return

        # ── 波次执行主循环 ──
        thread_id = self._get_thread_id(config)
        barrier: dict[str, int] = {}
        iterations = 0

        while ready_tasks:
            if iterations >= self._max_iterations:
                raise RuntimeError(
                    f"Graph execution exceeded maximum iterations ({self._max_iterations}). "
                    "This likely indicates an infinite loop in your graph."
                )

            # ── Phase 1: interrupt_before 检查（恢复时跳过） ──
            if not is_resuming:
                has_ib, ib_meta = self._check_interrupt_before(
                    ready_tasks, state, thread_id, iterations,
                )
                if has_ib:
                    self._save_checkpoint(thread_id, state, **ib_meta)
                    yield ("value", state)
                    return

            is_resuming = False

            # ── Phase 2: 执行所有 pending 任务 ──
            wave_updates: list[tuple[str, dict[str, Any]]] = []
            for task in ready_tasks:
                if task.state != "pending":
                    continue

                node = self._nodes[task.node]
                ctx: dict[str, Any] = {"node": task.node}
                if resume_value is not _SENTINEL:
                    ctx["resume"] = dynamic_resume_prefix + [resume_value]
                    ctx["interrupt_counter"] = 0

                _exec_ctx_token = _exec_ctx.set(ctx)
                try:
                    updates = node.action(state)
                except GraphInterrupt as gi:
                    if not self._checkpointer:
                        raise RuntimeError(
                            "interrupt() requires a checkpointer. "
                            "Pass checkpointer= to compile()."
                        )
                    consumed = list(
                        ctx.get("resume", [])[:ctx.get("interrupt_counter", 0)]
                    )
                    self._save_checkpoint(
                        thread_id, state,
                        **self._build_dynamic_interrupt_meta(
                            task, ready_tasks, iterations, gi, consumed,
                        ),
                    )
                    yield ("value", state)
                    return
                finally:
                    _exec_ctx.reset(_exec_ctx_token)

                if yield_updates:
                    yield ("update", task.node, dict(updates))

                _merge_state(state, updates, self._reducers)
                wave_updates.append((task.node, dict(updates)))
                task.state = "completed"

                # interrupt_after 检查
                if task.node in self._interrupt_after:
                    self._save_checkpoint(
                        thread_id, state,
                        node=task.node, step=iterations,
                        interrupt="after",
                        pending_tasks=self._build_interrupt_after_pending(
                            task, ready_tasks, state,
                        ),
                    )
                    yield ("value", state)
                    return

                # 保存 checkpoint
                self._save_checkpoint(
                    thread_id, state,
                    node=task.node, step=iterations,
                )

                yield ("value", dict(state))

            iterations += 1

            # ── Phase 3: 计算下一波次就绪节点 ──
            executed_nodes = {
                t.node for t in ready_tasks if t.state == "completed"
            }
            nodes_for_next = {
                node_name for node_name, _ in wave_updates
            } | executed_nodes
            ready_tasks = self._compute_next_wave(
                nodes_for_next, state, barrier, executed_nodes,
            )

    def update_state(
        self,
        config: dict[str, Any],
        values: dict[str, Any],
    ) -> None:
        """人工修改当前 checkpoint 的 state。"""
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

    def _restore_tasks(self, raw_pending: list[dict], state: dict) -> list[_Task]:
        """
        从 checkpoint 的 pending_tasks 恢复任务队列。

        包含条件边精确重路由逻辑：
        - 检查每个 source 是否仍指向原节点
        - 已重路由的 source 创建新任务
        - completed 任务保持原样（Phase 2 跳过执行，Phase 3 计入 barrier）
        - 非 completed 一律设为 pending（需要执行）
        """
        ready_tasks: list[_Task] = []
        for t in raw_pending:
            node = t["node"]
            task_state = t.get("state", "pending")
            sources = t.get("sources")
            rerouted = False

            # 只有 pending/interrupted 且有条件边来源的节点需要精确重路由
            # 检查每个 source：只要有一个 source 仍然指向该节点，就保留
            if task_state != "completed" and sources:
                still_targeting = []
                new_targets = []
                for src in sources:
                    result = self._eval_cond(src, state)
                    if not result or result == node or result == END:
                        still_targeting.append(src)
                    else:
                        new_targets.append((src, result))

                # 添加重路由的新目标（去重）
                for src, new_node in new_targets:
                    if not any(rt.node == new_node for rt in ready_tasks):
                        ready_tasks.append(_Task(node=new_node, sources=[src]))

                if still_targeting:
                    # 至少一个来源仍指向该节点 → 保留
                    if not any(rt.node == node for rt in ready_tasks):
                        ready_tasks.append(_Task(node=node, state="pending", sources=still_targeting))
                else:
                    # 所有来源都重路由了 → 不保留原节点
                    rerouted = True

            if not rerouted:
                if not any(rt.node == node for rt in ready_tasks):
                    # 恢复时：completed 保持原样（Phase 2 跳过，Phase 3 计入 barrier）
                    # 非 completed 一律设为 pending（需要执行）
                    restored_state = (
                        task_state if task_state == "completed"
                        else "pending"
                    )
                    ready_tasks.append(_Task(node=node, state=restored_state, sources=sources))
        return ready_tasks

    def _check_interrupt_before(
        self,
        ready_tasks: list[_Task],
        state: dict,
        thread_id: Optional[str],
        iterations: int,
    ) -> tuple[bool, dict[str, Any]]:
        """
        Phase 1: interrupt_before 检查。

        检查 ready_tasks 中是否有节点命中 interrupt_before。
        返回 (has_interrupt, metadata)：
        - has_interrupt=True 时，由 _run 负责保存 checkpoint 和 yield。
        - metadata 包含 checkpoint 所需的元数据。
        """
        has_interrupt = False
        for task in ready_tasks:
            if task.node in self._interrupt_before:
                task.state = "interrupted"
                has_interrupt = True

        if has_interrupt:
            metadata = {
                "node": next(
                    t.node for t in ready_tasks
                    if t.state == "interrupted"
                ),
                "step": iterations,
                "interrupt": "before",
                "pending_tasks": [t.to_dict() for t in ready_tasks],
            }
            return True, metadata

        return False, {}

    def _build_dynamic_interrupt_meta(
        self,
        task: _Task,
        ready_tasks: list[_Task],
        iterations: int,
        gi: GraphInterrupt,
        consumed: list[Any],
    ) -> dict[str, Any]:
        """构建动态中断（interrupt()）的 checkpoint metadata。"""
        return {
            "node": task.node,
            "step": iterations,
            "interrupt": "dynamic",
            "interrupt_value": gi.value,
            "consumed_resumes": consumed,
            "pending_tasks": [
                t.to_dict() | (
                    {"interrupt_value": gi.value}
                    if t is task else {}
                )
                for t in ready_tasks
            ],
        }

    def _save_checkpoint(
        self,
        thread_id: Optional[str],
        state: dict[str, Any],
        *,
        node: str,
        step: int,
        interrupt: Optional[str] = None,
        pending_tasks: Optional[list[dict]] = None,
        **extra: Any,
    ) -> None:
        """保存 checkpoint（如果配置了 checkpointer 且有 thread_id）。"""
        if not self._checkpointer or not thread_id:
            return
        metadata: dict[str, Any] = {
            "node": node,
            "step": step,
            "pending_tasks": pending_tasks or [],
        }
        if interrupt:
            metadata["interrupt"] = interrupt
        metadata.update(extra)
        self._checkpointer.save(thread_id, state, metadata=metadata)

    def _build_interrupt_after_pending(
        self,
        task: _Task,
        ready_tasks: list[_Task],
        state: dict,
    ) -> list[dict]:
        """
        构建 interrupt_after 场景的 pending_tasks 列表。

        包含：当前波次剩余 pending 任务 + 后继节点任务。
        条件边后继会附加 sources 信息。
        """
        remaining = [t for t in ready_tasks if t.state == "pending"]
        successors = self._resolve_next(task.node, state)
        has_cond = task.node in self._conditional
        pending = [t.to_dict() for t in remaining]
        for s in successors:
            if s != END:
                existing_p = next(
                    (p for p in pending if p["node"] == s),
                    None,
                )
                if existing_p and has_cond:
                    srcs = existing_p.get("sources", [])
                    if task.node not in srcs:
                        srcs.append(task.node)
                    existing_p["sources"] = srcs
                elif not existing_p:
                    pending.append(_Task(
                        node=s,
                        sources=[task.node] if has_cond else None,
                    ).to_dict())
        return pending

    def _compute_next_wave(
        self,
        nodes_for_next: set[str],
        state: dict,
        barrier: dict[str, int],
        executed_nodes: set[str],
    ) -> list[_Task]:
        """
        Phase 3: 计算下一波次就绪节点。

        根据本波次执行完成的节点，计算其后继节点是否满足执行条件。
        barrier 字典会被 in-place 更新（记录固定边到达计数）。
        """
        next_tasks: list[_Task] = []
        for node_name in nodes_for_next:
            # 固定边后继（可能参与扇入 barrier）
            fixed_succs = self._adjacency.get(node_name, [])
            # 条件边后继（required=0，不受 barrier 限制）
            cond_succ = None
            if node_name in self._conditional:
                result = self._eval_cond(node_name, state)
                if result and result != END:
                    cond_succ = result

            # 处理条件边后继：不走 barrier，不受 executed_nodes 限制
            if cond_succ:
                if cond_succ not in self._nodes:
                    raise ValueError(
                        f"Conditional edge from '{node_name}' returned "
                        f"invalid node '{cond_succ}'. "
                        f"Valid nodes: {list(self._nodes.keys())}"
                    )
                existing = next(
                    (t for t in next_tasks if t.node == cond_succ),
                    None,
                )
                if existing:
                    # 同节点已有任务，追加 source
                    if existing.sources is None:
                        existing.sources = [node_name]
                    elif node_name not in existing.sources:
                        existing.sources.append(node_name)
                else:
                    next_tasks.append(_Task(node=cond_succ, sources=[node_name]))

            # 处理固定边后继：走 barrier + 扇入逻辑
            for succ in fixed_succs:
                if succ == END:
                    continue
                barrier[succ] = barrier.get(succ, 0) + 1
                required = self._incoming_count.get(succ, 0)
                if required == 0:
                    # START 直连的固定边（罕见），不受 executed 限制
                    if not any(t.node == succ for t in next_tasks):
                        next_tasks.append(_Task(node=succ))
                elif barrier[succ] >= required:
                    if succ not in executed_nodes and not any(
                        t.node == succ for t in next_tasks
                    ):
                        next_tasks.append(_Task(node=succ))

        return next_tasks

    def _get_thread_id(self, config: Optional[dict[str, Any]]) -> Optional[str]:
        if not config:
            return None
        return config.get("thread_id")

    def _eval_cond(self, source: str, state: dict) -> str:
        """求值条件边路由。返回目标节点名或 END。异常时返回空字符串。"""
        cond = self._conditional.get(source)
        if not cond:
            return ""
        try:
            result = cond.path_fn(state)
            if cond.path_map and result in cond.path_map:
                result = cond.path_map[result]
            return result
        except Exception:
            return ""

    def _resolve_next(self, node_name: str, state: dict) -> list[str]:
        """
        决定 node_name 执行完后去哪。

        ① 如果有条件边 → 调用路由函数，返回结果
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
            print(chunk)

    并行执行：
        graph.add_edge("a", "b")
        graph.add_edge("a", "c")   # a → [b, c] 扇出
        graph.add_edge("b", "d")
        graph.add_edge("c", "d")   # [b, c] → d 扇入
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
        编译图：验证结构 + 构建邻接表 + 计算入边数 + 配置断点。
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

        # 构建固定边邻接表
        adjacency: dict[str, list[str]] = {}
        for edge in self._edges:
            adjacency.setdefault(edge.src, []).append(edge.dst)

        # 计算每个节点的入边数（来自固定边，不含 START 的边）
        incoming_count: dict[str, int] = {}
        for edge in self._edges:
            if edge.src != START and edge.dst != END:
                incoming_count[edge.dst] = incoming_count.get(edge.dst, 0) + 1

        # 构建条件边字典
        conditional: dict[str, ConditionalEdge] = {}
        for ce in self._conditional_edges:
            conditional[ce.src] = ce

        # 可达性检查
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
            incoming_count=incoming_count,
            max_iterations=max_iterations,
            checkpointer=checkpointer,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
        )
