"""
Milestone 6 测试：流式输出 Streaming

TDD RED 阶段：测试 stream() 方法的两种模式：
- values 模式：每执行一个节点，yield 当前完整 state
- updates 模式：每执行一个节点，yield 该节点返回的增量 updates
"""

import pytest


# ============================================================
# 测试 1: stream values 模式 — 每次 yield 完整 state
# ============================================================
class TestStreamValues:

    def test_single_node_stream(self):
        """
        单节点图 stream，yield 一次：执行完节点后的完整 state。
        """
        from simple_langgraph import StateGraph, START, END

        def add_one(state: dict) -> dict:
            return {"count": state.get("count", 0) + 1}

        graph = StateGraph()
        graph.add_node("add", add_one)
        graph.add_edge(START, "add")
        graph.add_edge("add", END)

        compiled = graph.compile()
        chunks = list(compiled.stream({"count": 0}))

        assert len(chunks) == 1
        assert chunks[0] == {"count": 1}

    def test_two_node_stream(self):
        """
        两节点链 stream，yield 两次：每次是执行完该节点后的完整 state。
        """
        from simple_langgraph import StateGraph, START, END

        def step_a(state: dict) -> dict:
            return {"a": 1}

        def step_b(state: dict) -> dict:
            return {"b": 2}

        graph = StateGraph()
        graph.add_node("step_a", step_a)
        graph.add_node("step_b", step_b)
        graph.add_edge(START, "step_a")
        graph.add_edge("step_a", "step_b")
        graph.add_edge("step_b", END)

        compiled = graph.compile()
        chunks = list(compiled.stream({"input": "test"}))

        assert len(chunks) == 2
        assert chunks[0] == {"input": "test", "a": 1}
        assert chunks[1] == {"input": "test", "a": 1, "b": 2}

    def test_three_node_chain(self):
        """
        3 节点链，3 次 yield。
        """
        from simple_langgraph import StateGraph, START, END

        def a(state): return {"a": 1}
        def b(state): return {"b": 2}
        def c(state): return {"c": 3}

        graph = StateGraph()
        graph.add_node("a", a)
        graph.add_node("b", b)
        graph.add_node("c", c)
        graph.add_edge(START, "a")
        graph.add_edge("a", "b")
        graph.add_edge("b", "c")
        graph.add_edge("c", END)

        compiled = graph.compile()
        chunks = list(compiled.stream({}))

        assert len(chunks) == 3
        assert chunks[0] == {"a": 1}
        assert chunks[1] == {"a": 1, "b": 2}
        assert chunks[2] == {"a": 1, "b": 2, "c": 3}

    def test_stream_with_loop(self):
        """
        循环图 stream，每次循环都 yield。
        """
        from simple_langgraph import StateGraph, START, END

        def count(state: dict) -> dict:
            return {"counter": state.get("counter", 0) + 1}

        def router(state: dict) -> str:
            if state["counter"] >= 3:
                return END
            return "count"

        graph = StateGraph()
        graph.add_node("count", count)
        graph.add_edge(START, "count")
        graph.add_conditional_edges("count", router)

        compiled = graph.compile()
        chunks = list(compiled.stream({}))

        assert len(chunks) == 3
        assert chunks[0] == {"counter": 1}
        assert chunks[1] == {"counter": 2}
        assert chunks[2] == {"counter": 3}

    def test_stream_is_generator(self):
        """
        stream() 返回的是生成器，可以逐步消费，不用一次性 list()。
        """
        from simple_langgraph import StateGraph, START, END

        def step(state: dict) -> dict:
            return {"done": True}

        graph = StateGraph()
        graph.add_node("step", step)
        graph.add_edge(START, "step")
        graph.add_edge("step", END)

        compiled = graph.compile()
        gen = compiled.stream({})

        # 验证是生成器
        import types
        assert isinstance(gen, types.GeneratorType)

        chunk = next(gen)
        assert chunk == {"done": True}

        # 没有更多了
        with pytest.raises(StopIteration):
            next(gen)


# ============================================================
# 测试 2: stream updates 模式 — 每次 yield 增量更新
# ============================================================
class TestStreamUpdates:

    def test_updates_mode_yields_node_output(self):
        """
        updates 模式：每次 yield {节点名: 节点返回的 updates}。
        """
        from simple_langgraph import StateGraph, START, END

        def step_a(state: dict) -> dict:
            return {"a": 1}

        def step_b(state: dict) -> dict:
            return {"b": 2}

        graph = StateGraph()
        graph.add_node("step_a", step_a)
        graph.add_node("step_b", step_b)
        graph.add_edge(START, "step_a")
        graph.add_edge("step_a", "step_b")
        graph.add_edge("step_b", END)

        compiled = graph.compile()
        chunks = list(compiled.stream({"input": "x"}, mode="updates"))

        assert len(chunks) == 2
        assert chunks[0] == {"step_a": {"a": 1}}
        assert chunks[1] == {"step_b": {"b": 2}}

    def test_updates_mode_with_loop(self):
        """
        循环图的 updates 模式。
        """
        from simple_langgraph import StateGraph, START, END

        def count(state: dict) -> dict:
            return {"counter": state.get("counter", 0) + 1}

        def router(state: dict) -> str:
            if state["counter"] >= 2:
                return END
            return "count"

        graph = StateGraph()
        graph.add_node("count", count)
        graph.add_edge(START, "count")
        graph.add_conditional_edges("count", router)

        compiled = graph.compile()
        chunks = list(compiled.stream({}, mode="updates"))

        assert len(chunks) == 2
        assert chunks[0] == {"count": {"counter": 1}}
        assert chunks[1] == {"count": {"counter": 2}}

    def test_updates_mode_with_reducer(self):
        """
        updates 模式 + reducer：yield 的是节点返回的原始值，不是 reducer 合并后的值。
        """
        from operator import add
        from simple_langgraph import StateGraph, START, END

        def append_item(state: dict) -> dict:
            return {"items": ["new"]}

        graph = StateGraph(schema={"items": (list, add)})
        graph.add_node("append", append_item)
        graph.add_edge(START, "append")
        graph.add_edge("append", END)

        compiled = graph.compile()
        chunks = list(compiled.stream({"items": ["old"]}, mode="updates"))

        # updates 模式 yield 的是节点返回的原始值
        assert chunks[0] == {"append": {"items": ["new"]}}


# ============================================================
# 测试 3: stream + checkpointer
# ============================================================
class TestStreamWithCheckpointer:

    def test_stream_saves_checkpoints(self):
        """
        stream 模式下也保存 checkpoint。
        """
        from simple_langgraph import StateGraph, START, END
        from simple_langgraph.checkpoint import MemorySaver

        def step_a(state): return {"a": 1}
        def step_b(state): return {"b": 2}

        graph = StateGraph()
        graph.add_node("a", step_a)
        graph.add_node("b", step_b)
        graph.add_edge(START, "a")
        graph.add_edge("a", "b")
        graph.add_edge("b", END)

        saver = MemorySaver()
        compiled = graph.compile(checkpointer=saver)
        chunks = list(compiled.stream({}, config={"thread_id": "t1"}))

        assert len(chunks) == 2

        # checkpoint 已保存
        state = compiled.get_state(config={"thread_id": "t1"})
        assert state == {"a": 1, "b": 2}

    def test_stream_with_interrupt(self):
        """
        stream 模式下遇到断点也暂停。
        恢复后继续 stream 剩余节点。
        """
        from simple_langgraph import StateGraph, START, END
        from simple_langgraph.checkpoint import MemorySaver

        def step_a(state): return {"a": 1}
        def step_b(state): return {"b": 2}
        def step_c(state): return {"c": 3}

        graph = StateGraph()
        graph.add_node("a", step_a)
        graph.add_node("b", step_b)
        graph.add_node("c", step_c)
        graph.add_edge(START, "a")
        graph.add_edge("a", "b")
        graph.add_edge("b", "c")
        graph.add_edge("c", END)

        saver = MemorySaver()
        compiled = graph.compile(
            checkpointer=saver,
            interrupt_after=["a"],
        )

        # 第一次 stream：执行 a 后暂停
        chunks1 = list(compiled.stream({}, config={"thread_id": "t1"}))
        assert len(chunks1) == 1
        assert chunks1[0] == {"a": 1}

        # 恢复 stream：执行 b 和 c
        chunks2 = list(compiled.stream(None, config={"thread_id": "t1"}))
        assert len(chunks2) == 2
        assert chunks2[0] == {"a": 1, "b": 2}
        assert chunks2[1] == {"a": 1, "b": 2, "c": 3}


# ============================================================
# 测试 4: 边界情况
# ============================================================
class TestStreamEdgeCases:

    def test_stream_empty_input(self):
        """空 state 也能 stream"""
        from simple_langgraph import StateGraph, START, END

        def noop(state): return {}

        graph = StateGraph()
        graph.add_node("noop", noop)
        graph.add_edge(START, "noop")
        graph.add_edge("noop", END)

        compiled = graph.compile()
        chunks = list(compiled.stream({}))

        assert len(chunks) == 1
        assert chunks[0] == {}

    def test_stream_no_nodes_after_start(self):
        """START 直接连 END → stream 空"""
        from simple_langgraph import StateGraph, START, END

        graph = StateGraph()
        graph.add_edge(START, END)

        compiled = graph.compile()
        chunks = list(compiled.stream({"x": 1}))

        assert chunks == []

    def test_invalid_stream_mode_raises(self):
        """不支持的 stream mode → 报错"""
        from simple_langgraph import StateGraph, START, END

        def step(state): return {"done": True}

        graph = StateGraph()
        graph.add_node("step", step)
        graph.add_edge(START, "step")
        graph.add_edge("step", END)

        compiled = graph.compile()

        with pytest.raises(ValueError, match="mode"):
            list(compiled.stream({}, mode="invalid"))
