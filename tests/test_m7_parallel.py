"""
Milestone 7 测试：并行 Fan-out / Fan-in

TDD RED 阶段：测试并行执行、扇出扇入、状态合并。
"""

import pytest


# ============================================================
# 测试 1: 基础钻石模式 A → [B, C] → D
# ============================================================
class TestBasicFanOutFanIn:

    def test_diamond_all_nodes_execute(self):
        """
        A → [B, C] → D
        A 执行完后，B 和 C 都执行，最后 D 执行。
        所有节点的更新都应该出现在最终 state 中。
        """
        from simple_langgraph import StateGraph, START, END

        call_order = []

        def node_a(state):
            call_order.append("a")
            return {"a": 1}

        def node_b(state):
            call_order.append("b")
            return {"b": 2}

        def node_c(state):
            call_order.append("c")
            return {"c": 3}

        def node_d(state):
            call_order.append("d")
            return {"d": 4}

        graph = StateGraph()
        graph.add_node("a", node_a)
        graph.add_node("b", node_b)
        graph.add_node("c", node_c)
        graph.add_node("d", node_d)
        graph.add_edge(START, "a")
        graph.add_edge("a", "b")
        graph.add_edge("a", "c")
        graph.add_edge("b", "d")
        graph.add_edge("c", "d")
        graph.add_edge("d", END)

        compiled = graph.compile()
        result = compiled.invoke({})

        assert "a" in result
        assert "b" in result
        assert "c" in result
        assert "d" in result
        assert result["a"] == 1
        assert result["b"] == 2
        assert result["c"] == 3
        assert result["d"] == 4

    def test_diamond_execution_order(self):
        """
        A 先执行，B 和 C 在 A 之后执行（顺序不限），D 在 B 和 C 之后。
        """
        from simple_langgraph import StateGraph, START, END

        call_order = []

        def node_a(state):
            call_order.append("a")
            return {"a": 1}

        def node_b(state):
            call_order.append("b")
            return {"b": 2}

        def node_c(state):
            call_order.append("c")
            return {"c": 3}

        def node_d(state):
            call_order.append("d")
            return {"d": 4}

        graph = StateGraph()
        graph.add_node("a", node_a)
        graph.add_node("b", node_b)
        graph.add_node("c", node_c)
        graph.add_node("d", node_d)
        graph.add_edge(START, "a")
        graph.add_edge("a", "b")
        graph.add_edge("a", "c")
        graph.add_edge("b", "d")
        graph.add_edge("c", "d")
        graph.add_edge("d", END)

        compiled = graph.compile()
        compiled.invoke({})

        # A 必须第一个，D 必须最后一个
        assert call_order[0] == "a"
        assert call_order[-1] == "d"
        # B 和 C 在中间，顺序不限
        assert set(call_order[1:3]) == {"b", "c"}

    def test_fan_out_to_end(self):
        """
        A → [B, C] → END
        B 和 C 都执行，结果合并到最终 state。
        """
        from simple_langgraph import StateGraph, START, END

        def node_a(state): return {"a": 1}
        def node_b(state): return {"b": 2}
        def node_c(state): return {"c": 3}

        graph = StateGraph()
        graph.add_node("a", node_a)
        graph.add_node("b", node_b)
        graph.add_node("c", node_c)
        graph.add_edge(START, "a")
        graph.add_edge("a", "b")
        graph.add_edge("a", "c")
        graph.add_edge("b", END)
        graph.add_edge("c", END)

        compiled = graph.compile()
        result = compiled.invoke({})

        assert result == {"a": 1, "b": 2, "c": 3}

    def test_three_way_fan_out(self):
        """
        A → [B, C, D] → E：三条并行分支。
        """
        from simple_langgraph import StateGraph, START, END

        def node_a(state): return {"a": 1}
        def node_b(state): return {"b": 2}
        def node_c(state): return {"c": 3}
        def node_d(state): return {"d": 4}
        def node_e(state): return {"e": 5}

        graph = StateGraph()
        graph.add_node("a", node_a)
        graph.add_node("b", node_b)
        graph.add_node("c", node_c)
        graph.add_node("d", node_d)
        graph.add_node("e", node_e)
        graph.add_edge(START, "a")
        graph.add_edge("a", "b")
        graph.add_edge("a", "c")
        graph.add_edge("a", "d")
        graph.add_edge("b", "e")
        graph.add_edge("c", "e")
        graph.add_edge("d", "e")
        graph.add_edge("e", END)

        compiled = graph.compile()
        result = compiled.invoke({})

        assert result == {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}


# ============================================================
# 测试 2: 并行状态合并
# ============================================================
class TestParallelStateMerge:

    def test_parallel_branches_merge_state(self):
        """
        B 和 C 各自添加不同的 key，D 能看到所有。
        """
        from simple_langgraph import StateGraph, START, END

        def enrich_name(state): return {"name": state.get("input", "").upper()}
        def enrich_greeting(state): return {"greeting": "Hello"}
        def combine(state): return {"done": True}

        graph = StateGraph()
        graph.add_node("name", enrich_name)
        graph.add_node("greeting", enrich_greeting)
        graph.add_node("combine", combine)
        graph.add_edge(START, "name")
        graph.add_edge(START, "greeting")
        graph.add_edge("name", "combine")
        graph.add_edge("greeting", "combine")
        graph.add_edge("combine", END)

        compiled = graph.compile()
        result = compiled.invoke({"input": "world"})

        assert result["name"] == "WORLD"
        assert result["greeting"] == "Hello"
        assert result["done"] is True

    def test_parallel_with_reducer(self):
        """
        B 和 C 都往同一个列表追加，reducer 合并。
        """
        from operator import add
        from simple_langgraph import StateGraph, START, END

        def node_a(state): return {"items": ["a"]}
        def node_b(state): return {"items": ["b"]}
        def node_c(state): return {"items": ["c"]}

        graph = StateGraph(schema={"items": (list, add)})
        graph.add_node("a", node_a)
        graph.add_node("b", node_b)
        graph.add_node("c", node_c)
        graph.add_edge(START, "a")
        graph.add_edge("a", "b")
        graph.add_edge("a", "c")
        graph.add_edge("b", END)
        graph.add_edge("c", END)

        compiled = graph.compile()
        result = compiled.invoke({"items": []})

        assert result["items"] == ["a", "b", "c"]

    def test_parallel_same_key_no_reducer_last_wins(self):
        """
        B 和 C 都写同一个 key，没有 reducer。
        行为：两个都合并到 state，后者覆盖前者（顺序不确定）。
        至少保证有一个值存在。
        """
        from simple_langgraph import StateGraph, START, END

        def node_a(state): return {"x": 1}
        def node_b(state): return {"x": 10}
        def node_c(state): return {"x": 20}

        graph = StateGraph()
        graph.add_node("a", node_a)
        graph.add_node("b", node_b)
        graph.add_node("c", node_c)
        graph.add_edge(START, "a")
        graph.add_edge("a", "b")
        graph.add_edge("a", "c")
        graph.add_edge("b", END)
        graph.add_edge("c", END)

        compiled = graph.compile()
        result = compiled.invoke({})

        # 并行分支都写了 "x"，最终值是其中一个
        assert result["x"] in (10, 20)


# ============================================================
# 测试 3: START 多个出边（并行入口）
# ============================================================
class TestParallelStart:

    def test_start_fan_out(self):
        """
        START → [A, B] → C
        两个节点同时从 START 开始执行。
        """
        from simple_langgraph import StateGraph, START, END

        def node_a(state): return {"a": 1}
        def node_b(state): return {"b": 2}
        def node_c(state): return {"c": 3}

        graph = StateGraph()
        graph.add_node("a", node_a)
        graph.add_node("b", node_b)
        graph.add_node("c", node_c)
        graph.add_edge(START, "a")
        graph.add_edge(START, "b")
        graph.add_edge("a", "c")
        graph.add_edge("b", "c")
        graph.add_edge("c", END)

        compiled = graph.compile()
        result = compiled.invoke({})

        assert result == {"a": 1, "b": 2, "c": 3}


# ============================================================
# 测试 4: Stream + 并行
# ============================================================
class TestParallelStreaming:

    def test_stream_values_with_diamond(self):
        """
        stream values 模式：每个节点执行后 yield 完整 state。
        钻石图 A → [B, C] → D 应该 yield 4 次。
        """
        from simple_langgraph import StateGraph, START, END

        def node_a(state): return {"a": 1}
        def node_b(state): return {"b": 2}
        def node_c(state): return {"c": 3}
        def node_d(state): return {"d": 4}

        graph = StateGraph()
        graph.add_node("a", node_a)
        graph.add_node("b", node_b)
        graph.add_node("c", node_c)
        graph.add_node("d", node_d)
        graph.add_edge(START, "a")
        graph.add_edge("a", "b")
        graph.add_edge("a", "c")
        graph.add_edge("b", "d")
        graph.add_edge("c", "d")
        graph.add_edge("d", END)

        compiled = graph.compile()
        chunks = list(compiled.stream({}))

        assert len(chunks) == 4
        # 第一个必须是 a
        assert chunks[0] == {"a": 1}
        # 最后一个包含所有
        assert chunks[-1] == {"a": 1, "b": 2, "c": 3, "d": 4}

    def test_stream_updates_with_diamond(self):
        """
        stream updates 模式：每个节点 yield {节点名: updates}。
        """
        from simple_langgraph import StateGraph, START, END

        def node_a(state): return {"a": 1}
        def node_b(state): return {"b": 2}
        def node_c(state): return {"c": 3}
        def node_d(state): return {"d": 4}

        graph = StateGraph()
        graph.add_node("a", node_a)
        graph.add_node("b", node_b)
        graph.add_node("c", node_c)
        graph.add_node("d", node_d)
        graph.add_edge(START, "a")
        graph.add_edge("a", "b")
        graph.add_edge("a", "c")
        graph.add_edge("b", "d")
        graph.add_edge("c", "d")
        graph.add_edge("d", END)

        compiled = graph.compile()
        chunks = list(compiled.stream({}, mode="updates"))

        assert len(chunks) == 4
        node_names = [list(c.keys())[0] for c in chunks]
        assert node_names[0] == "a"
        assert set(node_names[1:3]) == {"b", "c"}
        assert node_names[3] == "d"


# ============================================================
# 测试 5: 并行 + Checkpointer
# ============================================================
class TestParallelWithCheckpointer:

    def test_diamond_with_checkpointer(self):
        """
        并行执行也保存 checkpoint。
        """
        from simple_langgraph import StateGraph, START, END
        from simple_langgraph.checkpoint import MemorySaver

        def node_a(state): return {"a": 1}
        def node_b(state): return {"b": 2}
        def node_c(state): return {"c": 3}
        def node_d(state): return {"d": 4}

        graph = StateGraph()
        graph.add_node("a", node_a)
        graph.add_node("b", node_b)
        graph.add_node("c", node_c)
        graph.add_node("d", node_d)
        graph.add_edge(START, "a")
        graph.add_edge("a", "b")
        graph.add_edge("a", "c")
        graph.add_edge("b", "d")
        graph.add_edge("c", "d")
        graph.add_edge("d", END)

        saver = MemorySaver()
        compiled = graph.compile(checkpointer=saver)
        result = compiled.invoke({}, config={"thread_id": "t1"})

        assert result == {"a": 1, "b": 2, "c": 3, "d": 4}

        # checkpoint 保存了最终 state
        saved = compiled.get_state(config={"thread_id": "t1"})
        assert saved == {"a": 1, "b": 2, "c": 3, "d": 4}

    def test_diamond_checkpoint_history(self):
        """
        并行图的 checkpoint 历史记录了每一步。
        """
        from simple_langgraph import StateGraph, START, END
        from simple_langgraph.checkpoint import MemorySaver

        def node_a(state): return {"a": 1}
        def node_b(state): return {"b": 2}
        def node_c(state): return {"c": 3}
        def node_d(state): return {"d": 4}

        graph = StateGraph()
        graph.add_node("a", node_a)
        graph.add_node("b", node_b)
        graph.add_node("c", node_c)
        graph.add_node("d", node_d)
        graph.add_edge(START, "a")
        graph.add_edge("a", "b")
        graph.add_edge("a", "c")
        graph.add_edge("b", "d")
        graph.add_edge("c", "d")
        graph.add_edge("d", END)

        saver = MemorySaver()
        compiled = graph.compile(checkpointer=saver)
        compiled.invoke({}, config={"thread_id": "t1"})

        history = compiled.get_state_history(config={"thread_id": "t1"})
        # 应该有 4 个 checkpoint（a, b, c, d 各一个）
        assert len(history) == 4


# ============================================================
# 测试 6: 并行 + 条件边混合
# ============================================================
class TestParallelWithConditional:

    def test_conditional_then_parallel(self):
        """
        条件边选择进入并行区域。
        router → [B, C] → END
        """
        from simple_langgraph import StateGraph, START, END

        def router(state): return "branch"
        def node_b(state): return {"b": 1}
        def node_c(state): return {"c": 2}

        graph = StateGraph()
        graph.add_node("router", router)
        graph.add_node("b", node_b)
        graph.add_node("c", node_c)
        graph.add_edge(START, "router")
        graph.add_conditional_edges("router", lambda s: "parallel")
        # 注意：这里 router 用条件边路由到 "parallel"，
        # 但 "parallel" 不是节点名。需要不同的测试策略。
        # 改为：条件边路由到 B，然后 B 扇出到 C

        # 实际测试：条件边 → 节点 → 扇出
        graph2 = StateGraph()
        graph2.add_node("router", lambda s: {"path": "go"})
        graph2.add_node("b", lambda s: {"b": 1})
        graph2.add_node("c", lambda s: {"c": 2})
        graph2.add_node("d", lambda s: {"d": 3})
        graph2.add_edge(START, "router")
        graph2.add_conditional_edges("router", lambda s: "b")
        graph2.add_edge("b", "c")
        graph2.add_edge("b", "d")
        graph2.add_edge("c", END)
        graph2.add_edge("d", END)

        compiled = graph2.compile()
        result = compiled.invoke({})
        assert result["path"] == "go"
        assert result["b"] == 1
        assert "c" in result
        assert "d" in result


# ============================================================
# 测试 7: 复杂 DAG
# ============================================================
class TestComplexDAG:

    def test_two_separate_paths(self):
        """
        START → A, A → [B, C], B → D, C → E, [D, E] → F → END
        两条独立的路径最终汇聚到 F。
        """
        from simple_langgraph import StateGraph, START, END

        def a(s): return {"a": 1}
        def b(s): return {"b": 2}
        def c(s): return {"c": 3}
        def d(s): return {"d": 4}
        def e(s): return {"e": 5}
        def f(s): return {"f": 6}

        graph = StateGraph()
        for name, fn in [("a", a), ("b", b), ("c", c), ("d", d), ("e", e), ("f", f)]:
            graph.add_node(name, fn)

        graph.add_edge(START, "a")
        graph.add_edge("a", "b")
        graph.add_edge("a", "c")
        graph.add_edge("b", "d")
        graph.add_edge("c", "e")
        graph.add_edge("d", "f")
        graph.add_edge("e", "f")
        graph.add_edge("f", END)

        compiled = graph.compile()
        result = compiled.invoke({})

        assert result == {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6}

    def test_invoke_still_works_for_linear_graph(self):
        """
        加入并行支持后，线性图仍然正常工作。
        回归测试。
        """
        from simple_langgraph import StateGraph, START, END

        def a(s): return {"a": 1}
        def b(s): return {"b": 2}

        graph = StateGraph()
        graph.add_node("a", a)
        graph.add_node("b", b)
        graph.add_edge(START, "a")
        graph.add_edge("a", "b")
        graph.add_edge("b", END)

        compiled = graph.compile()
        result = compiled.invoke({})
        assert result == {"a": 1, "b": 2}

    def test_conditional_edges_still_work(self):
        """
        条件边回归测试。
        """
        from simple_langgraph import StateGraph, START, END

        def a(s): return {"x": 10}
        def pass_node(s): return {"result": "pass"}
        def fail_node(s): return {"result": "fail"}

        graph = StateGraph()
        graph.add_node("a", a)
        graph.add_node("pass", pass_node)
        graph.add_node("fail", fail_node)
        graph.add_edge(START, "a")
        graph.add_conditional_edges("a", lambda s: "pass" if s["x"] > 5 else "fail")
        graph.add_edge("pass", END)
        graph.add_edge("fail", END)

        compiled = graph.compile()
        result = compiled.invoke({})
        assert result["result"] == "pass"

    def test_cycles_still_work(self):
        """
        循环回归测试。
        """
        from simple_langgraph import StateGraph, START, END

        def count(s): return {"counter": s.get("counter", 0) + 1}
        def router(s): return END if s["counter"] >= 3 else "count"

        graph = StateGraph()
        graph.add_node("count", count)
        graph.add_edge(START, "count")
        graph.add_conditional_edges("count", router)

        compiled = graph.compile()
        result = compiled.invoke({})
        assert result["counter"] == 3
