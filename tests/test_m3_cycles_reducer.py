"""
Milestone 3 测试：循环 + Reducer

TDD RED 阶段：测试图的循环执行和 reducer 状态聚合。
"""

import operator
import pytest


# ============================================================
# 测试 1: 基本循环 — 条件边回指之前的节点
# ============================================================
class TestBasicCycles:

    def test_loop_with_counter(self):
        """
        graph:  START → count → (条件) → count（循环）
                            ↓
                          END（count >= 3 时退出）

        count 节点每次给 counter 加 1，到 3 时退出循环。
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
        result = compiled.invoke({})

        assert result["counter"] == 3

    def test_loop_accumulates_state(self):
        """
        graph:  START → append → (条件) → append（循环）
                            ↓
                          END（items 有 3 个时退出）

        每次循环给 items 列表加一个元素。
        验证 state 在循环中正确累积。
        """
        from simple_langgraph import StateGraph, START, END

        def append(state: dict) -> dict:
            items = state.get("items", [])
            return {"items": items + [len(items)]}

        def router(state: dict) -> str:
            if len(state["items"]) >= 3:
                return END
            return "append"

        graph = StateGraph()
        graph.add_node("append", append)
        graph.add_edge(START, "append")
        graph.add_conditional_edges("append", router)

        compiled = graph.compile()
        result = compiled.invoke({})

        assert result["items"] == [0, 1, 2]

    def test_two_node_loop(self):
        """
        graph:  START → a → b → (条件) → a（循环）
                                ↓
                              END

        两个节点交替执行，形成 a → b → a → b → ... 的循环。
        """
        from simple_langgraph import StateGraph, START, END

        def a(state: dict) -> dict:
            return {"turn": "a", "count": state.get("count", 0) + 1}

        def b(state: dict) -> dict:
            return {"turn": "b"}

        def router(state: dict) -> str:
            if state["count"] >= 3:
                return END
            return "a"

        graph = StateGraph()
        graph.add_node("a", a)
        graph.add_node("b", b)
        graph.add_edge(START, "a")
        graph.add_edge("a", "b")
        graph.add_conditional_edges("b", router)

        compiled = graph.compile()
        result = compiled.invoke({})

        assert result["count"] == 3
        assert result["turn"] == "b"


# ============================================================
# 测试 2: 最大迭代保护 — 防止无限循环
# ============================================================
class TestMaxIterations:

    def test_infinite_loop_raises(self):
        """
        路由函数永远不返回 END → 应该在达到最大迭代时抛异常。
        """
        from simple_langgraph import StateGraph, START, END

        def noop(state: dict) -> dict:
            return {}

        def always_loop(state: dict) -> str:
            return "noop"  # 永远不退出

        graph = StateGraph()
        graph.add_node("noop", noop)
        graph.add_edge(START, "noop")
        graph.add_conditional_edges("noop", always_loop)

        compiled = graph.compile()

        with pytest.raises(RuntimeError, match="maximum.*iteration"):
            compiled.invoke({})

    def test_custom_max_iterations(self):
        """
        用户可以自定义最大迭代次数。
        """
        from simple_langgraph import StateGraph, START, END

        call_count = 0

        def tick(state: dict) -> dict:
            nonlocal call_count
            call_count += 1
            return {"n": state.get("n", 0) + 1}

        def router(state: dict) -> str:
            return "tick"  # 永远循环

        graph = StateGraph()
        graph.add_node("tick", tick)
        graph.add_edge(START, "tick")
        graph.add_conditional_edges("tick", router)

        compiled = graph.compile(max_iterations=5)

        with pytest.raises(RuntimeError, match="5"):
            compiled.invoke({})

        # 循环执行了 5 次后停止
        assert call_count == 5


# ============================================================
# 测试 3: Reducer — 自定义状态合并策略
# ============================================================
class TestReducer:

    def test_list_reducer_with_operator_add(self):
        """
        用 schema 声明列表类型的 reducer。
        节点返回列表时，不是覆盖，而是追加合并。
        """
        from simple_langgraph import StateGraph, START, END

        def step1(state: dict) -> dict:
            return {"items": ["a"]}

        def step2(state: dict) -> dict:
            return {"items": ["b"]}

        def step3(state: dict) -> dict:
            return {"items": ["c"]}

        graph = StateGraph(schema={"items": (list, operator.add)})
        graph.add_node("step1", step1)
        graph.add_node("step2", step2)
        graph.add_node("step3", step3)
        graph.add_edge(START, "step1")
        graph.add_edge("step1", "step2")
        graph.add_edge("step2", "step3")
        graph.add_edge("step3", END)

        compiled = graph.compile()
        result = compiled.invoke({"items": []})

        assert result["items"] == ["a", "b", "c"]

    def test_list_reducer_in_loop(self):
        """
        循环中每次返回一个元素，reducer 把它们追加成列表。
        """
        from simple_langgraph import StateGraph, START, END

        def append_item(state: dict) -> dict:
            n = len(state["items"])
            return {"items": [n]}

        def router(state: dict) -> str:
            if len(state["items"]) >= 4:
                return END
            return "append"

        graph = StateGraph(schema={"items": (list, operator.add)})
        graph.add_node("append", append_item)
        graph.add_edge(START, "append")
        graph.add_conditional_edges("append", router)

        compiled = graph.compile()
        result = compiled.invoke({"items": []})

        assert result["items"] == [0, 1, 2, 3]

    def test_mixed_reducer_and_overwrite(self):
        """
        有的 key 用 reducer（追加），有的 key 用默认策略（覆盖）。
        """
        from simple_langgraph import StateGraph, START, END

        def step(state: dict) -> dict:
            return {"logs": ["entry"], "status": "processing"}

        def finish(state: dict) -> dict:
            return {"logs": ["done"], "status": "complete"}

        # 只有 logs 用 reducer，status 默认覆盖
        graph = StateGraph(schema={"logs": (list, operator.add), "status": str})
        graph.add_node("step", step)
        graph.add_node("finish", finish)
        graph.add_edge(START, "step")
        graph.add_edge("step", "finish")
        graph.add_edge("finish", END)

        compiled = graph.compile()
        result = compiled.invoke({"logs": [], "status": "init"})

        assert result["logs"] == ["entry", "done"]  # reducer: 追加
        assert result["status"] == "complete"        # 默认: 覆盖

    def test_custom_reducer_function(self):
        """
        自定义 reducer：比如取最大值。
        """
        from simple_langgraph import StateGraph, START, END

        def max_reducer(old, new):
            return max(old, new)

        def update(state: dict) -> dict:
            return {"score": state["score"] + 3}

        def update2(state: dict) -> dict:
            return {"score": state["score"] + 5}

        graph = StateGraph(schema={"score": (int, max_reducer)})
        graph.add_node("update", update)
        graph.add_node("update2", update2)
        graph.add_edge(START, "update")
        graph.add_edge("update", "update2")
        graph.add_edge("update2", END)

        compiled = graph.compile()
        result = compiled.invoke({"score": 0})

        # update: max(0, 3) = 3
        # update2: max(3, 8) = 8
        assert result["score"] == 8

    def test_reducer_without_schema_uses_overwrite(self):
        """
        不传 schema 时，保持原有的 last-writer-wins 行为。
        """
        from simple_langgraph import StateGraph, START, END

        def step(state: dict) -> dict:
            return {"value": [1]}

        def step2(state: dict) -> dict:
            return {"value": [2]}

        graph = StateGraph()  # 无 schema
        graph.add_node("step", step)
        graph.add_node("step2", step2)
        graph.add_edge(START, "step")
        graph.add_edge("step", "step2")
        graph.add_edge("step2", END)

        compiled = graph.compile()
        result = compiled.invoke({})

        assert result["value"] == [2]  # 覆盖，不是追加
