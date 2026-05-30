"""
Milestone 1 测试：核心 StateGraph + 节点 + 固定边 + invoke

TDD RED 阶段：先写测试，这些测试当前会失败，
因为 simple_langgraph 模块还不存在。
"""

import pytest


# ============================================================
# 测试 1: 最简单的线性图 — 一个节点，START → node → END
# ============================================================
class TestBasicLinearGraph:
    """最基础的一线穿三点的线性流程"""

    def test_single_node_graph(self):
        """
        graph:  START → greet → END

        节点 greet 接收 state，返回 {"greeting": "hello"}
        最终 state 应该合并这个返回值
        """
        from simple_langgraph import StateGraph, START, END

        def greet(state: dict) -> dict:
            return {"greeting": "hello"}

        graph = StateGraph()
        graph.add_node("greet", greet)
        graph.add_edge(START, "greet")
        graph.add_edge("greet", END)

        compiled = graph.compile()
        result = compiled.invoke({"name": "world"})

        # 原始输入保留，节点返回值合并进来
        assert result == {"name": "world", "greeting": "hello"}

    def test_two_node_linear_graph(self):
        """
        graph:  START → step_a → step_b → END

        step_a: 添加字段 "a"
        step_b: 添加字段 "b"
        """
        from simple_langgraph import StateGraph, START, END

        def step_a(state: dict) -> dict:
            return {"a": state["x"] + 1}

        def step_b(state: dict) -> dict:
            return {"b": state["a"] * 2}

        graph = StateGraph()
        graph.add_node("step_a", step_a)
        graph.add_node("step_b", step_b)
        graph.add_edge(START, "step_a")
        graph.add_edge("step_a", "step_b")
        graph.add_edge("step_b", END)

        compiled = graph.compile()
        result = compiled.invoke({"x": 5})

        # x=5 → step_a: a=6 → step_b: b=12
        assert result == {"x": 5, "a": 6, "b": 12}

    def test_three_node_chain(self):
        """
        graph:  START → add → multiply → format → END
        模拟一个数据处理管道
        """
        from simple_langgraph import StateGraph, START, END

        def add(state: dict) -> dict:
            return {"result": state["x"] + state["y"]}

        def multiply(state: dict) -> dict:
            return {"result": state["result"] * 10}

        def format_output(state: dict) -> dict:
            return {"output": f"Answer: {state['result']}"}

        graph = StateGraph()
        graph.add_node("add", add)
        graph.add_node("multiply", multiply)
        graph.add_node("format", format_output)
        graph.add_edge(START, "add")
        graph.add_edge("add", "multiply")
        graph.add_edge("multiply", "format")
        graph.add_edge("format", END)

        compiled = graph.compile()
        result = compiled.invoke({"x": 3, "y": 4})

        # 3+4=7 → 7*10=70 → "Answer: 70"
        assert result["output"] == "Answer: 70"
        assert result["result"] == 70


# ============================================================
# 测试 2: 节点函数可以返回空 dict（不修改状态）
# ============================================================
class TestNodeBehavior:
    """测试节点函数的不同行为"""

    def test_node_returns_empty_dict(self):
        """节点返回空 dict 时，state 不变"""
        from simple_langgraph import StateGraph, START, END

        def passthrough(state: dict) -> dict:
            return {}

        graph = StateGraph()
        graph.add_node("passthrough", passthrough)
        graph.add_edge(START, "passthrough")
        graph.add_edge("passthrough", END)

        compiled = graph.compile()
        result = compiled.invoke({"value": 42})

        assert result == {"value": 42}

    def test_node_overwrites_existing_key(self):
        """节点返回已有 key 时，覆盖旧值（last-writer-wins）"""
        from simple_langgraph import StateGraph, START, END

        def overwrite(state: dict) -> dict:
            return {"value": state["value"] + 1}

        graph = StateGraph()
        graph.add_node("overwrite", overwrite)
        graph.add_edge(START, "overwrite")
        graph.add_edge("overwrite", END)

        compiled = graph.compile()
        result = compiled.invoke({"value": 10})

        assert result == {"value": 11}

    def test_invoke_with_empty_state(self):
        """初始输入为空 dict 时，节点仍可正常工作"""
        from simple_langgraph import StateGraph, START, END

        def init(state: dict) -> dict:
            return {"started": True}

        graph = StateGraph()
        graph.add_node("init", init)
        graph.add_edge(START, "init")
        graph.add_edge("init", END)

        compiled = graph.compile()
        result = compiled.invoke({})

        assert result == {"started": True}


# ============================================================
# 测试 3: 边的验证 — 编译时检查图的结构合法性
# ============================================================
class TestGraphValidation:
    """编译阶段的图结构验证"""

    def test_compile_without_start_edge_raises(self):
        """没有从 START 出发的边 → 编译失败"""
        from simple_langgraph import StateGraph, END

        graph = StateGraph()
        graph.add_node("a", lambda s: {})
        graph.add_edge("a", END)
        # 缺少 START → a 的边

        with pytest.raises(ValueError, match="START"):
            graph.compile()

    def test_compile_with_disconnected_node_raises(self):
        """有节点没有入边也没有出边 → 编译失败"""
        from simple_langgraph import StateGraph, START, END

        graph = StateGraph()
        graph.add_node("a", lambda s: {})
        graph.add_node("b", lambda s: {})  # b 是孤立的
        graph.add_edge(START, "a")
        graph.add_edge("a", END)

        with pytest.raises(ValueError, match="unreachable|orphan|b"):
            graph.compile()

    def test_add_edge_to_nonexistent_node_raises(self):
        """添加边时引用不存在的节点 → 立即报错"""
        from simple_langgraph import StateGraph, START

        graph = StateGraph()

        with pytest.raises(ValueError):
            graph.add_edge(START, "nonexistent")

    def test_add_duplicate_node_raises(self):
        """重复注册同名节点 → 报错"""
        from simple_langgraph import StateGraph

        graph = StateGraph()
        graph.add_node("a", lambda s: {})

        with pytest.raises(ValueError, match="already exists"):
            graph.add_node("a", lambda s: {})


# ============================================================
# 测试 4: 分支与汇合（fan-out / fan-in 的简单形式）
# ============================================================
class TestBranchingGraph:
    """测试图的分支结构"""

    def test_two_branches_merge(self):
        """
        graph:
            START → process → END

        先用单链验证，后续 milestone 再加条件分支。
        这里测试的是：多个节点串行，state 持续累积。
        """
        from simple_langgraph import StateGraph, START, END

        def enrich(state: dict) -> dict:
            return {"enriched": True}

        def validate(state: dict) -> dict:
            return {"validated": True}

        def finalize(state: dict) -> dict:
            return {"finalized": True}

        graph = StateGraph()
        graph.add_node("enrich", enrich)
        graph.add_node("validate", validate)
        graph.add_node("finalize", finalize)
        graph.add_edge(START, "enrich")
        graph.add_edge("enrich", "validate")
        graph.add_edge("validate", "finalize")
        graph.add_edge("finalize", END)

        compiled = graph.compile()
        result = compiled.invoke({"data": "test"})

        assert result == {
            "data": "test",
            "enriched": True,
            "validated": True,
            "finalized": True,
        }
