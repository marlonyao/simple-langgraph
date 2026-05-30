"""
Milestone 2 测试：条件路由 Conditional Edges

TDD RED 阶段：测试 add_conditional_edges 的各种场景。
"""

import pytest


# ============================================================
# 测试 1: 基本条件路由 — 根据状态走不同分支
# ============================================================
class TestBasicConditionalRouting:

    def test_route_by_score_pass(self):
        """
        graph:  START → classify → (pass_route) → END
                            ↓
                       (fail_route) → END

        classify 节点打分，条件边根据分数决定走 pass 还是 fail 分支。
        """
        from simple_langgraph import StateGraph, START, END

        def classify(state: dict) -> dict:
            return {"grade": "pass" if state["score"] >= 60 else "fail"}

        def pass_route(state: dict) -> dict:
            return {"result": "恭喜通过！"}

        def fail_route(state: dict) -> dict:
            return {"result": "继续加油！"}

        def router(state: dict) -> str:
            return "pass_route" if state["grade"] == "pass" else "fail_route"

        graph = StateGraph()
        graph.add_node("classify", classify)
        graph.add_node("pass_route", pass_route)
        graph.add_node("fail_route", fail_route)
        graph.add_edge(START, "classify")
        graph.add_conditional_edges("classify", router)
        graph.add_edge("pass_route", END)
        graph.add_edge("fail_route", END)

        compiled = graph.compile()

        # 高分走 pass
        result = compiled.invoke({"score": 95})
        assert result["grade"] == "pass"
        assert result["result"] == "恭喜通过！"
        assert "result" in result

    def test_route_by_score_fail(self):
        """同上，低分走 fail"""
        from simple_langgraph import StateGraph, START, END

        def classify(state: dict) -> dict:
            return {"grade": "pass" if state["score"] >= 60 else "fail"}

        def pass_route(state: dict) -> dict:
            return {"result": "恭喜通过！"}

        def fail_route(state: dict) -> dict:
            return {"result": "继续加油！"}

        def router(state: dict) -> str:
            return "pass_route" if state["grade"] == "pass" else "fail_route"

        graph = StateGraph()
        graph.add_node("classify", classify)
        graph.add_node("pass_route", pass_route)
        graph.add_node("fail_route", fail_route)
        graph.add_edge(START, "classify")
        graph.add_conditional_edges("classify", router)
        graph.add_edge("pass_route", END)
        graph.add_edge("fail_route", END)

        compiled = graph.compile()

        result = compiled.invoke({"score": 30})
        assert result["grade"] == "fail"
        assert result["result"] == "继续加油！"


# ============================================================
# 测试 2: 条件路由直接返回 END
# ============================================================
class TestConditionalRouteToEnd:

    def test_conditional_edge_can_return_end(self):
        """
        graph:  START → check → (条件) → done → END
                            ↓
                         END（直接结束）

        有时条件路由可以直接返回 END，跳过后续节点。
        """
        from simple_langgraph import StateGraph, START, END

        def check(state: dict) -> dict:
            return {"checked": True}

        def done(state: dict) -> dict:
            return {"finished": True}

        def router(state: dict) -> str:
            if state.get("skip"):
                return END
            return "done"

        graph = StateGraph()
        graph.add_node("check", check)
        graph.add_node("done", done)
        graph.add_edge(START, "check")
        graph.add_conditional_edges("check", router)
        graph.add_edge("done", END)

        compiled = graph.compile()

        # skip=True → 直接结束，不执行 done
        result = compiled.invoke({"skip": True})
        assert result == {"skip": True, "checked": True}

        # skip=False → 执行 done
        result = compiled.invoke({"skip": False})
        assert result == {"skip": False, "checked": True, "finished": True}


# ============================================================
# 测试 3: path_map — 路由函数返回简写，映射到节点名
# ============================================================
class TestPathMap:

    def test_path_map_translates_return_values(self):
        """
        路由函数返回 "a" / "b" / "c" 这样的简写，
        path_map 把它们映射到真正的节点名。
        """
        from simple_langgraph import StateGraph, START, END

        def decide(state: dict) -> dict:
            return {"choice": state.get("pick", "left")}

        def handle_left(state: dict) -> dict:
            return {"path": "went left"}

        def handle_right(state: dict) -> dict:
            return {"path": "went right"}

        def router(state: dict) -> str:
            return state["choice"]

        graph = StateGraph()
        graph.add_node("decide", decide)
        graph.add_node("handle_left", handle_left)
        graph.add_node("handle_right", handle_right)
        graph.add_edge(START, "decide")
        graph.add_conditional_edges(
            "decide",
            router,
            path_map={"left": "handle_left", "right": "handle_right"}
        )
        graph.add_edge("handle_left", END)
        graph.add_edge("handle_right", END)

        compiled = graph.compile()

        result = compiled.invoke({"pick": "left"})
        assert result["path"] == "went left"

        result = compiled.invoke({"pick": "right"})
        assert result["path"] == "went right"

    def test_path_map_with_end(self):
        """path_map 的值可以是 END"""
        from simple_langgraph import StateGraph, START, END

        def decide(state: dict) -> dict:
            return {}

        def handle(state: dict) -> dict:
            return {"handled": True}

        def router(state: dict) -> str:
            return "skip" if state.get("skip") else "go"

        graph = StateGraph()
        graph.add_node("decide", decide)
        graph.add_node("handle", handle)
        graph.add_edge(START, "decide")
        graph.add_conditional_edges(
            "decide",
            router,
            path_map={"skip": END, "go": "handle"}
        )
        graph.add_edge("handle", END)

        compiled = graph.compile()

        result = compiled.invoke({"skip": True})
        assert "handled" not in result

        result = compiled.invoke({"skip": False})
        assert result["handled"] == True


# ============================================================
# 测试 4: 多级条件路由 — 链式决策
# ============================================================
class TestChainedConditionalRouting:

    def test_two_level_routing(self):
        """
        graph:
            START → level1 → (条件) → level2a → (条件) → done_a → END
                              ↓                    ↓
                           level2b             done_b → END
                              ↓
                             END

        两层条件路由，验证状态在多级决策中正确传递。
        """
        from simple_langgraph import StateGraph, START, END

        def level1(state: dict) -> dict:
            return {"l1": "A" if state["x"] > 0 else "B"}

        def level2a(state: dict) -> dict:
            return {"l2": "X" if state["y"] > 0 else "Y"}

        def level2b(state: dict) -> dict:
            return {"l2": "Z"}

        def done_a(state: dict) -> dict:
            return {"final": f"{state['l1']}-{state['l2']}"}

        def done_b(state: dict) -> dict:
            return {"final": f"{state['l1']}-{state['l2']}"}

        def router1(state: dict) -> str:
            return "level2a" if state["l1"] == "A" else "level2b"

        def router2a(state: dict) -> str:
            return "done_a"

        graph = StateGraph()
        graph.add_node("level1", level1)
        graph.add_node("level2a", level2a)
        graph.add_node("level2b", level2b)
        graph.add_node("done_a", done_a)
        graph.add_node("done_b", done_b)
        graph.add_edge(START, "level1")
        graph.add_conditional_edges("level1", router1)
        graph.add_conditional_edges("level2a", router2a)
        graph.add_edge("level2b", "done_b")
        graph.add_edge("done_a", END)
        graph.add_edge("done_b", END)

        compiled = graph.compile()

        # x=5, y=3 → l1=A → level2a → l2=X → done_a → "A-X"
        result = compiled.invoke({"x": 5, "y": 3})
        assert result["final"] == "A-X"

        # x=-1, y=0 → l1=B → level2b → l2=Z → done_b → "B-Z"
        result = compiled.invoke({"x": -1, "y": 0})
        assert result["final"] == "B-Z"


# ============================================================
# 测试 5: 混合使用固定边和条件边
# ============================================================
class TestMixedEdges:

    def test_fixed_then_conditional(self):
        """
        graph:  START → preprocess → analyze → (条件) → good → END
                                                  ↓
                                               bad → END
        前半段固定边，后半段条件边。
        """
        from simple_langgraph import StateGraph, START, END

        def preprocess(state: dict) -> dict:
            return {"value": state["raw"] * 2}

        def analyze(state: dict) -> dict:
            return {"quality": "high" if state["value"] > 10 else "low"}

        def good(state: dict) -> dict:
            return {"tag": "premium"}

        def bad(state: dict) -> dict:
            return {"tag": "standard"}

        def router(state: dict) -> str:
            return "good" if state["quality"] == "high" else "bad"

        graph = StateGraph()
        graph.add_node("preprocess", preprocess)
        graph.add_node("analyze", analyze)
        graph.add_node("good", good)
        graph.add_node("bad", bad)
        graph.add_edge(START, "preprocess")
        graph.add_edge("preprocess", "analyze")
        graph.add_conditional_edges("analyze", router)
        graph.add_edge("good", END)
        graph.add_edge("bad", END)

        compiled = graph.compile()

        result = compiled.invoke({"raw": 8})  # value=16 > 10 → good
        assert result["tag"] == "premium"

        result = compiled.invoke({"raw": 3})  # value=6 ≤ 10 → bad
        assert result["tag"] == "standard"


# ============================================================
# 测试 6: 验证条件边的错误处理
# ============================================================
class TestConditionalEdgeValidation:

    def test_conditional_edges_source_must_exist(self):
        """条件边的源节点必须已注册"""
        from simple_langgraph import StateGraph

        graph = StateGraph()

        with pytest.raises(ValueError):
            graph.add_conditional_edges("nonexistent", lambda s: "x")

    def test_router_returns_invalid_node_raises(self):
        """路由函数返回不存在的节点名 → 运行时报错"""
        from simple_langgraph import StateGraph, START, END

        def bad_router(state: dict) -> str:
            return "this_node_does_not_exist"

        graph = StateGraph()
        graph.add_node("a", lambda s: {})
        graph.add_edge(START, "a")
        graph.add_conditional_edges("a", bad_router)

        compiled = graph.compile()

        with pytest.raises(ValueError, match="this_node_does_not_exist"):
            compiled.invoke({})
