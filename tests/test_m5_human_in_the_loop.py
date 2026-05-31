"""
Milestone 5 测试：Human-in-the-loop 断点机制

TDD RED 阶段：测试 interrupt_before / interrupt_after、
暂停恢复、人工修改状态后继续执行。
"""

import pytest


# ============================================================
# 测试 1: interrupt_before — 在指定节点前暂停
# ============================================================
class TestInterruptBefore:

    def test_interrupt_before_stops_before_node(self):
        """
        3 节点链：A → B → C
        编译时设 interrupt_before=["B"]
        第一次 invoke 执行完 A 就暂停，返回 A 的输出。
        """
        from simple_langgraph import StateGraph, START, END
        from simple_langgraph.checkpoint import MemorySaver

        def node_a(state: dict) -> dict:
            return {"a": True}

        def node_b(state: dict) -> dict:
            return {"b": True}

        def node_c(state: dict) -> dict:
            return {"c": True}

        graph = StateGraph()
        graph.add_node("a", node_a)
        graph.add_node("b", node_b)
        graph.add_node("c", node_c)
        graph.add_edge(START, "a")
        graph.add_edge("a", "b")
        graph.add_edge("b", "c")
        graph.add_edge("c", END)

        saver = MemorySaver()
        compiled = graph.compile(
            checkpointer=saver,
            interrupt_before=["b"],
        )

        # 第一次 invoke：应该执行 A，然后在 B 之前暂停
        result = compiled.invoke({"input": "test"}, config={"thread_id": "t1"})
        assert result == {"input": "test", "a": True}
        assert "b" not in result  # B 没执行

    def test_interrupt_before_resume_continues(self):
        """
        暂停后，再次 invoke 同一个 thread_id，从断点继续执行。
        """
        from simple_langgraph import StateGraph, START, END
        from simple_langgraph.checkpoint import MemorySaver

        def node_a(state: dict) -> dict:
            return {"a": True}

        def node_b(state: dict) -> dict:
            return {"b": True}

        def node_c(state: dict) -> dict:
            return {"c": True}

        graph = StateGraph()
        graph.add_node("a", node_a)
        graph.add_node("b", node_b)
        graph.add_node("c", node_c)
        graph.add_edge(START, "a")
        graph.add_edge("a", "b")
        graph.add_edge("b", "c")
        graph.add_edge("c", END)

        saver = MemorySaver()
        compiled = graph.compile(
            checkpointer=saver,
            interrupt_before=["b"],
        )

        # 第一次：执行 A，暂停
        result1 = compiled.invoke({}, config={"thread_id": "t1"})
        assert result1 == {"a": True}

        # 第二次：从断点恢复，执行 B → C
        result2 = compiled.invoke(None, config={"thread_id": "t1"})
        assert result2 == {"a": True, "b": True, "c": True}

    def test_interrupt_before_multiple_breakpoints(self):
        """
        设两个断点：interrupt_before=["b", "c"]
        需要三次 invoke 才能跑完。
        """
        from simple_langgraph import StateGraph, START, END
        from simple_langgraph.checkpoint import MemorySaver

        def node_a(state: dict) -> dict:
            return {"a": 1}

        def node_b(state: dict) -> dict:
            return {"b": 2}

        def node_c(state: dict) -> dict:
            return {"c": 3}

        graph = StateGraph()
        graph.add_node("a", node_a)
        graph.add_node("b", node_b)
        graph.add_node("c", node_c)
        graph.add_edge(START, "a")
        graph.add_edge("a", "b")
        graph.add_edge("b", "c")
        graph.add_edge("c", END)

        saver = MemorySaver()
        compiled = graph.compile(
            checkpointer=saver,
            interrupt_before=["b", "c"],
        )

        # 第一次：A 执行完，B 前暂停
        r1 = compiled.invoke({}, config={"thread_id": "t1"})
        assert r1 == {"a": 1}

        # 第二次：B 执行完，C 前暂停
        r2 = compiled.invoke(None, config={"thread_id": "t1"})
        assert r2 == {"a": 1, "b": 2}

        # 第三次：C 执行完，结束
        r3 = compiled.invoke(None, config={"thread_id": "t1"})
        assert r3 == {"a": 1, "b": 2, "c": 3}


# ============================================================
# 测试 2: interrupt_after — 在指定节点后暂停
# ============================================================
class TestInterruptAfter:

    def test_interrupt_after_stops_after_node(self):
        """
        interrupt_after=["a"]
        执行完 A 后暂停（B 和 C 不执行）。
        """
        from simple_langgraph import StateGraph, START, END
        from simple_langgraph.checkpoint import MemorySaver

        def node_a(state: dict) -> dict:
            return {"a": True}

        def node_b(state: dict) -> dict:
            return {"b": True}

        graph = StateGraph()
        graph.add_node("a", node_a)
        graph.add_node("b", node_b)
        graph.add_edge(START, "a")
        graph.add_edge("a", "b")
        graph.add_edge("b", END)

        saver = MemorySaver()
        compiled = graph.compile(
            checkpointer=saver,
            interrupt_after=["a"],
        )

        result = compiled.invoke({}, config={"thread_id": "t1"})
        assert result == {"a": True}

    def test_interrupt_after_resume(self):
        """暂停后恢复，走完剩余节点"""
        from simple_langgraph import StateGraph, START, END
        from simple_langgraph.checkpoint import MemorySaver

        def node_a(state: dict) -> dict:
            return {"a": True}

        def node_b(state: dict) -> dict:
            return {"b": True}

        graph = StateGraph()
        graph.add_node("a", node_a)
        graph.add_node("b", node_b)
        graph.add_edge(START, "a")
        graph.add_edge("a", "b")
        graph.add_edge("b", END)

        saver = MemorySaver()
        compiled = graph.compile(
            checkpointer=saver,
            interrupt_after=["a"],
        )

        r1 = compiled.invoke({}, config={"thread_id": "t1"})
        assert r1 == {"a": True}

        r2 = compiled.invoke(None, config={"thread_id": "t1"})
        assert r2 == {"a": True, "b": True}


# ============================================================
# 测试 3: update_state — 人工修改状态后继续
# ============================================================
class TestUpdateState:

    def test_update_state_then_resume(self):
        """
        人工审批场景：
        1. 执行到断点暂停
        2. 人工用 update_state 修改 state
        3. 恢复执行，后续节点看到修改后的 state
        """
        from simple_langgraph import StateGraph, START, END
        from simple_langgraph.checkpoint import MemorySaver

        def review(state: dict) -> dict:
            return {"draft": "需要审批的内容"}

        def publish(state: dict) -> dict:
            return {"published": state.get("approved", False)}

        graph = StateGraph()
        graph.add_node("review", review)
        graph.add_node("publish", publish)
        graph.add_edge(START, "review")
        graph.add_edge("review", "publish")
        graph.add_edge("publish", END)

        saver = MemorySaver()
        compiled = graph.compile(
            checkpointer=saver,
            interrupt_after=["review"],
        )

        # 第一步：review 执行完暂停
        r1 = compiled.invoke({}, config={"thread_id": "t1"})
        assert r1 == {"draft": "需要审批的内容"}

        # 人工审批：批准
        compiled.update_state(
            config={"thread_id": "t1"},
            values={"approved": True},
        )

        # 恢复执行：publish 看到 approved=True
        r2 = compiled.invoke(None, config={"thread_id": "t1"})
        assert r2 == {"draft": "需要审批的内容", "approved": True, "published": True}

    def test_update_state_overwrites_existing_key(self):
        """
        人工修改已有 key 的值，然后继续。
        """
        from simple_langgraph import StateGraph, START, END
        from simple_langgraph.checkpoint import MemorySaver

        def set_score(state: dict) -> dict:
            return {"score": 50}

        def adjust(state: dict) -> dict:
            return {"result": state["score"] * 2}

        graph = StateGraph()
        graph.add_node("set_score", set_score)
        graph.add_node("adjust", adjust)
        graph.add_edge(START, "set_score")
        graph.add_edge("set_score", "adjust")
        graph.add_edge("adjust", END)

        saver = MemorySaver()
        compiled = graph.compile(
            checkpointer=saver,
            interrupt_after=["set_score"],
        )

        r1 = compiled.invoke({}, config={"thread_id": "t1"})
        assert r1 == {"score": 50}

        # 人工把 score 改成 100
        compiled.update_state(
            config={"thread_id": "t1"},
            values={"score": 100},
        )

        r2 = compiled.invoke(None, config={"thread_id": "t1"})
        assert r2 == {"score": 100, "result": 200}


# ============================================================
# 测试 4: 条件边 + 断点
# ============================================================
class TestInterruptWithConditionalEdges:

    def test_interrupt_in_conditional_graph(self):
        """
        条件路由图 + 断点：
        START → classify → (条件边) → approve / reject
        在 approve 前断点，人工审核后继续。
        """
        from simple_langgraph import StateGraph, START, END
        from simple_langgraph.checkpoint import MemorySaver

        def classify(state: dict) -> dict:
            return {"label": "positive"}

        def approve(state: dict) -> dict:
            return {"status": "approved"}

        def reject(state: dict) -> dict:
            return {"status": "rejected"}

        def router(state: dict) -> str:
            return "approve" if state["label"] == "positive" else "reject"

        graph = StateGraph()
        graph.add_node("classify", classify)
        graph.add_node("approve", approve)
        graph.add_node("reject", reject)
        graph.add_edge(START, "classify")
        graph.add_conditional_edges("classify", router)
        graph.add_edge("approve", END)
        graph.add_edge("reject", END)

        saver = MemorySaver()
        compiled = graph.compile(
            checkpointer=saver,
            interrupt_before=["approve", "reject"],
        )

        # classify 执行完，approve/reject 前暂停
        r1 = compiled.invoke({}, config={"thread_id": "t1"})
        assert r1 == {"label": "positive"}

        # 人工修改 label
        compiled.update_state(
            config={"thread_id": "t1"},
            values={"label": "negative"},
        )

        # 恢复：路由到 reject
        r2 = compiled.invoke(None, config={"thread_id": "t1"})
        assert r2 == {"label": "negative", "status": "rejected"}


# ============================================================
# 测试 5: 边界情况
# ============================================================
class TestInterruptEdgeCases:

    def test_no_breakpoints_runs_to_end(self):
        """没有断点，正常执行到底"""
        from simple_langgraph import StateGraph, START, END
        from simple_langgraph.checkpoint import MemorySaver

        def node_a(state: dict) -> dict:
            return {"a": 1}

        graph = StateGraph()
        graph.add_node("a", node_a)
        graph.add_edge(START, "a")
        graph.add_edge("a", END)

        saver = MemorySaver()
        compiled = graph.compile(checkpointer=saver)

        result = compiled.invoke({}, config={"thread_id": "t1"})
        assert result == {"a": 1}

    def test_interrupt_after_last_node(self):
        """最后一个节点后设断点，第一次就执行完但暂停"""
        from simple_langgraph import StateGraph, START, END
        from simple_langgraph.checkpoint import MemorySaver

        def node_a(state: dict) -> dict:
            return {"a": 1}

        graph = StateGraph()
        graph.add_node("a", node_a)
        graph.add_edge(START, "a")
        graph.add_edge("a", END)

        saver = MemorySaver()
        compiled = graph.compile(
            checkpointer=saver,
            interrupt_after=["a"],
        )

        r1 = compiled.invoke({}, config={"thread_id": "t1"})
        assert r1 == {"a": 1}

        # 恢复：已经到 END，直接返回
        r2 = compiled.invoke(None, config={"thread_id": "t1"})
        assert r2 == {"a": 1}

    def test_invoke_fresh_input_on_resumed_thread(self):
        """
        恢复时传了新的 input（不是 None），
        应该被忽略，用 checkpoint 的 state 继续。
        """
        from simple_langgraph import StateGraph, START, END
        from simple_langgraph.checkpoint import MemorySaver

        def node_a(state: dict) -> dict:
            return {"a": 1}

        def node_b(state: dict) -> dict:
            return {"b": 2}

        graph = StateGraph()
        graph.add_node("a", node_a)
        graph.add_node("b", node_b)
        graph.add_edge(START, "a")
        graph.add_edge("a", "b")
        graph.add_edge("b", END)

        saver = MemorySaver()
        compiled = graph.compile(
            checkpointer=saver,
            interrupt_after=["a"],
        )

        r1 = compiled.invoke({"input": "hello"}, config={"thread_id": "t1"})
        assert r1 == {"input": "hello", "a": 1}

        # 恢复时传了新 input，应该被忽略
        r2 = compiled.invoke({"input": "ignored"}, config={"thread_id": "t1"})
        assert r2 == {"input": "hello", "a": 1, "b": 2}

    def test_update_state_without_checkpointer_raises(self):
        """没有 checkpointer 时调用 update_state → 报错"""
        from simple_langgraph import StateGraph, START, END

        graph = StateGraph()
        graph.add_node("a", lambda s: {})
        graph.add_edge(START, "a")
        graph.add_edge("a", END)

        compiled = graph.compile()

        with pytest.raises(ValueError, match="checkpointer"):
            compiled.update_state(
                config={"thread_id": "t1"},
                values={"x": 1},
            )


# ============================================================
# 测试 6: 扇出 + 断点（任务队列模型）
# ============================================================
class TestFanOutInterrupt:

    def test_fanout_interrupt_before_resumes_all(self):
        """
        菱形图：A → [B, C] → D，interrupt_before=[B, C]
        第一次：执行 A，在 B 和 C 之前停住
        恢复：B 和 C 都应该执行，然后 D 执行
        """
        from simple_langgraph import StateGraph, START, END
        from simple_langgraph.checkpoint import MemorySaver

        graph = StateGraph()
        graph.add_node("a", lambda s: {"steps": s.get("steps", []) + ["a"]})
        graph.add_node("b", lambda s: {"steps": s.get("steps", []) + ["b"]})
        graph.add_node("c", lambda s: {"steps": s.get("steps", []) + ["c"]})
        graph.add_node("d", lambda s: {"steps": s.get("steps", []) + ["d"]})
        graph.add_edge(START, "a")
        graph.add_edge("a", "b")
        graph.add_edge("a", "c")
        graph.add_edge("b", "d")
        graph.add_edge("c", "d")
        graph.add_edge("d", END)

        app = graph.compile(
            checkpointer=MemorySaver(),
            interrupt_before=["b", "c"],
        )

        # 第一次：执行 A，停在 B 和 C 之前
        r1 = app.invoke({"steps": []}, config={"thread_id": "t1"})
        assert r1["steps"] == ["a"]

        # 恢复：B 和 C 都执行，然后 D
        r2 = app.invoke(None, config={"thread_id": "t1"})
        assert set(r2["steps"]) == {"a", "b", "c", "d"}

    def test_fanout_interrupt_after_resumes_all_successors(self):
        """
        扇出：START → A → [B, C]，interrupt_after=[A]
        第一次：执行 A，停住
        恢复：B 和 C 都应该执行
        """
        from simple_langgraph import StateGraph, START, END
        from simple_langgraph.checkpoint import MemorySaver

        graph = StateGraph()
        graph.add_node("a", lambda s: {"steps": s.get("steps", []) + ["a"]})
        graph.add_node("b", lambda s: {"steps": s.get("steps", []) + ["b"]})
        graph.add_node("c", lambda s: {"steps": s.get("steps", []) + ["c"]})
        graph.add_edge(START, "a")
        graph.add_edge("a", "b")
        graph.add_edge("a", "c")
        graph.add_edge("b", END)
        graph.add_edge("c", END)

        app = graph.compile(
            checkpointer=MemorySaver(),
            interrupt_after=["a"],
        )

        # 第一次：执行 A，停住
        r1 = app.invoke({"steps": []}, config={"thread_id": "t1"})
        assert r1["steps"] == ["a"]

        # 恢复：B 和 C 都执行
        r2 = app.invoke(None, config={"thread_id": "t1"})
        assert set(r2["steps"]) == {"a", "b", "c"}

    def test_fanout_interrupt_before_one_of_two_saves_wave(self):
        """
        扇出：A → [B, C]，interrupt_before=[B]（只拦 B，不拦 C）
        第一次：B 被拦，但整个波次应该保存（C 也保存）
        恢复：B 和 C 都执行
        """
        from simple_langgraph import StateGraph, START, END
        from simple_langgraph.checkpoint import MemorySaver

        graph = StateGraph()
        graph.add_node("a", lambda s: {"steps": s.get("steps", []) + ["a"]})
        graph.add_node("b", lambda s: {"steps": s.get("steps", []) + ["b"]})
        graph.add_node("c", lambda s: {"steps": s.get("steps", []) + ["c"]})
        graph.add_node("d", lambda s: {"steps": s.get("steps", []) + ["d"]})
        graph.add_edge(START, "a")
        graph.add_edge("a", "b")
        graph.add_edge("a", "c")
        graph.add_edge("b", "d")
        graph.add_edge("c", "d")
        graph.add_edge("d", END)

        app = graph.compile(
            checkpointer=MemorySaver(),
            interrupt_before=["b"],
        )

        # 第一次：A 执行，B 被拦（C 也应该在 checkpoint 里）
        r1 = app.invoke({"steps": []}, config={"thread_id": "t1"})
        assert r1["steps"] == ["a"]

        # 恢复：B 和 C 都执行，然后 D
        r2 = app.invoke(None, config={"thread_id": "t1"})
        assert set(r2["steps"]) == {"a", "b", "c", "d"}
