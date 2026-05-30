"""
Milestone 4 测试：Checkpointer 持久化 + 时间旅行

TDD RED 阶段：测试 checkpointer 的保存、加载、历史回溯。
"""

import pytest


# ============================================================
# 测试 1: MemorySaver — 基本 save / load
# ============================================================
class TestMemorySaverBasic:

    def test_save_and_load_checkpoint(self):
        """
        保存一个 checkpoint，然后通过 thread_id 加载回来。
        """
        from simple_langgraph.checkpoint import MemorySaver

        saver = MemorySaver()
        saver.save("thread-1", {"step": 0, "value": "hello"}, metadata={"node": "start"})

        cp = saver.load("thread-1")
        assert cp is not None
        assert cp["state"] == {"step": 0, "value": "hello"}
        assert cp["metadata"]["node"] == "start"

    def test_load_nonexistent_returns_none(self):
        """加载一个不存在的 thread_id → 返回 None"""
        from simple_langgraph.checkpoint import MemorySaver

        saver = MemorySaver()
        assert saver.load("no-such-thread") is None

    def test_save_overwrites_latest(self):
        """同一个 thread_id 多次 save，load 返回最新的"""
        from simple_langgraph.checkpoint import MemorySaver

        saver = MemorySaver()
        saver.save("t1", {"step": 1}, metadata={"node": "a"})
        saver.save("t1", {"step": 2}, metadata={"node": "b"})

        cp = saver.load("t1")
        assert cp["state"]["step"] == 2
        assert cp["metadata"]["node"] == "b"

    def test_different_threads_isolated(self):
        """不同 thread_id 的 checkpoint 互不干扰"""
        from simple_langgraph.checkpoint import MemorySaver

        saver = MemorySaver()
        saver.save("t1", {"data": "one"})
        saver.save("t2", {"data": "two"})

        assert saver.load("t1")["state"]["data"] == "one"
        assert saver.load("t2")["state"]["data"] == "two"


# ============================================================
# 测试 2: 历史记录 — 时间旅行
# ============================================================
class TestCheckpointHistory:

    def test_list_history_returns_all_versions(self):
        """
        同一个 thread_id 多次 save，
        list_history 返回所有历史版本（最新在前）。
        """
        from simple_langgraph.checkpoint import MemorySaver

        saver = MemorySaver()
        saver.save("t1", {"step": 1}, metadata={"node": "a"})
        saver.save("t1", {"step": 2}, metadata={"node": "b"})
        saver.save("t1", {"step": 3}, metadata={"node": "c"})

        history = saver.list_history("t1")

        assert len(history) == 3
        # 最新在前
        assert history[0]["state"]["step"] == 3
        assert history[1]["state"]["step"] == 2
        assert history[2]["state"]["step"] == 1

    def test_list_history_empty_for_nonexistent(self):
        """不存在的 thread_id → 空列表"""
        from simple_langgraph.checkpoint import MemorySaver

        saver = MemorySaver()
        assert saver.list_history("nope") == []

    def test_history_has_metadata(self):
        """每个历史条目都带 metadata"""
        from simple_langgraph.checkpoint import MemorySaver

        saver = MemorySaver()
        saver.save("t1", {"v": 1}, metadata={"node": "step1", "step": 0})
        saver.save("t1", {"v": 2}, metadata={"node": "step2", "step": 1})

        history = saver.list_history("t1")
        assert history[0]["metadata"]["node"] == "step2"
        assert history[1]["metadata"]["node"] == "step1"

    def test_history_has_timestamp(self):
        """每个历史条目都有一个递增的 checkpoint_id"""
        from simple_langgraph.checkpoint import MemorySaver

        saver = MemorySaver()
        saver.save("t1", {"v": 1})
        saver.save("t1", {"v": 2})
        saver.save("t1", {"v": 3})

        history = saver.list_history("t1")
        ids = [h["checkpoint_id"] for h in history]
        # 最新在前，所以 id 递减
        assert ids[0] > ids[1] > ids[2]


# ============================================================
# 测试 3: 带 Checkpointer 的图执行
# ============================================================
class TestGraphWithCheckpointer:

    def test_invoke_saves_checkpoints(self):
        """
        编译时传入 checkpointer，
        invoke 执行完后可以通过 get_state 拿到最终状态。
        """
        from simple_langgraph import StateGraph, START, END
        from simple_langgraph.checkpoint import MemorySaver

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

        saver = MemorySaver()
        compiled = graph.compile(checkpointer=saver)
        result = compiled.invoke({"input": "test"}, config={"thread_id": "t1"})

        assert result == {"input": "test", "a": 1, "b": 2}

        # 通过 checkpointer 拿到最终状态
        cp = saver.load("t1")
        assert cp["state"] == {"input": "test", "a": 1, "b": 2}

    def test_invoke_with_loop_saves_each_step(self):
        """
        循环图中，每次循环都保存 checkpoint。
        历史记录可以看到每一轮的 state。
        """
        from simple_langgraph import StateGraph, START, END
        from simple_langgraph.checkpoint import MemorySaver

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

        saver = MemorySaver()
        compiled = graph.compile(checkpointer=saver)
        compiled.invoke({}, config={"thread_id": "loop1"})

        history = saver.list_history("loop1")
        # 应该有 3 条记录（每次 count 执行后保存）
        assert len(history) == 3
        assert history[0]["state"]["counter"] == 3
        assert history[1]["state"]["counter"] == 2
        assert history[2]["state"]["counter"] == 1

    def test_get_state_convenience_method(self):
        """
        CompiledGraph.get_state(config) 直接返回最新 state。
        """
        from simple_langgraph import StateGraph, START, END
        from simple_langgraph.checkpoint import MemorySaver

        def enrich(state: dict) -> dict:
            return {"enriched": True}

        graph = StateGraph()
        graph.add_node("enrich", enrich)
        graph.add_edge(START, "enrich")
        graph.add_edge("enrich", END)

        saver = MemorySaver()
        compiled = graph.compile(checkpointer=saver)
        compiled.invoke({"data": "x"}, config={"thread_id": "t1"})

        state = compiled.get_state(config={"thread_id": "t1"})
        assert state == {"data": "x", "enriched": True}

    def test_get_state_history_convenience_method(self):
        """
        CompiledGraph.get_state_history(config) 返回所有历史 state。
        """
        from simple_langgraph import StateGraph, START, END
        from simple_langgraph.checkpoint import MemorySaver

        def step1(state: dict) -> dict:
            return {"step": 1}

        def step2(state: dict) -> dict:
            return {"step": 2}

        graph = StateGraph()
        graph.add_node("step1", step1)
        graph.add_node("step2", step2)
        graph.add_edge(START, "step1")
        graph.add_edge("step1", "step2")
        graph.add_edge("step2", END)

        saver = MemorySaver()
        compiled = graph.compile(checkpointer=saver)
        compiled.invoke({}, config={"thread_id": "t1"})

        states = compiled.get_state_history(config={"thread_id": "t1"})
        assert len(states) == 2
        assert states[0]["step"] == 2  # 最新
        assert states[1]["step"] == 1

    def test_no_checkpointer_get_state_returns_none(self):
        """没有 checkpointer 时 get_state 返回 None"""
        from simple_langgraph import StateGraph, START, END

        graph = StateGraph()
        graph.add_node("a", lambda s: {})
        graph.add_edge(START, "a")
        graph.add_edge("a", END)

        compiled = graph.compile()
        assert compiled.get_state(config={"thread_id": "t1"}) is None

    def test_invoke_without_config_still_works(self):
        """没有 config 时，不带 checkpointer 的图照常工作"""
        from simple_langgraph import StateGraph, START, END

        def step(state: dict) -> dict:
            return {"done": True}

        graph = StateGraph()
        graph.add_node("step", step)
        graph.add_edge(START, "step")
        graph.add_edge("step", END)

        compiled = graph.compile()
        result = compiled.invoke({"x": 1})
        assert result == {"x": 1, "done": True}

    def test_invoke_with_checkpointer_but_no_config_raises(self):
        """有 checkpointer 但 invoke 没传 config → 报错"""
        from simple_langgraph import StateGraph, START, END
        from simple_langgraph.checkpoint import MemorySaver

        graph = StateGraph()
        graph.add_node("a", lambda s: {})
        graph.add_edge(START, "a")
        graph.add_edge("a", END)

        saver = MemorySaver()
        compiled = graph.compile(checkpointer=saver)

        with pytest.raises(ValueError, match="thread_id"):
            compiled.invoke({"x": 1})
