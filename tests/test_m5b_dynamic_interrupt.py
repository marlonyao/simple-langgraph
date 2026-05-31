"""
M5b: 动态 interrupt() — 节点内部按条件中断

对比静态 interrupt_before/interrupt_after：
- 静态：编译时固定，compile(interrupt_before=[...])
- 动态：运行时按条件，节点内调用 interrupt(value)

核心 API：
- interrupt(value) — 节点内调用，暂停执行并传值给客户端
- Command(resume=value) — 恢复时传入值，interrupt() 返回该值
- 恢复时节点重新执行，interrupt() 返回 resume 值
"""
import pytest


class TestBasicDynamicInterrupt:
    def test_interrupt_pauses_before_node_completes(self):
        """
        节点内调用 interrupt() → 暂停，节点不执行完。

        graph: START → a → b → END
        node_b 调用 interrupt("请确认")，图暂停，返回 a 的 state。
        """
        from simple_langgraph import StateGraph, START, END
        from simple_langgraph.graph import interrupt
        from simple_langgraph.checkpoint import MemorySaver

        def node_a(state):
            return {"x": 1}

        def node_b(state):
            answer = interrupt("请确认")
            return {"answer": answer}

        graph = StateGraph()
        graph.add_node("a", node_a)
        graph.add_node("b", node_b)
        graph.add_edge(START, "a")
        graph.add_edge("a", "b")

        saver = MemorySaver()
        compiled = graph.compile(checkpointer=saver)

        # 第一次执行：在 node_b 的 interrupt 处暂停
        r1 = compiled.invoke({}, config={"thread_id": "t1"})
        assert r1 == {"x": 1}

    def test_interrupt_resume_returns_value(self):
        """
        Command(resume=value) 恢复 → interrupt() 返回该值 → 节点执行完。
        """
        from simple_langgraph import StateGraph, START, END
        from simple_langgraph.graph import interrupt, Command
        from simple_langgraph.checkpoint import MemorySaver

        def node_b(state):
            answer = interrupt("请确认")
            return {"answer": answer}

        graph = StateGraph()
        graph.add_node("b", node_b)
        graph.add_edge(START, "b")

        saver = MemorySaver()
        compiled = graph.compile(checkpointer=saver)

        # 第一次：暂停
        r1 = compiled.invoke({}, config={"thread_id": "t1"})
        assert r1 == {}

        # 恢复：传值
        r2 = compiled.invoke(Command(resume="yes"), config={"thread_id": "t1"})
        assert r2 == {"answer": "yes"}

    def test_interrupt_value_saved_in_checkpoint(self):
        """
        interrupt(value) 的 value 存在 checkpoint metadata 中，
        客户端可以通过 get_state 读取。
        """
        from simple_langgraph import StateGraph, START, END
        from simple_langgraph.graph import interrupt
        from simple_langgraph.checkpoint import MemorySaver

        def node_b(state):
            interrupt("请输入您的年龄")
            return {}

        graph = StateGraph()
        graph.add_node("b", node_b)
        graph.add_edge(START, "b")

        saver = MemorySaver()
        compiled = graph.compile(checkpointer=saver)

        compiled.invoke({}, config={"thread_id": "t1"})

        cp = saver.load("t1")
        # metadata 应该包含 interrupt 信息
        assert cp["metadata"]["interrupt"] == "dynamic"
        tasks = cp["metadata"]["pending_tasks"]
        assert len(tasks) == 1
        assert tasks[0]["node"] == "b"
        assert tasks[0]["interrupt_value"] == "请输入您的年龄"

    def test_full_flow_with_interrupt_in_middle(self):
        """
        完整流程：a → b(interrupt) → c
        第一次停在 b，恢复后 b 完成 → c 执行。
        """
        from simple_langgraph import StateGraph, START, END
        from simple_langgraph.graph import interrupt, Command
        from simple_langgraph.checkpoint import MemorySaver

        def node_a(state):
            return {"x": 1}

        def node_b(state):
            val = interrupt("需要审核")
            return {"approved": val}

        def node_c(state):
            return {"done": True}

        graph = StateGraph()
        graph.add_node("a", node_a)
        graph.add_node("b", node_b)
        graph.add_node("c", node_c)
        graph.add_edge(START, "a")
        graph.add_edge("a", "b")
        graph.add_edge("b", "c")

        saver = MemorySaver()
        compiled = graph.compile(checkpointer=saver)

        r1 = compiled.invoke({}, config={"thread_id": "t1"})
        assert r1 == {"x": 1}

        r2 = compiled.invoke(Command(resume="ok"), config={"thread_id": "t1"})
        assert r2 == {"x": 1, "approved": "ok", "done": True}


class TestConditionalInterrupt:
    def test_interrupt_only_when_risky(self):
        """
        条件中断：只有高风险才 interrupt。
        低风险直接通过，不中断。
        """
        from simple_langgraph import StateGraph, START, END
        from simple_langgraph.graph import interrupt, Command
        from simple_langgraph.checkpoint import MemorySaver

        def review(state):
            if state.get("risk", 0) > 0.8:
                answer = interrupt("高风险请求，需人工审核")
                return {"approved": answer}
            return {"approved": "auto"}

        graph = StateGraph()
        graph.add_node("review", review)
        graph.add_edge(START, "review")

        saver = MemorySaver()
        compiled = graph.compile(checkpointer=saver)

        # 低风险：不中断
        r1 = compiled.invoke({"risk": 0.3}, config={"thread_id": "safe"})
        assert r1["approved"] == "auto"

        # 高风险：中断
        r2 = compiled.invoke({"risk": 0.9}, config={"thread_id": "risky"})
        assert "approved" not in r2

        r3 = compiled.invoke(Command(resume="deny"), config={"thread_id": "risky"})
        assert r3["approved"] == "deny"

    def test_interrupt_with_none_resume(self):
        """
        interrupt 后用 Command(resume=None) 恢复，interrupt() 返回 None。
        """
        from simple_langgraph import StateGraph, START, END
        from simple_langgraph.graph import interrupt, Command
        from simple_langgraph.checkpoint import MemorySaver

        def node_a(state):
            val = interrupt("确认？")
            return {"result": val}

        graph = StateGraph()
        graph.add_node("a", node_a)
        graph.add_edge(START, "a")

        saver = MemorySaver()
        compiled = graph.compile(checkpointer=saver)

        compiled.invoke({}, config={"thread_id": "t1"})
        r = compiled.invoke(Command(resume=None), config={"thread_id": "t1"})
        assert r["result"] is None


class TestDynamicInterruptEdgeCases:
    def test_interrupt_without_checkpointer_raises(self):
        """
        没有 checkpointer 时调用 interrupt() → 运行时报错。
        """
        from simple_langgraph import StateGraph, START, END
        from simple_langgraph.graph import interrupt

        def node_a(state):
            interrupt("没有 checkpointer")
            return {}

        graph = StateGraph()
        graph.add_node("a", node_a)
        graph.add_edge(START, "a")

        compiled = graph.compile()

        with pytest.raises(RuntimeError, match="checkpointer"):
            compiled.invoke({})

    def test_resume_without_pending_returns_state(self):
        """
        没有待恢复的中断时，invoke(Command(resume=...)) 直接返回 state。
        """
        from simple_langgraph import StateGraph, START, END
        from simple_langgraph.graph import Command
        from simple_langgraph.checkpoint import MemorySaver

        def node_a(state):
            return {"x": 1}

        graph = StateGraph()
        graph.add_node("a", node_a)
        graph.add_edge(START, "a")

        saver = MemorySaver()
        compiled = graph.compile(checkpointer=saver)

        # 正常执行
        r1 = compiled.invoke({}, config={"thread_id": "t1"})
        assert r1 == {"x": 1}

        # 再次 invoke 带 Command(resume=...) → 已完成，直接返回
        r2 = compiled.invoke(Command(resume="ignored"), config={"thread_id": "t1"})
        assert r2 == {"x": 1}

    def test_interrupt_outside_node_raises(self):
        """
        interrupt() 在节点外调用 → RuntimeError。
        """
        from simple_langgraph.graph import interrupt

        with pytest.raises(RuntimeError, match="node"):
            interrupt("不在节点里")

    def test_multiple_interrupts_in_same_node(self):
        """
        同一个节点多次 interrupt — 按顺序恢复。
        第一个 interrupt 先答，第二个 interrupt 后答。
        """
        from simple_langgraph import StateGraph, START, END
        from simple_langgraph.graph import interrupt, Command
        from simple_langgraph.checkpoint import MemorySaver

        def node_a(state):
            name = interrupt("你叫什么？")
            age = interrupt("你多大？")
            return {"name": name, "age": age}

        graph = StateGraph()
        graph.add_node("a", node_a)
        graph.add_edge(START, "a")

        saver = MemorySaver()
        compiled = graph.compile(checkpointer=saver)

        # 第一次：停在第一个 interrupt
        r1 = compiled.invoke({}, config={"thread_id": "t1"})
        assert r1 == {}

        # 恢复第一个 interrupt → 停在第二个
        r2 = compiled.invoke(Command(resume="Alice"), config={"thread_id": "t1"})
        assert r2 == {}

        # 恢复第二个 interrupt → 完成
        r3 = compiled.invoke(Command(resume="25"), config={"thread_id": "t1"})
        assert r3 == {"name": "Alice", "age": "25"}

    def test_stream_with_dynamic_interrupt(self):
        """
        stream 模式下动态中断也正常工作。
        """
        from simple_langgraph import StateGraph, START, END
        from simple_langgraph.graph import interrupt, Command
        from simple_langgraph.checkpoint import MemorySaver

        def node_a(state):
            return {"x": 1}

        def node_b(state):
            answer = interrupt("确认？")
            return {"answer": answer}

        graph = StateGraph()
        graph.add_node("a", node_a)
        graph.add_node("b", node_b)
        graph.add_edge(START, "a")
        graph.add_edge("a", "b")

        saver = MemorySaver()
        compiled = graph.compile(checkpointer=saver)

        chunks = list(compiled.stream({}, config={"thread_id": "t1"}))
        # node_a 执行后 yield 一次，node_b interrupt 时 yield 当前 state
        # 所以收到 2 个 chunks（都是中断前的 state）
        assert len(chunks) == 2
        assert chunks[0] == {"x": 1}  # node_a 完成
        assert chunks[1] == {"x": 1}  # node_b 中断时的 state

        # 恢复
        chunks2 = list(compiled.stream(
            Command(resume="ok"), config={"thread_id": "t1"}
        ))
        assert chunks2[-1] == {"x": 1, "answer": "ok"}


class TestDynamicAndStaticCoexist:
    def test_static_and_dynamic_interrupt_in_same_graph(self):
        """
        静态 interrupt_before 和动态 interrupt() 共存。
        """
        from simple_langgraph import StateGraph, START, END
        from simple_langgraph.graph import interrupt, Command
        from simple_langgraph.checkpoint import MemorySaver

        def node_a(state):
            return {"x": 1}

        def node_b(state):
            return {"y": 2}

        def node_c(state):
            answer = interrupt("动态中断")
            return {"answer": answer}

        graph = StateGraph()
        graph.add_node("a", node_a)
        graph.add_node("b", node_b)
        graph.add_node("c", node_c)
        graph.add_edge(START, "a")
        graph.add_edge("a", "b")
        graph.add_edge("b", "c")

        saver = MemorySaver()
        compiled = graph.compile(
            checkpointer=saver,
            interrupt_before=["b"],
        )

        # 第一次：静态拦截在 b 前
        r1 = compiled.invoke({}, config={"thread_id": "t1"})
        assert r1 == {"x": 1}

        # 恢复静态 → 执行 b → c 的动态 interrupt
        r2 = compiled.invoke(None, config={"thread_id": "t1"})
        assert r2 == {"x": 1, "y": 2}

        # 恢复动态 → 完成
        r3 = compiled.invoke(Command(resume="done"), config={"thread_id": "t1"})
        assert r3 == {"x": 1, "y": 2, "answer": "done"}


class TestConcurrencySafety:
    """并发安全：多个图同时 interrupt，上下文互不干扰。"""

    def test_two_graphs_concurrent_interrupt_isolated(self):
        """
        两个图分别在不同线程中调用 interrupt()，各自 resume 值互不干扰。

        - 模块级 _exec_stack：所有线程共享同一个 list，后 push 的覆盖前一个
        - ContextVar：每个线程有独立副本，天然隔离
        """
        import threading
        from simple_langgraph import StateGraph, START, END
        from simple_langgraph.graph import interrupt, Command
        from simple_langgraph.checkpoint import MemorySaver

        results = {}
        errors = {}

        def make_node(label, resume_val):
            def node(state):
                answer = interrupt(f"{label} 中断")
                return {"answer": answer, "label": label}
            return node

        def run_graph(label, resume_val):
            try:
                graph = StateGraph()
                graph.add_node("n", make_node(label, resume_val))
                graph.add_edge(START, "n")

                saver = MemorySaver()
                compiled = graph.compile(checkpointer=saver)

                # 第一次执行 → interrupt
                r1 = compiled.invoke({}, config={"thread_id": f"{label}_t"})
                # 恢复 → 传入各自的 resume 值
                r2 = compiled.invoke(
                    Command(resume=resume_val),
                    config={"thread_id": f"{label}_t"},
                )
                results[label] = r2
            except Exception as e:
                errors[label] = e

        t1 = threading.Thread(target=run_graph, args=("A", "resume_A"))
        t2 = threading.Thread(target=run_graph, args=("B", "resume_B"))
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        assert not errors, f"线程报错: {errors}"
        assert results["A"]["answer"] == "resume_A"
        assert results["B"]["answer"] == "resume_B"

    def test_exec_context_thread_local_isolation(self):
        """
        直接验证：线程 A 设置执行上下文后，线程 B 的 interrupt() 读不到它。

        这是 _exec_stack vs ContextVar 的核心区别测试。
        模拟两个线程先后执行节点，如果用共享 list，
        线程 B 会在线程 A pop 之前读到 A 的上下文。
        """
        import threading
        from simple_langgraph.graph import interrupt

        read_ctx = {}
        errors = {}

        def thread_a():
            """线程 A：设置上下文后阻塞，让线程 B 有机会读到它"""
            try:
                # 模拟图执行设置上下文
                from simple_langgraph.graph import _exec_ctx
                token = _exec_ctx.set({"node": "A", "resume": ["val_A"], "interrupt_counter": 0})
                sync_a_ready.set()
                sync_a_continue.wait(timeout=3)
                _exec_ctx.reset(token)
            except Exception as e:
                errors["A"] = e

        def thread_b():
            """线程 B：试图读取 interrupt 上下文"""
            try:
                sync_a_ready.wait(timeout=3)
                # 此时线程 A 的上下文应该对 B 不可见
                try:
                    from simple_langgraph.graph import _exec_ctx
                    ctx = _exec_ctx.get()
                    read_ctx["B"] = ctx
                except LookupError:
                    read_ctx["B"] = None  # 正确：B 没有设置上下文
            except Exception as e:
                errors["B"] = e
            finally:
                sync_a_continue.set()

        sync_a_ready = threading.Event()
        sync_a_continue = threading.Event()

        ta = threading.Thread(target=thread_a)
        tb = threading.Thread(target=thread_b)
        ta.start()
        tb.start()
        ta.join(timeout=5)
        tb.join(timeout=5)

        assert not errors, f"线程报错: {errors}"
        # B 不应该读到 A 的上下文
        assert read_ctx.get("B") is None, \
            f"线程 B 不应读到线程 A 的上下文，但读到了: {read_ctx['B']}"

    def test_interrupt_outside_node_raises_even_after_graph_run(self):
        """
        图执行完毕后，interrupt() 不应泄漏上下文。
        模块级 _exec_stack 执行完如果没 pop 干净会泄漏。
        ContextVar 在 reset 后自动恢复空状态。
        """
        from simple_langgraph import StateGraph, START, END
        from simple_langgraph.graph import interrupt, GraphInterrupt
        from simple_langgraph.checkpoint import MemorySaver

        def node_a(state):
            return {"x": 1}

        graph = StateGraph()
        graph.add_node("a", node_a)
        graph.add_edge(START, "a")
        compiled = graph.compile(checkpointer=MemorySaver())
        compiled.invoke({}, config={"thread_id": "t1"})

        # 图执行完毕，interrupt 应该报 RuntimeError
        with pytest.raises(RuntimeError, match="interrupt"):
            interrupt("不应被调用")
